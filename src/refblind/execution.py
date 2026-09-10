"""Fixed-geometry job planning and resumable, explicitly authorized execution."""

from __future__ import annotations

import importlib.metadata
import shutil
from dataclasses import replace
from pathlib import Path

from .calculators.orca import OrcaConfig, render_orca_input, run_orca
from .datasets import Dataset, save_calculations
from .jobstore import JobStore
from .provenance import digest, file_digest, read_json, write_json
from .schema import CalcStatus, Calculation, Structure
from .validation import integer, positive


def make_plan(dataset: Dataset, config: dict, output: str | Path) -> dict:
    backend = config.get("backend")
    if backend not in {"orca", "uma", "mace_omol", "orbmol", "pyscf"}:
        raise ValueError("unsupported backend")
    if "method_id" not in config:
        raise ValueError("method_id required")
    if backend == "orca":
        OrcaConfig(**{k: v for k, v in config.items() if k != "backend"})
    jobs = []
    for structure in dataset.structures:
        job = {"structure": structure.to_dict(), "config": config}
        job["id"] = digest(job)
        jobs.append(job)
    plan = {
        "schema_version": "1.0",
        "dataset_hash": dataset.fingerprint,
        "dataset_name": dataset.name,
        "jobs": jobs,
        "execution": "not_run",
        "synthetic": bool(dataset.metadata.get("synthetic", False)),
    }
    write_json(output, plan, overwrite=False)
    if backend == "orca":
        conf = OrcaConfig(**{k: v for k, v in config.items() if k != "backend"})
        from .provenance import atomic_text

        inputs = Path(output).parent / (Path(output).stem + "-inputs")
        for structure, job in zip(dataset.structures, jobs):
            atomic_text(
                inputs / f"{job['id']}.inp", render_orca_input(structure, conf), overwrite=False
            )
    return plan


def load_plan(path: str | Path) -> dict:
    plan = read_json(path)
    if plan.get("schema_version") != "1.0" or not plan.get("jobs"):
        raise ValueError("invalid or empty job plan")
    seen = set()
    for job in plan["jobs"]:
        expected = digest({"structure": job["structure"], "config": job["config"]})
        if job["id"] != expected or expected in seen:
            raise ValueError("job plan hash mismatch or duplicate job")
        seen.add(expected)
        Structure.from_dict(job["structure"])
    return plan


def runtime_identity(config: dict) -> dict:
    backend = config["backend"]
    if backend == "orca":
        executable = shutil.which(config.get("executable", "orca"))
        if executable is None:
            raise FileNotFoundError("ORCA is not installed/on PATH")
        return {"executable_sha256": file_digest(executable)}
    package = {
        "uma": "fairchem-core",
        "mace_omol": "mace-torch",
        "orbmol": "orb-models",
        "pyscf": "pyscf",
    }[backend]
    try:
        return {
            "package": package,
            "version": importlib.metadata.version(package),
            "checkpoint_sha256": config.get("checkpoint_sha256"),
        }
    except importlib.metadata.PackageNotFoundError as exc:
        raise ImportError(f"optional backend {package} is not installed") from exc


def execute_plan(
    path: str | Path,
    output_dir: str | Path,
    *,
    execute: bool = False,
    limit: int = 1,
    budget: float = 100.0,
    retry_failed: bool = False,
    job_index: int | None = None,
) -> dict:
    """Budget is a per-invocation configured estimate, not a billed-cost guarantee."""
    plan = load_plan(path)
    integer(limit, "limit", 1)
    positive(budget, "budget")
    if not execute:
        return {
            "dry_run": True,
            "jobs": len(plan["jobs"]),
            "will_execute": 0,
            "message": "No chemistry calculations or model downloads were started.",
        }
    jobs = plan["jobs"]
    if job_index is not None:
        integer(job_index, "job_index", 0)
        if job_index >= len(jobs):
            raise ValueError("job_index out of bounds")
        jobs = [jobs[job_index]]
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    spent, launched, cached, failures = 0.0, 0, 0, 0
    records, calculators = [], {}
    with JobStore(out / "jobs.sqlite") as store:
        for job in jobs:
            structure = Structure.from_dict(job["structure"])
            config = job["config"]
            identity = runtime_identity(config)
            key = store.register({"job": job, "runtime": identity})
            cache = store.cached_result(key)
            if cache is not None:
                records.append(Calculation.from_dict(read_json(cache)))
                cached += 1
                continue
            if retry_failed:
                store.retry(key)
            state = store.get(key)
            if state["status"] != "pending":
                if state["status"] == "failed" and state["result_path"]:
                    if file_digest(state["result_path"]) != state["result_sha256"]:
                        raise ValueError("failed result was altered")
                    records.append(Calculation.from_dict(read_json(state["result_path"])))
                continue
            estimate = float(config.get("relative_cost", 1.0))
            from .validation import nonnegative

            nonnegative(estimate, "relative_cost")
            if launched >= limit or spent + estimate > budget:
                continue
            token = store.claim(key)
            if token is None:
                continue
            launched += 1
            spent += estimate
            attempt = store.get(key)["attempts"]
            work = out / "raw" / key / f"attempt-{attempt}"
            try:
                backend = config["backend"]
                if backend == "orca":
                    conf = OrcaConfig(**{k: v for k, v in config.items() if k != "backend"})
                    result = run_orca(structure, conf, work)
                elif backend == "pyscf":
                    from .calculators.pyscf_backend import calculate_pyscf

                    result = calculate_pyscf(structure, config, work)
                else:
                    from .calculators.ase_backend import calculate_ase, make_calculator

                    fingerprint = digest(config)
                    if fingerprint not in calculators:
                        calculators[fingerprint] = make_calculator(config)
                    result = calculate_ase(
                        structure,
                        calculators[fingerprint],
                        method_id=config["method_id"],
                        config=config,
                    )
                result = replace(
                    result,
                    provenance={
                        **result.provenance,
                        "runtime_identity": identity,
                        "job_id": job["id"],
                        "attempt": attempt,
                    },
                )
                record_path = work / "result.json"
                write_json(record_path, result.to_dict(), overwrite=False)
                success = result.status == CalcStatus.SUCCEEDED
                store.finish(key, token, record_path, success=success, error=result.message)
                records.append(result)
                failures += not success
            except BaseException as exc:
                result = Calculation(
                    structure.id,
                    structure.geometry_hash,
                    config["method_id"],
                    config.get(
                        "fidelity",
                        "mlip" if config["backend"] in {"uma", "mace_omol", "orbmol"} else "dft",
                    ),
                    CalcStatus.FAILED,
                    None,
                    cost=estimate,
                    message=f"{type(exc).__name__}: {exc}",
                    provenance={
                        "job_id": job["id"],
                        "attempt": attempt,
                        "runtime_identity": identity,
                    },
                )
                record_path = work / "exception-result.json"
                write_json(record_path, result.to_dict(), overwrite=False)
                store.finish(key, token, record_path, success=False, error=result.message)
                records.append(result)
                failures += 1
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
        summary = {
            "dry_run": False,
            "launched": launched,
            "cached": cached,
            "failed_this_run": failures,
            "configured_cost_spent": spent,
            "budget_scope": "this invocation only",
            "ledger": store.counts(),
        }
    # Array workers use disjoint summary/record filenames.
    suffix = f"-{job_index}" if job_index is not None else ""
    save_calculations(out / f"calculations{suffix}.json", records)
    write_json(out / f"execution-summary{suffix}.json", summary)
    return summary
