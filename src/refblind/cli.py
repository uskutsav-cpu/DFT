"""Command-line interface. Chemistry execution always needs explicit --execute."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

from .configuration import load_config
from .datasets import Dataset, load_calculations, save_calculations
from .errors import decompose_error
from .provenance import environment_record, freeze_protocol, read_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="refblind", description="Auditable MLIP / DFT reference-risk research"
    )
    parser.add_argument("--version", action="version", version="refblind 0.2.0")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="inspect local optional dependencies; no downloads")
    demo = sub.add_parser("demo", help="run the SYNTHETIC statistical integration test")
    demo.add_argument("--output", type=Path)
    demo.add_argument("--groups", type=int, default=60)
    demo.add_argument("--rows-per-group", type=int, default=4)
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--plots", action="store_true")
    validate = sub.add_parser("validate", help="validate a dataset or study table")
    validate.add_argument("path", type=Path)
    validate.add_argument("--table", action="store_true")
    importer = sub.add_parser(
        "import-gmtkn", help="import a local GMTKN55 subset; never execute .res"
    )
    importer.add_argument("--checkout", type=Path, required=True)
    importer.add_argument("--subset", default="BH76")
    importer.add_argument("--kind", default="barrier")
    importer.add_argument("--expected-commit")
    importer.add_argument("--output", type=Path, required=True)
    plan = sub.add_parser("plan", help="prepare fixed-geometry calculations without running them")
    plan.add_argument("--dataset", type=Path, required=True)
    plan.add_argument("--config", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run", help="execute/resume a prepared plan with explicit guards")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--execute", action="store_true")
    run.add_argument("--limit", type=int, default=1)
    run.add_argument("--budget", type=float, default=100.0)
    run.add_argument("--retry-failed", action="store_true")
    run.add_argument("--job-index", type=int)
    status = sub.add_parser("status", help="show local job ledger counts")
    status.add_argument("database", type=Path)
    parse = sub.add_parser("parse-orca", help="inspect one single-point output")
    parse.add_argument("output_file", type=Path)
    parse.add_argument("--multiplicity", type=int, default=1)
    exchange = sub.add_parser("import-orca", help="import matched plan/input/output directories")
    exchange.add_argument("--plan", type=Path, required=True)
    exchange.add_argument("--root", type=Path, required=True)
    exchange.add_argument("--output", type=Path, required=True)
    assemble = sub.add_parser("assemble", help="construct balanced observable-level study rows")
    assemble.add_argument("--dataset", type=Path, required=True)
    assemble.add_argument("--calculations", type=Path, required=True, action="append")
    assemble.add_argument("--mlip-method", required=True, action="append")
    assemble.add_argument("--dft-method", required=True)
    assemble.add_argument("--high-level-method")
    assemble.add_argument("--allow-unverified-teacher", action="store_true")
    assemble.add_argument("--high-level-cost", type=float, default=20.0)
    assemble.add_argument("--output", type=Path, required=True)
    attach = sub.add_parser("attach-features", help="attach separately computed, costed features")
    attach.add_argument("--table", type=Path, required=True)
    attach.add_argument("--attachments", type=Path, required=True)
    attach.add_argument("--output", type=Path, required=True)
    review = sub.add_parser("review-references", help="apply explicit human reference reviews")
    review.add_argument("--dataset", type=Path, required=True)
    review.add_argument("--reviews", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True)
    for name in ("fit", "study"):
        action = sub.add_parser(
            name, help="fit/freeze" if name == "fit" else "fit/freeze then evaluate held-out test"
        )
        action.add_argument("--table", type=Path, required=True)
        action.add_argument("--config", type=Path)
        action.add_argument("--exploratory", action="store_true")
        action.add_argument("--output", type=Path, required=True)
        if name == "study":
            action.add_argument("--plots", action="store_true")
    evaluate = sub.add_parser(
        "evaluate", help="evaluate the frozen policies on the untouched test partition"
    )
    evaluate.add_argument("--table", type=Path, required=True)
    evaluate.add_argument("--artifact", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--plots", action="store_true")
    plot = sub.add_parser("plot", help="render saved report figures")
    plot.add_argument("results", type=Path)
    freeze = sub.add_parser(
        "freeze", help="snapshot configuration and file hashes without overwriting"
    )
    freeze.add_argument("--config", type=Path, required=True)
    freeze.add_argument("--input", type=Path, action="append", default=[])
    freeze.add_argument("--output", type=Path, required=True)
    errors = sub.add_parser("errors", help="decompose one scalar observable in consistent units")
    errors.add_argument("--mlip", type=float, required=True)
    errors.add_argument("--dft", type=float, required=True)
    errors.add_argument("--high-level", type=float, required=True)
    return parser


def dispatch(args) -> dict:
    command = args.command
    if command == "attach-features":
        from .attachments import FeatureAttachment, attach_features
        from .tables import StudyTable

        output = attach_features(
            StudyTable.load(args.table),
            [FeatureAttachment(**v) for v in read_json(args.attachments)],
        )
        if args.output.exists():
            raise FileExistsError(args.output)
        output.save(args.output)
        return {"table": str(args.output), "rows": len(output.rows)}
    if command == "review-references":
        from .review import apply_reference_reviews

        output = apply_reference_reviews(Dataset.load(args.dataset), read_json(args.reviews))
        if args.output.exists():
            raise FileExistsError(args.output)
        output.save(args.output)
        return {"dataset": str(args.output), "reference_review_is_human_supplied": True}
    if command == "doctor":
        modules = {
            m: importlib.util.find_spec(m) is not None
            for m in ("numpy", "yaml", "ase", "pyscf", "matplotlib")
        }
        return {
            **environment_record(Path.cwd()),
            "optional_modules": modules,
            "orca_on_path": shutil.which("orca") is not None,
            "chemistry_backend_note": "No models downloaded; ORCA/MLIP/PySCF backends not validated by this check.",
        }
    if command == "demo":
        from .reporting import render_figures
        from .study import StudyConfig, run_study
        from .synthetic import synthetic_table

        out = args.output or Path("results") / (
            "synthetic-" + datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        table = synthetic_table(
            groups=args.groups, rows_per_group=args.rows_per_group, seed=args.seed
        )
        data_path = out.with_name(out.name + "-table.json")
        if out.exists() or data_path.exists():
            raise FileExistsError("use a fresh demo output path")
        table.save(data_path)
        report = run_study(table, StudyConfig(seed=args.seed), out)
        figures = render_figures(out) if args.plots else []
        return {
            "status": "SYNTHETIC SOFTWARE TEST ONLY",
            "output": str(out),
            "test_rows": report["test_rows"],
            "figures": figures,
            "empirical_chemistry_claims": False,
        }
    if command == "validate":
        if args.table:
            from .tables import StudyTable

            value = StudyTable.load(args.path)
            return {"valid": True, "rows": len(value.rows), "sha256": value.fingerprint}
        dataset = Dataset.load(args.path)
        return {
            "valid": True,
            "structures": len(dataset.structures),
            "observables": len(dataset.observables),
            "sha256": dataset.fingerprint,
        }
    if command == "import-gmtkn":
        from .gmtkn import import_subset

        if args.output.exists():
            raise FileExistsError(args.output)
        dataset = import_subset(
            args.checkout, args.subset, kind=args.kind, expected_commit=args.expected_commit
        )
        dataset.save(args.output)
        return {
            "output": str(args.output),
            "structures": len(dataset.structures),
            "observables": len(dataset.observables),
            "reference_review_required": True,
            "source_commit": dataset.metadata["source_commit"],
        }
    if command == "plan":
        from .execution import make_plan

        plan = make_plan(Dataset.load(args.dataset), load_config(args.config), args.output)
        return {"jobs": len(plan["jobs"]), "output": str(args.output), "executed": False}
    if command == "run":
        from .execution import execute_plan

        return execute_plan(
            args.plan,
            args.output,
            execute=args.execute,
            limit=args.limit,
            budget=args.budget,
            retry_failed=args.retry_failed,
            job_index=args.job_index,
        )
    if command == "status":
        from .jobstore import JobStore

        if not args.database.exists():
            raise FileNotFoundError(args.database)
        with JobStore(args.database) as store:
            return store.counts()
    if command == "parse-orca":
        from .calculators.orca import parse_orca_output

        return parse_orca_output(
            args.output_file.read_text(encoding="utf-8", errors="replace"),
            multiplicity=args.multiplicity,
        )
    if command == "import-orca":
        from .exchange import import_orca_plan

        if args.output.exists():
            raise FileExistsError(args.output)
        records = import_orca_plan(args.plan, args.root)
        save_calculations(args.output, records)
        return {"records": len(records), "output": str(args.output)}
    if command == "assemble":
        from .assembly import assemble_table

        if args.output.exists():
            raise FileExistsError(args.output)
        records = [r for path in args.calculations for r in load_calculations(path)]
        table = assemble_table(
            Dataset.load(args.dataset),
            records,
            mlip_methods=args.mlip_method,
            dft_method=args.dft_method,
            high_level_method=args.high_level_method,
            require_verified_teacher=not args.allow_unverified_teacher,
            high_level_cost=args.high_level_cost,
        )
        table.save(args.output)
        return {
            "rows": len(table.rows),
            "output": str(args.output),
            "teacher_audit_required": args.allow_unverified_teacher,
        }
    if command in {"fit", "study", "evaluate"}:
        from .reporting import render_figures
        from .study import StudyConfig, evaluate_study, fit_study, run_study
        from .tables import StudyTable

        table = StudyTable.load(args.table)
        if command == "evaluate":
            report = evaluate_study(table, read_json(args.artifact), args.output)
        else:
            settings = load_config(args.config) if args.config else {}
            if args.exploratory:
                settings["exploratory"] = True
            config = StudyConfig(**settings)
            if command == "fit":
                artifact = fit_study(table, config, args.output)
                return {
                    "artifact": str(args.output),
                    "sha256": artifact["artifact_sha256"],
                    "test_used": False,
                }
            report = run_study(table, config, args.output)
        figures = render_figures(args.output) if getattr(args, "plots", False) else []
        return {
            "output": str(args.output),
            "status": report["research_status"],
            "figures": figures,
            "test_rows": report["test_rows"],
            "synthetic": report["synthetic"],
            "requires_scientific_validation": True,
        }
    if command == "plot":
        from .reporting import render_figures

        return {"figures": render_figures(args.results)}
    if command == "freeze":
        return freeze_protocol(load_config(args.config), args.input, args.output)
    if command == "errors":
        from dataclasses import asdict

        return asdict(decompose_error(args.mlip, args.dft, args.high_level))
    raise ValueError(f"unknown command {command}")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Preserve the v0.1 `python -m refblind.cli --mlip ...` entry point.
    if "--mlip" in argv and argv and argv[0].startswith("--"):
        argv.insert(0, "errors")
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = dispatch(args)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 1 if isinstance(result, dict) and result.get("failed_this_run", 0) else 0
    except (ValueError, OSError, ImportError, KeyError) as exc:
        print(f"refblind: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
