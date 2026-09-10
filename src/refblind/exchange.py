"""File-based exchange with ORCA jobs run manually, on SLURM, or via ChemRefine.

No private/unstable ChemRefine Python API is assumed. Matching generated input
files are required before importing outputs into a paired fixed-geometry study.
"""

from __future__ import annotations

from pathlib import Path

from .calculators.orca import (
    OrcaConfig,
    enforce_correlation_convergence,
    parse_engrad,
    parse_orca_output,
    render_orca_input,
)
from .execution import load_plan
from .provenance import file_digest
from .schema import CalcStatus, Calculation, Structure
from .units import EnergyUnit


def import_orca_plan(plan_path: str | Path, root: str | Path) -> list[Calculation]:
    plan = load_plan(plan_path)
    root = Path(root)
    records = []
    for job in plan["jobs"]:
        if job["config"]["backend"] != "orca":
            raise ValueError("ORCA exchange needs an ORCA job plan")
        config = OrcaConfig(**{k: v for k, v in job["config"].items() if k != "backend"})
        structure = Structure.from_dict(job["structure"])
        directory = root / job["id"]
        expected_input = render_orca_input(structure, config)
        input_file, output_file = directory / "input.inp", directory / "orca.out"
        if not output_file.is_file():
            records.append(
                Calculation(
                    structure.id,
                    structure.geometry_hash,
                    config.method_id,
                    config.fidelity,
                    CalcStatus.FAILED,
                    None,
                    settings_hash=config.fingerprint,
                    message="output missing",
                    cost=0.0,
                    teacher_id=config.teacher_id,
                    teacher_verified=config.teacher_verified,
                    provenance={"expected_job_id": job["id"]},
                )
            )
            continue
        if not input_file.is_file() or input_file.read_text(encoding="utf-8") != expected_input:
            raise ValueError(f"cannot import unmatched/changed input for job {job['id']}")
        for name, expected in config.asset_sha256.items():
            if file_digest(directory / name) != expected:
                raise ValueError(f"external basis/asset mismatch in {job['id']}")
        parsed = parse_orca_output(
            output_file.read_text(encoding="utf-8", errors="replace"),
            multiplicity=structure.multiplicity,
        )
        parsed = enforce_correlation_convergence(
            parsed, output_file.read_text(encoding="utf-8", errors="replace"), config
        )
        forces = None
        if parsed["status"] == CalcStatus.SUCCEEDED and (directory / "input.engrad").is_file():
            forces = parse_engrad((directory / "input.engrad").read_text(), len(structure.symbols))
        records.append(
            Calculation(
                structure.id,
                structure.geometry_hash,
                config.method_id,
                config.fidelity,
                parsed["status"],
                parsed["energy_hartree"],
                EnergyUnit.HARTREE,
                teacher_id=config.teacher_id,
                teacher_verified=config.teacher_verified,
                forces_ev_angstrom=forces,
                diagnostics=parsed["diagnostics"],
                cost=config.relative_cost,
                software="ORCA",
                software_version=parsed["software_version"],
                settings_hash=config.fingerprint,
                raw_sha256=file_digest(output_file),
                message=parsed["message"],
                provenance={
                    "input_sha256": file_digest(input_file),
                    "imported_job_id": job["id"],
                    "wall_time_not_recorded": True,
                    "cost_semantics": "configured estimate",
                    "teacher_audit_note": config.teacher_audit_note,
                },
            )
        )
    return records
