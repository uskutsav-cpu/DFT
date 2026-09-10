"""Render (never submit) a small SLURM array for an audited calculation plan."""

from __future__ import annotations

import argparse
import re
import shlex
from pathlib import Path

from refblind.execution import load_plan
from refblind.provenance import atomic_text
from refblind.validation import integer, positive


def render(
    plan_path: Path,
    output_root: Path,
    *,
    python_executable: Path,
    cpus: int = 1,
    memory_mb: int = 2000,
    max_jobs: int = 1,
    concurrent: int = 1,
    budget_per_task: float = 20.0,
    time_limit: str = "01:00:00",
) -> str:
    """One isolated local ledger per array task; no shared SQLite on network storage."""
    plan = load_plan(plan_path)
    for name, value in (
        ("cpus", cpus),
        ("memory_mb", memory_mb),
        ("max_jobs", max_jobs),
        ("concurrent", concurrent),
    ):
        integer(value, name, 1)
    positive(budget_per_task, "budget_per_task")
    if not re.fullmatch(r"\d{1,3}:[0-5]\d:[0-5]\d", time_limit):
        raise ValueError("SLURM time limit must be HHH:MM:SS")
    count = min(len(plan["jobs"]), max_jobs)
    for job in plan["jobs"][:count]:
        cfg = job["config"]
        if cfg["backend"] == "orca":
            from refblind.calculators.orca import OrcaConfig

            cfg = OrcaConfig(**{k: v for k, v in cfg.items() if k != "backend"}).to_dict()
        need_cpus = cfg.get("nprocs", cfg.get("threads", 1))
        need_mem = cfg.get("memory_budget_mb", cfg.get("memory_mb", 0))
        if need_cpus > cpus or need_mem > memory_mb:
            raise ValueError("scheduler allocation is smaller than the calculation configuration")
        if cfg.get("device", "cpu") not in {"cpu"}:
            raise ValueError(
                "this conservative SLURM renderer supports CPU jobs only; review GPU allocation separately"
            )
    q = shlex.quote
    lines = [
        "#!/usr/bin/env bash",
        "# Generated for review; not submitted by RefBlind.",
        "# Budget is PER TASK, not a global spending ceiling.",
        f"#SBATCH --array=0-{count - 1}%{min(concurrent, count)}",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={memory_mb}M",
        f"#SBATCH --time={time_limit}",
        "#SBATCH --job-name=refblind-pilot",
        "set -euo pipefail",
        ': "${SLURM_ARRAY_TASK_ID:?Run as a reviewed SLURM array}"',
        f"export OMP_NUM_THREADS={cpus}",
        f"export MKL_NUM_THREADS={cpus}",
        f"PYTHON={q(str(python_executable.resolve()))}",
        f"PLAN={q(str(plan_path.resolve()))}",
        f"ROOT={q(str(output_root.resolve()))}",
        '"$PYTHON" -m refblind run --plan "$PLAN" '
        '--output "$ROOT/task-$SLURM_ARRAY_TASK_ID" --job-index "$SLURM_ARRAY_TASK_ID" '
        f"--execute --limit 1 --budget {budget_per_task:g}",
        "",
    ]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--python", type=Path, required=True)
    p.add_argument("--script", type=Path, required=True)
    p.add_argument("--cpus", type=int, default=1)
    p.add_argument("--memory-mb", type=int, default=2000)
    p.add_argument("--max-jobs", type=int, default=1)
    p.add_argument("--concurrent", type=int, default=1)
    p.add_argument("--budget-per-task", type=float, default=20.0)
    p.add_argument("--time", default="01:00:00")
    a = p.parse_args()
    text = render(
        a.plan,
        a.output_root,
        python_executable=a.python,
        cpus=a.cpus,
        memory_mb=a.memory_mb,
        max_jobs=a.max_jobs,
        concurrent=a.concurrent,
        budget_per_task=a.budget_per_task,
        time_limit=a.time,
    )
    atomic_text(a.script, text, overwrite=False)
    print(f"Wrote {a.script}; inspect it before separately running sbatch. No job was submitted.")


if __name__ == "__main__":
    main()
