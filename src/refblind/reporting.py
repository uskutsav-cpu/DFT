"""Human-readable and figure outputs from versioned analysis artifacts."""

from __future__ import annotations

from pathlib import Path

from .provenance import read_json


def _fmt(value):
    return "undefined" if value is None else f"{value:.4f}"


def markdown_report(report: dict) -> str:
    title = (
        "SYNTHETIC SOFTWARE TEST — NOT CHEMISTRY RESULTS"
        if report["synthetic"]
        else "RefBlind benchmark report"
    )
    lines = [
        f"# {title}",
        "",
        f"Status: **{report['research_status']}**",
        "",
        f"Input rows: {report['input_rows']}; eligible: {report['eligible_rows']}; "
        f"held-out test: {report['test_rows']} rows in {report['test_groups']} groups.",
        "",
        report["evaluation_population"],
        "",
        "## Reference-blind audit",
        "",
        f"Flagged {report['reference_blind_count']} / {report['test_rows']} test observations "
        "under the configured observable-error definition. This is not automatically a mechanism reversal.",
        "",
        "## Held-out detection",
        "",
        "| Model / target | AUROC | Average precision | Brier |",
        "|---|---:|---:|---:|",
    ]
    for name, score in report["detection"].items():
        lines.append(
            f"| {name} | {_fmt(score['auroc'])} | {_fmt(score['average_precision'])} | {_fmt(score['brier'])} |"
        )
    lines.extend(
        [
            "",
            "## Policy replay",
            "",
            "Abstentions are unresolved and count as failures in eligible-test accuracy. This is conditional on the audited complete-case evaluation population.",
            "",
            "| Policy | Eligible-test accuracy | Coverage | Mean cost | HL fraction |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, score in report["policies"].items():
        lines.append(
            f"| {name} | {_fmt(score['accuracy_all_cases'])} | {_fmt(score['coverage'])} | "
            f"{_fmt(score['mean_cost'])} | {_fmt(score['high_level_call_fraction'])} |"
        )
    lines.extend(
        [
            "",
            f"Cost unit: `{report['cost_unit']}`. {report['cost_semantics']}",
            "",
            "## Limitations",
            "",
            *report["warnings"],
            "",
            "Excluded observations remain in metrics.json with reasons. Review denominators before publication.",
            "",
            f"Frozen artifact SHA-256: `{report['artifact_sha256']}`",
            "",
        ]
    )
    return "\n".join(lines)


def render_figures(results_dir: str | Path) -> list[str]:
    """Separate figures; use matplotlib defaults rather than hidden style settings."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(results_dir)
    report = read_json(root / "metrics.json")
    audit = read_json(root / "test-audit.json")
    suffix = " — SYNTHETIC TEST" if report["synthetic"] else ""
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    paths = []

    def save(name):
        path = figures / name
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        paths.append(str(path))

    plt.figure(figsize=(7, 5))
    plt.scatter(
        [abs(r["surrogate_error"]) for r in audit],
        [abs(r["reference_error"]) for r in audit],
        alpha=0.7,
    )
    plt.xlabel("Absolute surrogate error (kcal/mol)")
    plt.ylabel("Absolute reference-method error (kcal/mol)")
    plt.title("Paired error decomposition" + suffix)
    save("error-decomposition.png")
    plt.figure(figsize=(8, 5))
    for name, score in report["policies"].items():
        plt.scatter([score["mean_cost"]], [score["accuracy_all_cases"]], label=name)
    plt.xlabel(f"Mean cost ({report['cost_unit']})")
    plt.ylabel("Eligible-test accuracy (abstentions unresolved)")
    plt.title("Policy cost–accuracy" + suffix)
    plt.legend(fontsize=7)
    save("cost-accuracy.png")
    plt.figure(figsize=(7, 5))
    for name, curve in report["reference_risk_coverage"].items():
        plt.plot([r["coverage"] for r in curve], [r["risk"] for r in curve], label=name)
    plt.xlabel("Coverage")
    plt.ylabel("Reference-error event rate among accepted cases")
    plt.title("Reference-risk coverage" + suffix)
    plt.legend(fontsize=8)
    save("risk-coverage.png")
    return paths
