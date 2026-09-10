"""Train/calibrate/select/freeze/test orchestration for an observable benchmark.

The development and final evaluation entry points are separate. Test labels are
not used to select thresholds, fit preprocessing, train models, or calibrate.
"""

from __future__ import annotations

import csv
import io
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .features import Stage, validate_feature_names
from .learning import RiskModel
from .metrics import (
    auroc,
    average_precision,
    classification_report,
    cluster_bootstrap,
    pareto_frontier,
    regression_report,
    risk_coverage,
)
from .policy import CostLedger, SequentialPolicy
from .provenance import atomic_text, digest, environment_record, write_json
from .replay import replay_row, summarize_replay
from .schema import ReferenceQuality
from .splits import SplitPlan, grouped_split
from .tables import StudyRow, StudyTable, target_label
from .validation import integer, nonnegative, positive, probability


@dataclass(frozen=True)
class StudyConfig:
    seed: int = 42
    surrogate_tolerance: float = 1.0
    reference_tolerance: float = 3.0
    decision_tolerance: float = 3.0
    cost_weight: float = 0.05
    cost_scale: float = 20.0
    l2: float = 1.0
    bootstrap_repetitions: int = 500
    exploratory: bool = False
    stage1_features: tuple[str, ...] = ("ensemble_std", "ood_distance", "cheap_screen")
    stage2_features: tuple[str, ...] = (
        "ensemble_std",
        "ood_distance",
        "cheap_screen",
        "homo_lumo_gap_ev",
        "spin_contamination",
        "scf_cycles",
        "fod",
    )
    thresholds: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)

    def __post_init__(self):
        integer(self.seed, "seed", 0)
        for name in (
            "surrogate_tolerance",
            "reference_tolerance",
            "decision_tolerance",
            "cost_weight",
        ):
            nonnegative(getattr(self, name), name)
        positive(self.cost_scale, "cost_scale")
        positive(self.l2, "l2")
        integer(self.bootstrap_repetitions, "bootstrap_repetitions", 2)
        validate_feature_names(list(self.stage1_features), Stage.MLIP)
        validate_feature_names(list(self.stage2_features), Stage.DFT)
        if not self.thresholds:
            raise ValueError("at least one validation threshold is required")
        for threshold in self.thresholds:
            probability(threshold, "threshold")
        if not isinstance(self.exploratory, bool):
            raise ValueError("exploratory must be boolean")

    def to_dict(self):
        return asdict(self)


def eligible_rows(table: StudyTable, config: StudyConfig) -> tuple[list[StudyRow], list[dict]]:
    rows, exclusions = [], []
    for row in table.rows:
        reasons = []
        if row.mlip is None or row.dft is None or row.reference is None:
            reasons.append("missing or failed paired calculation/reference")
        if row.metadata.get("duplicate_of"):
            reasons.append(f"duplicate observable of {row.metadata['duplicate_of']}")
        if row.reference_quality in {
            ReferenceQuality.QUESTIONABLE,
            ReferenceQuality.MULTIREFERENCE_REQUIRED,
        }:
            reasons.append("reference-quality exclusion")
        if row.reference_quality == ReferenceQuality.UNREVIEWED and not config.exploratory:
            reasons.append("reference needs review or explicit exploratory mode")
        if row.reference_quality == ReferenceQuality.SYNTHETIC and not table.metadata.get(
            "synthetic", False
        ):
            reasons.append("synthetic row in a table not marked synthetic")
        if not reasons:
            targets = labels_for([row], config)
            if any(value[0] is None for value in targets.values()):
                reasons.append("reference uncertainty makes a training target ambiguous")
        if reasons:
            exclusions.append({"id": row.id, "group_id": row.group_id, "reasons": reasons})
        else:
            rows.append(row)
    if not rows:
        raise ValueError(
            "no eligible paired rows; inspect calculation failures and reference review status"
        )
    return rows, exclusions


def labels_for(rows: list[StudyRow], config: StudyConfig) -> dict[str, list[int | None]]:
    return {
        "surrogate": [target_label(r.mlip, r.dft, config.surrogate_tolerance) for r in rows],
        "reference": [
            target_label(r.dft, r.reference, config.reference_tolerance, r.reference_uncertainty)
            for r in rows
        ],
        "total": [
            target_label(r.mlip, r.reference, config.decision_tolerance, r.reference_uncertainty)
            for r in rows
        ],
    }


def _model_rows(rows: list[StudyRow]) -> list[dict]:
    return [r.all_features for r in rows]


def _fit_models(
    train: list[StudyRow], calibration: list[StudyRow], config: StudyConfig
) -> dict[str, RiskModel]:
    train_labels, calibration_labels = labels_for(train, config), labels_for(calibration, config)
    names_without_screen = [n for n in config.stage1_features if n != "cheap_screen"]
    if not names_without_screen:
        raise ValueError("a genuine MLIP-only baseline feature is required")
    definitions = {
        "surrogate": (names_without_screen, Stage.MLIP, "surrogate"),
        "total": (list(config.stage1_features), Stage.MLIP, "total"),
        "reference_mlip_only": (names_without_screen, Stage.MLIP, "reference"),
        "reference_augmented": (list(config.stage2_features), Stage.DFT, "reference"),
    }
    models = {}
    for name, (features, stage, target) in definitions.items():
        model = RiskModel(features, stage, l2=config.l2)
        model.fit(_model_rows(train), train_labels[target], [r.id for r in train])
        model.calibrate(
            _model_rows(calibration), calibration_labels[target], [r.id for r in calibration]
        )
        models[name] = model
    return models


def _run_policy(
    rows: list[StudyRow],
    models: dict[str, RiskModel],
    definition: dict,
    config: StudyConfig,
    cost_unit: str,
):
    policy = SequentialPolicy(**definition.get("policy", {}))
    mode = definition.get("mode", "sequential")
    stage1 = definition.get("stage1", "total")
    stage2 = definition.get("stage2", "reference_augmented")
    p1 = models[stage1].predict(_model_rows(rows))
    use_screen = bool(definition.get("use_screen", False))
    auxiliary_names = {"fod", "method_disagreement"}
    use_diagnostics = bool(auxiliary_names & set(models[stage2].preprocessor.names))
    ledger = CostLedger(cost_unit=cost_unit)
    results = []
    for i, row in enumerate(rows):

        def later(row=row):
            return float(models[stage2].predict([row.all_features])[0])

        results.append(
            replay_row(
                row,
                policy,
                mlip_risk=float(p1[i]),
                reference_risk=later,
                tolerance=config.decision_tolerance,
                ledger=ledger,
                use_screen=use_screen,
                use_diagnostics=use_diagnostics,
                use_reference_screen="cheap_screen" in models[stage2].preprocessor.names,
                mode=mode,
            )
        )
    return results, summarize_replay(
        results, rows, cost_weight=config.cost_weight, cost_scale=config.cost_scale
    )


def fit_study(table: StudyTable, config: StudyConfig, output: str | Path) -> dict:
    """Develop on train/calibration/validation only and write a frozen artifact."""
    rows, exclusions = eligible_rows(table, config)
    split = grouped_split([r.id for r in rows], [r.group_id for r in rows], seed=config.seed)
    partitions = {p: [r for r in rows if split.assignments[r.id] == p] for p in split.fractions}
    models = _fit_models(partitions["train"], partitions["calibration"], config)
    selected = {
        "always_mlip": {"mode": "always_mlip"},
        "always_dft": {"mode": "always_dft"},
        "always_high_level_observed": {"mode": "always_high_level"},
    }
    validation_scores = []
    variants = {
        "uq_only": {"mode": "uq_only", "stage1": "surrogate", "use_screen": False},
        "uq_reference": {"mode": "sequential", "stage1": "surrogate", "use_screen": False},
        "sequential_reference": {
            "mode": "sequential",
            "stage1": "total",
            "use_screen": "cheap_screen" in config.stage1_features,
        },
        "screen_all_reference": {"mode": "sequential", "stage1": "total", "use_screen": False},
    }
    for name, base in variants.items():
        candidates = []
        first = (0.0,) if name == "screen_all_reference" else config.thresholds
        second = (1.0,) if name == "uq_only" else config.thresholds
        for t1 in first:
            for t2 in second:
                policy = SequentialPolicy(t1, t2, screen_all=name == "screen_all_reference")
                definition = {**base, "policy": policy.to_dict()}
                _, summary = _run_policy(
                    partitions["validation"], models, definition, config, table.cost_unit
                )
                score = {"name": name, "definition": definition, **summary}
                validation_scores.append(score)
                candidates.append(score)
        best = min(
            candidates, key=lambda x: (x["objective"], x["mean_cost"], digest(x["definition"]))
        )
        selected[name] = best["definition"]
    calibration_audit = {}
    for name, model in models.items():
        target = "reference" if name.startswith("reference") else name
        ytrain = labels_for(partitions["train"], config)[target]
        ycal = labels_for(partitions["calibration"], config)[target]
        calibration_audit[name] = {
            "train_n": len(ytrain),
            "train_positive": sum(ytrain),
            "calibration_n": len(ycal),
            "calibration_positive": sum(ycal),
            "training_single_class": len(set(ytrain)) == 1,
            "calibration_single_class": len(set(ycal)) == 1,
            "optimizer_converged": model.model.converged,
            "calibration_optimizer_converged": model.calibrator.converged,
        }
    content = {
        "schema_version": "1.0",
        "table_hash": table.fingerprint,
        "config": config.to_dict(),
        "split": split.to_dict(),
        "models": {k: m.to_dict() for k, m in models.items()},
        "selected_policies": selected,
        "validation_results": validation_scores,
        "exclusions": exclusions,
        "calibration_audit": calibration_audit,
        "test_used_for_selection": False,
        "synthetic": bool(table.metadata.get("synthetic", False)),
        "research_status": "synthetic-integration-test"
        if table.metadata.get("synthetic")
        else ("exploratory" if config.exploratory else "frozen-evaluation-pending"),
    }
    artifact = {**content, "artifact_sha256": digest(content)}
    write_json(output, artifact, overwrite=False)
    return artifact


def _verify_artifact(artifact: dict, table: StudyTable):
    content = {k: v for k, v in artifact.items() if k != "artifact_sha256"}
    if artifact.get("artifact_sha256") != digest(content):
        raise ValueError("frozen artifact hash mismatch")
    if artifact.get("schema_version") != "1.0" or artifact["table_hash"] != table.fingerprint:
        raise ValueError("frozen model is not paired with this exact study table")


def evaluate_study(table: StudyTable, artifact: dict, output_dir: str | Path) -> dict:
    _verify_artifact(artifact, table)
    config = StudyConfig(**artifact["config"])
    rows, exclusions = eligible_rows(table, config)
    split = SplitPlan(**artifact["split"])
    split.validate([r.id for r in rows], [r.group_id for r in rows])
    test = [r for r in rows if split.assignments[r.id] == "test"]
    models = {name: RiskModel.from_dict(value) for name, value in artifact["models"].items()}
    test_ids = {r.id for r in test}
    for model in models.values():
        if test_ids & (model.training_ids | model.calibration_ids):
            raise ValueError("test IDs leaked into model development")
    labels = labels_for(test, config)
    predictions = {name: model.predict(_model_rows(test)) for name, model in models.items()}
    detection = {}
    for name, probabilities in predictions.items():
        target = "reference" if name.startswith("reference") else name
        detection[name] = classification_report(labels[target], probabilities)
    # This is an explicit offline diagnostic. It is not used by the controller.
    raw_pairs = [(r, r.mlip_features.get("ensemble_std")) for r in test]
    known = [(r, score) for r, score in raw_pairs if score is not None]
    raw_uq = None
    if known:
        raw_rows = [r for r, _ in known]
        raw_labels = labels_for(raw_rows, config)
        scores = [s for _, s in known]
        raw_uq = {
            target: {
                "n": len(scores),
                "auroc": auroc(raw_labels[target], scores),
                "average_precision": average_precision(raw_labels[target], scores),
            }
            for target in ("surrogate", "reference", "total")
        }
    policy_results, summaries = {}, {}
    for name, definition in artifact["selected_policies"].items():
        results, summary = _run_policy(test, models, definition, config, table.cost_unit)
        policy_results[name] = results
        summaries[name] = summary
    names = list(summaries)
    frontier = pareto_frontier(
        [summaries[n]["mean_cost"] for n in names],
        [1 - summaries[n]["accuracy_all_cases"] for n in names],
    )
    paired_ci = None
    if len({r.group_id for r in test}) > 1:
        delta = [
            float(a.correct) - float(b.correct)
            for a, b in zip(policy_results["sequential_reference"], policy_results["always_dft"])
        ]
        paired_ci = cluster_bootstrap(
            delta,
            [r.group_id for r in test],
            repetitions=config.bootstrap_repetitions,
            seed=config.seed,
        )
    rb = [
        (abs(r.mlip - r.dft) <= config.surrogate_tolerance and labels["reference"][i] == 1)
        for i, r in enumerate(test)
    ]
    curves = {
        name: risk_coverage(labels["reference"], predictions[name])
        for name in ("reference_mlip_only", "reference_augmented")
    }
    report = {
        "evaluation_population": "Complete, eligible, grouped held-out test observables. Excluded raw observations are audited separately.",
        "schema_version": "1.0",
        "synthetic": artifact["synthetic"],
        "research_status": artifact["research_status"],
        "table_hash": table.fingerprint,
        "artifact_sha256": artifact["artifact_sha256"],
        "input_rows": len(table.rows),
        "eligible_rows": len(rows),
        "excluded_rows": exclusions,
        "test_rows": len(test),
        "test_groups": len({r.group_id for r in test}),
        "reference_blind_count": int(sum(rb)),
        "reference_blind_fraction": float(np.mean(rb)),
        "reference_uncertainty_missing_test_rows": sum(
            r.reference_uncertainty is None for r in test
        ),
        "detection": detection,
        "raw_ensemble_uq": raw_uq,
        "regression": {
            "mlip_vs_dft": regression_report([r.mlip for r in test], [r.dft for r in test]),
            "dft_vs_reference": regression_report(
                [r.dft for r in test], [r.reference for r in test]
            ),
            "mlip_vs_reference": regression_report(
                [r.mlip for r in test], [r.reference for r in test]
            ),
        },
        "policies": summaries,
        "pareto_policies": [names[i] for i in frontier],
        "paired_accuracy_difference_vs_always_dft": paired_ci,
        "reference_risk_coverage": curves,
        "cost_unit": table.cost_unit,
        "cost_semantics": table.metadata.get(
            "cost_semantics", "user-supplied costs; not independently verified"
        ),
        "warnings": [
            "Policy scores are empirical, not correctness guarantees.",
            "Missing high-level outputs produce abstentions, never oracle credit.",
            "Held-out groups support only the generalization represented by the supplied grouping.",
            "Screening bias: downstream reference-risk metrics are shown offline for all complete test rows.",
        ],
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "metrics.json", report)
    write_json(output / "environment.json", environment_record(Path.cwd()))
    write_json(output / "split.json", split.to_dict())
    write_json(output / "frozen-artifact.json", artifact)
    buffer = io.StringIO()
    fields = [
        "policy",
        "row_id",
        "group_id",
        "action",
        "prediction",
        "cost",
        "ran_dft",
        "ran_high_level",
        "reason",
        "absolute_error",
        "correct",
        "abstained",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for name, results in policy_results.items():
        writer.writerows({"policy": name, **r.to_dict()} for r in results)
    atomic_text(output / "policy-decisions.csv", buffer.getvalue())
    audit = [
        {
            "id": r.id,
            "group_id": r.group_id,
            "mlip": r.mlip,
            "dft": r.dft,
            "reference": r.reference,
            "reference_blind": bool(rb[i]),
            "surrogate_error": r.mlip - r.dft,
            "reference_error": r.dft - r.reference,
            "total_error": r.mlip - r.reference,
            **{f"risk_{name}": float(value[i]) for name, value in predictions.items()},
        }
        for i, r in enumerate(test)
    ]
    write_json(output / "test-audit.json", audit)
    from .reporting import markdown_report

    atomic_text(output / "REPORT.md", markdown_report(report))
    return report


def run_study(table: StudyTable, config: StudyConfig, output: str | Path) -> dict:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"use a new output directory; refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    frozen_path = output.with_name(output.name + "-frozen.json")
    artifact = fit_study(table, config, frozen_path)
    return evaluate_study(table, artifact, output)
