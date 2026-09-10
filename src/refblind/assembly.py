"""Join species calculations to published observable-level references."""

from __future__ import annotations

from .datasets import Dataset
from .features import FEATURES, Stage
from .observables import assemble, ensemble_observable_std, require_teacher_match
from .schema import CalcStatus, Calculation, Fidelity
from .tables import StudyRow, StudyTable
from .units import EnergyUnit, convert_energy


def assemble_table(
    dataset: Dataset,
    records: list[Calculation],
    *,
    mlip_methods: list[str],
    dft_method: str,
    high_level_method: str | None = None,
    require_verified_teacher: bool = True,
    high_level_cost: float = 20.0,
) -> StudyTable:
    from .validation import nonnegative

    nonnegative(high_level_cost, "high_level_cost")
    if not mlip_methods or len(set(mlip_methods)) != len(mlip_methods):
        raise ValueError("unique MLIP method IDs required")
    if (
        dft_method in mlip_methods
        or high_level_method in mlip_methods
        or high_level_method == dft_method
    ):
        raise ValueError("fidelity method IDs must be distinct")
    if len({(r.structure_id, r.method_id) for r in records}) != len(records):
        raise ValueError("duplicate calculation records")
    mlip_records = [r for r in records if r.method_id in mlip_methods]
    dft_records = [r for r in records if r.method_id == dft_method]
    if any(r.fidelity != Fidelity.MLIP for r in mlip_records) or any(
        r.fidelity != Fidelity.DFT for r in dft_records
    ):
        raise ValueError("method/fidelity mismatch")
    if high_level_method and any(
        r.fidelity != Fidelity.HIGH_LEVEL for r in records if r.method_id == high_level_method
    ):
        raise ValueError("high-level method/fidelity mismatch")
    if require_verified_teacher:
        require_teacher_match(mlip_records, dft_records)
    cost_units = {r.cost_unit for r in records}
    if len(cost_units) != 1:
        raise ValueError("all calculation costs must use the same unit")
    cost_unit = next(iter(cost_units))
    index = {(r.structure_id, r.method_id): r for r in records}
    rows = []
    for observable in dataset.observables:
        failures = []

        def try_assemble(method):
            if method is None:
                return None
            try:
                return assemble(observable, records, dataset.structure_map, method_id=method)
            except ValueError as exc:
                failures.append({"method": method, "reason": str(exc)})
                return None

        ensemble = [try_assemble(method) for method in mlip_methods]
        dft = try_assemble(dft_method)
        hl = try_assemble(high_level_method)
        # Primary MLIP = explicitly first method, never silently swap after failure.
        primary = ensemble[0]
        valid_members = [e for e in ensemble if e is not None]
        uq = (
            ensemble_observable_std(valid_members)
            if len(valid_members) == len(ensemble) and len(ensemble) >= 2
            else None
        )
        diagnostics = {}
        molecules = [index.get((term.structure_id, dft_method)) for term in observable.terms]
        for name, spec in FEATURES.items():
            if spec.stage != Stage.DFT:
                continue
            if name in {"fod", "method_disagreement"}:
                # Separate electronic calculations must enter through costed feature attachments.
                diagnostics[name] = None
                continue
            values = [
                r.diagnostics.get(name)
                for r in molecules
                if r is not None and r.status == CalcStatus.SUCCEEDED
            ]
            # Do not turn partial missing diagnostics into deceptively low risk.
            if len(values) != len(molecules) or any(v is None for v in values):
                diagnostics[name] = None
            else:
                diagnostics[name] = min(values) if name == "homo_lumo_gap_ev" else max(values)
        operations = {}
        for stage, methods in (
            ("mlip", mlip_methods),
            ("dft", [dft_method]),
            ("high_level", [high_level_method] if high_level_method else []),
        ):
            ops = {}
            for term in observable.terms:
                for method in methods:
                    record = index.get((term.structure_id, method))
                    if record is not None:
                        # Include alias/state/settings identity to align cache and cost accounting.
                        ops[
                            f"{record.structure_id}/{record.geometry_hash}/{method}/{record.settings_hash}"
                        ] = record.cost
            if ops:
                operations[stage] = ops
        costs = {
            stage: sum(operations.get(stage, {}).values())
            for stage in ("mlip", "dft", "high_level")
        }
        if not high_level_method:
            costs["high_level"] = high_level_cost
        costs.update(screen=0.0, diagnostics=0.0)
        reference = observable.reference
        rows.append(
            StudyRow(
                observable.id,
                observable.group_id,
                primary.value if primary else None,
                dft.value if dft else None,
                convert_energy(reference.value, reference.unit, EnergyUnit.KCAL_MOL)
                if reference
                else None,
                {"ensemble_std": uq},
                diagnostics,
                costs,
                reference_quality=reference.quality if reference else "unreviewed",
                reference_uncertainty=convert_energy(
                    reference.uncertainty, reference.unit, EnergyUnit.KCAL_MOL
                )
                if reference and reference.uncertainty is not None
                else None,
                high_level_value=hl.value if hl else None,
                high_level_supported=high_level_method is not None,
                decision_group=observable.decision_group,
                operations=operations,
                metadata={
                    **observable.metadata,
                    "failures": failures,
                    "reference_source": reference.source if reference else None,
                    "reference_method": reference.method if reference else None,
                },
            )
        )
    return StudyTable(
        rows,
        {
            "dataset_hash": dataset.fingerprint,
            "dataset_metadata": dataset.metadata,
            "mlip_methods": mlip_methods,
            "dft_method": dft_method,
            "high_level_method": high_level_method,
            "teacher_audit_required": not require_verified_teacher,
            "synthetic": bool(dataset.metadata.get("synthetic", False)),
            "cost_semantics": "sum of supplied species costs; shared operation keys deduplicated in replay",
            "missing_high_level_semantics": "abstain, never replace with the reference label",
        },
        cost_unit=cost_unit,
    )
