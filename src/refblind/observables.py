"""Assemble atom/charge-balanced observables before comparing methods."""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum

import numpy as np

from .schema import CalcStatus, Calculation, Observable, Structure
from .units import EnergyUnit, convert_energy


@dataclass(frozen=True)
class ObservableResult:
    observable_id: str
    method_id: str
    value: float
    unit: EnergyUnit
    job_keys: tuple[str, ...]
    cost: float
    cost_unit: str


def assemble(
    observable: Observable,
    records: list[Calculation],
    structures: dict[str, Structure],
    *,
    method_id: str,
    unit: EnergyUnit | str = EnergyUnit.KCAL_MOL,
) -> ObservableResult:
    observable.validate_balance(structures)
    selected = [r for r in records if r.method_id == method_id]
    if len({r.structure_id for r in selected}) != len(selected):
        raise ValueError("duplicate results for one structure/method")
    index = {r.structure_id: r for r in selected}
    energies, costs, keys, cost_units, settings = [], [], [], set(), set()
    for term in observable.terms:
        if term.structure_id not in index:
            raise ValueError(f"missing calculation {term.structure_id}/{method_id}")
        record = index[term.structure_id]
        if record.status != CalcStatus.SUCCEEDED:
            raise ValueError(f"unusable calculation {term.structure_id}: {record.status.value}")
        structure = structures[term.structure_id]
        if record.geometry_hash != structure.geometry_hash:
            raise ValueError(f"geometry/state mismatch for {term.structure_id}")
        if record.forces_ev_angstrom is not None and len(record.forces_ev_angstrom) != len(
            structure.symbols
        ):
            raise ValueError("force atom count mismatch")
        energies.append(term.coefficient * convert_energy(record.energy, record.energy_unit, unit))
        costs.append(record.cost)
        cost_units.add(record.cost_unit)
        settings.add(record.settings_hash)
        keys.append(f"{record.geometry_hash}/{method_id}/{record.settings_hash}")
    if len(cost_units) != 1 or len(settings) != 1:
        raise ValueError("mixed cost units or settings within one observable")
    return ObservableResult(
        observable.id,
        method_id,
        fsum(energies),
        EnergyUnit(unit),
        tuple(keys),
        fsum(costs),
        cost_units.pop(),
    )


def ensemble_observable_std(values: list[ObservableResult]) -> float:
    """Disagreement on balanced observables, NOT unrelated absolute totals."""
    if len(values) < 2:
        raise ValueError("ensemble needs at least two members")
    if len({v.method_id for v in values}) != len(values):
        raise ValueError("duplicate ensemble members")
    if len({(v.observable_id, v.unit) for v in values}) != 1:
        raise ValueError("ensemble members must describe the same observable/unit")
    return float(np.std([v.value for v in values], ddof=1))


def require_teacher_match(mlip: list[Calculation], dft: list[Calculation]) -> str:
    """Reject a claimed teacher comparison unless metadata explicitly supports it."""
    if not mlip or not dft:
        raise ValueError("both MLIP and DFT records are required")
    teachers = {r.teacher_id for r in mlip + dft}
    if len(teachers) != 1 or None in teachers:
        raise ValueError("MLIP/DFT teacher identities do not match")
    if not all(r.teacher_verified for r in dft):
        raise ValueError("DFT teacher-equivalence audit has not been marked verified")
    return teachers.pop()
