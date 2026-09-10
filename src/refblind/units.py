"""Explicit energy units. Constants are fixed, not inferred from magnitudes.

2018 CODATA values are retained for reproducibility. Report precision should be
limited by the calculation/reference, not the number of digits here.
"""

from __future__ import annotations

from enum import Enum

from .validation import finite

HARTREE_TO_EV = 27.211386245988
EV_TO_KJ_MOL = 96.48533212331002
KCAL_TO_KJ = 4.184
HARTREE_TO_KCAL_MOL = HARTREE_TO_EV * EV_TO_KJ_MOL / KCAL_TO_KJ
BOHR_TO_ANGSTROM = 0.529177210903


class EnergyUnit(str, Enum):
    EV = "eV"
    HARTREE = "hartree"
    KCAL_MOL = "kcal/mol"
    KJ_MOL = "kJ/mol"


_TO_EV = {
    EnergyUnit.EV: 1.0,
    EnergyUnit.HARTREE: HARTREE_TO_EV,
    EnergyUnit.KCAL_MOL: KCAL_TO_KJ / EV_TO_KJ_MOL,
    EnergyUnit.KJ_MOL: 1.0 / EV_TO_KJ_MOL,
}


def convert_energy(value: float, source: EnergyUnit | str, target: EnergyUnit | str) -> float:
    return finite(value, "energy") * _TO_EV[EnergyUnit(source)] / _TO_EV[EnergyUnit(target)]
