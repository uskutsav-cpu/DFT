from dataclasses import replace

import pytest

from refblind.datasets import Dataset
from refblind.schema import Calculation, Observable, Reference, Structure, Term
from refblind.tables import StudyRow


@pytest.fixture
def molecules():
    a = Structure(
        "h2-r", ("H", "H"), ((0.0, 0.0, 0.0), (0.0, 0.0, 0.74)), 0, 1, "synthetic test fixture"
    )
    b = replace(a, id="h2-ts", positions_angstrom=((0.0, 0.0, 0.0), (0.0, 0.0, 1.2)))
    return {a.id: a, b.id: b}


@pytest.fixture
def dataset(molecules):
    observable = Observable(
        "barrier1",
        "family1",
        "relative_energy",
        (Term("h2-r", -1), Term("h2-ts", 1)),
        Reference(1.0, "kcal/mol", "synthetic", "test", quality="synthetic"),
    )
    return Dataset("test", list(molecules.values()), [observable], {"synthetic": True})


@pytest.fixture
def calculations(molecules):
    return [
        Calculation(
            s.id,
            s.geometry_hash,
            "dft-a",
            "dft",
            "succeeded",
            -1.0 + i * 0.01,
            "hartree",
            teacher_id="teacher1",
            teacher_verified=True,
            settings_hash="settings1",
            cost=1.0,
            diagnostics={"scf_cycles": 5.0},
        )
        for i, s in enumerate(molecules.values())
    ]


@pytest.fixture
def row():
    return StudyRow(
        "a",
        "g",
        5.0,
        5.1,
        0.0,
        {"ensemble_std": 0.1, "cheap_screen": 0.9},
        {"fod": 1.0},
        {"mlip": 1.0, "dft": 2.0, "screen": 3.0, "diagnostics": 4.0, "high_level": 5.0},
        reference_quality="synthetic",
        high_level_value=0.2,
    )
