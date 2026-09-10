import math
from dataclasses import replace

import pytest

from refblind import decompose_error, reference_blind_failure
from refblind.datasets import (
    Dataset,
    load_calculations,
    read_turbomole,
    read_xyz,
    save_calculations,
)
from refblind.diagnostics import weighted_risk
from refblind.evaluation import ordering_accuracy
from refblind.observables import assemble, ensemble_observable_std, require_teacher_match
from refblind.provenance import atomic_text, digest, read_json, write_json
from refblind.schema import Reference, Structure, Term
from refblind.units import EnergyUnit, convert_energy
from refblind.validation import finite, integer, matrix, vector


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True, "1", None])
def test_nonfinite_rejected(value):
    with pytest.raises((ValueError, TypeError)):
        finite(value)


@pytest.mark.parametrize("value", [1.5, True, "2", float("nan")])
def test_integer_strict(value):
    with pytest.raises((ValueError, TypeError)):
        integer(value)


@pytest.mark.parametrize("unit", list(EnergyUnit))
def test_units_roundtrip(unit):
    x = convert_energy(2.0, "hartree", unit)
    assert convert_energy(x, unit, "hartree") == pytest.approx(2.0)
    assert convert_energy(-2.0, unit, unit) == -2.0


@pytest.mark.parametrize("bad", [[], [float("nan")], [[1, 2]], [float("inf")]])
def test_vector_validation(bad):
    with pytest.raises(ValueError):
        vector(bad)


@pytest.mark.parametrize("bad", [[], [1, 2], [[1, float("nan")]]])
def test_matrix_validation(bad):
    with pytest.raises(ValueError):
        matrix(bad)


def test_legacy_fail_closed():
    with pytest.raises(ValueError):
        decompose_error(float("nan"), 0, 0)
    with pytest.raises(ValueError):
        weighted_risk({"x": 0.2}, {"x": float("nan")})
    with pytest.raises(ValueError):
        ordering_accuracy([1, 2], [1, 2], tie_tol=float("nan"))
    with pytest.raises(ValueError):
        ordering_accuracy([[1, 2]], [[1, 2]])
    with pytest.raises(ValueError):
        reference_blind_failure(
            decompose_error(1, 1, 0), surrogate_abs_max=float("nan"), reference_abs_min=1
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"multiplicity": 2},
        {"charge": 0.1},
        {"multiplicity": 0},
        {"symbols": ("Fake", "H")},
        {"positions_angstrom": ((0, 0, 0), (0, 0, 0))},
        {"positions_angstrom": ((0, 0, 0), (1, 2, float("inf")))},
        {"positions_angstrom": ((0, 0), (0, 1))},
        {"source": ""},
        {"id": "../danger"},
    ],
)
def test_structure_strict(molecules, changes):
    with pytest.raises((ValueError, TypeError)):
        replace(molecules["h2-r"], **changes)


def test_geometry_hash_state_and_roundtrip(molecules):
    a = molecules["h2-r"]
    assert Structure.from_dict(a.to_dict()) == a
    assert a.geometry_hash != replace(a, multiplicity=3).geometry_hash
    assert a.geometry_hash == replace(a, id="another").geometry_hash
    assert a.geometry_hash != molecules["h2-ts"].geometry_hash


def test_balanced_observables(dataset, molecules):
    dataset.observables[0].validate_balance(molecules)
    with pytest.raises(ValueError):
        replace(
            dataset.observables[0], terms=(Term("h2-r", -2), Term("h2-ts", 1))
        ).validate_balance(molecules)
    with pytest.raises(ValueError):
        replace(
            dataset.observables[0], terms=(Term("missing", -1), Term("h2-ts", 1))
        ).validate_balance(molecules)
    with pytest.raises(ValueError):
        Term("x", 0)
    with pytest.raises(ValueError):
        Reference(1, "kcal/mol", "method", "source", quality="reviewed")


def test_dataset_roundtrip(tmp_path, dataset):
    p = tmp_path / "data.json"
    dataset.save(p)
    assert Dataset.load(p).fingerprint == dataset.fingerprint
    d = dataset.to_dict()
    d["structures"].append(d["structures"][0])
    with pytest.raises(ValueError):
        Dataset.from_dict(d)


def test_duplicate_json_and_atomic_files(tmp_path):
    p = tmp_path / "a.json"
    p.write_text('{"x":1,"x":2}')
    with pytest.raises(ValueError):
        read_json(p)
    p.write_text('{"x":NaN}')
    with pytest.raises(ValueError):
        read_json(p)
    with pytest.raises(ValueError):
        write_json(p, {"nan": float("nan")})
    assert digest({"b": 1, "a": 2}) == digest({"a": 2, "b": 1})
    atomic_text(tmp_path / "new", "x", overwrite=False)
    with pytest.raises(FileExistsError):
        atomic_text(tmp_path / "new", "y", overwrite=False)
    assert (tmp_path / "new").read_text() == "x"


def test_geometry_readers(tmp_path):
    kwargs = dict(structure_id="test", charge=0, multiplicity=1, source="test")
    p = tmp_path / "s.xyz"
    p.write_text("2\ncomment\nH 0 0 0\nH 0 0 0.74\n")
    xyz = read_xyz(p, **kwargs)
    assert xyz.positions_angstrom[1][2] == 0.74
    p.write_text(p.read_text() + "1\nextra\nH 0 0 0\n")
    with pytest.raises(ValueError):
        read_xyz(p, **kwargs)
    p = tmp_path / "coord"
    p.write_text("$coord\n0 0 0 h\n0 0 1.4 h\n$end\n")
    assert read_turbomole(p, **kwargs).positions_angstrom[1][2] == pytest.approx(
        1.4 * 0.529177210903
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "failed"},
        {"energy": None},
        {"cost": -1},
        {"wall_seconds": float("nan")},
        {"geometry_hash": "bad"},
        {"teacher_verified": "yes"},
        {"energy": float("inf")},
        {"forces_ev_angstrom": ((1, 2),)},
    ],
)
def test_calculation_validation(calculations, changes):
    with pytest.raises((ValueError, TypeError)):
        replace(calculations[0], **changes)


def test_observable_units_and_offsets(dataset, calculations, molecules):
    a = assemble(dataset.observables[0], calculations, molecules, method_id="dft-a")
    assert a.value == pytest.approx(convert_energy(0.01, "hartree", "kcal/mol"))
    assert a.cost == 2
    shifted = [replace(c, energy=c.energy + 99) for c in calculations]
    assert assemble(
        dataset.observables[0], shifted, molecules, method_id="dft-a"
    ).value == pytest.approx(a.value)
    b = replace(a, method_id="b", value=a.value + 1)
    assert ensemble_observable_std([a, b]) == pytest.approx(math.sqrt(0.5))
    with pytest.raises(ValueError):
        ensemble_observable_std([a, a])


@pytest.mark.parametrize(
    "change", ["missing", "geometry", "settings", "costunit", "forces", "failure", "duplicate"]
)
def test_bad_assembly_rejected(dataset, calculations, molecules, change):
    records = calculations.copy()
    if change == "missing":
        records = records[:1]
    if change == "geometry":
        records[0] = replace(records[0], geometry_hash="0" * 64)
    if change == "settings":
        records[0] = replace(records[0], settings_hash="wrong")
    if change == "costunit":
        records[0] = replace(records[0], cost_unit="hours")
    if change == "forces":
        records[0] = replace(records[0], forces_ev_angstrom=((0.0, 0.0, 0.0),))
    if change == "failure":
        records[0] = replace(records[0], status="failed", energy=None)
    if change == "duplicate":
        records.append(records[0])
    with pytest.raises(ValueError):
        assemble(dataset.observables[0], records, molecules, method_id="dft-a")


def test_teacher_audit(calculations):
    mlip = [replace(c, method_id="mlip1", fidelity="mlip") for c in calculations]
    assert require_teacher_match(mlip, calculations) == "teacher1"
    with pytest.raises(ValueError):
        require_teacher_match(mlip, [replace(c, teacher_verified=False) for c in calculations])
    with pytest.raises(ValueError):
        require_teacher_match(mlip, [replace(c, teacher_id="other") for c in calculations])


def test_calculation_roundtrip(tmp_path, calculations):
    p = tmp_path / "calc.json"
    save_calculations(p, calculations)
    assert load_calculations(p) == calculations
