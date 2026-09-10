import sys
import types

import numpy as np
import pytest

from refblind.calculators.ase_backend import atoms_from_structure, calculate_ase, make_calculator
from refblind.calculators.orca import (
    OrcaConfig,
    parse_engrad,
    parse_orbital_gap,
    parse_orca_output,
    render_orca_input,
    run_orca,
)
from refblind.datasets import load_calculations
from refblind.exchange import import_orca_plan
from refblind.execution import execute_plan, load_plan, make_plan
from refblind.jobstore import JobStore
from refblind.provenance import file_digest, read_json, write_json
from refblind.schema import CalcStatus

# Deliberately fabricated output fixture to test parser contracts. Not an ORCA result.
SUCCESS = """Program Version 6.1.0
SCF CONVERGED AFTER 12 CYCLES
Expectation value of <S**2> : 0.8000
HOMO-LUMO GAP: 2.5 eV
FINAL SINGLE POINT ENERGY -1.123456789
ORCA TERMINATED NORMALLY
"""


@pytest.fixture
def fake_orca(tmp_path):
    p = tmp_path / "fake-orca"
    p.write_text(f"#!{sys.executable}\nprint({SUCCESS!r})\n")
    p.chmod(0o755)
    return str(p)


def test_orca_success_parse():
    p = parse_orca_output(SUCCESS, multiplicity=2)
    assert p["status"] == CalcStatus.SUCCEEDED and p["energy_hartree"] == -1.123456789
    assert p["diagnostics"]["scf_cycles"] == 12
    assert p["diagnostics"]["spin_contamination"] == pytest.approx(0.05)
    assert p["diagnostics"]["homo_lumo_gap_ev"] == 2.5


@pytest.mark.parametrize(
    "text,status",
    [
        (SUCCESS.replace("ORCA TERMINATED NORMALLY", ""), CalcStatus.FAILED),
        (SUCCESS + "SCF NOT CONVERGED", CalcStatus.NONCONVERGED),
        (SUCCESS + "ORCA finished by error termination", CalcStatus.FAILED),
        (SUCCESS + "FINAL SINGLE POINT ENERGY -1.2\n", CalcStatus.UNSUPPORTED),
        (SUCCESS.replace("SCF CONVERGED AFTER 12 CYCLES", ""), CalcStatus.NONCONVERGED),
        (SUCCESS + SUCCESS, CalcStatus.FAILED),
    ],
)
def test_orca_never_accepts_incomplete_or_multiple(text, status):
    p = parse_orca_output(text)
    assert p["status"] == status and p["energy_hartree"] is None


def test_orbital_table_gap():
    s = """ORBITAL ENERGIES
----------------
SPIN UP ORBITALS
NO OCC E(Eh) E(eV)
0 1.0000 -0.4 -10.8845
1 0.0000 0.1 2.7211
SPIN DOWN ORBITALS
NO OCC E(Eh) E(eV)
0 1.0000 -0.2 -5.4423
1 0.0000 0.1 2.7211
MULLIKEN ANALYSIS
"""
    assert parse_orbital_gap(s) == pytest.approx(8.1634)
    assert parse_orbital_gap(s.replace("1.0000", "0.9990")) is None
    assert parse_orbital_gap(s + "N_FOD = 1.0") is None
    assert parse_orbital_gap(s + s) is None
    parsed = parse_orca_output(SUCCESS + "N_FOD = 0.9")
    assert parsed["diagnostics"]["fod"] == 0.9 and parsed["diagnostics"]["homo_lumo_gap_ev"] is None


@pytest.mark.parametrize(
    "changes",
    [
        {"nprocs": 8, "memory_budget_mb": 2000},
        {"header": "! Opt B3LYP"},
        {"header": "! NEB B3LYP"},
        {"header": "! x\n! y"},
        {"blocks": "%pal nprocs 100 end"},
        {"blocks": "* xyz 0 1"},
        {"teacher_verified": True},
        {"assets": {"basis.bas": "b"}},
        {"timeout_seconds": -1},
    ],
)
def test_orca_config_guards(changes):
    with pytest.raises((ValueError, TypeError)):
        OrcaConfig("test", **changes)


def test_orca_input_geometry(molecules):
    text = render_orca_input(molecules["h2-r"], OrcaConfig("test"))
    assert "* xyz 0 1" in text and "%pal nprocs 1 end" in text
    assert text.endswith("*\n")


def test_engrad_sign_and_shape():
    forces = parse_engrad("# synthetic engrad\n2\n-1.0\n0.1\n0\n0\n-0.1\n0\n0\n", 2)
    assert forces[0][0] == pytest.approx(-0.1 * 27.211386245988 / 0.529177210903)
    assert forces[1][0] == -forces[0][0]
    with pytest.raises(ValueError):
        parse_engrad("1\n-1\n0\n", 2)


def test_fake_executable_contract(tmp_path, molecules, fake_orca):
    config = OrcaConfig("test", executable=fake_orca)
    result = run_orca(molecules["h2-r"], config, tmp_path / "job")
    assert result.status == CalcStatus.SUCCEEDED and result.software == "ORCA"
    assert (tmp_path / "job/orca.out").is_file()
    assert result.raw_sha256 == file_digest(tmp_path / "job/orca.out")
    with pytest.raises(FileExistsError):
        run_orca(molecules["h2-r"], config, tmp_path / "job")


def test_fake_executable_timeout(tmp_path, molecules):
    executable = tmp_path / "sleep-test"
    executable.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(20)\n")
    executable.chmod(0o755)
    c = run_orca(
        molecules["h2-r"],
        OrcaConfig("test", executable=str(executable), timeout_seconds=0.05),
        tmp_path / "job",
    )
    assert c.status == CalcStatus.TIMEOUT and c.energy is None


def test_jobstore_claims_retry_and_cache(tmp_path):
    db = tmp_path / "jobs.sqlite"
    with JobStore(db) as a, JobStore(db) as b:
        key = a.register({"job": 1})
        assert b.register({"job": 1}) == key
        token = a.claim(key)
        assert token and b.claim(key) is None
        with pytest.raises(ValueError):
            b.mark_failed(key, "bad-token", "bad")
        a.mark_failed(key, token, "test failure")
        assert b.retry(key) and b.get(key)["status"] == "pending"
        token = b.claim(key)
        out = tmp_path / "result.json"
        write_json(out, {"test": True})
        b.finish(key, token, out, success=True)
        assert a.cached_result(key) == out
        assert a.counts() == {"succeeded": 1}
        out.write_text("{}")
        with pytest.raises(ValueError):
            a.cached_result(key)


def test_plan_dryrun_does_not_start_chemistry(tmp_path, dataset):
    plan = tmp_path / "plan.json"
    make_plan(dataset, {"backend": "orca", "method_id": "test"}, plan)
    assert len(load_plan(plan)["jobs"]) == 2
    assert execute_plan(plan, tmp_path / "runs")["dry_run"] is True
    assert not (tmp_path / "runs").exists()
    value = read_json(plan)
    value["jobs"][0]["config"]["method_id"] = "changed"
    write_json(plan, value)
    with pytest.raises(ValueError):
        load_plan(plan)


def test_invalid_config_writes_no_plan(tmp_path, dataset):
    path = tmp_path / "bad.json"
    with pytest.raises(ValueError):
        make_plan(dataset, {"backend": "orca", "method_id": "test", "header": "! OPT"}, path)
    assert not path.exists()


def test_resumable_execution_with_mocked_chemistry(tmp_path, dataset, fake_orca):
    plan = tmp_path / "plan.json"
    runs = tmp_path / "runs"
    make_plan(dataset, {"backend": "orca", "method_id": "test", "executable": fake_orca}, plan)
    first = execute_plan(plan, runs, execute=True, limit=1)
    assert first["launched"] == 1 and first["ledger"] == {"pending": 1, "succeeded": 1}
    second = execute_plan(plan, runs, execute=True, limit=1)
    assert second["launched"] == 1 and second["cached"] == 1
    third = execute_plan(plan, runs, execute=True, limit=1)
    assert third["launched"] == 0 and third["cached"] == 2
    assert len(load_calculations(runs / "calculations.json")) == 2


def test_failed_runs_preserved_on_resume(tmp_path, dataset):
    executable = tmp_path / "bad-orca"
    executable.write_text(f'#!{sys.executable}\nprint("SCF NOT CONVERGED")\n')
    executable.chmod(0o755)
    plan = tmp_path / "plan.json"
    runs = tmp_path / "runs"
    make_plan(dataset, {"backend": "orca", "method_id": "bad", "executable": str(executable)}, plan)
    assert execute_plan(plan, runs, execute=True, limit=2)["failed_this_run"] == 2
    again = execute_plan(plan, runs, execute=True, limit=2)
    assert again["launched"] == 0
    records = load_calculations(runs / "calculations.json")
    assert len(records) == 2 and all(r.energy is None for r in records)


def test_orca_file_exchange_checks_input(tmp_path, dataset):
    plan = tmp_path / "plan.json"
    p = make_plan(dataset, {"backend": "orca", "method_id": "test"}, plan)
    root = tmp_path / "external"
    for job in p["jobs"]:
        directory = root / job["id"]
        directory.mkdir(parents=True)
        from refblind.schema import Structure

        (directory / "input.inp").write_text(
            render_orca_input(Structure.from_dict(job["structure"]), OrcaConfig("test"))
        )
        (directory / "orca.out").write_text(SUCCESS)
    assert all(r.status == CalcStatus.SUCCEEDED for r in import_orca_plan(plan, root))
    first = root / p["jobs"][0]["id"]
    (first / "input.inp").write_text("changed")
    with pytest.raises(ValueError):
        import_orca_plan(plan, root)


def test_missing_exchange_output_retained(tmp_path, dataset):
    plan = tmp_path / "p.json"
    make_plan(dataset, {"backend": "orca", "method_id": "test"}, plan)
    results = import_orca_plan(plan, tmp_path / "missing")
    assert len(results) == 2 and all(r.status == CalcStatus.FAILED for r in results)


@pytest.fixture
def fake_ase(monkeypatch):
    module = types.ModuleType("ase")

    class Atoms:
        def __init__(self, symbols, positions, pbc):
            self.symbols = symbols
            self.positions = positions
            self.pbc = pbc
            self.info = {}

        def get_potential_energy(self):
            return self.calc.energy

        def get_forces(self):
            return self.calc.forces

    module.Atoms = Atoms
    monkeypatch.setitem(sys.modules, "ase", module)
    return module


def test_ase_contract(fake_ase, molecules):
    structure = molecules["h2-r"]
    atoms = atoms_from_structure(structure)
    assert atoms.info == {"charge": 0, "spin": 1} and atoms.pbc is False
    calc = types.SimpleNamespace(energy=-2.0, forces=np.zeros((2, 3)))
    c = calculate_ase(structure, calc, method_id="test", config={"backend": "custom"})
    assert c.energy == -2 and c.status == CalcStatus.SUCCEEDED
    calc.energy = float("nan")
    assert calculate_ase(structure, calc, method_id="test", config={}).energy is None


def test_uma_and_mace_factory_contract(monkeypatch):
    core = types.ModuleType("fairchem.core")
    calls = []
    core.pretrained_mlip = types.SimpleNamespace(
        get_predict_unit=lambda name, device: calls.append((name, device)) or "model"
    )
    core.FAIRChemCalculator = lambda predictor, task_name: (predictor, task_name)
    monkeypatch.setitem(sys.modules, "fairchem", types.ModuleType("fairchem"))
    monkeypatch.setitem(sys.modules, "fairchem.core", core)
    with pytest.raises(ValueError):
        make_calculator({"backend": "uma", "model_name": "test"})
    c = make_calculator({"backend": "uma", "model_name": "test", "allow_model_download": True})
    assert c == ("model", "omol") and calls == [("test", "cpu")]
    module = types.ModuleType("mace.calculators")
    module.mace_omol = lambda **kw: kw
    monkeypatch.setitem(sys.modules, "mace", types.ModuleType("mace"))
    monkeypatch.setitem(sys.modules, "mace.calculators", module)
    with pytest.raises(ValueError):
        make_calculator({"backend": "mace_omol"})
    assert (
        make_calculator({"backend": "mace_omol", "allow_model_download": True})["default_dtype"]
        == "float64"
    )


@pytest.mark.integration
def test_optional_real_pyscf_h2(tmp_path, molecules):
    pytest.importorskip("pyscf")
    from refblind.calculators.pyscf_backend import calculate_pyscf

    c = calculate_pyscf(
        molecules["h2-r"],
        {"method_id": "hf-sto3g", "method": "HF", "basis": "sto-3g"},
        tmp_path / "real",
    )
    assert c.status == CalcStatus.SUCCEEDED and -1.2 < c.energy < -1.0
    assert c.teacher_verified is False
