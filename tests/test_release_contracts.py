"""Release-level contracts. External chemistry remains fabricated/mocked here."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

from refblind.calculators.ase_backend import make_calculator
from refblind.calculators.orca import OrcaConfig, enforce_correlation_convergence, parse_orca_output
from refblind.cli import main
from refblind.execution import make_plan
from refblind.provenance import file_digest
from refblind.reporting import render_figures
from refblind.schema import CalcStatus
from refblind.study import StudyConfig, run_study
from refblind.synthetic import synthetic_table


def test_highlevel_requires_correlated_convergence_contract():
    with pytest.raises(ValueError, match="audited"):
        OrcaConfig("ccsd", fidelity="high_level")
    cfg = OrcaConfig(
        "ccsd", fidelity="high_level", correlation_convergence_pattern=r"^AUDITED_CC_CONVERGED$"
    )
    text = (
        "SCF CONVERGED AFTER 7 CYCLES\nFINAL SINGLE POINT ENERGY -1.0\nORCA TERMINATED NORMALLY\n"
    )
    missing = enforce_correlation_convergence(parse_orca_output(text), text, cfg)
    assert missing["status"] == CalcStatus.NONCONVERGED and missing["energy_hartree"] is None
    text += "AUDITED_CC_CONVERGED\n"
    assert (
        enforce_correlation_convergence(parse_orca_output(text), text, cfg)["status"]
        == CalcStatus.SUCCEEDED
    )


def test_orbmol_current_factory_contract(monkeypatch, tmp_path):
    ff = types.ModuleType("orb_models.forcefield")
    calls = []
    ff.pretrained = types.SimpleNamespace(
        orb_v3_conservative_omol=lambda **kw: calls.append(kw) or ("model", "adapter")
    )
    calcmod = types.ModuleType("orb_models.forcefield.inference.calculator")
    calcmod.ORBCalculator = lambda model, atoms_adapter, device: (model, atoms_adapter, device)
    for name, module in [
        ("orb_models", types.ModuleType("orb_models")),
        ("orb_models.forcefield", ff),
        ("orb_models.forcefield.inference", types.ModuleType("orb_models.forcefield.inference")),
        ("orb_models.forcefield.inference.calculator", calcmod),
    ]:
        monkeypatch.setitem(sys.modules, name, module)
    with pytest.raises(ValueError):
        make_calculator({"backend": "orbmol"})
    out = make_calculator({"backend": "orbmol", "allow_model_download": True})
    assert out == ("model", "adapter", "cpu") and calls[-1]["compile"] is False
    model = tmp_path / "fake-weights"
    model.write_bytes(b"fabricated test weights; never loaded")
    cfg = {"backend": "orbmol", "model_path": str(model), "checkpoint_sha256": file_digest(model)}
    assert make_calculator(cfg) == out
    with pytest.raises(ValueError):
        make_calculator({**cfg, "checkpoint_sha256": "0" * 64})


def test_permission_is_not_truthy_string():
    with pytest.raises(ValueError, match="boolean"):
        make_calculator({"backend": "uma", "allow_model_download": "false"})
    with pytest.raises(ValueError, match="boolean"):
        OrcaConfig("test", teacher_verified="false")


def test_synthetic_report_figures_and_cli(tmp_path):
    pytest.importorskip("matplotlib")
    table = synthetic_table(groups=24, rows_per_group=2)
    root = tmp_path / "report"
    report = run_study(table, StudyConfig(bootstrap_repetitions=20), root)
    assert "eligible" in report["evaluation_population"]
    figures = render_figures(root)
    assert len(figures) == 3 and all(Path(p).stat().st_size > 1000 for p in figures)
    assert main(["plot", str(root)]) == 0
    assert "SYNTHETIC" in (root / "REPORT.md").read_text()


def test_h2_shipped_example_has_no_fake_label():
    from refblind.datasets import Dataset

    path = Path(__file__).resolve().parents[1] / "examples/h2-smoke.json"
    d = Dataset.load(path)
    assert len(d.structures) == 2
    assert all(o.reference is None and o.kind == "relative_energy" for o in d.observables)


def test_slurm_script_is_review_only_and_resource_checked(tmp_path, dataset):
    source = Path(__file__).resolve().parents[1] / "scripts/render_slurm.py"
    spec = importlib.util.spec_from_file_location("refblind_slurm_script", source)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    plan = tmp_path / "plan.json"
    make_plan(dataset, {"backend": "orca", "method_id": "candidate"}, plan)
    text = mod.render(plan, tmp_path / "runs", python_executable=Path(sys.executable))
    assert "#SBATCH --array=0-0%1" in text and "--execute --limit 1" in text
    assert "task-$SLURM_ARRAY_TASK_ID" in text
    with pytest.raises(ValueError):
        mod.render(plan, tmp_path / "runs", python_executable=Path(sys.executable), memory_mb=100)
    with pytest.raises(ValueError):
        mod.render(
            plan,
            tmp_path / "runs",
            python_executable=Path(sys.executable),
            time_limit="x\nmalicious",
        )
