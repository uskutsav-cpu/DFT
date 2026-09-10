"""ASE/UMA/MACE/OrbMol adapters. Optional packages and model downloads are explicit."""

from __future__ import annotations

import importlib.metadata
import time
from pathlib import Path

import numpy as np

from ..provenance import digest, file_digest
from ..schema import CalcStatus, Calculation, Fidelity, Structure
from ..validation import nonnegative


def atoms_from_structure(structure: Structure):
    try:
        from ase import Atoms
    except ImportError as exc:
        raise ImportError("Install refblind[ase] in your chemistry environment") from exc
    atoms = Atoms(
        symbols=list(structure.symbols), positions=structure.positions_angstrom, pbc=False
    )
    atoms.info.update(charge=structure.charge, spin=structure.multiplicity)
    return atoms


def make_calculator(config: dict):
    """No automatic task switching: only nonperiodic OMol chemistry is supported."""
    backend = config["backend"]
    if not isinstance(config.get("allow_model_download", False), bool):
        raise ValueError("allow_model_download must be a boolean")
    device = config.get("device", "cpu")
    if device not in {"cpu", "cuda", "mps"}:
        raise ValueError("unsupported device; choose an explicit cpu/cuda/mps backend")
    if backend == "uma":
        from fairchem.core import FAIRChemCalculator, pretrained_mlip

        if not config.get("allow_model_download", False):
            raise ValueError(
                "UMA loading may access a gated checkpoint; explicitly allow_model_download"
            )
        predictor = pretrained_mlip.get_predict_unit(config["model_name"], device=device)
        return FAIRChemCalculator(predictor, task_name="omol")
    if backend == "mace_omol":
        from mace.calculators import mace_omol

        model = config.get("model_path", "extra_large")
        if model == "extra_large" and not config.get("allow_model_download", False):
            raise ValueError("MACE checkpoint download requires explicit permission")
        if model != "extra_large":
            if not Path(model).is_file():
                raise FileNotFoundError(model)
            expected = config.get("checkpoint_sha256")
            if not expected or file_digest(model) != expected:
                raise ValueError("MACE local checkpoint requires matching checkpoint_sha256")
        return mace_omol(model=model, device=device, default_dtype=config.get("dtype", "float64"))
    if backend == "orbmol":
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.inference.calculator import ORBCalculator

        # Keep OMol25-only v1 distinct from the OMol25+OPoly26 v2 release.
        name = config.get("model_name", "orb_v3_conservative_omol")
        if name not in {"orb_v3_conservative_omol", "orbmol_v2"}:
            raise ValueError("select an explicitly supported OrbMol model")
        kwargs = {"device": device, "precision": config.get("dtype", "float64"), "compile": False}
        model_path = config.get("model_path")
        if model_path:
            if not Path(model_path).is_file():
                raise FileNotFoundError(model_path)
            if (
                not config.get("checkpoint_sha256")
                or file_digest(model_path) != config["checkpoint_sha256"]
            ):
                raise ValueError("OrbMol local checkpoint requires matching checkpoint_sha256")
            kwargs["weights_path"] = str(model_path)
        elif not config.get("allow_model_download", False):
            raise ValueError("OrbMol checkpoint download requires explicit permission")
        model, atoms_adapter = getattr(pretrained, name)(**kwargs)
        return ORBCalculator(model, atoms_adapter=atoms_adapter, device=device)
    raise ValueError(
        f"unsupported ASE backend: {backend}; use a reviewed custom ASE calculator in Python"
    )


def calculate_ase(structure: Structure, calculator, *, method_id: str, config: dict) -> Calculation:
    """Accept an ASE calculator instance; failed inference never becomes zero energy."""
    relative_cost = nonnegative(config.get("relative_cost", 0.01), "relative_cost")
    atoms = atoms_from_structure(structure)
    atoms.calc = calculator
    start = time.monotonic()
    status, energy, forces, message = CalcStatus.SUCCEEDED, None, None, ""
    try:
        energy = float(atoms.get_potential_energy())
        if not np.isfinite(energy):
            raise ValueError("non-finite energy")
        if config.get("forces", True):
            raw = np.asarray(atoms.get_forces(), dtype=float)
            if raw.shape != (len(structure.symbols), 3) or not np.isfinite(raw).all():
                raise ValueError("invalid force shape/values")
            forces = tuple(tuple(float(v) for v in xyz) for xyz in raw)
    except Exception as exc:
        status, energy, forces = CalcStatus.FAILED, None, None
        message = f"{type(exc).__name__}: {exc}"
    wall = time.monotonic() - start
    package = {"uma": "fairchem-core", "mace_omol": "mace-torch", "orbmol": "orb-models"}.get(
        config.get("backend"), "custom-calculator"
    )
    try:
        version = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        version = "custom-calculator"
    return Calculation(
        structure.id,
        structure.geometry_hash,
        method_id,
        Fidelity.MLIP,
        status,
        energy,
        teacher_id=config.get("teacher_id"),
        forces_ev_angstrom=forces,
        wall_seconds=wall,
        cost=relative_cost,
        software=package,
        software_version=version,
        settings_hash=digest(config),
        message=message,
        provenance={
            "checkpoint_sha256": config.get("checkpoint_sha256"),
            "model_name": config.get("model_name"),
            "checkpoint_identity_verified": bool(config.get("checkpoint_sha256")),
            "cost_semantics": "configured estimate; model loading time is not included",
        },
    )
