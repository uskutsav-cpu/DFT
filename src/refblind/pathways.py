"""Guarded endpoint interpolation and optional ASE NEB exploration.

A converged band is not a verified transition state. Frequency and IRC audits
must follow at the chosen chemistry level before any mechanism claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .datasets import xyz_text
from .provenance import atomic_text, write_json
from .schema import Structure
from .validation import integer, positive, vector


def interpolate_endpoints(
    start: Structure,
    end: Structure,
    *,
    images: int = 7,
    atom_mapping_verified: bool = False,
    minimum_distance: float = 0.1,
) -> list[Structure]:
    integer(images, "images", 3)
    positive(minimum_distance, "minimum_distance")
    if not isinstance(atom_mapping_verified, bool):
        raise ValueError("atom_mapping_verified must be boolean")
    if not atom_mapping_verified:
        raise ValueError("explicitly verify atom mapping before interpolating endpoints")
    if start.symbols != end.symbols or (start.charge, start.multiplicity) != (
        end.charge,
        end.multiplicity,
    ):
        raise ValueError(
            "endpoints require identical atom order, composition, charge and multiplicity"
        )
    a, b = np.asarray(start.positions_angstrom), np.asarray(end.positions_angstrom)
    result = []
    for i, fraction in enumerate(np.linspace(0, 1, images)):
        xyz = (1 - fraction) * a + fraction * b
        if len(xyz) > 1:
            distances = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
            distances += np.eye(len(xyz)) * 1e10
            if distances.min() < minimum_distance:
                raise ValueError(
                    "interpolation creates close/coincident nuclei; use a reviewed path initializer"
                )
        result.append(
            Structure(
                f"path:{i:03d}",
                start.symbols,
                tuple(map(tuple, xyz)),
                start.charge,
                start.multiplicity,
                f"linear interpolation of {start.id} and {end.id}; not a verified path",
            )
        )
    return result


@dataclass(frozen=True)
class TSAudit:
    imaginary_modes: int
    stationary_order_one: bool
    irc_endpoints_verified: bool
    verified_transition_state: bool
    warning: str


def audit_transition_state(
    frequencies_cm1,
    *,
    irc_endpoints_verified: bool,
    imaginary_threshold: float = 20.0,
    geometry_converged: bool = True,
) -> TSAudit:
    if not isinstance(irc_endpoints_verified, bool) or not isinstance(geometry_converged, bool):
        raise ValueError("transition-state audit flags must be boolean")
    positive(imaginary_threshold, "imaginary_threshold")
    frequencies = vector(frequencies_cm1, "frequencies")
    count = int(np.sum(frequencies < -imaginary_threshold))
    order_one = count == 1 and geometry_converged
    verified = order_one and irc_endpoints_verified
    return TSAudit(
        count,
        order_one,
        bool(irc_endpoints_verified),
        bool(verified),
        "Frequency/IRC flags require reviewed outputs and a physically appropriate electronic state.",
    )


def run_neb(
    start: Structure,
    end: Structure,
    calculator_factory,
    output_dir: str | Path,
    *,
    atom_mapping_verified: bool,
    images: int = 7,
    steps: int = 200,
    force_tolerance: float = 0.05,
) -> dict:
    from ase.mep import NEB
    from ase.optimize import FIRE

    from .calculators.ase_backend import atoms_from_structure

    integer(steps, "steps", 1)
    positive(force_tolerance, "force_tolerance")
    structures = interpolate_endpoints(
        start, end, images=images, atom_mapping_verified=atom_mapping_verified
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    atoms = [atoms_from_structure(s) for s in structures]
    for image in atoms:
        image.calc = (
            calculator_factory()
        )  # ASE NEB needs separate calculators unless explicitly shared.
    band = NEB(atoms, climb=True, method="improvedtangent")
    optimizer = FIRE(band, logfile=str(out / "neb.log"), trajectory=str(out / "neb.traj"))
    converged = bool(optimizer.run(fmax=force_tolerance, steps=steps))
    energies = [float(image.get_potential_energy()) for image in atoms]
    for i, image in enumerate(atoms):
        structure = Structure(
            f"neb:{i:03d}",
            start.symbols,
            tuple(map(tuple, image.positions)),
            start.charge,
            start.multiplicity,
            "ASE NEB candidate; unverified TS",
        )
        atomic_text(out / f"image-{i:03d}.xyz", xyz_text(structure), overwrite=False)
    result = {
        "neb_converged": converged,
        "steps": optimizer.nsteps,
        "energies_ev": energies,
        "candidate_index": int(np.argmax(energies)),
        "verified_transition_state": False,
        "required_next_steps": "DFT stationary-point optimization, frequency analysis and IRC",
    }
    write_json(out / "neb-result.json", result)
    return result
