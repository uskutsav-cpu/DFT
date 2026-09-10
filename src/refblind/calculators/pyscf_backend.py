"""Optional small-molecule PySCF adapter. Not an OMol25 teacher replacement.

FCI is exact only within its chosen finite orbital basis; CCSD(T) is not a
universal high-level reference. State/reference validation stays explicit.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

from ..features import orbital_gap_ev, spin_contamination
from ..provenance import digest, file_digest
from ..schema import CalcStatus, Calculation, Fidelity, Structure
from ..units import HARTREE_TO_EV, EnergyUnit
from ..validation import integer, nonnegative, positive


def calculate_pyscf(structure: Structure, config: dict, directory: str | Path) -> Calculation:
    import numpy as np
    import pyscf
    from pyscf import cc, dft, fci, gto, lib, scf

    work = Path(directory)
    work.mkdir(parents=True, exist_ok=True)
    log = work / "pyscf.log"
    if log.exists():
        raise FileExistsError(log)
    method = config.get("method", "HF").upper()
    if method not in {"HF", "DFT", "CCSD(T)", "FCI"}:
        raise ValueError("supported PySCF methods: HF, DFT, CCSD(T), FCI")
    expected_fidelity = (
        "high_level" if method in {"CCSD(T)", "FCI"} else "screen" if method == "HF" else "dft"
    )
    fidelity = Fidelity(config.get("fidelity", expected_fidelity))
    if fidelity.value != expected_fidelity:
        raise ValueError("PySCF method/fidelity mismatch")
    memory = integer(config.get("memory_mb", 1000), "memory_mb", 1)
    threads = integer(config.get("threads", 1), "threads", 1)
    max_atoms = integer(config.get("max_atoms", 12), "max_atoms", 1)
    nonnegative(config.get("relative_cost", 1.0), "relative_cost")
    if len(structure.symbols) > max_atoms:
        raise ValueError("molecule exceeds explicit PySCF atom guard")
    lib.num_threads(threads)
    molecule = gto.M(
        atom=list(zip(structure.symbols, structure.positions_angstrom)),
        unit="Angstrom",
        basis=config.get("basis", "sto-3g"),
        charge=structure.charge,
        spin=structure.multiplicity - 1,
        max_memory=memory,
        output=str(log),
        verbose=4,
    )
    unrestricted = molecule.spin != 0 or bool(config.get("unrestricted", False))
    if method == "DFT":
        mf = dft.UKS(molecule) if unrestricted else dft.RKS(molecule)
        mf.xc = config.get("functional", "PBE")
        mf.grids.level = integer(config.get("grid_level", 3), "grid_level", 0)
    else:
        mf = scf.UHF(molecule) if unrestricted else scf.RHF(molecule)
    mf.conv_tol = positive(config.get("conv_tol", 1e-10), "conv_tol")
    mf.max_cycle = integer(config.get("max_cycle", 100), "max_cycle", 1)
    start = time.monotonic()
    status, energy, message, diagnostics = CalcStatus.SUCCEEDED, None, "", {}
    try:
        energy = float(mf.kernel())
        if not mf.converged:
            status, energy, message = CalcStatus.NONCONVERGED, None, "SCF did not converge"
        else:
            gaps = []
            channels = zip(mf.mo_energy, mf.mo_occ) if unrestricted else [(mf.mo_energy, mf.mo_occ)]
            for energies, occupations in channels:
                try:
                    gaps.append(orbital_gap_ev(np.asarray(energies) * HARTREE_TO_EV, occupations))
                except ValueError:
                    pass
            diagnostics["homo_lumo_gap_ev"] = min(gaps) if gaps else None
            if unrestricted:
                diagnostics["spin_contamination"] = spin_contamination(
                    float(mf.spin_square()[0]), structure.multiplicity
                )
            if method == "CCSD(T)":
                solver = cc.CCSD(mf)
                solver.conv_tol = mf.conv_tol
                solver.kernel()
                if not solver.converged:
                    status, energy, message = CalcStatus.NONCONVERGED, None, "CCSD did not converge"
                else:
                    energy = float(solver.e_tot + solver.ccsd_t())
            elif method == "FCI":
                if unrestricted:
                    raise ValueError(
                        "this finite-basis FCI adapter currently requires a restricted reference"
                    )
                norb = mf.mo_coeff.shape[1]
                alpha, beta = molecule.nelec
                determinant_count = math.comb(norb, alpha) * math.comb(norb, beta)
                limit = integer(config.get("max_determinants", 100000), "max_determinants", 1)
                if determinant_count > limit:
                    raise ValueError(
                        f"FCI determinant guard exceeded: {determinant_count} > {limit}"
                    )
                solver = fci.FCI(mf)
                solver.conv_tol = mf.conv_tol
                energy = float(solver.kernel()[0])
                if not solver.converged:
                    status, energy, message = CalcStatus.NONCONVERGED, None, "FCI did not converge"
            if energy is not None and not np.isfinite(energy):
                raise ValueError("non-finite PySCF energy")
    except Exception as exc:
        status, energy, message = CalcStatus.FAILED, None, f"{type(exc).__name__}: {exc}"
    finally:
        molecule.stdout.flush()
        molecule.stdout.close()
    return Calculation(
        structure.id,
        structure.geometry_hash,
        config["method_id"],
        fidelity,
        status,
        energy,
        EnergyUnit.HARTREE,
        teacher_id=config.get("teacher_id"),
        teacher_verified=False,
        diagnostics=diagnostics,
        wall_seconds=time.monotonic() - start,
        cost=config.get("relative_cost", 1.0),
        software="PySCF",
        software_version=pyscf.__version__,
        settings_hash=digest(config),
        raw_sha256=file_digest(log),
        message=message,
        provenance={"finite_basis_only": True, "not_omol_teacher": True},
    )
