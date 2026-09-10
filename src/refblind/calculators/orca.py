"""ORCA single-point adapter with strict failure handling and immutable logs.

The built-in header is a *candidate* DFT setup, not audited OMol25 equivalence.
Use a reviewed template/settings plus basis assets to reproduce a teacher.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..features import spin_contamination
from ..provenance import atomic_text, digest, file_digest
from ..schema import CalcStatus, Calculation, Fidelity, Structure
from ..units import EnergyUnit
from ..validation import identifier, integer, nonnegative, positive

NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][-+]?\d+)?"


@dataclass(frozen=True)
class OrcaConfig:
    method_id: str
    header: str = "! wB97M-V def2-TZVPD TightSCF DEFGRID3"
    blocks: str = "%scf MaxIter 300 end"
    executable: str = "orca"
    fidelity: Fidelity | str = Fidelity.DFT
    nprocs: int = 1
    maxcore_mb: int = 1000
    memory_budget_mb: int = 2000
    timeout_seconds: float = 3600.0
    teacher_id: str | None = None
    teacher_verified: bool = False
    teacher_audit_note: str = ""
    relative_cost: float = 1.0
    correlation_convergence_pattern: str = ""
    assets: dict[str, str] = field(default_factory=dict)
    asset_sha256: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        identifier(self.method_id, "method_id")
        if not isinstance(self.teacher_verified, bool):
            raise ValueError("teacher_verified must be a boolean")
        object.__setattr__(self, "fidelity", Fidelity(self.fidelity))
        if self.fidelity == Fidelity.HIGH_LEVEL:
            if not self.correlation_convergence_pattern.strip():
                raise ValueError("high-level ORCA needs an audited correlation_convergence_pattern")
            if len(self.correlation_convergence_pattern) > 500:
                raise ValueError("convergence pattern is too long")
            re.compile(self.correlation_convergence_pattern)
        integer(self.nprocs, "nprocs", 1)
        integer(self.maxcore_mb, "maxcore_mb", 1)
        integer(self.memory_budget_mb, "memory_budget_mb", 1)
        positive(self.timeout_seconds, "timeout_seconds")
        nonnegative(self.relative_cost, "relative_cost")
        if self.nprocs * self.maxcore_mb > 0.8 * self.memory_budget_mb:
            raise ValueError("ORCA maxcore is per process; reserve >=20% memory overhead")
        if not self.header.startswith("!") or "\n" in self.header or "\r" in self.header:
            raise ValueError("header must be one ORCA ! keyword line")
        # Single points only: no inadvertent geometry moves in paired comparisons.
        tokens = self.header.upper().split()
        if any(
            t.startswith(("OPT", "NEB", "GOAT")) or t in {"MD", "FREQ", "NUMFREQ"} for t in tokens
        ):
            raise ValueError("this adapter supports fixed-geometry single points only")
        if re.search(r"(?im)^\s*\*|\$new_job|%pal|%maxcore|%coords|%geom", self.blocks):
            raise ValueError("blocks may not override jobs, geometry, or resource controls")
        if self.teacher_verified and (not self.teacher_id or not self.teacher_audit_note.strip()):
            raise ValueError("teacher verification requires an identity and documented audit")
        if set(self.assets) != set(self.asset_sha256):
            raise ValueError("each external basis/auxiliary asset needs a SHA-256")
        for name in self.assets:
            if Path(name).name != name or name in {"input.inp", "orca.out", "orca.err"}:
                raise ValueError("asset names must be safe, unique basenames")
        digest(asdict(self))

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        return digest(self.to_dict())


def render_orca_input(structure: Structure, config: OrcaConfig) -> str:
    lines = [
        config.header,
        config.blocks,
        f"%pal nprocs {config.nprocs} end",
        f"%maxcore {config.maxcore_mb}",
        f"* xyz {structure.charge} {structure.multiplicity}",
    ]
    lines.extend(
        f"{s} {x:.16g} {y:.16g} {z:.16g}"
        for s, (x, y, z) in zip(structure.symbols, structure.positions_angstrom)
    )
    return "\n".join(lines) + "\n*\n"


def _last_number(pattern: str, text: str) -> float | None:
    matches = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if not matches:
        return None
    from ..validation import finite

    return finite(float(matches[-1].replace("D", "E").replace("d", "e")))


def parse_orbital_gap(text: str) -> float | None:
    """Minimum complete spin-channel gap from one ordinary orbital-energy table.

    Fractional occupations and FOD/smearing calculations return missing. This is
    an orbital-gap descriptor, not a fundamental excitation gap or certificate.
    """
    if re.search(r"N_FOD|SMEAR(?:TEMP|ING)", text, re.I):
        return None
    if len(re.findall(r"^\s*ORBITAL ENERGIES\s*$", text, re.I | re.M)) != 1:
        return None
    section = re.split(r"^\s*ORBITAL ENERGIES\s*$", text, flags=re.I | re.M)[1]
    rows, channels, reading = [], [], False
    for line in section.splitlines():
        if re.search(r"NO\s+OCC\s+E\(Eh\)\s+E\(eV\)", line, re.I):
            if rows:
                channels.append(rows)
            rows, reading = [], True
            continue
        match = re.match(rf"^\s*\d+\s+({NUMBER})\s+({NUMBER})\s+({NUMBER})(?:\s|$)", line)
        if reading and match:
            values = [float(v.replace("D", "E").replace("d", "e")) for v in match.groups()]
            rows.append((values[2], values[0]))
        elif reading and rows and line.strip():
            channels.append(rows)
            rows, reading = [], False
    if rows:
        channels.append(rows)
    if not channels:
        return None
    from ..features import orbital_gap_ev

    gaps = []
    for channel in channels:
        if any(abs(occupation - round(occupation)) > 1e-5 for _, occupation in channel):
            return None
        try:
            gaps.append(orbital_gap_ev([e for e, _ in channel], [o for _, o in channel]))
        except ValueError:
            return None
    return min(gaps)


def parse_orca_output(text: str, *, multiplicity: int = 1) -> dict:
    """Parse only unambiguous single-job/single-point output.

    Missing diagnostics remain None. Multiple final energies are rejected rather
    than silently mixing different geometries or the wrong electronic states.
    """
    integer(multiplicity, "multiplicity", 1)
    normal = len(re.findall(r"ORCA TERMINATED NORMALLY", text, re.I))
    energies = re.findall(rf"^\s*FINAL SINGLE POINT ENERGY\s+({NUMBER})\s*$", text, re.I | re.M)
    version_match = re.search(r"Program Version\s+([A-Za-z0-9_.+-]+)", text, re.I)
    version = version_match.group(1) if version_match else "unknown"
    failed_scf = bool(
        re.search(r"SCF NOT CONVERGED|SCF DID NOT CONVERGE|SCF failed to converge", text, re.I)
    )
    explicit_error = bool(
        re.search(r"ORCA finished by error termination|ORCA TERMINATED ABNORMALLY", text, re.I)
    )
    converged = bool(
        re.search(
            r"SCF CONVERGED AFTER|SCF CONVERGENCE ACHIEVED|Energy Check signals convergence",
            text,
            re.I,
        )
    )
    status, message = CalcStatus.SUCCEEDED, ""
    if failed_scf:
        status, message = CalcStatus.NONCONVERGED, "SCF did not converge"
    elif explicit_error or normal != 1:
        status, message = (
            CalcStatus.FAILED,
            "missing/ambiguous normal termination or explicit error",
        )
    elif len(energies) != 1:
        status, message = CalcStatus.UNSUPPORTED, "expected exactly one final single-point energy"
    elif not converged:
        status, message = CalcStatus.NONCONVERGED, "no recognized SCF-convergence confirmation"
    energy = (
        float(energies[0].replace("D", "E").replace("d", "e"))
        if status == CalcStatus.SUCCEEDED
        else None
    )
    cycles = _last_number(r"SCF CONVERGED AFTER\s+(\d+)\s+CYCLES", text)
    s2 = _last_number(rf"Expectation value of <S(?:\*\*|\^)?2>\s*:?\s*({NUMBER})", text)
    fod = _last_number(rf"N_FOD\s*=\s*({NUMBER})", text)
    # FOD's printed smearing gap is NOT silently substituted for a ground-state gap.
    gap = parse_orbital_gap(text)
    if gap is None and not re.search(r"N_FOD|SMEAR(?:TEMP|ING)", text, re.I):
        gap = _last_number(rf"HOMO-LUMO GAP\s*[:=]\s*({NUMBER})\s*eV", text)
    return {
        "status": status,
        "energy_hartree": energy,
        "software_version": version,
        "scf_converged": converged and not failed_scf,
        "message": message,
        "diagnostics": {
            "scf_cycles": cycles,
            "s2": s2,
            "spin_contamination": spin_contamination(s2, multiplicity) if s2 is not None else None,
            "fod": fod,
            "homo_lumo_gap_ev": gap,
        },
    }


def enforce_correlation_convergence(parsed: dict, text: str, config: OrcaConfig) -> dict:
    """A normal SCF termination alone is not evidence of correlated-method convergence."""
    if config.fidelity == Fidelity.HIGH_LEVEL and parsed["status"] == CalcStatus.SUCCEEDED:
        if not re.search(config.correlation_convergence_pattern, text, flags=re.MULTILINE):
            parsed.update(
                status=CalcStatus.NONCONVERGED,
                energy_hartree=None,
                message="no match for reviewed correlated-method convergence pattern",
            )
    return parsed


def parse_engrad(text: str, number_of_atoms: int) -> tuple[tuple[float, float, float], ...]:
    """Convert ORCA gradients (Eh/bohr) to forces (eV/angstrom)."""
    from ..units import BOHR_TO_ANGSTROM, HARTREE_TO_EV

    integer(number_of_atoms, "number_of_atoms", 1)
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(lines) < 2 + 3 * number_of_atoms or int(lines[0]) != number_of_atoms:
        raise ValueError("malformed engrad atom count or gradient block")
    from ..validation import finite

    gradients = [
        finite(float(v.replace("D", "E")), "gradient") for v in lines[2 : 2 + 3 * number_of_atoms]
    ]
    scale = -HARTREE_TO_EV / BOHR_TO_ANGSTROM
    return tuple(
        tuple(scale * gradients[3 * i + j] for j in range(3)) for i in range(number_of_atoms)
    )


def run_orca(structure: Structure, config: OrcaConfig, directory: str | Path) -> Calculation:
    work = Path(directory).resolve()
    work.mkdir(parents=True, exist_ok=True)
    executable = shutil.which(config.executable)
    if executable is None:
        raise FileNotFoundError("ORCA executable not found; install separately and set executable")
    executable = str(Path(executable).resolve())
    for name, source in config.assets.items():
        if file_digest(source) != config.asset_sha256[name]:
            raise ValueError(f"asset hash mismatch: {name}")
        destination = work / name
        if destination.exists():
            raise FileExistsError(destination)
        shutil.copyfile(source, destination)
    input_text = render_orca_input(structure, config)
    atomic_text(work / "input.inp", input_text, overwrite=False)
    start = time.monotonic()
    timed_out = False
    # O_EXCL protects previous logs; never overwrite an old calculation.
    with (
        (work / "orca.out").open("x", encoding="utf-8") as out,
        (work / "orca.err").open("x", encoding="utf-8") as err,
    ):
        process = subprocess.Popen(
            [executable, "input.inp"],
            cwd=work,
            stdout=out,
            stderr=err,
            start_new_session=(os.name == "posix"),
        )
        try:
            code = process.wait(timeout=config.timeout_seconds)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait()
            if isinstance(exc, KeyboardInterrupt):
                raise
            timed_out = True
            code = process.returncode
    wall = time.monotonic() - start
    parsed = parse_orca_output(
        (work / "orca.out").read_text(encoding="utf-8", errors="replace"),
        multiplicity=structure.multiplicity,
    )
    parsed = enforce_correlation_convergence(
        parsed, (work / "orca.out").read_text(encoding="utf-8", errors="replace"), config
    )
    if timed_out:
        parsed.update(status=CalcStatus.TIMEOUT, energy_hartree=None, message="execution timed out")
    elif code != 0:
        parsed.update(
            status=CalcStatus.FAILED, energy_hartree=None, message=f"ORCA exit code {code}"
        )
    forces = None
    if parsed["status"] == CalcStatus.SUCCEEDED and (work / "input.engrad").exists():
        forces = parse_engrad((work / "input.engrad").read_text(), len(structure.symbols))
    return Calculation(
        structure.id,
        structure.geometry_hash,
        config.method_id,
        config.fidelity,
        parsed["status"],
        parsed["energy_hartree"],
        EnergyUnit.HARTREE,
        teacher_id=config.teacher_id,
        teacher_verified=config.teacher_verified,
        forces_ev_angstrom=forces,
        diagnostics=parsed["diagnostics"],
        wall_seconds=wall,
        cost=config.relative_cost,
        software="ORCA",
        software_version=parsed["software_version"],
        settings_hash=config.fingerprint,
        raw_sha256=file_digest(work / "orca.out"),
        message=parsed["message"],
        provenance={
            "input_sha256": digest(input_text),
            "executable_sha256": file_digest(executable),
            "teacher_audit_note": config.teacher_audit_note,
            "cost_semantics": "configured relative estimate; use wall_seconds for measured elapsed cost",
        },
    )
