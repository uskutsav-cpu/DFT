"""Versioned chemistry records, with explicit states, units and provenance.

No record implies a reference is exact. Published reference *observables* need
not (and usually do not) imply availability of high-level species energies.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from .provenance import digest
from .units import EnergyUnit
from .validation import finite, identifier, integer, nonnegative

ELEMENTS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
    "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn "
    "Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split()
ATOMIC_NUMBERS = {symbol: i + 1 for i, symbol in enumerate(ELEMENTS)}
SCHEMA_VERSION = "1.0"


class Fidelity(str, Enum):
    MLIP = "mlip"
    DFT = "dft"
    HIGH_LEVEL = "high_level"
    SCREEN = "screen"


class CalcStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    NONCONVERGED = "nonconverged"
    TIMEOUT = "timeout"
    UNSUPPORTED = "unsupported"


class ReferenceQuality(str, Enum):
    REVIEWED = "reviewed"
    UNREVIEWED = "unreviewed"
    QUESTIONABLE = "questionable"
    MULTIREFERENCE_REQUIRED = "multireference_required"
    SYNTHETIC = "synthetic"


@dataclass(frozen=True)
class Structure:
    id: str
    symbols: tuple[str, ...]
    positions_angstrom: tuple[tuple[float, float, float], ...]
    charge: int
    multiplicity: int
    source: str

    def __post_init__(self):
        identifier(self.id, "structure id")
        integer(self.charge, "charge")
        integer(self.multiplicity, "multiplicity", 1)
        if not self.source or not isinstance(self.source, str):
            raise ValueError("geometry source is required")
        symbols = tuple(self.symbols)
        if not symbols or any(s not in ATOMIC_NUMBERS for s in symbols):
            raise ValueError("symbols must contain recognized chemical elements")
        coords = tuple(
            tuple(finite(v, "coordinate") for v in xyz) for xyz in self.positions_angstrom
        )
        if len(coords) != len(symbols) or any(len(xyz) != 3 for xyz in coords):
            raise ValueError("positions must have shape (number of atoms, 3)")
        if len(set(coords)) != len(coords):
            raise ValueError("coincident nuclei are not supported")
        electrons = sum(ATOMIC_NUMBERS[s] for s in symbols) - self.charge
        unpaired = self.multiplicity - 1
        if electrons < 0 or unpaired > electrons or (electrons - unpaired) % 2:
            raise ValueError("charge/multiplicity is inconsistent with electron count")
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "positions_angstrom", coords)

    @property
    def geometry_hash(self) -> str:
        # Include charge and spin state: identical coordinates do not imply the same job.
        return digest(
            {
                "symbols": self.symbols,
                "positions_angstrom": self.positions_angstrom,
                "charge": self.charge,
                "multiplicity": self.multiplicity,
            }
        )

    @property
    def composition(self) -> dict[str, int]:
        return dict(Counter(self.symbols))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> Structure:
        return cls(**value)


@dataclass(frozen=True)
class Term:
    structure_id: str
    coefficient: float

    def __post_init__(self):
        identifier(self.structure_id, "structure id")
        if finite(self.coefficient, "stoichiometric coefficient") == 0:
            raise ValueError("zero stoichiometric terms must be removed")


@dataclass(frozen=True)
class Reference:
    value: float
    unit: EnergyUnit | str
    method: str
    source: str
    quality: ReferenceQuality | str = ReferenceQuality.UNREVIEWED
    uncertainty: float | None = None
    review_note: str = ""

    def __post_init__(self):
        finite(self.value, "reference value")
        object.__setattr__(self, "unit", EnergyUnit(self.unit))
        object.__setattr__(self, "quality", ReferenceQuality(self.quality))
        if not self.method or not self.source:
            raise ValueError("reference method and source are required")
        if self.uncertainty is not None:
            nonnegative(self.uncertainty, "reference uncertainty")
        if self.quality == ReferenceQuality.REVIEWED and not self.review_note.strip():
            raise ValueError("reviewed reference requires a review note")


@dataclass(frozen=True)
class Observable:
    id: str
    group_id: str
    kind: str
    terms: tuple[Term, ...]
    reference: Reference | None = None
    decision_group: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        identifier(self.id, "observable id")
        identifier(self.group_id, "split group")
        if self.kind not in {
            "barrier",
            "reaction_energy",
            "state_gap",
            "pathway_gap",
            "relative_energy",
        }:
            raise ValueError("unknown observable kind")
        terms = tuple(t if isinstance(t, Term) else Term(**t) for t in self.terms)
        if len(terms) < 2 or len({t.structure_id for t in terms}) != len(terms):
            raise ValueError("observable needs >=2 unique, nonzero species terms")
        if not any(t.coefficient > 0 for t in terms) or not any(t.coefficient < 0 for t in terms):
            raise ValueError("observable must include positive and negative terms")
        object.__setattr__(self, "terms", terms)
        if isinstance(self.reference, dict):
            object.__setattr__(self, "reference", Reference(**self.reference))
        if self.decision_group is not None:
            identifier(self.decision_group, "decision group")
        digest(self.metadata)

    def validate_balance(self, structures: dict[str, Structure]) -> None:
        balance = Counter()
        charge_balance = 0.0
        for term in self.terms:
            if term.structure_id not in structures:
                raise ValueError(f"missing structure {term.structure_id} in {self.id}")
            structure = structures[term.structure_id]
            for symbol, count in structure.composition.items():
                balance[symbol] += term.coefficient * count
            charge_balance += term.coefficient * structure.charge
        if any(abs(v) > 1e-8 for v in balance.values()) or abs(charge_balance) > 1e-8:
            raise ValueError(f"unbalanced stoichiometry or charge in {self.id}: {dict(balance)}")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Calculation:
    structure_id: str
    geometry_hash: str
    method_id: str
    fidelity: Fidelity | str
    status: CalcStatus | str
    energy: float | None
    energy_unit: EnergyUnit | str = EnergyUnit.EV
    teacher_id: str | None = None
    teacher_verified: bool = False
    forces_ev_angstrom: tuple[tuple[float, float, float], ...] | None = None
    diagnostics: dict[str, float | None] = field(default_factory=dict)
    wall_seconds: float = 0.0
    cost: float = 0.0
    cost_unit: str = "relative"
    software: str = "unknown"
    software_version: str = "unknown"
    settings_hash: str = ""
    raw_sha256: str | None = None
    message: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        identifier(self.structure_id, "structure id")
        identifier(self.method_id, "method id")
        object.__setattr__(self, "fidelity", Fidelity(self.fidelity))
        object.__setattr__(self, "status", CalcStatus(self.status))
        object.__setattr__(self, "energy_unit", EnergyUnit(self.energy_unit))
        if len(self.geometry_hash) != 64 or any(
            c not in "0123456789abcdef" for c in self.geometry_hash
        ):
            raise ValueError("geometry_hash must be a lowercase SHA-256 digest")
        if self.status == CalcStatus.SUCCEEDED and self.energy is None:
            raise ValueError("successful calculation requires an energy")
        if self.energy is not None:
            finite(self.energy, "energy")
        if self.status != CalcStatus.SUCCEEDED and self.energy is not None:
            raise ValueError("failed calculations must not expose a usable energy")
        nonnegative(self.wall_seconds, "wall_seconds")
        nonnegative(self.cost, "cost")
        if not self.cost_unit:
            raise ValueError("cost_unit is required")
        if not isinstance(self.teacher_verified, bool):
            raise ValueError("teacher_verified must be bool")
        for name, value in self.diagnostics.items():
            if value is not None:
                finite(value, f"diagnostic {name}")
        if self.forces_ev_angstrom is not None:
            forces = tuple(
                tuple(finite(v, "force") for v in xyz) for xyz in self.forces_ev_angstrom
            )
            if not forces or any(len(xyz) != 3 for xyz in forces):
                raise ValueError("forces must have shape (n_atoms, 3)")
            object.__setattr__(self, "forces_ev_angstrom", forces)
        digest(self.provenance)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> Calculation:
        return cls(**value)
