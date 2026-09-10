"""Benchmark I/O and geometry parsing. Files are data, never shell programs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .provenance import digest, read_json, write_json
from .schema import SCHEMA_VERSION, Calculation, Observable, Structure
from .units import BOHR_TO_ANGSTROM


@dataclass
class Dataset:
    name: str
    structures: list[Structure]
    observables: list[Observable]
    metadata: dict = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self):
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema version {self.schema_version}")
        if not self.name or not self.structures or not self.observables:
            raise ValueError("dataset needs a name, structures and observables")
        if len({s.id for s in self.structures}) != len(self.structures):
            raise ValueError("duplicate structure ids")
        if len({o.id for o in self.observables}) != len(self.observables):
            raise ValueError("duplicate observable ids")
        index = self.structure_map
        for observable in self.observables:
            observable.validate_balance(index)
        # Related competing decisions must not be split across partitions.
        decisions = {}
        for observable in self.observables:
            if observable.decision_group:
                existing = decisions.setdefault(observable.decision_group, observable.group_id)
                if existing != observable.group_id:
                    raise ValueError("a decision group spans different split groups")
        digest(self.metadata)

    @property
    def structure_map(self) -> dict[str, Structure]:
        return {s.id: s for s in self.structures}

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "structures": [s.to_dict() for s in self.structures],
            "observables": [o.to_dict() for o in self.observables],
            "metadata": self.metadata,
        }

    @property
    def fingerprint(self) -> str:
        return digest(self.to_dict())

    def save(self, path: str | Path) -> None:
        write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> Dataset:
        return cls.from_dict(read_json(path))

    @classmethod
    def from_dict(cls, value: dict) -> Dataset:
        value = dict(value)
        value["structures"] = [Structure.from_dict(s) for s in value["structures"]]
        value["observables"] = [Observable(**o) for o in value["observables"]]
        return cls(**value)


def read_xyz(
    path: str | Path,
    *,
    structure_id: str,
    charge: int,
    multiplicity: int,
    source: str | None = None,
) -> Structure:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise ValueError("XYZ file is too short")
    try:
        count = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError("XYZ first line must contain the atom count") from exc
    if count < 1 or len(lines) < count + 2 or any(line.strip() for line in lines[count + 2 :]):
        raise ValueError("XYZ atom count mismatch or multiple XYZ frames")
    symbols, coords = [], []
    for line in lines[2 : count + 2]:
        parts = line.split()
        if len(parts) != 4:
            raise ValueError("expected plain XYZ with four columns per atom")
        symbols.append(parts[0])
        coords.append(tuple(float(x.replace("D", "E")) for x in parts[1:]))
    return Structure(
        structure_id, tuple(symbols), tuple(coords), charge, multiplicity, source or str(path)
    )


def xyz_text(structure: Structure) -> str:
    rows = [
        str(len(structure.symbols)),
        f"{structure.id} charge={structure.charge} multiplicity={structure.multiplicity}",
    ]
    rows += [
        f"{s} {x:.16g} {y:.16g} {z:.16g}"
        for s, (x, y, z) in zip(structure.symbols, structure.positions_angstrom)
    ]
    return "\n".join(rows) + "\n"


def read_turbomole(
    path: str | Path,
    *,
    structure_id: str,
    charge: int,
    multiplicity: int,
    source: str | None = None,
) -> Structure:
    symbols, coords = [], []
    reading = False
    scale = BOHR_TO_ANGSTROM
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.lower().startswith("$coord"):
            if reading or symbols:
                raise ValueError("multiple coordinate blocks")
            qualifier = line.lower().split()[1:]
            if qualifier not in ([], ["angs"], ["bohr"]):
                raise ValueError("unsupported Turbomole coordinate qualifier")
            scale = 1.0 if qualifier == ["angs"] else BOHR_TO_ANGSTROM
            reading = True
            continue
        if reading and line.startswith("$"):
            reading = False
        if not reading or not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 4:
            raise ValueError("expected four columns in Turbomole coordinates")
        coords.append(tuple(float(x.replace("D", "E")) * scale for x in parts[:3]))
        symbols.append(parts[3].capitalize())
    return Structure(
        structure_id, tuple(symbols), tuple(coords), charge, multiplicity, source or str(path)
    )


def save_calculations(path: str | Path, records: list[Calculation]) -> None:
    keys = [(r.structure_id, r.method_id) for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate calculation keys")
    write_json(
        path, {"schema_version": SCHEMA_VERSION, "calculations": [r.to_dict() for r in records]}
    )


def load_calculations(path: str | Path) -> list[Calculation]:
    value = read_json(path)
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported calculation schema")
    records = [Calculation.from_dict(r) for r in value["calculations"]]
    keys = [(r.structure_id, r.method_id) for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate calculation keys")
    return records
