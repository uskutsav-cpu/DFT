"""Import GMTKN55's on-disk .res format without executing its shell wrapper.

The grammar is restricted to `$tmer species/$f ... x coefficients ... $w value`.
Unknown command forms fail loudly. Data is loaded from a user-supplied, pinned
checkout; this module never downloads or evaluates arbitrary shell text.
"""

from __future__ import annotations

import re
import shlex
import subprocess
from collections import defaultdict
from pathlib import Path

from .datasets import Dataset, read_turbomole, read_xyz
from .provenance import digest, file_digest
from .schema import Observable, Reference, Term
from .units import EnergyUnit

_SPECIES = re.compile(r"([A-Za-z0-9][A-Za-z0-9_+.-]*)/\$(?:f|\{f\})$")


def parse_res(text: str) -> list[dict]:
    rows = []
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped.startswith(("$tmer", "${tmer}")):
            # Non-data preamble is deliberately ignored, never executed.
            continue
        tokens = shlex.split(stripped, comments=True, posix=True)
        if tokens[0] not in {"$tmer", "${tmer}"} or tokens.count("x") != 1:
            raise ValueError(f"unsupported reference-row grammar at line {number}")
        sep = tokens.index("x")
        names = []
        for token in tokens[1:sep]:
            match = _SPECIES.fullmatch(token)
            if not match or ".." in match.group(1):
                raise ValueError(f"unsafe/unsupported species token at line {number}: {token}")
            names.append(match.group(1))
        tail = tokens[sep + 1 :]
        if len(tail) != len(names) + 2 or tail[-2] not in {"$w", "${w}"}:
            raise ValueError(f"coefficient count mismatch at line {number}")
        coefficients = [float(c) for c in tail[:-2]]
        reference = float(tail[-1])
        merged = defaultdict(float)
        for name, coefficient in zip(names, coefficients):
            merged[name] += coefficient
        terms = [(name, coefficient) for name, coefficient in merged.items() if coefficient]
        if len(terms) < 2:
            raise ValueError(f"degenerate reference row at line {number}")
        rows.append(
            {"terms": terms, "reference": reference, "line": number, "source_line": stripped}
        )
    if not rows:
        raise ValueError("no supported GMTKN55 reference rows found")
    return rows


def _integer_sidecar(directory: Path, name: str, default: int) -> int:
    path = directory / name
    if not path.exists():
        return default
    value = path.read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"[+-]?\d+", value):
        raise ValueError(f"invalid integer metadata in {path}")
    return int(value)


def import_subset(
    checkout: str | Path,
    subset: str = "BH76",
    *,
    kind: str = "barrier",
    expected_commit: str | None = None,
) -> Dataset:
    root = Path(checkout).resolve()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", subset):
        raise ValueError("invalid subset name")
    directory = root / subset
    path = directory / ".res"
    rows = parse_res(path.read_text(encoding="utf-8"))
    commit = None
    dirty = None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            capture_output=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--", subset],
                cwd=root,
                text=True,
                capture_output=True,
                check=True,
                timeout=5,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        if expected_commit is not None:
            raise ValueError("cannot verify the requested dataset commit") from None
    if expected_commit is not None and (commit != expected_commit or dirty):
        raise ValueError("dataset checkout does not match the clean expected commit")
    revision = commit or "unversioned-local-copy"
    base_url = f"https://github.com/grimme-lab/GMTKN55/blob/{revision}/{subset}"
    structures, sources = [], {str(path.relative_to(root)): file_digest(path)}
    names = sorted({name for row in rows for name, _ in row["terms"]})
    for name in names:
        species = directory / name
        if species.is_symlink() or not species.resolve().is_relative_to(root):
            raise ValueError("species directories must stay within the checkout")
        charge = _integer_sidecar(species, ".CHRG", 0)
        unpaired = _integer_sidecar(species, ".UHF", 0)
        if unpaired < 0:
            raise ValueError("negative .UHF count")
        geom = species / "struc.xyz"
        reader = read_xyz
        if not geom.exists():
            geom = species / "coord"
            reader = read_turbomole
        kwargs = {
            "structure_id": f"{subset}:{name}",
            "charge": charge,
            "multiplicity": unpaired + 1,
            "source": f"{base_url}/{name}/{geom.name}",
        }
        structures.append(reader(geom, **kwargs))
        for source in (geom, species / ".CHRG", species / ".UHF"):
            if source.exists():
                sources[str(source.relative_to(root))] = file_digest(source)
    observables, canonical = [], {}
    for index, row in enumerate(rows, 1):
        terms = tuple(Term(f"{subset}:{name}", coefficient) for name, coefficient in row["terms"])
        # Both forward and reverse barriers at the same TS share one split group.
        # This is TS grouping, NOT a claim of reaction-family independence.
        positive = sorted(t.structure_id for t in terms if t.coefficient > 0)
        group = f"{subset}:ts-{digest(positive)[:16]}"
        key = digest(sorted((t.structure_id, t.coefficient) for t in terms))
        oid = f"{subset}:{index:03d}"
        metadata = {
            "source_line": row["line"],
            "subset": subset,
            "duplicate_of": canonical.get(key),
            "source_expression": row["source_line"],
        }
        canonical.setdefault(key, oid)
        reference = Reference(
            row["reference"],
            EnergyUnit.KCAL_MOL,
            "published GMTKN55 reference; inspect source .res and .bib",
            f"{base_url}/.res#L{row['line']}",
        )
        observables.append(Observable(oid, group, kind, terms, reference, metadata=metadata))
    return Dataset(
        f"GMTKN55/{subset}",
        structures,
        observables,
        {
            "source_repository": "https://github.com/grimme-lab/GMTKN55",
            "source_commit": commit,
            "source_dirty": dirty,
            "input_sha256": sources,
            "source_license": "CC-BY-4.0 (dataset)",
            "citation": "Goerigk et al., PCCP 2017, DOI:10.1039/C7CP04913G; plus per-subset .bib",
            "grouping": "shared positive/transition-state species; not full reaction-family holdout",
            "reference_review_required": True,
            "synthetic": False,
        },
    )
