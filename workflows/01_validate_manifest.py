from __future__ import annotations

import csv
from pathlib import Path

REQUIRED = {
    "system_id",
    "reaction_class",
    "charge",
    "multiplicity",
    "mlip_model",
    "dft_method",
    "basis",
    "high_level_method",
    "decision_type",
}


def validate(path: str | Path) -> int:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"manifest missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("manifest contains no systems")
    return len(rows)


if __name__ == "__main__":
    count = validate("benchmarks/reactions.example.csv")
    print(f"validated {count} manifest row(s)")
