# Data contract

Versioned JSON, not pickle, is the interchange format. Unknown malformed schemas,
duplicate JSON/YAML keys, and NaN/Infinity are rejected. Missing values are `null`.

## Dataset

A dataset contains `schema_version`, `name`, `structures`, `observables`, and
`metadata`. `examples/h2-smoke.json` is a runnable illustration, not research data.
Each structure carries `id`, `symbols`, `positions_angstrom`, `charge`,
`multiplicity`, and source/provenance. Each observable carries an `id`,
`group_id`, `kind`, signed `terms`, optional `reference`, optional
`decision_group`, and metadata.

A Reference specifies its numerical value/unit, source, method, quality,
optional uncertainty, and review note. Supported qualities are defined by
`ReferenceQuality` in `schema.py`. Imports are unreviewed. Do not mark a source
trusted because its name contains CCSD(T); method/state/basis suitability matters.

## Calculation records

Save/load using `save_calculations` / `load_calculations`. Supply each
`(structure_id, method_id)` once. Success requires finite energy; failures have
`energy=null` and an explicit status. Same method ID must mean identical method
settings for the species being combined. Do not combine relaxed MLIP coordinates
with fixed DFT coordinates in a paired single-point error decomposition.

## Study tables

`assemble` produces one row per observable with separate `mlip`, `dft`,
`reference`, `high_level_value`, and `high_level_supported` fields. Features are
separate from labels. All observable energies are normalized to kcal/mol.
The `operations` ledger permits shared species computations to be charged once.
Do not merge operation IDs for physically distinct jobs, and do not assign
inconsistent prices to one ID.

A `cheap_screen` feature must be a user-supplied probability-like score in [0,1].
The program validates range, not empirical calibration. An ensemble standard
deviation is a raw descriptor, not automatically an error probability.

## Reference review interchange

A review file is a JSON list. Each record specifies an actual `observable_id`,
`reviewer`, `quality`, `note`, and optionally `uncertainty`. The review operation
changes provenance and quality, not the reference energy. A reviewer must check
the actual paper/data and explain the choice; fabricated review records are not
included in this repository.

## Auxiliary features

A feature attachment has `row_id`, stage (`screen` or `diagnostics`), `features`,
`operations` (identity → nonnegative cost), and `provenance`. Provenance needs a
`method` and 64-character SHA-256 `source_sha256`. Existing nonmissing features
cannot be overwritten silently. Use `attach-features --help` for file arguments.

## Units

Supported energy units are eV, hartree, kcal/mol, and kJ/mol. Coordinates are
angstrom internally. Turbomole coordinates are converted from bohr when needed.
ORCA engrad values are gradients in hartree/bohr; the parser changes sign and
converts to forces in eV/angstrom. No zero-point, thermal, solvent, or entropy
correction is inferred from electronic energies. Such corrections require their
own reviewed bookkeeping and consistent observables.
