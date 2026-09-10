# Architecture and trust boundaries

## 1. Calculation layer

`Structure` identifies symbols, fixed coordinates (angstrom), total charge, and
spin multiplicity. Its geometry hash also includes electronic state. A
`Calculation` records method identity, settings hash, software/version, units,
status, energy/forces/diagnostics, raw-output hash, wall time, and supplied cost.
A failed calculation has no numeric energy. A zero energy is never a failure code.

The planner writes immutable JSON jobs; ORCA plans also emit reviewable input
files. Execution is disabled without `--execute`. Job identities include input
configuration; runtime identities include the backend version or executable
hash. Local file checkpoint hashes are verified where supported. Remote model
aliases remain a limitation: record and audit resolved checkpoint bytes before
using caches for a frozen study. Identical alias text does not prove identical
weights across time.

SQLite WAL and claim tokens prevent two local workers from claiming one pending
job. A crashed `running` claim is not automatically stolen: confirm its process is
dead and reconcile the ledger deliberately. SQLite is not a distributed HPC
scheduler; do not assume WAL works safely on arbitrary network filesystems.

## 2. Observable layer

`Observable` is a signed sum of species energies. Atoms and charge must balance.
For a bimolecular activation barrier the reactant side includes both reactants,
not just whichever species is convenient. `assemble` checks geometry, method,
settings, fidelity, and availability and converts to kcal/mol. Model ensemble
spread is computed on the observable, avoiding arbitrary model energy offsets.

Published reference values stay in the `Reference` record. A computed high-level
candidate stays in a separate `Calculation` and later `high_level_value`. Assembly
never converts a literature reference into a fictitious calculation.

## 3. Feature layer

MLIP-stage features and DFT-stage features are explicitly whitelisted. Reference
energies, signed errors, T1/D1 diagnostics, and high-level results are forbidden
as cheap gate inputs. A DFT-derived descriptor is paid for only after DFT.

FOD and cross-method disagreement generally need additional calculations. They
are not silently read from a DFT record as free features. Import them through
`attach-features` with operation IDs/costs and a source hash. The source hash is a
traceability record, not proof that the supplied physical descriptor is correct.

## 4. Learning and replay

Development uses grouped train, calibration, validation, and test partitions.
Preprocessing is fitted on training rows only. A regularized logistic classifier
is calibrated on its separate calibration partition. Policies are selected on
validation data; all model parameters and choices are frozen before evaluation.

The stage-2 predictor is invoked by replay only after the DFT and auxiliary
operations have been purchased. A shared operation key is charged once across
related observations. Costs are expressed in one consistent unit; default
relative values are estimates, not wall-clock guarantees. Training/reference
acquisition costs are outside inference replay and must be reported separately.

## 5. Evaluation population

The implemented detection study is conditional on complete paired MLIP/DFT/
reference observations and eligible reference quality. Duplicate, failed,
unreviewed (unless exploratory), questionable, and interval-ambiguous rows are
audited separately. Within the evaluated cohort, abstentions count as unresolved
in all-case accuracy. Do not call this unconditional coverage of the raw benchmark.

A reported reference-blind event is a thresholded observable error. Chemical
mechanism reversal additionally needs a genuinely comparable decision group,
credible reference ordering, and numerical uncertainty margins.
