# Validation scope

`python -m pytest -q -m 'not integration'` runs the portable suite. Tests cover
unit conversion, data integrity, electronic-state/balance validation, observable
assembly, malformed files, leakage rejection, train-only fitting, calibration,
serialization, exact-score ties, sklearn cross-checks, sequential information
access, cost deduplication, abstention, job claims/cache/retries, timeout handling,
fabricated ORCA parsing, mocked ASE factories, CLI workflows and synthetic reports.

`python -m pytest -q -m integration` is a separate real small-system PySCF smoke
test when PySCF is installed. Skipping that test is not a passed chemistry run.
The repository includes no real ORCA executable, model weights or secret access.

The release's companion validation artifact records the actual environment,
command outputs, test count and coverage from the authoring container. Coverage
is a line-execution measure, not scientific verification; mocks count as executed
lines. Do not describe parser/mock tests as quantum-chemistry validation.

The authoring environment did not contain Ruff and could not download it.
Therefore local Ruff success is **not** asserted by this release. The terminal
installer installs Ruff and runs formatting, safe auto-fixes, lint, tests, and the
synthetic demo before committing; CI separately checks formatting/lint/tests.

Optional chemistry packages have their own interpreter/hardware constraints.
The core supports Python >=3.10; the actual authoring run used Python 3.13.
CI is configured for 3.10–3.13 but configured coverage is not proof those jobs
have run. Pin a complete resolved environment for a frozen scientific study.
