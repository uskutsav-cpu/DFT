# RefBlind
### Reference-blind error analysis and sequential MLIP → DFT → high-level escalation

**v0.2.0 · Research software · No empirical chemistry result is claimed.**

RefBlind asks whether a machine-learning interatomic potential (MLIP) can reproduce
its DFT teacher while that teacher still mispredicts a chemically important
observable. It provides an auditable benchmark pipeline, not an assertion that
this happens often or that a proposed diagnostic solves it.

For a balanced reaction observable `Q` (a barrier, reaction energy, or state gap):

```text
surrogate error = Q_MLIP − Q_DFT
reference error = Q_DFT  − Q_reference
total error     = Q_MLIP − Q_reference
```

Use **the same geometries, states, stoichiometry, and units** at every level.
Comparing arbitrary absolute energies from different implementations is not the
benchmark. A published high-level observable is an evaluation label, **not** a
completed high-level calculation that an escalation policy can receive for free.

## Run immediately: portable synthetic integration test

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,analysis]'
python -m pytest -q -m 'not integration'
refblind doctor
refblind demo --output results/synthetic-demo --plots
```

Use a fresh output directory on repeated runs. The demo creates a marked synthetic
study table, grouped splits, fitted/calibrated models, a frozen policy artifact,
held-out predictions, metrics, CSV/Markdown reports, and three figures. It is a
software test with intentionally designed signal—not evidence about real DFT.

## Implemented

| Layer | Working functionality |
|---|---|
| Data | Typed structures/calculations/observables; charge–multiplicity and atom/charge balance checks; explicit units; immutable hashes; strict JSON/YAML |
| Import | Local GMTKN55 `.res`, XYZ/coord, `.CHRG` and `.UHF` import; no execution of downloaded shell text; explicit reference-review records |
| Chemistry interfaces | ORCA single-point input/output and timeout handling; optional UMA, MACE-OMOL, OrbMol ASE adapters; optional small-system PySCF HF/DFT/CCSD(T)/finite-basis FCI |
| Jobs | Dry-run plans; SQLite claim tokens; cached result integrity; explicit retries; failure retention; per-invocation resource/budget guards; file exchange with externally run ORCA jobs |
| Features | Ensemble summaries, Mahalanobis OOD, orbital-gap/spin helpers; costed auxiliary feature import; stage whitelist; no label-only features |
| Modeling | Train-only imputation/scaling; regularized logistic models; separate calibration set; JSON model serialization; group-max conformal diagnostic |
| Evaluation | Exact-tie AUROC/AP, Brier/calibration, risk–coverage, grouped bootstrap, error decomposition, decision-order auditing, policy Pareto analysis |
| Decision policy | Actual two-stage access; screening-all-DFT baseline; optional costed early screen; explicit accept/execute/abstain; shared operation cost deduplication |
| Reproducibility | Separate fit/evaluate commands, frozen hashes and splits, environment capture, auditable exclusions, manuscript-oriented CSV/JSON/Markdown/PNG outputs |

**Integration status matters.** The portable tests exercise fabricated chemistry
outputs and mocked external APIs. No ORCA/UMA/MACE/OrbMol computation was executed
when this upgrade was authored. PySCF is optional, and its real smoke test is
marked `integration`. See [validation scope](docs/validation.md).

## The sequential decision is not a one-shot threshold

```text
MLIP outputs (+ optional separately charged cheap screen)
  ├─ accept MLIP
  └─ run DFT
       ├─ accept DFT
       └─ request state-appropriate high level
             ├─ use an actually available high-level result
             └─ abstain if unsupported, failed, absent, or over budget
```

Low MLIP uncertainty can miss a reference-blind failure entirely. The comparison
therefore includes both a pure-UQ route and a DFT-screen-all route; the software
does not pretend that DFT diagnostics were available before DFT. A calibrated
early screen must be supplied and paid for; the code does not invent one.

## Start a real pilot without launching compute

```bash
refblind validate examples/h2-smoke.json
refblind plan --dataset examples/h2-smoke.json \
  --config configs/orca_candidate.yaml --output plans/h2-orca.json
refblind run --plan plans/h2-orca.json --output runs/h2-orca
```

The last command is a **dry run**. The H2 geometries are illustrative input
fixtures with no high-level label and no transition-state claim. Only add
`--execute --limit 1` after installing the backend and auditing its configuration.
The full benchmark path is documented in [real pilot](docs/real_pilot.md).

The included ORCA header is **not verified OMol25 teacher equivalence**. Matching a
functional/basis name is insufficient: basis assets, numerical settings, electronic
state conventions, and implementation all require review. Assembly refuses a
matched-teacher claim unless explicitly verified, or the user chooses the clearly
marked `--allow-unverified-teacher` exploratory route.

## Repository map

```text
src/refblind/
  schema.py, datasets.py, units.py     data contracts and chemistry bookkeeping
  gmtkn.py, review.py                 import and explicit reference review
  calculators/                       optional computational engines
  execution.py, jobstore.py           resumable calculation workflow
  exchange.py, attachments.py         audited external outputs/features
  observables.py, assembly.py         species → balanced observables → study table
  features.py, learning.py            stage-safe preprocessing and risk models
  splits.py, uncertainty.py           grouped partitions and UQ helpers
  policy.py, replay.py, study.py      staged cost-aware evaluation
  metrics.py, reporting.py            statistics, figures, and reports
  pathways.py                        mapped interpolation / optional ASE NEB
  cli.py                             refblind command-line interface
configs/   examples/   scripts/   tests/   docs/
```

The original v0.1 scalar/error and one-shot-policy APIs remain available for
compatibility. New research workflows use `policy.py` and `study.py`, not the
legacy `EscalationPolicy.decide()` as a deployable sequential algorithm.

## Research scope and limitations

The default 1/3 kcal/mol tolerances and relative costs are **illustrative settings,
not preregistered or experimentally calibrated values**. The primary study reports
an observable-error event and conditional eligible-test accuracy; it does not
turn a large error into a proven mechanism reversal. Failed, duplicate,
questionable-reference, and ambiguous-label observations remain in exclusion
reports. A clean complete-case result is not a reliability claim over every raw
calculation. No finite data set or conformal diagnostic here certifies all chemistry.

Read [architecture](docs/architecture.md), [data contract](docs/data_contract.md),
[teacher audit](docs/teacher_audit.md), [limitations](docs/limitations.md), and the
[completion roadmap](docs/roadmap.md). Primary software sources are recorded in
[docs/sources.md](docs/sources.md).

## Development

```bash
bash scripts/run_checks.sh
```

No model weights, licensed ORCA executable, benchmark archive, private transcript,
or credential is bundled. Model/code/data licenses must be checked independently.
No project-wide license decision is made by this upgrade. No institutional
ownership, collaboration agreement, or publication acceptance is implied.
