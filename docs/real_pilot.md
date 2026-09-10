# From the software to a real pilot

The commands below are user-run workflow commands. Paths under data/plans/runs
are generated and ignored by Git. No real benchmark data or model is bundled.
Use separate environments for incompatible chemistry/model dependencies.

## 1. Check installation and a label-free fixture

```bash
refblind doctor
refblind validate examples/h2-smoke.json
refblind plan --dataset examples/h2-smoke.json \
  --config configs/pyscf_h2_smoke.yaml --output plans/h2-pyscf.json
refblind run --plan plans/h2-pyscf.json --output runs/h2-pyscf
```

This final command is dry-run only. Optional real small-system PySCF execution:

```bash
python -m pip install -e '.[pyscf]'
refblind run --plan plans/h2-pyscf.json --output runs/h2-pyscf \
  --execute --limit 1 --budget 2
```

That is a PBE/STO-3G engineering check, not the OMol teacher or a reaction benchmark.
The same command resumes without rerunning a verified successful cache entry.
Use `--retry-failed` deliberately after reviewing a failure. The budget is a
per-invocation configured estimate, not a billing or global scheduler limit.

## 2. Import a reviewed upstream subset

```bash
bash scripts/fetch_gmtkn.sh external/GMTKN55
refblind import-gmtkn --checkout external/GMTKN55 --subset BH76 \
  --expected-commit "$(git -C external/GMTKN55 rev-parse HEAD)" \
  --output data/bh76-import.json
refblind validate data/bh76-import.json
```

The script checks out the upstream `v2` ref and prints the exact commit. Review
that commit and the references, correction history, license and `.res` semantics
before freezing. The importer parses `.res` as text; it never executes its shell
instructions. It recognizes the inspected GMTKN55 grammar and deliberately
fails on unsupported variants. It does not promise every future subset format.

Forward/reverse and shared-transition-state observations are grouped, and exact
duplicate observables are marked. This is not automatically a stringent chemical
family/scaffold holdout. Establish a chemically reviewed grouping rule before the
confirmatory study. Check possible benchmark/pretraining overlap; an unfamiliar
geometry to this repo is not proof it was unseen in MLIP training.

Reference values are initially unreviewed. Apply actual human review records with
`refblind review-references`; do not manufacture trusted flags. Published
observable references are sufficient for error labels, but not for simulating
already-computed high-level predictions.

## 3. Audit and run one actual backend at a time

UMA/MACE/OrbMol configs default to **no checkpoint download permission**. Install
an appropriate optional extra in a compatible environment, review the model
terms, pin the model and code, record resolved checkpoint hashes, and explicitly
change `allow_model_download` only when authorized. UMA may also require external
account access. Do not put tokens into the YAML or repository.

Audit ORCA teacher equivalence before creating `configs/local-teacher.yaml`.
Keep machine-specific paths out of shared commits. `plan` renders reviewable
inputs without execution. A real launch requires `run --execute --limit 1`.
After one species succeeds, independently compare raw and parsed energy, units,
state, convergence, and geometry. Then run a tiny multi-species observable.
Only scale after checking that observable against an independent calculation.

## 4. Bring outputs together

```bash
refblind assemble --dataset data/bh76-reviewed.json \
  --calculations runs/uma/calculations.json \
  --calculations runs/mace/calculations.json \
  --calculations runs/orca/calculations.json \
  --mlip-method uma-primary --mlip-method mace-secondary \
  --dft-method audited-teacher --output data/paired-table.json
```

Method IDs above are examples and must match your actual configs. With no
computed high-level results supplied, escalation to high level must abstain.
Add a separate high-level calculation file and `--high-level-method` only when
those calculations exist and their applicability has been reviewed.

Cheap electronic features can be supplied through `attach-features` with real
source hashes and explicit costs. Orbital gap/spin helpers are descriptors, not
universal certificates. Separate FOD calculations, alternate DFAs, or HF-density
calculations are not automatically executed by the study command.

For jobs launched through ChemRefine or another system, use the file exchange
contract: `<root>/<job_id>/input.inp`, `orca.out`, optional `input.engrad`, and
matching basis assets, then `refblind import-orca`. This is an implemented
file-based integration, not a claimed native ChemRefine API plugin.

## 5. Develop, freeze, then evaluate

```bash
refblind fit --table data/paired-table.json --config configs/study_v1.yaml \
  --output results/frozen-policy.json
refblind evaluate --table data/paired-table.json \
  --artifact results/frozen-policy.json --output results/heldout --plots
```

Replace the illustrative study tolerances with a prospectively documented
protocol. Pilot data used to choose the protocol must not become an untouched
confirmatory test set. Freeze source data, grouping, reference review, features,
thresholds, and model identities. The program protects its frozen artifact from
accidental mismatch; it cannot prevent a researcher from repeatedly inspecting
test outcomes and redesigning the experiment outside the program.

## 6. Analyze failures before expanding

Check the reference-blind quadrant, base rates, teacher mismatch, reference
uncertainty and exclusions before training more complex models. Sparse positives
may not support a stable classifier. Compare the cost of actually acquired
features, raw failure rates, and source overlap. Report null findings and cases
where a high-level method does not settle the reference.
