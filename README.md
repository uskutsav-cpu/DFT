# RefBlind: Reference-Blind Failure Detection for MLIP/QM Workflows

**Status: ACTIVE RESEARCH / PRE-RESULTS.** This repository defines hypotheses, benchmark interfaces, metrics, and an escalation-policy scaffold. It does **not** claim that reference-blind failures are common, that the proposed diagnostics work, or that the proposed policy outperforms baselines until those claims are supported by benchmark results.

## Research question

When a DFT-trained machine-learning interatomic potential (MLIP) agrees with its DFT teacher, can the underlying DFT reference still be chemically wrong enough to change a barrier, pathway, state ordering, or mechanistic conclusion—and can those cases be detected cheaply enough to escalate only the calculations that need higher-level electronic structure?

## Core decomposition

For an energy-like observable,

\[
\varepsilon_{\mathrm{surrogate}} = E_{\mathrm{MLIP}} - E_{\mathrm{DFT}}
\]

\[
\varepsilon_{\mathrm{reference}} = E_{\mathrm{DFT}} - E_{\mathrm{HL}}
\]

and therefore

\[
E_{\mathrm{MLIP}}-E_{\mathrm{HL}}
= \varepsilon_{\mathrm{surrogate}} + \varepsilon_{\mathrm{reference}}.
\]

Here `HL` means a **state-appropriate higher-level reference**, not automatically CCSD(T). Systems with substantial multireference character require a method appropriate to that regime.

A **reference-blind failure** is operationally a case where MLIP-vs-DFT error is small, but DFT-vs-higher-level error is chemically consequential—ideally defined by a changed decision such as barrier/pathway/state ordering, not by a universal kcal/mol cutoff alone.

## Hypotheses

1. MLIP uncertainty/OOD scores are primarily sensitive to surrogate-model mismatch and may miss reference-method error.
2. Inexpensive electronic-structure diagnostics may provide complementary signal about DFT-reference reliability.
3. A calibrated escalation policy may preserve chemically correct decisions at lower computational cost than always using the highest-fidelity method.

## Planned pipeline

```text
reactive structures / pathways
          |
          v
     foundation MLIP
          |
          +---- MLIP uncertainty / OOD features
          |
          v
          DFT
          |
          +---- inexpensive electronic diagnostics
          |
          v
    escalation policy
      /          \
  accept       higher-level
 MLIP/DFT   electronic structure
      \          /
       chemical decision
```

The primary evaluation target is **decision preservation under compute constraints**, not just pointwise energy MAE.

## Repository layout

```text
src/refblind/       Core scientific bookkeeping and policy logic
configs/            Explicit benchmark and escalation configuration
benchmarks/         Manifest schema + example reaction entries
scripts/            Reproducible smoke/demo utilities
workflows/          Pipeline entry-point templates
tests/              Unit tests for error and decision logic
docs/               Research question, benchmark protocol, reproducibility
.github/workflows/   CI
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
python scripts/smoke_demo.py
```

## Baselines to compare

| Policy | MLIP UQ | Electronic diagnostics | DFT escalation | HL escalation |
|---|---:|---:|---:|---:|
| Always MLIP | no | no | no | no |
| Always DFT | no | no | yes | no |
| MLIP-UQ threshold | yes | no | yes | no |
| Fixed hierarchical thresholds | yes | yes | yes | yes |
| Calibrated/learned gate | yes | yes | yes | yes |
| Oracle ceiling | eval only | eval only | yes | yes |

The oracle is an analysis ceiling only, never a deployable method.

## Primary metrics

- reference-blind failure prevalence under a pre-registered operational definition;
- AUROC/AUPRC for detecting chemically consequential reference failures;
- calibration and risk-coverage behavior;
- barrier/pathway/state-ordering preservation;
- fraction of DFT and higher-level calls;
- compute-normalized error / utility;
- stratification by chemistry, charge, multiplicity, reaction class, method, and diagnostic regime.

## Scientific guardrails

- Never use higher-level labels as gate features at evaluation time.
- Keep raw electronic-structure outputs immutable.
- Record method, basis, code/version, charge, multiplicity, convergence flags, geometry provenance, and random seeds where applicable.
- Separate model-selection data from final test systems.
- Treat failed SCF/optimization/TS calculations as outcomes with provenance, not silently dropped rows.
- Report negative results and regimes where the gate fails.
- Do not convert exploratory thresholds into post-hoc “pre-registered” thresholds.

See [`docs/benchmark_protocol.md`](docs/benchmark_protocol.md), [`docs/research_question.md`](docs/research_question.md), and [`docs/reproducibility.md`](docs/reproducibility.md).
