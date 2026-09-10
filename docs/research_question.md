# Research question and falsifiable claims

## Main question

Can a DFT-trained foundation MLIP appear reliable relative to its DFT teacher while the DFT reference is wrong enough to alter a chemically meaningful conclusion, and can a low-cost diagnostic layer identify those cases well enough to justify selective escalation to higher-level electronic structure?

## What would count as a positive result?

A convincing positive result requires more than finding isolated DFT errors. At minimum, the study should show all of the following on held-out chemical systems:

1. a non-trivial set of cases with low MLIP→DFT discrepancy and chemically consequential DFT→HL discrepancy;
2. conventional MLIP UQ/OOD metrics fail to rank at least some of these cases reliably;
3. added electronic diagnostics improve detection or risk calibration beyond MLIP-only baselines;
4. an escalation policy preserves more chemistry-level decisions at a matched compute budget (or uses less compute at matched decision accuracy) than transparent baselines.

## What would falsify or weaken the project?

The central thesis is weakened if any of the following dominate the benchmark:

- reference-blind failures are too rare to matter in the target chemistry;
- MLIP uncertainty already predicts reference failure once ordinary confounders are controlled;
- cheap electronic diagnostics add no out-of-sample information;
- higher-level labels are too method-dependent to support a stable target;
- the selective gate cannot beat simple always-DFT / fixed-threshold baselines under realistic cost accounting.

Negative outcomes are scientifically informative and must be reported rather than optimized away.

## Unit of analysis

Prefer chemically meaningful observables and decisions (activation barriers, competing pathways, electronic/spin states, or intermediate ordering) over isolated single-point energies whenever possible.
