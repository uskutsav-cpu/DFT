# Benchmark protocol

## 1. Freeze the target question

Before final evaluation, freeze:

- target chemistry / reaction families;
- accepted MLIP families and checkpoints;
- DFT methods and basis settings;
- definition of the state-appropriate higher-level reference;
- operational reference-blind criteria;
- train/validation/test split rule;
- primary decision and cost metrics.

Exploratory revisions should be versioned rather than silently replacing the frozen protocol.

## 2. Data provenance

Each structure/calculation should record at minimum:

- stable system/configuration ID;
- parent reaction/pathway ID;
- geometry provenance and units;
- charge and multiplicity;
- electronic-structure program and version;
- method, basis/pseudopotential, dispersion model, and key numerical settings;
- convergence status and diagnostics;
- MLIP model/version/checkpoint;
- random seed where stochasticity exists;
- wall time / compute resource metadata when cost is analyzed.

Raw output files are immutable. Parsed tables are derived artifacts with a reproducible parser version.

## 3. Label separation

Higher-level calculations may define evaluation labels, but **must not leak into gate features**. Any feature available to the escalation model must be computable at the fidelity available at decision time.

Examples:

- MLIP-stage features: ensemble disagreement, representation distance, force disagreement, geometry/OOD summaries.
- DFT-stage features: inexpensive wavefunction/electronic diagnostics available from the DFT calculation or deliberately cheap auxiliary calculations.
- prohibited test-time feature: any quantity requiring the held-out higher-level label that the gate is supposed to decide whether to compute.

Feature normalization/calibration parameters are learned only on training/validation data.

## 4. Splitting

Random structure-level splits are usually too optimistic when neighboring geometries from one pathway appear in multiple splits. Group by reaction/pathway/family so correlated configurations cannot leak across train and test. Report the exact grouping rule.

## 5. Failures are data

Do not silently remove SCF failures, TS optimization failures, spin contamination, non-converged coupled-cluster jobs, or broken pathways. Record them with explicit failure codes and decide prospectively how each enters evaluation.

## 6. Evaluation

Primary evaluation should pair chemistry correctness with compute cost:

- barrier/pathway/state-ordering accuracy;
- reference-blind failure detection AUROC/AUPRC;
- calibration / risk-coverage;
- DFT call fraction;
- high-level call fraction;
- relative/wall-clock compute;
- correctness at matched budget and cost at matched correctness.

Compare against always-MLIP, always-DFT, MLIP-UQ thresholding, fixed hierarchical thresholds, and an oracle analysis ceiling.

## 7. Robustness

At minimum stratify by reaction class, charge, multiplicity, DFT method, higher-level method, and diagnostic regime. If a result depends on one narrow subset, report that limitation explicitly.
