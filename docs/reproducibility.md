# Reproducibility policy

## Immutable layers

Recommended data flow:

```text
raw_outputs/        immutable program outputs
        ↓
parsed/             machine-readable normalized records
        ↓
features/           MLIP UQ + inexpensive electronic diagnostics
        ↓
results/            metrics, tables, calibrated policies
        ↓
figures/            publication artifacts
```

Large raw outputs should not be committed to Git unless deliberately managed with an appropriate data/versioning system.

## Configuration

Every benchmark run should capture the exact configuration file, code commit SHA, software environment, and random seed. Thresholds in `configs/` are scaffolding defaults until a protocol version is frozen.

## Determinism

Where numerical chemistry software is not bitwise deterministic, distinguish numerical reproducibility from exact binary reproducibility. Record convergence tolerances and important hardware/library information when it can affect results.

## Result claims

A figure/table is considered reproducible only when there is a documented command or workflow that regenerates it from versioned parsed data and configuration. Manual spreadsheet edits should not sit between raw results and manuscript figures.
