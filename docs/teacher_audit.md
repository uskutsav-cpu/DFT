# Teacher-equivalence audit

**The shipped ORCA config is a candidate, not a replication claim.** OMol25's
published nominal level is ωB97M-V/def2-TZVPD in ORCA 6. The upstream generation
code also specifies auxiliary/basis assets, numerical controls, occupations and
state-dependent settings. Functional/basis labels alone do not establish exact
teacher equivalence.

Before setting `teacher_verified: true`, record in `teacher_audit_note` the actual
comparison and freeze at least the following:

- upstream generation-code commit and selected state-specific branch;
- ORCA release, executable hash, functional/dispersion/nonlocal treatment;
- orbital and auxiliary basis files (including the upstream basis asset), all
  corresponding file hashes, and any pseudopotential/ECP treatment;
- quadrature, integral thresholds, SCF convergence/initialization, symmetry, and
  RIJCOSX settings;
- charge, multiplicity, restricted/unrestricted conventions, broken-symmetry or
  occupation handling when relevant;
- same-geometry comparison on a small accessible reference subset, residual
  discrepancies, and the acceptance tolerance chosen before the full study.

External assets are supplied by the user through the ORCA `assets` mapping and
matching `asset_sha256`; the adapter verifies them before use. No copyrighted
basis bundle or licensed executable is redistributed here.

Use one identical explicit `teacher_id` for the MLIP and its audited DFT records.
IDs name the audited setup, not simply the model family. Multiple MLIPs trained on
a nominally similar corpus are not automatically a calibrated ensemble or proof
of identical reference conventions.

`assemble --allow-unverified-teacher` exists for engineering/exploration. It sets
an audit-required warning and must not support a clean "surrogate vs teacher"
scientific claim. It is not a shortcut to validating this project.

For non-DFT high-level ORCA configurations, an explicit reviewed
`correlation_convergence_pattern` is required. It must match the correct
method/version's positive convergence output. The generic SCF/normal-termination
check alone cannot establish correlated-method convergence. This adapter does
not choose the right high-level method or active space.

Sources: FAIR Chemistry OMol25 documentation and the original `om-data` ORCA
calculation generator, linked in `sources.md`.
