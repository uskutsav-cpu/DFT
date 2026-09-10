# Known limitations — read before reporting results

1. **No empirical chemistry validation in this release.** Statistical tests,
   mocks, and fabricated outputs cannot validate an MLIP, DFT setup, ORCA parser
   across arbitrary output variants, transition state, or high-level reference.
2. **Candidate teacher only.** The shipped ORCA header does not reproduce every
   OMol25 generation convention. Explicit audited equivalence remains necessary.
3. **Optional API compatibility.** UMA, MACE-OMOL and OrbMol factories were checked
   against official source/docs. No checkpoint inference was run. Upstream main
   APIs may differ from an installed tag; pin and smoke-test a compatible release.
4. **No automatic cheap reference-risk invention.** Orbital and spin helpers plus
   costed imports are implemented. FOD, alternate-functional, density-sensitivity,
   and active-space decisions still require reviewed calculations and physics.
5. **Reference quality is not ground truth.** A reviewed flag records human review.
   CCSD(T), localized approximations and finite-basis FCI have different regimes
   and limitations. The code does not select active spaces or certify states.
6. **Complete eligible cohort.** Modeling excludes missing/failed pairs,
   questionable labels, exact duplicates, and threshold-ambiguous intervals.
   Exclusion reasons are retained, but policy accuracy is conditional on the
   resulting test cohort, not all input jobs. Report both denominators.
7. **Fixed geometry first.** The initial study uses balanced electronic-energy
   observables. No solvent, ZPE, entropy, kinetics, or free-energy correction is
   inferred. A reaction barrier ordering alone may not establish mechanism.
8. **Grouping and overlap.** Shared-TS import grouping prevents a specific leak,
   not all chemical-family overlap or MLIP pretraining contamination. Supply and
   review stronger family/scaffold groups for the final study.
9. **Compute costs.** Defaults are relative estimates. Wall times exclude model
   load time in ASE inference, and imported ORCA jobs have no measured wall time.
   Training/data-acquisition costs and shared cluster scheduling overhead are not
   priced by replay. Array-task budgets are per task, not global.
10. **High-level availability.** Literature references cannot stand in for an
    obtained high-level prediction. Missing, failed or unsupported high-level
    branches abstain. No simulated oracle is presented as a deployable policy.
11. **Live controller.** Replay models the sequential information/cost boundary;
    it is not a closed-loop production scheduler that decides and launches all
    chemistry automatically. The calculation planner/runner and learned-policy
    replay are separate, interoperable components.
12. **Pathways.** Interpolation requires confirmed atom mapping; optional ASE NEB
    has safety checks but was not run. A converged band is not a confirmed TS.
    Frequencies and IRC evidence are supplied separately; no automatic mechanistic
    interpretation or transition-state certification is implemented.
13. **Cache scope.** Input/settings hashes and recorded runtime identity help,
    but an unpinned remote checkpoint alias or changed external executable
    dependency can invalidate reproducibility. Resolve/audit hashes before freeze.
14. **Statistics.** The group-max conformal helper assumes exchangeable groups and
    returns unbounded intervals when the calibration sample cannot support the
    requested level. It is not a chemistry safety certificate. Bootstrap intervals
    from very few groups can be unstable. Hypotheses remain falsifiable.
15. **Licensing.** Data, model weights, chemistry executables and any eventual
    project license are separate decisions. No redistribution rights are assumed.
