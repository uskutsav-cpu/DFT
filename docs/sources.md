# Primary implementation sources

Checked 2026-09-10. These links informed API/contracts; inclusion does not mean
this release performed live calculations or fully validated the scientific method.
Pin reviewed upstream commits/releases rather than relying on moving `main` refs.

| Component | Primary source and what was checked |
|---|---|
| OMol25 | https://fair-chem.github.io/omol25/ — nominal reference level, units and generation context |
| Original ORCA generator | https://github.com/Open-Catalyst-Project/om-data/blob/main/omdata/orca/calc.py — basis asset, numerical and state-specific settings |
| UMA | https://fair-chem.github.io/uma-tutorial/ — `get_predict_unit`, `FAIRChemCalculator`, `omol`, charge/spin, gated access |
| MACE-OMOL | https://github.com/ACEsuit/mace/blob/main/mace/calculators/foundations_models.py — `mace_omol`, local weights, device/dtype |
| OrbMol | https://github.com/orbital-materials/orb-models/blob/main/README.md and https://github.com/orbital-materials/orb-models/blob/main/orb_models/forcefield/pretrained.py — current tuple-return factory and inference calculator; v1/v2 distinction |
| GMTKN55 | https://github.com/grimme-lab/GMTKN55/tree/v2 — inspected BH76 `.res`, `.UHF`, coordinate files and reference comments; importer does not execute shell commands |
| ORCA FOD | https://www.faccts.de/docs/orca/6.1/manual/contents/spectroscopyproperties/fod.html — diagnostic nature and separate calculation settings |
| ORCA correlated methods | https://orca-manual.mpi-muelheim.mpg.de/contents/modelchemistries/mdci.html — correlated method scope and diagnostic limitations |
| PySCF | https://pyscf.org/quickstart.html ; https://pyscf.org/user/cc.html ; https://pyscf.org/user/ci.html — molecular SCF/DFT/CC/FCI interfaces |
| ASE NEB | https://docs.ase-lib.org/ase/neb.html — `ase.mep.NEB`, FIRE, method choice; explicitly set improvedtangent rather than relying on a changing default |
| ChemRefine | https://sterling-group.github.io/ChemRefine/ — workflow integration context; implemented bridge is file-based, not a guessed private Python API |
| Metrics | https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html ; https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html — test-oracle conventions for AP/AUROC |
| Ruff | https://docs.astral.sh/ruff/ — optional development lint/format tooling |

Dataset papers and reference corrections must be cited separately in an actual
scientific manuscript. This is an implementation-source register, not an exhaustive
novelty search or a claim of literature priority.
