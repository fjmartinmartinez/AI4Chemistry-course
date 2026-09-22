# Week 02 — Data and molecules in code (pandas, SMILES, RDKit, descriptors)

_Record of the approved design (syllabus v0.1). Not a proposal._

## Learning objectives
By the end of week 02 students can:
1. Load tabular chemical data into a pandas `DataFrame`; inspect, select, filter,
   sort and aggregate it (`groupby`). [LO1]
2. Convert SMILES to RDKit molecule objects, canonicalise them, draw them, and
   handle invalid input gracefully. [LO2]
3. Compute molecular descriptors (MW, logP, TPSA, H-bond donors/acceptors,
   rotatable bonds) and Morgan fingerprints, and attach them to a dataframe. [LO2]
4. (Session B) Assemble a clean, deduplicated, featurised molecular dataset from
   raw data and explain each cleaning decision. [LO2]
5. (Session B) Visualise chemical space with PCA and cluster molecules by
   fingerprint similarity (Butina / Tanimoto). [LO2, LO4]

## Prerequisites assumed
- Week 01: Python control flow and functions, NumPy arrays, matplotlib,
  reading a CSV.
- First-year concepts: partition coefficient / lipophilicity, hydrogen bonding,
  aqueous solubility, functional groups.

## Concept sequence
| # | Concept | Chemistry anchor | Where |
|---|---------|------------------|-------|
| 1 | DataFrame: read_csv, head/info/describe, dtypes | ESOL solubility table | notebook A |
| 2 | Selection, `.loc`/`.iloc`, boolean filter, sort | most/least soluble compounds | notebook A |
| 3 | New columns, `groupby`, aggregation | solubility vs number of rings | notebook A |
| 4 | SMILES ↔ RDKit mol; canonical SMILES; drawing; invalid input | tautomer/duplicate detection | notebook A |
| 5 | Descriptors via `apply`; RDKit vs dataset columns | MW, logP, TPSA, HBD/HBA | notebook A |
| 6 | Cleaning: NaN, invalid SMILES, duplicates, ranges | data hygiene for ML | notebook B |
| 7 | Featurisation: Morgan fingerprints, descriptor block | molecule → vector | notebook B |
| 8 | Chemical space: PCA scatter | coverage / outliers | notebook B |
| 9 | Similarity + Butina clustering; representatives | diversity selection | notebook B |
| 10 | `make_dataset()` pipeline → X, y, df | reproducible input for Week 03 | notebook B |

## Notebook plan
- **Session A** (`week02_a_data-and-molecules_*`): dataset `esol_delaney.csv`.
  Worked example: end-to-end inspection + descriptor augmentation of ESOL.
  4 exercises.
- **Session B** (`week02_b_data-and-molecules_*`): same raw dataset → produces
  `sources/datasets/esol_clean.csv` (committed) + a `make_dataset()` function.
  5 sections, 3 exercises + mini-challenge (reusable pipeline).
- Compute budget: PCA + Butina on ~1128 molecules, < 60 s on laptop CPU.

## Slides plan
- Session A ~8 slides: why dataframes; SMILES; RDKit mol objects; descriptors;
  "a molecule is a point in descriptor space". Demo → notebook A §1–§5.
- Session B ~8 slides: data hygiene checklist; fingerprints; chemical space;
  clustering; leakage preview. Demo → notebook B §1–§5.
- Figures: `solubility_vs_logp.png`, `descriptor_hist.png`,
  `chemical_space_pca.png`, `cluster_sizes.png`.

## Assessment hooks
- Examinable: reading/filtering a dataframe; canonical SMILES and why they
  matter; what each descriptor measures; why deduplication must use canonical
  SMILES; what PCA of fingerprints does and does not show.

## Sources used
- SciCompforChemists ch. 5 (pandas).
- EPFL ai4chem 01b (pandas), 01d (RDKit basics) — Colab alternatives; ESOL load.
- TeachOpenCADD T001 (ChEMBL — mentioned, not run live), T002 (Ro5 / ADME
  descriptors), T005 (fingerprints, Tanimoto, Butina clustering).
- Dataset: ESOL / Delaney (`esol_delaney.csv`, checksum-pinned download).

## Open questions for instructor
- Confirm we standardise on **Morgan (ECFP-like) radius 2, 2048 bits** as the
  course-default fingerprint (used here and reused in weeks 03–07).
- ChEMBL querying (T001) is described but not executed in the self-guided
  notebook to avoid a network dependency — is a cached ChEMBL subset wanted for
  a later week instead?
