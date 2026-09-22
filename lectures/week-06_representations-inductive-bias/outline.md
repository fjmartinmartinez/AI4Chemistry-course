# Week 06 — Representations and inductive bias

_Record of the approved design (syllabus v0.1). Not a proposal._

## Learning objectives
By the end of week 06 students can:
1. Explain why raw atomic coordinates are not directly usable as ML features,
   using the concrete failure mode of translational variance. [LO6]
2. State the formal definitions of permutation/translation/rotation
   invariance and equivariance, and classify a given descriptor or
   architecture against them. [LO6]
3. Place hand-engineered descriptors, fingerprints and (later) learned
   graph/point-cloud representations on a "how much is hand-engineered vs
   learned" spectrum. [LO6]
4. Run a representation shoot-out: hold the model fixed and vary the
   representation (or vice versa) to isolate which one drives performance. [LO4, LO6]
5. State, qualitatively, what an equivariant neural network is and why 3D
   molecular ML cares about SE(3) — no implementation this week (deliberate
   scope cut, per `syllabus.md`). [LO6]
6. (Session B) Scope, plan and begin an end-to-end mini-project — the
   module's own applied research task, assigned this week. [LO8]

## Prerequisites assumed
- Week 02: descriptors, Morgan fingerprints, canonical SMILES.
- Week 03: leaderboard pattern (fixed split, multiple models).
- Week 05: MLP and the "MLP vs random forest" honest comparison.

## Concept sequence
| # | Concept | Chemistry anchor | Where |
|---|---------|------------------|-------|
| 1 | Why raw coordinates fail: translational variance | a molecule's energy should not depend on where you put the origin | notebook A |
| 2 | Formal invariance/equivariance definitions | permutation, translation, rotation | notebook A |
| 3 | Building invariant descriptors from a point cloud | pairwise distances, radius of gyration | notebook A |
| 4 | Descriptors vs fingerprints vs learned features: a spectrum | Weeks 02/03/05 revisited | notebook A |
| 5 | Equivariant NNs (qualitative only) | SE(3), Tensor Field Networks/E3NN (named, not implemented) | notebook A |
| 6 | Representation shoot-out: same model, 3 representations | ESOL, descriptors vs MACCS vs Morgan | notebook B |
| 7 | Representation × model grid | 3 representations × 3 models | notebook B |
| 8 | Chemical space per representation | PCA per representation | notebook B |
| 9 | Mini-project assigned | end-to-end study, Weeks 06→10 | notebook B |

## Notebook plan
- **Session A** (`week06_a_representations-inductive-bias_*`): reuses
  `sources/datasets/ethanol.xyz` (Week 01) for the invariance demonstrations;
  reuses ESOL descriptors/fingerprints for the representation-spectrum
  discussion. 4 exercises.
- **Session B** (`week06_b_representations-inductive-bias_*`): dataset
  `esol_clean.csv`; three representations (7 descriptors, MACCS 167-bit,
  Morgan 2048-bit) × three models (Ridge, kNN, random forest). 3 exercises +
  mini-challenge, then the mini-project brief.
- Compute budget: fitting 9 (representation × model) combinations on ≤ 1117
  molecules; < 15 s total.

## Slides plan
- Session A ~8 slides: translational variance demo; invariance/equivariance
  definitions; the representation spectrum; equivariant NNs (survey slide,
  explicitly marked no-implementation). Demo → notebook A §1–§4.
- Session B ~8 slides: the shoot-out grid; chemical space per representation;
  mini-project brief walkthrough. Demo → notebook B §1–§3; project brief on
  its own slide with a link to `assessments/mini-project-brief.md`.
- Figures: `invariant_vs_variant.png` (session A);
  `representation_shootout_grid.png`, `chemical_space_by_representation.png`
  (session B).

## Assessment hooks
- Examinable: the three invariance/equivariance definitions and being able to
  classify a new descriptor against them; why a representation shoot-out must
  hold the model fixed to be informative; the mini-project rubric (see
  `assessments/mini-project-brief.md`).
- **Mini-project assigned this week** (`assessments/mini-project-brief.md`),
  worth 40-60% per `syllabus.md`'s assessment skeleton; presented/submitted in
  Week 10.

## Sources used
- dmol.pub *Input data and equivariances* (translational-variance opening
  example, invariance/equivariance definitions, invariant-descriptor
  strategies: pairwise distances, radius-of-gyration-style reductions, RBF
  featurisation).
- dmol.pub *Equivariant neural networks* — survey only, per syllabus scope
  cut; no implementation.
- DeepChem *An Introduction to MoleculeNet* — representation-comparison
  framing (ECFP vs GraphConv vs Weave featurizers); **we reproduce the same
  idea (compare representations, model held fixed) using RDKit + scikit-learn
  directly**, rather than adding the `deepchem` package to
  `env/environment.yml` (a large dependency with its own TensorFlow/PyTorch
  version constraints).

## Open questions for instructor
- DeepChem itself is not installed; the "shoot-out" is built from RDKit
  descriptors/MACCS/Morgan fingerprints + scikit-learn models instead of
  DeepChem's `GraphConv`/`Weave` featurizers. Confirm this substitution, or
  add `deepchem` as a dependency if its featurizers are wanted directly (note:
  DeepChem's own graph features are superseded by Week 07's GNN anyway).
- Please review `assessments/mini-project-brief.md` (new this week) —
  dataset choices, rubric weighting and the Week 10 deliverable format are
  drafted from the syllabus's assessment skeleton and need instructor
  sign-off before being given to students.
