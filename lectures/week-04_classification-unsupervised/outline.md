# Week 04 — Classification and unsupervised learning

_Record of the approved design (syllabus v0.1). Not a proposal._

## Learning objectives
By the end of week 04 students can:
1. Explain the classification framing (labels, sigmoid, log-odds, decision
   boundary) and fit logistic regression. [LO4]
2. Compute and choose between classification metrics (accuracy, precision,
   recall, F1, ROC-AUC) and explain why accuracy misleads under class
   imbalance. [LO4]
3. Use `class_weight` and decision-threshold tuning to trade precision for
   recall on an imbalanced task. [LO4]
4. Apply PCA and k-means to a labelled dataset and recognise when unsupervised
   structure does or does not align with a supervised target. [LO4]
5. State, qualitatively, what a kernel is and why it lets a model use
   similarity instead of explicit features — no implementation this week
   (deliberate scope cut, per `syllabus.md`). [LO4]
6. (Session B) Run a small classifier comparison on a genuinely imbalanced
   toxicity task, tune a decision threshold against a stated screening
   requirement, and read false negatives structurally. [LO4, LO8]

## Prerequisites assumed
- Week 03: train/test split, cross-validation, regression metrics, ridge,
  random forest, leakage/cluster-split reasoning.
- Week 02: RDKit descriptors, canonical SMILES, PCA on standardised features.

## Concept sequence
| # | Concept | Chemistry anchor | Where |
|---|---------|------------------|-------|
| 1 | Classification framing: sigmoid, log-odds, cross-entropy | BBB penetration (yes/no) | notebook A |
| 2 | Metrics: confusion matrix, accuracy/precision/recall/F1, ROC-AUC | class imbalance in BBBP | notebook A |
| 3 | `class_weight="balanced"`; the precision/recall trade-off | minority-class recall | notebook A |
| 4 | PCA + k-means on a labelled set; ARI vs true label | BBBP chemical space | notebook A |
| 5 | Kernel idea (qualitative only) | Tanimoto / RBF similarity | notebook A |
| 6 | Baseline + LogReg + RF leaderboard on a severely imbalanced task | Tox21 SR-p53 | notebook B |
| 7 | Threshold tuning against a stated recall requirement | screening priorities | notebook B |
| 8 | Error analysis: false negatives, structures | missed toxic compounds | notebook B |

## Notebook plan
- **Session A** (`week04_a_classification-unsupervised_*`): dataset
  `BBBP.csv` (blood-brain-barrier penetration, MoleculeNet, checksum-pinned).
  Cleaning extends Week 02's checklist with **desalting** (keep the largest
  fragment) — BBBP and Tox21 both contain salts. 4 exercises.
- **Session B** (`week04_b_classification-unsupervised_*`): dataset
  `tox21.csv.gz`, task `SR-p53` (~6% positive — genuinely imbalanced).
  3 exercises + mini-challenge (meet a stated recall constraint).
- Compute budget: logistic regression + random forest on ≤ 7000 molecules,
  7 descriptors; < 15 s total.

## Slides plan
- Session A ~8 slides: sigmoid/log-odds; confusion matrix; imbalance trap;
  PCA/k-means vs label; kernel idea (one slide, explicitly marked
  qualitative/no-code). Demo → notebook A §1–§4.
- Session B ~8 slides: leaderboard; precision-recall trade-off; error
  analysis. Demo → notebook B §1–§4.
- Figures: `bbbp_decision_boundary.png`, `roc_curve.png`,
  `pca_chemical_space_bbbp.png`, `tox21_pr_curve.png`,
  `tox21_leaderboard.png`, `false_negatives_grid.png`.

## Assessment hooks
- Examinable: why accuracy is a poor single metric under imbalance; reading a
  confusion matrix and ROC/PR curve; effect of `class_weight`/threshold on
  precision vs recall; what ARI does and does not tell you about clusters vs
  labels; the qualitative kernel idea.

## Sources used
- dmol.pub *Classification* (sigmoid, cross-entropy, confusion matrix,
  ROC-AUC, class imbalance, ClinTox example — we substitute BBBP/Tox21 to
  avoid the Mordred dependency dmol uses).
- dmol.pub *Kernel learning* — qualitative only, per syllabus scope cut; no
  implementation.
- Pat Walters `classification_model.ipynb`, `comparing_classification_models.ipynb`
  (metrics/plots pattern) and TeachOpenCADD T007 (sensitivity/specificity/AUC
  comparison pattern; threshold-based activity labelling idea).
- Datasets: BBBP and Tox21 (MoleculeNet, via the same DeepChem mirror as
  Week 02's ESOL), both checksum-pinned.

## Open questions for instructor
- dmol's own worked classification example uses ClinTox + Mordred descriptors
  (~1500 features); we use BBBP/Tox21 + the Week 02 7-descriptor block to
  avoid adding `mordred` to `env/environment.yml`. Confirm, or add the
  dependency if Mordred's richer descriptor set is wanted.
- Confirm BBBP (session A) and Tox21/SR-p53 (session B) as the two
  classification tasks, or substitute ClinTox/HIV if a specific toxicological
  narrative is preferred for the lecture.
