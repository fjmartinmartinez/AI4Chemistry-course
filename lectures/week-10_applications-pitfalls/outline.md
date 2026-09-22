# Week 10 — Frontier applications, pitfalls, and the mini-project hackathon

_Record of the approved design (syllabus v0.1). Not a proposal._

## Learning objectives
By the end of week 10 students can:
1. Train a GNN to predict a real DFT-computed property (QM9 HOMO-LUMO gap)
   and report its error honestly against a baseline. [LO4, LO7]
2. Name modern invariant/equivariant molecular-potential architectures
   (SchNet, ANI, NequIP, TorchMD-NET) and explain, at survey level, why
   equivariant internal layers tend to improve accuracy (Week 06). [LO6, LO7]
3. Quantify how much a leaderboard ranking depends on the specific
   train/test split, using repeated splits — not just assert that splits
   matter. [LO3, LO8]
4. Audit a small ML study against a pitfalls checklist (leakage, benchmark
   choice, reproducibility) built from the whole course. [LO8]
5. (Session B) Apply the full course workflow, end to end, to a chemical
   dataset, against the mini-project's own rubric. [LO1-LO8]

## Prerequisites assumed
- Week 07: the `GCNLayer`/`GCN` pattern, reused directly on a new dataset.
- Week 03B/04/06: leakage, imbalance, and representation-choice lessons this
  week's pitfalls section reviews and extends.
- The mini-project brief (`assessments/mini-project-brief.md`, assigned
  Week 06B).

## Concept sequence
| # | Concept | Chemistry anchor | Where |
|---|---------|------------------|-------|
| 1 | A GNN for a real DFT property | QM9 HOMO-LUMO gap (eV) | notebook A |
| 2 | Modern invariant/equivariant potentials (survey) | SchNet/ANI vs NequIP/TorchMD-NET | notebook A |
| 3 | Split variance: how much does the split matter? | repeated random splits, QM9/ESOL | notebook A |
| 4 | A pitfalls checklist, applied | leakage/benchmark/reproducibility audit of a flawed study | notebook A |
| 5 | The mini-project, worked end to end | BBBP (Week 04A), every rubric element | notebook B |

## Notebook plan
- **Session A** (`week10_a_applications-pitfalls_*`): dataset
  `qm9_subset.csv` (3000 QM9 molecules, checksum-pinned via
  `lectures/_build/make_datasets_wk10.py`); reuses Week 07's `GCNLayer`/`GCN`
  classes directly. 4 exercises.
- **Session B** (`week10_b_applications-pitfalls_*`): **no new taught
  material** (per `syllabus.md`: "mini-project hackathon — no new material").
  Instead of a lecture-style notebook, this is a **fully worked mini-project
  example** on BBBP, hitting every required element from
  `assessments/mini-project-brief.md`'s checklist — a concrete reference
  students can consult while building their own. 3 "checklist" exercises +
  a mini-challenge (assemble the whole pipeline into one reusable function),
  rather than new concepts.
- Compute budget: Session A's GNN trains on 2400 QM9 molecules (up to 9 heavy
  atoms each — smaller than ESOL) in well under a minute; Session B reuses
  already-tested Week 02/03/04 patterns on BBBP, < 30 s total.

## Slides plan
- Session A ~8 slides: QM9 and DFT properties; GNN result; modern
  architecture survey (invariant vs equivariant); split-variance result;
  pitfalls checklist. Demo → notebook A §1-4.
- Session B ~6 slides: the mini-project rubric walkthrough; the worked BBBP
  example as a live reference; presentation logistics for the hackathon.
  Demo → notebook B, whole notebook as the walkthrough.
- Figures: `qm9_gap_parity_plot.png`, `split_variance_boxplot.png` (session A);
  `mini_project_worked_leaderboard.png`, `mini_project_error_analysis.png`
  (session B).

## Assessment hooks
- Examinable: reading a parity plot for a real DFT property; the
  invariant-vs-equivariant accuracy trade-off (Week 06 applied); why
  split-to-split variance matters when comparing two close leaderboard
  entries; the full pitfalls checklist.
- Session B doubles as the mini-project's own worked reference — the
  hackathon's assessment is the mini-project itself
  (`assessments/mini-project-brief.md`), not this notebook.

## Sources used
- dmol.pub *Predicting DFT energies with GNNs* (QM9: size, properties,
  featurisation, message-passing equations, honestly-reported "undertrained"
  result) — we reuse Week 07's own `GCNLayer`/`GCN` (2D bond graph, no 3D
  distances) rather than dmol's 3D inverse-distance edge features, a
  disclosed scope simplification (see Open questions).
- dmol.pub *Modern molecular neural networks* (SchNet, ANI, NequIP,
  TorchMD-NET; invariant vs equivariant internal layers) — survey only.
- Pat Walters, `comparing_regression_models.ipynb` (model-comparison rigor;
  motivates Section 3's repeated-split variance study).
- `assessments/mini-project-brief.md` (this course) — Session B's checklist.

## Open questions for instructor
- dmol's own QM9 chapter featurises 3D geometry (inverse pairwise distances
  as edge features); we reuse Week 07's 2D bond-graph `GCNLayer` instead, to
  avoid introducing new 3D-graph machinery in the course's final week.
  Confirm this simplification, or request a 3D/distance-weighted extension.
- QM9's full 134k molecules are not cached (only a fixed, checksum-derived
  3000-molecule subsample is, kept small for the repository) — confirm this
  is an acceptable trade-off versus caching (or dynamically re-downloading)
  the full dataset for a larger training set.
- Session B departs from the course's usual "new concept + exercises"
  notebook shape, since the syllabus explicitly assigns no new material this
  session — confirm the "fully worked mini-project reference" format is the
  right way to spend a self-guided notebook slot here, versus, e.g., no
  notebook at all for this session.
