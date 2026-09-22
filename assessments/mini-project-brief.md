# Mini-project brief — AI for Chemistry

> **Status: DRAFT — needs instructor sign-off before being given to students.**
> Drafted 2026-09-11 from `syllabus.md`'s assessment skeleton ("Mini-project
> (wk 06→10): small end-to-end study on a chemical dataset; ~40–60%"). The
> exact weighting, group size and submission mechanics are `TODO(verify)`
> against the approved module spec.

**Assigned:** Week 06, Session B. **Presented/submitted:** Week 10, Session B
(mini-project hackathon). **Weight:** `TODO(verify)` — syllabus gives a
40–60% range; confirm the exact figure and how it splits against any
remaining coursework/exam component in `admin/learning-outcomes.md`'s
assessment skeleton.

## Objective

Carry out a small, honest, end-to-end machine-learning study on a chemical
dataset, using the workflow this module has built week by week: clean data →
featurise → model → evaluate → critique. The point is not to produce the
highest leaderboard score — it is to demonstrate you can run the *whole*
pipeline correctly and say, precisely, how much to trust the result.

## Choose a dataset

Pick **one** option (or propose your own to the instructor for approval):

1. **A cached course dataset, different task.** E.g. a Tox21 task other than
   `SR-p53` (Week 04B), or ESOL with a representation/model combination not
   covered in class.
2. **A MoleculeNet dataset not yet used in this course.** Several are plain
   CSV/CSV.GZ files on the same DeepChem mirror used for `esol_delaney.csv`,
   `BBBP.csv` and `tox21.csv.gz` (e.g. Lipophilicity, HIV, SIDER, BACE,
   ClinTox). Download it yourself and **checksum-pin** it exactly as
   `lectures/_build/make_datasets_wk04.py` does — do not trust an
   unauthenticated download without recording its SHA-256.
   `TODO(verify)`: confirm with the instructor which specific
   MoleculeNet tasks are pre-approved, to avoid a dataset that is too large
   for a laptop or too easy/hard for the time budget.
3. **A dataset of your own**, with instructor approval — e.g. from a project
   you already work on, or a public dataset relevant to your chemistry
   background. Must come with measured (not simulated-only) labels unless
   agreed otherwise.

## Required elements

Your notebook (see **Deliverables**) must include, in this order, and must
**run top to bottom without errors**:

1. **Data hygiene** (Week 02/04 checklist): parse SMILES, drop rows that fail
   to parse, canonicalise, desalt if relevant, deduplicate by structure,
   handle missing labels explicitly, sanity-check target ranges. State how
   many rows were removed at each step and why.
2. **At least two representations** (e.g. a descriptor block and a
   fingerprint, or two fingerprint types) — justify the choice chemically,
   not just computationally.
3. **At least two models**, including one simple baseline (mean/majority
   predictor, or linear/logistic regression) — never report a result without
   a baseline to compare against (Week 03/04).
4. **Two evaluation splits**: a random split *and* a structure-aware split
   (Butina cluster or scaffold split, Week 02B/03B/04) — report both, and
   discuss the gap between them explicitly (Week 03B's leakage lesson).
5. **Metrics appropriate to the task type** (regression: MAE/RMSE/R²;
   classification: confusion matrix, precision/recall/F1, ROC-AUC or
   average precision as appropriate to any class imbalance — Week 03A/04).
6. **Error analysis** (Week 04B): look at specific mispredicted molecules as
   structures, not only as a metric, and offer a grounded hypothesis for at
   least one failure mode.
7. **A stated limitation.** One paragraph: what would make you distrust this
   model in a real application (applicability domain, dataset size/bias,
   representation choice, missing uncertainty estimate, ...)?
8. **Reproducibility**: fixed random seeds throughout; state the environment
   used (`env/environment.yml`); no unpinned downloads.

Everything through week 08–09's material (deep learning, GNNs, sequence
models, generative models, XAI) is **optional extension**, not required — a
rigorous classical-ML study that does all of the above well outscores a
flashier one that skips a required element.

## Deliverables

- **One Jupyter notebook**, `mini_project_<your-name-or-group>.ipynb`,
  executable top to bottom (`Restart & Run All`), following the same
  documentation standard as the course notebooks: a header cell (objective,
  dataset, representations/models used), a markdown cell before every code
  cell, and a short summary cell at the end.
- **A short write-up** (`TODO(verify)` exact format expected by the
  instructor — e.g. a 1–2 page PDF, or the notebook's own summary section may
  suffice) covering: the question you asked, what you found, and your stated
  limitation.
- **A short presentation** in the Week 10 hackathon session.
  `TODO(verify)` exact time slot per group/individual.

## Suggested timeline

| When | Milestone |
|---|---|
| Week 06 | Dataset chosen; data-hygiene step drafted. |
| Week 07–08 | Representations + baseline model working end to end. |
| Week 09 | Second model + both evaluation splits done; error analysis started. |
| Week 10 | Final notebook, write-up and presentation. |

`TODO(verify)`: whether an informal instructor check-in is required at one of
the intermediate milestones.

## Indicative marking rubric (draft — needs instructor sign-off)

| Component | Weight (draft) |
|---|---:|
| Data hygiene and reasoning about each cleaning decision | 15% |
| Representation(s): choice and justification | 15% |
| Modelling and evaluation rigor (baseline, ≥2 models, both split types) | 30% |
| Error analysis and critical discussion (incl. the stated limitation) | 25% |
| Reproducibility and clarity of the notebook/write-up | 15% |

`TODO(verify)`: these weights are a draft split of the module's mini-project
allocation and are not yet approved — replace with the module-spec marking
scheme, or confirm this one, before distributing to students.

## Academic integrity

Work individually unless the instructor specifies group work.
`TODO(verify)`: group size/policy is not specified in `syllabus.md` and must
be confirmed before this brief is distributed. Cite any code adapted from
course materials or external sources (as the course notebooks themselves do
in their own "Attribution" cells).
