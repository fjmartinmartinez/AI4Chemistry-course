# Mini-project brief — AI for Chemistry

> **Status: DRAFT — needs instructor sign-off before being given to students.**
> Drafted 2026-09-11 from `syllabus.md`'s assessment skeleton; weighting and
> two-part structure confirmed by instructor 2026-09-22 (mini-project is the
> **sole summative assessment** for the module — no separate coursework/exam
> component). Dataset pre-approval list, group-size policy, exact write-up
> format and presentation slot length are still `TODO(verify)`.

**Assigned:** Week 06, Session B. **Presented/submitted:** Week 10, Session B
(mini-project hackathon). **Weight:** 100% of the module's summative
assessment, split into two equally-weighted parts:

| Part | What | Weight |
|---|---|---:|
| **1 — Notebook** | The technical, end-to-end analysis (data → representation → model → evaluation) | 50% |
| **2 — Write-up + presentation** | Written critical discussion and the Week 10 hackathon presentation | 50% |

Weekly notebook exercises (wk 01–10) remain formative/auto-checked only and
carry no summative weight.

## Objective

Carry out a small, honest, end-to-end machine-learning study on a chemical
dataset, using the workflow this module has built week by week: clean data →
featurise → model → evaluate → critique. The point is not to produce the
highest leaderboard score — it is to demonstrate you can run the *whole*
pipeline correctly (Part 1) and say, precisely, how much to trust the result
(Part 2).

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
   `TODO(verify)`: confirm with the instructor which specific MoleculeNet
   tasks are pre-approved (size/difficulty appropriate to the time budget),
   and explicitly rule out any dataset that arrives already featurised or
   pre-split — the point of Part 1 is to build the pipeline, not skip it.
3. **A dataset of your own**, with instructor approval — e.g. from a project
   you already work on, or a public dataset relevant to your chemistry
   background. Must come with measured (not simulated-only) labels unless
   agreed otherwise, and must not arrive pre-featurised/pre-split.

## Part 1 — Notebook (50%)

Your notebook must include, in this order, and must **run top to bottom
without errors**:

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
   classification: confusion matrix, precision/recall/F1, ROC-AUC or average
   precision as appropriate to any class imbalance — Week 03A/04).
6. **Error analysis** (Week 04B): look at specific mispredicted molecules as
   structures, not only as a metric, and offer a grounded hypothesis for at
   least one failure mode.
7. **A brief stated-limitation note** in the notebook's own summary cell
   (2–3 sentences is enough here — develop this properly in the Part 2
   write-up, don't duplicate effort).
8. **Reproducibility** (gating requirement): fixed random seeds throughout;
   state the environment used (`env/environment.yml`); no unpinned
   downloads. A notebook that fails `Restart & Run All`, or that depends on
   an unpinned/unchecksummed download, cannot be marked on the rows above it
   — the results are not verifiable as genuine.

Everything through week 08–09's material (deep learning, GNNs, sequence
models, generative models, XAI) is **optional extension** to Part 1, not
required — a rigorous classical-ML notebook that does all of the above well
outscores a flashier one that skips a required element.

## Part 2 — Write-up + presentation (50%)

This is not a summary of the notebook — it is the critical, communicable
account of the study, and is assessed independently of Part 1's technical
execution. It must cover:

- **The question and its motivation**: what you asked, and why it matters
  chemically (not just "because the dataset was available").
- **What you found**: a concise, non-code account of the results — a
  reader should understand the outcome without opening the notebook.
- **Critical discussion**, developed properly (this is the bulk of Part 2):
  - Interpret the gap between the random and structure-aware splits — what
    does it imply about how the model would perform on genuinely new
    chemistry?
  - Applicability domain: where would you *not* trust this model?
  - Dataset size/bias, representation choice, and any missing uncertainty
    estimate.
- **A stated limitation**, expanded from the notebook's brief note into a
  real paragraph: what would make you distrust this model in a real
  application?

**Format:** `TODO(verify)` exact format expected by the instructor (e.g. a
1–2 page PDF, or a dedicated markdown/notebook section distinct from the
Part 1 summary cell — it must NOT simply be Part 1's summary cell copied
over, since that would double-count the same content against both parts).

**Presentation:** a short presentation in the Week 10 hackathon session,
covering the same content as the write-up. `TODO(verify)` exact time slot
per group/individual, and whether Q&A response counts toward this part's
mark or is formative only.

## Deliverables

- **One Jupyter notebook**, `mini_project_<your-name-or-group>.ipynb`
  (Part 1), following the same documentation standard as the course
  notebooks: a header cell (objective, dataset, representations/models
  used), a markdown cell before every code cell, and a short summary cell
  at the end.
- **A write-up** (Part 2) — format `TODO(verify)`, see above.
- **A short presentation** (Part 2) in the Week 10 hackathon session —
  slot length `TODO(verify)`.

## Suggested timeline

| When | Milestone |
|---|---|
| Week 06 | Dataset chosen; data-hygiene step drafted. |
| Week 07–08 | Representations + baseline model working end to end. |
| Week 09 | Second model + both evaluation splits done; error analysis started; write-up drafted. |
| Week 10 | Final notebook, write-up and presentation. |

`TODO(verify)`: whether an informal instructor check-in is required at one of
the intermediate milestones.

## Marking rubric (draft — needs instructor sign-off)

### Part 1 — Notebook (50% of module total)

| Component | Weight |
|---|---:|
| Data hygiene and reasoning about each cleaning decision | 10% |
| Representation(s): choice and justification (≥2) | 10% |
| Modelling and evaluation rigour (baseline, ≥2 models, both split types, appropriate metrics) | 20% |
| Error analysis (specific mispredictions, grounded hypothesis) | 10% |

Reproducibility (required element 8) is a **gating** requirement, not a
separately weighted row: a notebook that doesn't run top-to-bottom or relies
on an unpinned download forfeits marks on the rows above that its results
feed into.

### Part 2 — Write-up + presentation (50% of module total)

| Component | Weight |
|---|---:|
| Critical discussion (split-gap interpretation, applicability domain, stated limitation) | 20% |
| Write-up clarity, structure and reproducibility statement | 15% |
| Presentation delivery and response to Q&A | 15% |

`TODO(verify)`: these weights are a draft split of the confirmed 50/50
part-level allocation and are not yet approved — replace with the approved
module-spec marking scheme, or confirm this one, before distributing to
students.

## Academic integrity

Work individually unless the instructor specifies group work.
`TODO(verify)`: group size/policy is not specified in `syllabus.md` and must
be confirmed before this brief is distributed — it affects both parts (Part
1's workload expectation and Part 2's presentation format). Cite any code
adapted from course materials or external sources (as the course notebooks
themselves do in their own "Attribution" cells).
