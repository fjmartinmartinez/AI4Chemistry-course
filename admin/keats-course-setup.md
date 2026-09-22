# KEATS course setup — AI for Chemistry

> Copy-paste source for structuring and populating the KEATS (Moodle) page.
> Every block below is meant to be lifted, as-is or lightly edited, into the
> corresponding KEATS field. Blocks needing a KCL-specific detail I cannot
> know or verify are marked `[PLACEHOLDER]` or `TODO(verify)` — fill those in
> before publishing to students, per `CLAUDE.md`'s "never invent" rule.
>
> Source of truth for everything here: `admin/syllabus.md`,
> `admin/learning-outcomes.md` (DRAFT, needs sign-off),
> `assessments/mini-project-brief.md` (DRAFT, needs sign-off), and each
> week's `lectures/week-NN_*/outline.md`.

---

## 0. Action items before you publish

1. **Sign off `admin/learning-outcomes.md` and `assessments/mini-project-brief.md`** —
   both are still marked DRAFT. The text below quotes them as they stand;
   if you change the wording or the rubric weights, update this file too.
2. **Decide the notebook release policy**: release the `_student.ipynb` at
   the start of each session and the `_solutions.ipynb` after it (protects
   the exercises), or release both together. §4 assumes staggered release;
   change it if you'd rather release both at once.
3. ~~Create a GitHub repository for the course materials~~ — done
   (`https://github.com/fjmartinmartinez/AI4Chemistry-course`). All ten
   weeks run as Jupyter notebooks in the local `ai4chem` environment; there
   is no Google Colab path, so no "Open in Colab" badges are needed.
4. **Fill in the placeholders**: `[MODULE CODE]`, `[ACADEMIC YEAR]`,
   `[SESSION A DAY/TIME]`, `[SESSION B DAY/TIME]`, `[ROOM/LOCATION]`,
   `[SUBMISSION DEADLINE]`, `[PRESENTATION SLOT]`.

---

## 1. Module landing page text

### 1.1 Module description
*(paste into the KEATS course summary / "About this module" block)*

> This module gives you a practical, hands-on introduction to machine
> learning and deep learning as applied to chemistry — from your first line
> of Python through to graph neural networks, chemical language models and
> generative molecular design. No prior programming experience is assumed:
> Week 1 starts from variables and loops and builds up, session by session,
> to the methods used in current AI-for-chemistry research.
>
> Every session is **notebook-driven**. Session A (2.5 h) interleaves short
> concept blocks with guided notebook work; Session B (2.0 h) is a hands-on
> workshop applying that week's ideas to a real chemical dataset, ending in
> a short debrief. All notebooks are **self-guided**: read the markdown cell
> before every code cell, run it, check your answer against the stated
> "what to look for", and use the built-in "Check yourself" cells to confirm
> you're on track before moving on.

### 1.2 Module aims
*(paste into the module handbook / "aims" block, above the learning outcomes — aims are the broad "why this module exists" statements; the learning outcomes in §1.3 are the specific, assessable ones)*

> This module aims to:
>
> 1. Equip you with practical programming and data-analysis skills, starting
>    from **no prior programming experience**, sufficient to read, clean and
>    critically analyse real chemical data.
> 2. Introduce the core methodology of machine learning — from classical
>    regression and classification through to deep learning — grounded
>    throughout in chemical datasets and chemically meaningful evaluation,
>    not taught as abstract computer science.
> 3. Develop fluency with the molecular representations and model
>    architectures used in current AI-for-chemistry research (descriptors,
>    fingerprints, graph neural networks, sequence/attention models,
>    generative models), including working directly with pretrained models.
> 4. Build critical judgement about the reliability of AI-driven chemical
>    predictions — data leakage, benchmark and metric choice, representation
>    bias, and the limits of explainability — so you use these tools
>    critically, not as an unquestioned "black box".
> 5. Give you direct, first-hand experience of the end-to-end
>    AI-for-chemistry research workflow (data → representation → model →
>    evaluation → critique) through a self-directed mini-project on a real
>    chemical dataset.
>
> *(`TODO(verify)`: align with the approved KCL module-spec aims once signed
> off — see `admin/learning-outcomes.md`.)*

### 1.3 Learning outcomes
*(paste into the module handbook / learning outcomes block)*

> On successful completion of this module, you will be able to:
>
> 1. **Program in Python for chemical data** — read, transform and visualise
>    chemical data files, from no prior programming experience.
> 2. **Represent molecules computationally** — SMILES, RDKit, molecular
>    descriptors and fingerprints, and build a clean, ML-ready dataset.
> 3. **Apply the supervised ML workflow correctly** — train/validation/test
>    splitting, cross-validation, over/under-fitting, and choosing the right
>    metric for the task.
> 4. **Build and evaluate classical ML models** — regression, classification,
>    clustering and PCA for molecular property prediction and analysis.
> 5. **Implement neural networks** — tensors, layers, backpropagation; train
>    and regularise your own models in PyTorch.
> 6. **Reason about representation and inductive bias** — fixed vs learned
>    features, sequences vs graphs, invariance and equivariance.
> 7. **Use modern architectures for chemistry** — graph neural networks,
>    attention/transformers and generative models, including pretrained
>    models.
> 8. **Assess reliability and deliver a study** — data leakage, benchmark
>    choice, explainability, reproducibility, and a complete end-to-end
>    mini-project.
>
> *(`TODO(verify)`: replace with the approved KCL module-spec wording once
> signed off — see `admin/learning-outcomes.md`.)*

### 1.4 Weekly schedule
*(paste as a table into the course overview page)*

| Week | Session A (2.5 h) | Session B (2.0 h) |
|---|---|---|
| 0 *(optional, before Week 1)* | — | Python primer (self-study, ungraded) |
| 1 | Python bootcamp I | NumPy + matplotlib workshop |
| 2 | pandas; molecules in code (SMILES, RDKit) | Build a clean molecular dataset |
| 3 | First ML: regression & model assessment | Solubility-prediction lab; leaderboard |
| 4 | Classification + unsupervised learning | Toxicity classification mini-competition |
| 5 | Deep learning I: tensors, MLPs, PyTorch | Train an MLP; loss curves, early stopping |
| 6 | Representations & inductive bias | Representation shoot-out — **mini-project assigned** |
| 7 | Graph neural networks | GNN lab; compare to Week 5's MLP |
| 8 | Sequences & attention; chemical LLMs | Pretrained transformer for property prediction |
| 9 | Generative models (VAE); explainability (XAI) | Generative RNN lab; sample & assess validity |
| 10 | Frontier applications & pitfalls | **Mini-project hackathon + presentations** |

`[PLACEHOLDER]`: add calendar dates once term dates are confirmed.

### 1.5 Assessment overview
*(paste into the assessment / grading block)*

> - **Weekly notebook exercises** — formative, auto-checked with `assert`
>   statements as you go; not submitted or marked, but the same checks a
>   marker would use to confirm a technique works.
> - **Mini-project** — the module's **sole summative assessment (100%)**,
>   assigned Week 6, presented/submitted Week 10, in two equally-weighted
>   parts: **Part 1 (50%)** the technical notebook; **Part 2 (50%)** the
>   write-up and Week 10 hackathon presentation. See
>   `assessments/mini-project-brief.md` for the full brief and rubric.

### 1.6 Software and setup
*(paste into a "Getting started" page/section)*

> This module runs entirely in **Jupyter notebooks**, on your own laptop —
> no Google Colab. Install the course environment once:
> ```
> conda env create -f env/environment.yml
> conda activate ai4chem
> jupyter lab
> ```
> This same environment covers every week, including Weeks 8–9's pretrained
> transformer and from-scratch generative model — both are sized to run
> comfortably on a laptop CPU.
>
> **New to programming entirely?** Do the optional **Week 0 primer** first
> (self-study, ungraded, no installation needed — runs anywhere Python 3 does).

---

## 2. Weekly KEATS section/topic text

Each block below is: **section title** → **summary** (student-facing,
paste into the topic/section summary field) → **files to upload**. Learning
objective tags refer to `admin/learning-outcomes.md`.

### Week 0 — Python primer *(optional, before Week 1)*
> Never written a line of code before? Start here, in your own time. A short
> (~50 min), ungraded notebook covering notebooks/cells, `print()`,
> variables, arithmetic, and the most common beginner error messages —
> so Week 1 isn't the first Python you've ever typed. If you've programmed
> in any language before, skip straight to Week 1.
- Files: `week00_python-primer_student.ipynb` (release immediately;
  solutions optional to withhold, since it's self-study)

### Week 1 — Python bootcamp
> **Session A**: variables, lists, loops and functions — and reading real
> chemical data files (`.xyz` geometries, a CSV of alkane properties).
> **Session B**: NumPy arrays and matplotlib — vectorised maths, an IR
> spectrum, and fitting a Beer–Lambert calibration line. *(LO1)*
- Files: `week01_a_python-bootcamp_student.ipynb`,
  `week01_b_python-bootcamp_student.ipynb` (+ solutions, per your release policy)

### Week 2 — Data and molecules in code
> **Session A**: pandas dataframes, SMILES, RDKit, and your first molecular
> descriptors, on the ESOL solubility dataset. **Session B**: turn raw data
> into a clean, ML-ready dataset — cleaning checklist, Morgan fingerprints,
> a chemical-space map, and a leakage-aware train/test split. *(LO1, LO2)*
- Files: `week02_a_data-and-molecules_student.ipynb`,
  `week02_b_data-and-molecules_student.ipynb` (+ solutions)

### Week 3 — First ML: regression and model assessment
> **Session A**: the supervised-learning framing, train/test splitting,
> overfitting, cross-validation, and regression metrics. **Session B**: a
> solubility-prediction leaderboard (linear → ridge → random forest), and a
> concrete demonstration of data leakage. *(LO3, LO4)*
- Files: `week03_a_regression-model-assessment_student.ipynb`,
  `week03_b_regression-model-assessment_student.ipynb` (+ solutions)

### Week 4 — Classification and unsupervised learning
> **Session A**: logistic regression, why accuracy misleads under class
> imbalance, and PCA/k-means on a labelled dataset. **Session B**: a
> genuinely imbalanced toxicity-classification task, threshold tuning, and
> error analysis on the molecules the model gets wrong. *(LO3, LO4)*
- Files: `week04_a_classification-unsupervised_student.ipynb`,
  `week04_b_classification-unsupervised_student.ipynb` (+ solutions)

### Week 5 — Deep learning I
> **Session A**: tensors, dense layers, why non-linearity matters, and
> backpropagation versus `autograd` — building blocks only. **Session B**:
> train an MLP properly — train/val/test, loss curves, early stopping, a
> hyperparameter sweep, and an honest comparison to Week 3's leaderboard.
> *(LO5)*
- Files: `week05_a_deep-learning-i_student.ipynb`,
  `week05_b_deep-learning-i_student.ipynb` (+ solutions)

### Week 6 — Representations and inductive bias · 📋 mini-project assigned
> **Session A**: why raw coordinates fail, formal invariance/equivariance,
> and where descriptors, fingerprints and learned features sit on one
> spectrum. **Session B**: a representation shoot-out — **and the
> mini-project is assigned this session**. Read
> `assessments/mini-project-brief.md` in full before Session B. *(LO4, LO6)*
- Files: `week06_a_representations-inductive-bias_student.ipynb`,
  `week06_b_representations-inductive-bias_student.ipynb` (+ solutions),
  `mini-project-brief.pdf` (export of the brief)

### Week 7 — Graph neural networks
> **Session A**: molecules as graphs, message passing, and a minimal graph
> convolutional network — verified numerically to respect atom-order
> symmetry. **Session B**: train it properly and compare, honestly, to
> Week 5's MLP and Week 3's classical leaderboard. *(LO6, LO7)*
- Files: `week07_a_graph-neural-networks_student.ipynb`,
  `week07_b_graph-neural-networks_student.ipynb` (+ solutions)

### Week 8 — Sequences, attention and chemical language models
> **Session A**: SMILES as sequences, the attention mechanism from first
> principles, and a survey of chemical language models. **Session B**: load
> a real pretrained chemical transformer and use it for property
> prediction, in the course's own Jupyter environment. *(LO7)*
- Files: `week08_a_sequences-attention_student.ipynb`,
  `week08_b_sequences-attention_student.ipynb` (+ solutions)

### Week 9 — Generative models and explainability
> **Session A**: variational autoencoders, a real generative failure mode,
> and explaining predictions (gradient saliency, integrated gradients).
> **Session B**: train your own generative model for SMILES and assess what
> it produces — validity, uniqueness, novelty. *(LO7, LO8)*
- Files: `week09_a_generative-models-xai_student.ipynb`,
  `week09_b_generative-models-xai_student.ipynb` (+ solutions)

### Week 10 — Frontier applications, pitfalls, and the hackathon
> **Session A**: a GNN on a real DFT-computed property, modern
> architecture survey, and a pitfalls checklist built from the whole
> module. **Session B**: the **mini-project hackathon** — no new material;
> a fully worked reference example is provided, then the rest of the
> session is your own project and presentations. *(LO7, LO8)*
- Files: `week10_a_applications-pitfalls_student.ipynb`,
  `week10_b_applications-pitfalls_student.ipynb` (+ solutions)

---

## 3. Notebook access strategy

**Recommendation**: upload every `.ipynb` directly to KEATS as a File
resource. All ten weeks run in the same local Jupyter environment
(`env/environment.yml`) — no Google Colab path, so no "Open in Colab"
badges are needed; the GitHub repository
(`https://github.com/fjmartinmartinez/AI4Chemistry-course`) exists as a
source-control backup and for your own reference, not as a student-facing
access route.

**Release timing**: recommend releasing `_student.ipynb` at the start of
each session and `_solutions.ipynb` 24–48 h after Session B closes, so the
"Check yourself" cells and exercises retain some challenge. Both files are
generated together from the same source (`lectures/_build/nbbuild.py`), so
they never diverge — releasing them at different times is a scheduling
choice only, not a content-consistency risk.

---

## 4. Mini-project — KEATS Assignment activity setup

*(Everything in this section is DRAFT per `assessments/mini-project-brief.md` — confirm weights, group policy and submission format before publishing.)*

**Activity name**: `Mini-project — AI for Chemistry`

**Availability**: Open from Week 6, Session B. Due: `[SUBMISSION DEADLINE]`
(brief suggests end of Week 10). Presentations: `[PRESENTATION SLOT]`,
Week 10 Session B.

**Submission type**: File submission. Accepted types: `.ipynb` (required)
+ one of `.pdf`/`.docx` for the write-up (`TODO(verify)` exact format — the
brief allows the notebook's own summary section to suffice instead).
**Submission**: `TODO(verify)` individual vs group — brief defaults to
individual unless you specify otherwise.

### 4.1 Assignment description
*(paste into the KEATS Assignment "Description" rich-text field)*

> Carry out a small, honest, end-to-end machine-learning study on a chemical
> dataset, using the workflow this module has built week by week: clean
> data → featurise → model → evaluate → critique. The point is not the
> highest leaderboard score — it is demonstrating you can run the *whole*
> pipeline correctly and state precisely how much to trust your result.
>
> **Choose one dataset**: a cached course dataset on a new task, an unused
> MoleculeNet dataset (checksum-pinned yourself), or your own
> (instructor approval required for the last option).
>
> **Your notebook must include, and must run top to bottom without errors:**
> 1. Data hygiene (parsing, cleaning, deduplication — with row counts stated
>    at each step)
> 2. At least two molecular representations, chemically justified
> 3. At least two models, including a baseline
> 4. Two evaluation splits: random **and** structure-aware (cluster/scaffold)
> 5. Metrics appropriate to your task type
> 6. Error analysis on specific mispredicted molecules
> 7. One stated limitation, in your own words
> 8. Fixed random seeds and a stated environment (reproducibility)
>
> Full detail, required-element descriptions and suggested timeline: see
> the mini-project brief (linked above / attached).
>
> Everything from Week 8 onward (deep learning, GNNs, sequence models,
> generative models, XAI) is **optional extension**, not required.

### 4.2 Marking rubric
*(paste into KEATS Assignment → Advanced grading → Rubric; 100 points shown, rescale to your actual assignment weight)*

| Criterion | Excellent (100%) | Good (75%) | Adequate (50%) | Inadequate (0–25%) | Points |
|---|---|---|---|---|---:|
| **Data hygiene** | Every cleaning step justified; row counts at each step; a non-obvious cleaning decision handled well | Cleaning done correctly but reasoning less explicit | Basic cleaning done, but one required step missing or unjustified | Data hygiene checklist not followed | 15 |
| **Representations** | ≥2 representations, chemically well justified, correctly implemented | ≥2 representations, adequately justified | ≥2 representations present but justification weak/missing | Fewer than 2 representations, or incorrectly implemented | 15 |
| **Modelling & evaluation rigour** | Baseline + ≥2 models; both split types; leakage/split gap discussed insightfully | Baseline + ≥2 models; both splits present, discussion present but thin | One of {baseline, 2nd model, 2nd split} missing | Multiple required modelling elements missing | 30 |
| **Error analysis & critical discussion** | Specific, structurally-grounded error analysis; limitation is precise and consequential | Error analysis present; limitation stated but generic | Error analysis superficial (metric only, no structures/cases) | No error analysis or stated limitation | 25 |
| **Reproducibility & clarity** | Runs top-to-bottom; seeds fixed throughout; environment stated; notebook clearly documented | Runs top-to-bottom with fixed seeds; documentation adequate | Runs with minor manual fixes needed; documentation sparse | Does not run top-to-bottom, or seeds/environment unstated | 15 |

`TODO(verify)`: rescale points to the module's actual mini-project weight
once confirmed (`assessments/mini-project-brief.md` gives a 40–60% range).

---

## 5. Welcome / joining announcement

*(Ready to post to the KEATS "Announcements" forum before Week 1. Signature
and placeholders to fill in.)*

> **Subject: Welcome to AI for Chemistry — before Week 1**
>
> Hello everyone, and welcome to *AI for Chemistry*.
>
> This module is entirely hands-on: every session works through a Jupyter
> notebook, and by the end you'll have built — from scratch — everything
> from a first Python script to a graph neural network and a generative
> model for molecules. **No prior programming experience is assumed.**
> Week 1 starts from variables and loops.
>
> **Before Session 1, please:**
> 1. Install the course software environment (instructions are on the
>    module's "Getting started" page) — the whole module runs on your own
>    laptop; there is no Google Colab component.
> 2. If you have **never written any code before, in any language**,
>    complete the optional **Week 0 primer** first — it's about 50 minutes,
>    ungraded, and needs no installation at all. If you've coded before
>    (in any language), you can skip it.
>
> **How sessions work:** Session A (2.5 h) mixes short concept explanations
> with guided notebook work; Session B (2.0 h) is a hands-on workshop
> applying that week's ideas to a real chemical dataset. Every notebook is
> self-guided — read the text before each code cell, run it, and use the
> built-in "Check yourself" cells to confirm your answer before moving on.
>
> **Assessment**: weekly notebook exercises are formative (not submitted);
> the assessed component is a mini-project, assigned in Week 6 and
> presented in Week 10 — more detail nearer the time.
>
> Looking forward to working with you.
>
> — Fran Martin-Martinez
> `[MODULE CODE]`, `[ACADEMIC YEAR]`

---

## 6. Where each piece of this document comes from

| Section | Source file |
|---|---|
| §1.2 Module aims | `admin/learning-outcomes.md` (DRAFT) |
| §1.3 Learning outcomes | `admin/learning-outcomes.md` (DRAFT) |
| §1.4 Weekly schedule | `admin/syllabus.md` |
| §1.5 Assessment overview | `admin/syllabus.md` + `assessments/mini-project-brief.md` |
| §2 Weekly summaries | Each week's `lectures/week-NN_*/outline.md` |
| §4 Mini-project | `assessments/mini-project-brief.md` (DRAFT) |

Keep this file in sync if you edit any of the source files above.
