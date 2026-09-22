# Syllabus — AI for Chemistry (DRAFT v0.1, 2026-09-02)

10 weeks × (Session A: 2.5 h + Session B: 2.0 h) = 45 contact hours.
Format: every session is notebook-driven. Session A interleaves short concept blocks (15–20 min, slides) with guided notebook work (25–30 min); Session B is a hands-on workshop on the same topic (exercises → mini-challenge), ending with a 10-min debrief.
Assumes NO prior programming. Core text from week 4: dmol.pub (Deep Learning for Molecules & Materials).

| Wk | Session A (2.5 h) | Session B (2.0 h) | Backbone sources |
|----|-------------------|-------------------|------------------|
| 01 | Python bootcamp I: variables, lists, loops, functions; reading chemical data files | Numpy + matplotlib workshop: arrays, plotting spectra/EoS data | MolSSI python_scripting_cms; intro_python_chemists |
| 02 | Python bootcamp II: pandas dataframes; molecules in code — SMILES, RDKit, descriptors | Build a clean molecular dataset (ESOL-style): parse, featurise, visualise chemical space | SciCompforChemists; TeachOpenCADD T001–T002/T005 |
| 03 | First ML: regression & model assessment — train/test split, overfitting, cross-validation, metrics | Solubility-prediction lab: linear → random forest on wk-02 dataset; leaderboard | dmol.pub ch. B (regression); ML4chemArg Gentle Intro |
| 04 | Classification + unsupervised: logistic regression, clustering, PCA on chemical space; kernel idea (qualitative) | Toxicity/activity classification mini-competition; error analysis | dmol.pub ch. B (classification, kernels); ML-in-chemistry-101 |
| 05 | Deep learning I: tensors & shapes, MLPs, backprop intuition, PyTorch basics | Train an MLP property predictor; loss curves, early stopping, hyperparameter sweep | dmol.pub ch. A + C (overview, standard layers) |
| 06 | Representations & inductive bias: descriptors vs learned features; invariance/equivariance (conceptual) | Representation shoot-out lab: same target, three representations; MINI-PROJECT ASSIGNED | dmol.pub (input data & equivariances) |
| 07 | Graph neural networks: molecules as graphs, message passing, GCN | GNN lab on ESOL/QM9 subset; compare to wk-05 MLP | dmol.pub (GNN chapter); DeepChem graph-conv tutorial |
| 08 | Sequences & attention: SMILES language models, transformers, chemical LLMs (survey) | Use a pretrained transformer for property prediction / reaction task | dmol.pub (sequences, attention); schwallergroup ai4chem_course |
| 09 | Generative models: VAE core; flows/diffusion (survey); explaining predictions (XAI) | Generative lab: molecular VAE / generative RNN in browser; sample & assess validity | dmol.pub (VAE, explaining predictions, ch. D) |
| 10 | Frontier applications & practice: GNNs for DFT energies, foundation models; pitfalls — leakage, benchmarks, reproducibility | Mini-project hackathon + presentations (assessed) | dmol.pub ch. D; PatWalters tutorials (pitfalls) |

## Deliberate scope cuts (from dmol.pub)
- Kernel learning: qualitative only (wk 04). Equivariant NNs: conceptual only (wk 06), no implementation. Normalizing flows: survey slide only (wk 09).

## Assessment skeleton
- Weekly notebook exercises (formative, auto-checked with asserts; no summative weight).
- Mini-project (wk 06→10): **sole summative assessment, 100% of module weight**, in two equally-weighted parts — Part 1 (50%): the technical notebook; Part 2 (50%): write-up + Week 10 hackathon presentation. Confirmed by instructor 2026-09-22; full brief and rubric in `assessments/mini-project-brief.md`.

## Environment
- Weeks 1–7 run in `env/environment.yml` on laptops; weeks 7–9 notebooks need Colab (free GPU) fallbacks — keep both paths per CLAUDE.md standards.
