# Module aims and learning outcomes

> **Status: DRAFT — needs instructor / module-spec sign-off.**
> Drafted 2026-09-02 (aims added 2026-09-13) by course-development
> collaborator from the approved syllabus (`syllabus.md` v0.1). These are
> working aims/outcomes so that lecture outlines and notebooks have a stable
> mapping target; replace the wording with the approved KCL module-spec
> text when available and re-check every `[LOn]` reference in
> `lectures/**/outline.md` and the notebook header cells.

## Module aims

Aims are broad statements of intent — why the module exists — distinct from
the specific, assessable learning outcomes below.

This module aims to:

1. Equip chemistry students with practical programming and data-analysis
   skills, starting from **no prior programming experience**, sufficient to
   read, clean and critically analyse real chemical data.
2. Introduce the core methodology of machine learning — from classical
   regression and classification through to deep learning — grounded
   throughout in chemical datasets and chemically meaningful evaluation,
   rather than taught as abstract computer science.
3. Develop fluency with the molecular representations and model
   architectures used in current AI-for-chemistry research (descriptors,
   fingerprints, graph neural networks, sequence/attention models,
   generative models), including working directly with pretrained models.
4. Build critical judgement about the reliability of AI-driven chemical
   predictions — recognising data leakage, benchmark and metric choice,
   representation bias, and the limits of explainability — so that
   graduates use these tools critically rather than as an unquestioned
   "black box".
5. Give students direct, first-hand experience of the end-to-end
   AI-for-chemistry research workflow (data → representation → model →
   evaluation → critique) through a self-directed mini-project on a real
   chemical dataset.

`TODO(verify)`: align this wording with the approved KCL module-spec aims
before publishing to students (see `admin/keats-course-setup.md` §1.1 for
where this text is reused).

## Learning outcomes

On successful completion of this module, students will be able to:

1. **[LO1] Program in Python for chemical data.** Write, run and debug Python code
   (variables, control flow, functions, `numpy`, `pandas`, `matplotlib`) to read,
   transform and visualise chemical data files, starting from no prior
   programming experience.

2. **[LO2] Represent molecules computationally.** Convert between chemical
   structures and machine-readable representations (SMILES, RDKit molecule
   objects, molecular descriptors, fingerprints), and assemble a clean,
   documented, featurised molecular dataset suitable for machine learning.

3. **[LO3] Apply the supervised ML workflow correctly.** Explain and carry out
   train/validation/test splitting, cross-validation and the diagnosis of
   over- and under-fitting, and select regression or classification performance
   metrics appropriate to a given chemical prediction task.

4. **[LO4] Build and evaluate classical ML models.** Train, tune and compare
   linear and logistic regression, random forests, clustering and PCA for
   molecular property prediction, activity classification and chemical-space
   analysis, and perform basic error analysis on the results.

5. **[LO5] Implement neural networks.** Explain the core mechanics of deep
   learning (tensors and shapes, linear/non-linear layers, loss functions,
   gradient descent, backpropagation) and implement, train and regularise
   multilayer perceptrons in PyTorch on a laptop CPU.

6. **[LO6] Reason about representation and inductive bias.** Compare fixed
   descriptors with learned features, and sequence with graph representations of
   molecules, and explain the role of invariance and equivariance in models for
   molecules and materials.

7. **[LO7] Use modern architectures for chemistry.** Describe and apply graph
   neural networks, attention/transformer-based sequence models and generative
   models (variational autoencoders; flows and diffusion at a survey level),
   including running, evaluating or fine-tuning pretrained models.

8. **[LO8] Assess reliability and deliver a study.** Evaluate an AI-for-chemistry
   result for data leakage, benchmark and metric choice, applicability domain,
   explainability and reproducibility, and complete and communicate a small
   end-to-end study on a chemical dataset.

## Indicative mapping to sessions

| LO  | Primarily developed in |
|-----|------------------------|
| LO1 | Wk 01 A/B, Wk 02 A |
| LO2 | Wk 02 A/B, Wk 06 A/B |
| LO3 | Wk 03 A/B, Wk 04 A |
| LO4 | Wk 03 B, Wk 04 A/B |
| LO5 | Wk 05 A/B |
| LO6 | Wk 06 A/B, Wk 07 A, Wk 08 A |
| LO7 | Wk 07–09 A/B |
| LO8 | Wk 09 A, Wk 10 A/B |
