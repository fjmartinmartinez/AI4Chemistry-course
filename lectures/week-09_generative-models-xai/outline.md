# Week 09 — Generative models and explainability

_Record of the approved design (syllabus v0.1). Not a proposal._

## Learning objectives
By the end of week 09 students can:
1. Explain the VAE architecture (encoder, latent prior, decoder), the
   reparameterisation trick, and the ELBO loss (reconstruction + KL). [LO7]
2. Train a small VAE, compare its learned latent space to Week 02B/06B's PCA,
   and identify a concrete failure mode of sampling from its prior. [LO4, LO7]
3. State, at survey level, what normalising flows and diffusion models are
   and how they relate to VAEs (deliberate scope cut — no implementation). [LO7]
4. Compute gradient saliency and integrated gradients for a trained model,
   verify integrated gradients' completeness property, and describe (survey
   level) SHAP, LIME and counterfactual explanations. [LO7, LO8]
5. (Session B) Train a character-level generative RNN on SMILES, sample new
   strings, and assess them by validity, uniqueness and novelty. [LO4, LO7]

## Prerequisites assumed
- Week 08: sequences, tokenisation, causal masking (now used for real
  generation).
- Week 05: `nn.Module`, autograd, training loops.
- Week 02B/06B: PCA on a molecular representation.

## Concept sequence
| # | Concept | Chemistry anchor | Where |
|---|---------|------------------|-------|
| 1 | VAE: encoder/decoder, reparameterisation, ELBO | ESOL descriptor block | notebook A |
| 2 | Sampling from the VAE prior; a concrete failure mode | chemically impossible decoded values | notebook A |
| 3 | Flows and diffusion (survey only) | named, not implemented | notebook A |
| 4 | Gradient saliency; integrated gradients (completeness check) | which descriptor drives one prediction | notebook A |
| 5 | SHAP / LIME / counterfactuals (survey) | named, not implemented | notebook A |
| 6 | Character-level generative RNN (LSTM), teacher forcing | SMILES as a sequence (Week 08) | notebook B |
| 7 | Temperature-controlled sampling | validity vs diversity trade-off | notebook B |
| 8 | Assessing generated molecules: validity, uniqueness, novelty | standard generative-chemistry metrics | notebook B |

## Notebook plan
- **Session A** (`week09_a_generative-models-xai_*`): dataset = ESOL
  7-descriptor block (Weeks 02-03). A small VAE (2D latent) and a small MLP
  (for XAI) are both trained in-notebook, briefly. 4 exercises.
- **Session B** (`week09_b_generative-models-xai_*`): dataset = ESOL SMILES
  strings (character sequences, Week 08A's tokenisation extended with
  `<start>`/`<end>`). Guarded Colab install (RDKit is not preinstalled on
  Colab) + reduced local training-epoch fallback, per the Weeks 08-09
  Colab-compatibility requirement. 3 exercises + mini-challenge.
- Compute budget: Session A < 10 s; Session B ~30-40 s locally (40 epochs,
  reduced from a Colab default of 80).

## Slides plan
- Session A ~8 slides: VAE architecture; reparameterisation; ELBO; latent
  space vs PCA; the "VAE can decode nonsense" failure mode; flows/diffusion
  survey slide; gradient saliency; integrated gradients; XAI-method survey.
  Demo → notebook A §1-4.
- Session B ~8 slides: char-RNN architecture; training curve; temperature
  sampling; validity/uniqueness/novelty results. Demo → notebook B §1-3.
- Figures: `vae_latent_space.png`, `vae_vs_pca_latent.png`,
  `saliency_bar.png`, `integrated_gradients_bar.png` (session A);
  `char_rnn_training_curve.png`, `validity_vs_temperature.png`,
  `generated_property_distribution.png` (session B).

## Assessment hooks
- Examinable: the ELBO's two terms and what each does; why sampling a VAE's
  prior can decode to invalid/impossible outputs; gradient saliency vs
  integrated gradients vs global feature importance (Week 03B) — when do
  local and global explanations agree or disagree; validity/uniqueness/
  novelty as the standard generative-chemistry evaluation trio.

## Sources used
- dmol.pub *Variational autoencoders* (encoder/decoder, ELBO derivation,
  reparameterisation trick, beta-VAE) — the course's own examples use
  synthetic/polymer data; we apply the same architecture to the ESOL
  descriptor block instead, to keep a chemistry-labelled, previously-used
  dataset.
- dmol.pub *Explaining predictions* (gradient methods, integrated gradients,
  SHAP, LIME, counterfactuals — we implement the first two, survey the rest,
  matching the chapter's own emphasis).
- dmol.pub *Normalizing flows* — survey only, per syllabus scope cut.
- EPFL ai4chem *Molecular generative models* / *SMILES-LSTM walkthrough*, and
  dmol.pub's own in-browser *MolGenerator* demo — the char-level generative
  RNN pattern Session B implements directly in PyTorch (no browser demo or
  external service reproduced).

## Open questions for instructor
- dmol.pub's own VAE chapter does not use molecular/SMILES data (synthetic
  and polymer examples only); we apply its architecture to the ESOL
  descriptor block for continuity with the rest of the course. Confirm this
  substitution, or prefer a SMILES-based VAE (a substantially larger scope
  increase: discrete-sequence VAEs need a different decoder design than the
  continuous case dmol.pub itself teaches).
- Session B's char-RNN reaches roughly 80% validity at `temperature=0.7`
  after 40 epochs (~31 s) in local testing — confirm this level of polish is
  sufficient for a first generative-chemistry exposure, or whether a longer
  training budget (Colab-only) should be the headline number instead.
- No pretrained generative model (e.g. a public SMILES-VAE checkpoint) is
  used — Session B trains entirely from scratch in-session, which keeps the
  notebook self-contained but means generation quality is modest compared to
  published, fully-trained chemical generative models.
