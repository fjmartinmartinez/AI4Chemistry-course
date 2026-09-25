# Week 08 — Sequences, attention and chemical language models

_Record of the approved design (syllabus v0.1). Not a proposal._

## Learning objectives
By the end of week 08 students can:
1. Tokenise SMILES at character level and explain why the *order* of tokens
   is chemically meaningful (unlike a fingerprint's bit order). [LO7]
2. State and implement scaled dot-product attention
   ($\mathrm{softmax}(\vec q\cdot K/\sqrt d)\cdot V$) from its query/key/value
   definitions. [LO7]
3. Build a minimal sequence-to-vector model (embedding + GRU) for property
   prediction, and a causal attention mask for autoregressive next-token
   prediction. [LO7]
4. Describe, at survey level, how transformer blocks, multi-head attention
   and masked-language-model pretraining combine into chemical language
   models (ChemBERTa, MolBERT, ...). [LO7]
5. (Session B) Load a pretrained chemical transformer, use it (frozen) for
   property prediction, and inspect its own internal attention weights. [LO4, LO7]

## Prerequisites assumed
- Week 07: `nn.Module` patterns extended to a new data type (sequences,
  instead of graphs).
- Week 02: SMILES strings, canonicalisation (why more than one string can
  encode the same molecule — now relevant to tokenisation too).

## Concept sequence
| # | Concept | Chemistry anchor | Where |
|---|---------|------------------|-------|
| 1 | SMILES as sequences: character tokenisation, padding | order-dependence of SMILES | notebook A |
| 2 | Scaled dot-product attention (Q/K/V) | attention weights over a SMILES string | notebook A |
| 3 | Seq2vec: embedding + GRU for property prediction | ESOL solubility | notebook A |
| 4 | Causal masking for autoregressive next-token prediction | sets up Week 09's generative RNN | notebook A |
| 5 | Transformers, multi-head attention, pretraining (survey) | ChemBERTa/MolBERT named, not built | notebook A |
| 6 | A pretrained chemical transformer (ChemBERTa), frozen | subword tokenisation vs Section 1's char-level | notebook B |
| 7 | Frozen embeddings for property prediction | ESOL, compared to Weeks 03/06/07 | notebook B |
| 8 | Inspecting a pretrained model's own attention weights | real attention, not the toy of notebook A | notebook B |

## Notebook plan
- **Session A** (`week08_a_sequences-attention_*`): pure PyTorch, no pretrained
  model; mechanics of sequences/attention/seq2vec, mirroring dmol.pub's own
  worked example (embedding+GRU on ESOL, which dmol itself reports
  underperforming linear regression — reproduced honestly, not "fixed"). 4
  exercises.
- **Session B** (`week08_b_sequences-attention_*`): loads
  `seyonec/ChemBERTa-zinc-base-v1` (44M-parameter RoBERTa-style model
  pretrained on ~ZINC SMILES) via `transformers`, frozen (no fine-tuning —
  keeps runtime laptop-appropriate). Reduced-size (`N_MAX`) working set to
  keep embedding extraction laptop-CPU-friendly (course runs on Jupyter
  locally; no Colab path). 3 exercises + mini-challenge.
- Compute budget: Session A < 10 s; Session B ~15-30 s locally
  (`N_MAX=400` molecules).

## Slides plan
- Session A ~8 slides: SMILES tokenisation; attention Q/K/V; seq2vec
  architecture; causal masking; the transformer/pretraining survey slide
  (named architectures, explicitly marked survey-only for hands-on work).
  Demo → notebook A §1-4.
- Session B ~8 slides: loading a pretrained model; frozen-embedding
  workflow; the chemical-space PCA; a real attention-weight heatmap from
  ChemBERTa itself. Demo → notebook B §1-3.
- Figures: `attention_weights_toy.png`, `causal_mask.png` (session A);
  `chembert_embedding_pca.png`, `chembert_attention_heatmap.png`,
  `chembert_vs_leaderboard.png` (session B).

## Assessment hooks
- Examinable: the scaled dot-product attention formula and what Q/K/V mean;
  why SMILES token order matters; the seq2vec architecture; the difference
  between frozen-embedding probing and fine-tuning (survey level).

## Sources used
- dmol.pub *Attention* (Q/K/V, scaled dot-product formula, multi-head
  attention, self- vs cross-attention).
- dmol.pub *Deep learning on sequences* (SMILES tokenisation, RNN/GRU/LSTM,
  seq2vec/seq2seq, autoregressive next-token prediction, the course's own
  GRU-on-ESOL worked example and its reported underperformance vs linear
  regression, SELFIES).
- dmol.pub *Pretraining* — survey only, per syllabus wording ("chemical LLMs
  (survey)"); named architectures (ChemBERTa, MolBERT) are not built from
  scratch, but Session B *uses* one directly (loading a real pretrained
  model, not surveying it, is the syllabus's B-session ask).
- EPFL ai4chem *reaction prediction (template-free)* — the syllabus's B-core
  link; **not reproduced directly** (see Open questions) in favour of a
  property-prediction task using the same "pretrained transformer" ask.

## Open questions for instructor
- The syllabus's Session B core link is EPFL's *template-free reaction
  prediction* notebook; we instead use a frozen pretrained ChemBERTa for
  **property prediction** (the syllabus row's other stated option: "for
  property prediction / reaction task"). Reaction prediction needs a
  seq2seq/encoder-decoder setup and reaction-specific pretrained checkpoints
  that add materially more scope; property prediction on ESOL keeps Session B
  comparable to every earlier week's leaderboard. Confirm this choice, or
  request reaction prediction as well/instead.
- This is the first week to add a real dependency, `transformers` (Hugging
  Face), to `env/environment.yml` — unlike prior substitutions (LightGBM,
  Mordred, PyTorch Lightning, DeepChem, PyTorch Geometric), a "pretrained
  transformer" cannot be reproduced without a transformer library and a real
  pretrained checkpoint, so this dependency is load-bearing, not optional.
  Confirm network access to Hugging Face Hub is available where the course is
  taught (first run downloads a small model, ~180 MB).
- Our own CPU timing showed the full 1117-molecule ESOL set actually runs
  in well under a minute on a laptop CPU (frozen inference only, no
  fine-tuning) — `N_MAX=400` is kept as a deliberately smaller working set
  regardless, since Section 2's point is teaching the reduced-size-working-
  set pattern itself; confirm whether to keep the reduction or widen `N_MAX`
  to the full dataset, now that laptop timing is confirmed comfortable
  either way.
