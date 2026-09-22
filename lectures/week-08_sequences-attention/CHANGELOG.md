# Changelog
- 2026-09-12: created from template.
- 2026-09-12: outline.md filled from approved syllabus (v0.1).
- 2026-09-12: added `transformers>=4.40,<5` to `env/environment.yml` — the
  first genuine new dependency this course adds (needed to load a real
  pretrained chemical language model; unlike earlier substitutions, no
  from-scratch reimplementation is a faithful stand-in for "use a pretrained
  transformer").
- 2026-09-12: built self-guided notebooks for sessions A and B
  (`week08_{a,b}_sequences-attention_{solutions,student}.ipynb`), generated
  from `notebook/build_week08_{a,b}.py` via `lectures/_build/nbbuild.py`.
  Solutions notebooks execute clean (`nbconvert --execute`): A < 5 s, B ~31 s.
  Session A: char-level SMILES tokenisation, hand-implemented scaled
  dot-product attention (verified as a genuine convex combination), a
  seq2vec embedding+GRU model (dmol.pub's own architecture, including its
  honestly-reported underperformance vs linear regression), causal masking
  for autoregressive prediction, transformers/pretraining covered at survey
  level. Session B: loads `seyonec/ChemBERTa-zinc-base-v1` (frozen) with a
  guarded Colab install cell and an `N_MAX` reduced-size local fallback
  (400 molecules locally, full 1117 on Colab); frozen-embedding property
  prediction compared honestly against RDKit descriptors (descriptors still
  win on this dataset scale); inspects the model's own real attention
  weights, contrasted with Session A's untrained toy example.
  Substituted property prediction for the syllabus's reaction-prediction
  core link (open question logged in outline.md).
  Slide figures (shared `slides/figures/`): `attention_weights_toy.png`,
  `causal_mask.png` (A); `chembert_embedding_pca.png`,
  `chembert_attention_heatmap.png`, `chembert_vs_leaderboard.png` (B).
- 2026-09-22: removed Session B's Google Colab compatibility path (course
  decision: all workshops run as Jupyter notebooks in the `ai4chem`
  environment; Colab is not used). Dropped the `IN_COLAB` detection and
  guarded `pip install` for `transformers` (already a pinned dependency);
  `N_MAX` is now a fixed 400-molecule working set rather than a
  Colab-vs-local conditional — no change to the executed/verified local
  behaviour, since 400 was already the local-path value. Updated
  `build_week08_b.py` and both `week08_b_*.ipynb` variants (source cells
  and one stale stdout line only; all other stored outputs unchanged) and
  `outline.md`. Edited in a container without the `ai4chem` environment
  available, so **not** re-verified with a fresh `nbconvert --execute`
  pass — `TODO(verify)`: run `Restart & Run All` before next use to
  confirm.
