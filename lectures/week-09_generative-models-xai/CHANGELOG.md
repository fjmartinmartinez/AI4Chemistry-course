# Changelog
- 2026-09-12: created from template.
- 2026-09-12: outline.md filled from approved syllabus (v0.1).
- 2026-09-12: built self-guided notebooks for sessions A and B
  (`week09_{a,b}_generative-models-xai_{solutions,student}.ipynb`), generated
  from `notebook/build_week09_{a,b}.py` via `lectures/_build/nbbuild.py`.
  Solutions notebooks execute clean (`nbconvert --execute`): A < 5 s, B ~39 s.
  Session A: a VAE (encoder/decoder, reparameterisation trick, ELBO) trained
  on the ESOL descriptor block, compared to PCA; a genuine, checkable
  sampling failure mode (decoded prior samples can have negative/impossible
  descriptor values); flows/diffusion covered at survey level per syllabus
  scope cut; gradient saliency and integrated gradients implemented and
  completeness-checked; SHAP/LIME/counterfactuals surveyed. Session B: a
  from-scratch character-level LSTM generative model, guarded RDKit install
  for Colab (not preinstalled there) + `N_EPOCHS` reduced-size local
  fallback (40 vs 80), validity/uniqueness/novelty assessment (76%/96%/62%
  at temperature 0.7 in local testing), a temperature/validity-diversity
  trade-off sweep, and a property-distribution sanity check.
  Slide figures (shared `slides/figures/`): `vae_vs_pca_latent.png`,
  `saliency_bar.png`, `integrated_gradients_bar.png` (A);
  `char_rnn_training_curve.png`, `validity_vs_temperature.png`,
  `generated_property_distribution.png` (B). No new environment dependencies
  beyond `transformers` (already added Week 08) and `rdkit` (already in
  `env/environment.yml`; only newly *guarded* for Colab in this notebook).
- 2026-09-22: removed Session B's Google Colab compatibility path (course
  decision: all workshops run as Jupyter notebooks in the `ai4chem`
  environment; Colab is not used). Dropped the `IN_COLAB` detection and
  guarded `pip install` for `rdkit` (already a pinned conda dependency);
  `N_EPOCHS` is now a fixed value of 40 rather than a Colab-vs-local
  conditional — no change to the executed/verified local behaviour, since
  40 was already the local-path value. Replaced the now-moot "why guard
  against Colab" conceptual question with one on the `<start>`/`<end>`
  reserved-vocabulary-slot design already covered in the Common errors box.
  Updated `build_week09_b.py` and both `week09_b_*.ipynb` variants (source
  cells and one stale stdout line only; all other stored outputs
  unchanged) and `outline.md`. Edited in a container without the
  `ai4chem` environment available, so **not** re-verified with a fresh
  `nbconvert --execute` pass — `TODO(verify)`: run `Restart & Run All`
  before next use to confirm.
