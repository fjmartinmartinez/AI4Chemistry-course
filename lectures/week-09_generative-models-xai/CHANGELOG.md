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
