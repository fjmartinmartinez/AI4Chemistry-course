# Changelog
- 2026-09-11: created from template.
- 2026-09-11: outline.md filled from approved syllabus (v0.1).
- 2026-09-11: built self-guided notebooks for sessions A and B
  (`week03_{a,b}_regression-model-assessment_{solutions,student}.ipynb`),
  generated from `notebook/build_week03_{a,b}.py` via `lectures/_build/nbbuild.py`.
  Solutions notebooks execute clean (`nbconvert --execute`): A < 5 s, B ~13 s.
  Both notebooks self-heal `sources/datasets/esol_clean.csv` from
  `esol_delaney.csv` if Week 02B has not been run.
  Session B reuses Week 02B's Butina clustering (cutoff 0.4) to compare a
  random split against a leakage-aware cluster split.
  Slide figures exported to `slides/figures/`: `overfitting_train_test_gap.png`,
  `cv_fold_scores.png`, `parity_plot_linear.png`, `regularisation_shrinkage.png`
  (session A); `feature_importance.png`, `leaderboard.png`,
  `leakage_random_vs_cluster_split.png`, `rf_rmse_vs_trees.png` (session B).
  No new environment dependencies (scikit-learn already in
  `env/environment.yml`); deliberately used `RandomForestRegressor` instead of
  LightGBM to avoid adding one — flagged as an open question in outline.md.
