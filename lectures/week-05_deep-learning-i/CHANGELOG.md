# Changelog
- 2026-09-11: created from template.
- 2026-09-11: outline.md filled from approved syllabus (v0.1).
- 2026-09-11: built self-guided notebooks for sessions A and B
  (`week05_{a,b}_deep-learning-i_{solutions,student}.ipynb`), generated from
  `notebook/build_week05_{a,b}.py` via `lectures/_build/nbbuild.py`.
  Solutions notebooks execute clean (`nbconvert --execute`): A < 3 s, B ~14 s.
  Session A is mechanics-only (tensors/shapes, dense layers, backprop vs
  autograd, one manual training step); Session B is the full train/val/test
  workflow (loss curves, early stopping, hyperparameter sweep) on the exact
  Week 03 test split, ending with an honest MLP-vs-random-forest comparison.
  Adapted EPFL's `01_intro_to_dl.ipynb` (PyTorch Lightning + Weights & Biases)
  to plain PyTorch, since neither dependency is in `env/environment.yml`
  (open question logged in outline.md).
  Slide figures exported to `slides/figures/`: `activation_functions.png`
  (session A); `mlp_loss_curve_overfit.png`, `hparam_sweep_heatmap.png`,
  `mlp_vs_week03_leaderboard.png` (session B). No new environment
  dependencies (torch CPU already in `env/environment.yml`).
