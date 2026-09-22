# Changelog
- 2026-09-11: created from template.
- 2026-09-11: outline.md filled from approved syllabus (v0.1).
- 2026-09-11: built self-guided notebooks for sessions A and B
  (`week07_{a,b}_graph-neural-networks_{solutions,student}.ipynb`), generated
  from `notebook/build_week07_{a,b}.py` via `lectures/_build/nbbuild.py`.
  Solutions notebooks execute clean (`nbconvert --execute`): A < 3 s, B ~88 s
  (full GCN training + a 5-point depth ablation; well within the <10 min
  budget). Session A: molecule-to-graph conversion, a minimal `GCNLayer`
  verified numerically permutation-equivariant, mean/sum readout, one manual
  training step — mechanics only. Session B: full mini-batch (gradient
  accumulation) training, an honest comparison to Weeks 03/05's leaderboard
  (GCN beats the trivial baseline but not linear regression/MLP/random forest
  here — reported honestly, not adjusted to force a win), and a depth
  ablation explained via molecule diameter rather than an unearned
  "over-smoothing confirmed" claim.
  Implemented a minimal from-scratch GCN in plain PyTorch (dmol.pub's own
  approach) rather than adding PyTorch Geometric/DGL/Chemprop as a dependency
  (open question logged in outline.md). QM9 deferred to Week 10 (its own
  dmol.pub core link) rather than duplicated here.
  Slide figures (shared `slides/figures/` across both sessions):
  `readout_pooling_comparison.png` (A); `gcn_training_curve.png`,
  `gcn_vs_leaderboard.png`, `gcn_depth_ablation.png` (B). No new environment
  dependencies (torch CPU already in `env/environment.yml`).
