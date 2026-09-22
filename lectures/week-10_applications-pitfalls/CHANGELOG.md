# Changelog
- 2026-09-12: created from template.
- 2026-09-12: outline.md filled from approved syllabus (v0.1).
- 2026-09-12: cached a fixed 3000-molecule QM9 subsample
  (`sources/datasets/qm9_subset.csv`) via
  `lectures/_build/make_datasets_wk10.py` (full `qm9.csv` checksum-pinned,
  only the small subsample committed).
- 2026-09-12: built self-guided notebooks for sessions A and B
  (`week10_{a,b}_applications-pitfalls_{solutions,student}.ipynb`), generated
  from `notebook/build_week10_{a,b}.py` via `lectures/_build/nbbuild.py`.
  Solutions notebooks execute clean (`nbconvert --execute`): A ~61 s, B ~11 s.
  Session A reuses Week 07's `GCNLayer`/`GCN` unchanged on QM9 HOMO-LUMO gap
  (2D bond graph, not dmol's 3D distance features — disclosed scope
  simplification); surveys invariant vs equivariant potentials
  (SchNet/ANI vs NequIP/TorchMD-NET); quantifies split-to-split variance with
  10 repeated splits; applies a whole-course pitfalls checklist to its own
  QM9 study. Session B is a **fully worked mini-project reference example**
  on BBBP (no new taught material, per syllabus.md) hitting every required
  element from `assessments/mini-project-brief.md`.
  Slide figures (shared `slides/figures/`): `qm9_gap_parity_plot.png`,
  `split_variance_boxplot.png` (A); `mini_project_worked_leaderboard.png`,
  `mini_project_error_analysis.png` (B). No new environment dependencies.
- 2026-09-12: **course build complete** — all 10 weeks x 2 sessions (20
  notebook pairs, 40 files) built, executed clean, and student versions
  derived.
