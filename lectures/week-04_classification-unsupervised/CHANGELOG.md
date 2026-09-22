# Changelog
- 2026-09-11: created from template.
- 2026-09-11: outline.md filled from approved syllabus (v0.1).
- 2026-09-11: built self-guided notebooks for sessions A and B
  (`week04_{a,b}_classification-unsupervised_{solutions,student}.ipynb`),
  generated from `notebook/build_week04_{a,b}.py` via `lectures/_build/nbbuild.py`.
  Solutions notebooks execute clean (`nbconvert --execute`): A < 5 s, B ~10 s.
  Datasets: `BBBP.csv` (session A, blood-brain-barrier penetration) and
  `tox21.csv.gz` task SR-p53 (session B), both MoleculeNet/DeepChem mirror,
  checksum-pinned via `lectures/_build/make_datasets_wk04.py`.
  Cleaning extends Week 02's checklist with desalting (keep largest fragment).
  Kernel learning covered qualitatively only (markdown, no code), per
  `syllabus.md` scope cut.
  Slide figures exported to `slides/figures/`: `bbbp_decision_boundary.png`,
  `roc_curve.png`, `pca_chemical_space_bbbp.png` (session A);
  `tox21_leaderboard.png`, `tox21_pr_curve.png`, `false_negatives_grid.png`
  (session B). No new environment dependencies (scikit-learn already in
  `env/environment.yml`); dmol's own ClinTox+Mordred example substituted with
  BBBP/Tox21 + the Week 02 descriptor block to avoid adding `mordred` —
  flagged as an open question in outline.md.
