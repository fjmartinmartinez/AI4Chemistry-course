# Changelog
- 2026-09-11: created from template.
- 2026-09-11: outline.md filled from approved syllabus (v0.1).
- 2026-09-11: built self-guided notebooks for sessions A and B
  (`week06_{a,b}_representations-inductive-bias_{solutions,student}.ipynb`),
  generated from `notebook/build_week06_{a,b}.py` via `lectures/_build/nbbuild.py`.
  Solutions notebooks execute clean (`nbconvert --execute`): A < 3 s, B ~10 s.
  Session A: concrete translational-variance demo + formal invariance/
  equivariance definitions using Week 01's `ethanol.xyz`; equivariant NNs
  covered qualitatively only (no code), per syllabus scope cut. Session B:
  representation shoot-out (descriptors/MACCS/Morgan-2048 x Ridge/kNN/RF) on
  ESOL, substituting RDKit+scikit-learn for DeepChem's own featurizers (open
  question logged in outline.md) — ends by assigning the mini-project.
  Added `assessments/mini-project-brief.md` (new — DRAFT, needs instructor
  sign-off; several `TODO(verify)` items: exact weighting, group-size
  policy, submission format).
  Slide figures exported to `slides/figures/`: `invariant_vs_variant.png`
  (session A); `representation_shootout_grid.png`,
  `chemical_space_by_representation.png` (session B). No new environment
  dependencies.
