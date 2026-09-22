# Changelog
- 2026-09-02: created from template.
- 2026-09-02: outline.md filled from approved syllabus (v0.1).
- 2026-09-02: built self-guided notebooks for sessions A and B
  (`week01_{a,b}_python-bootcamp_{solutions,student}.ipynb`), generated from
  `notebook/build_week01_{a,b}.py` via `lectures/_build/nbbuild.py`.
  Solutions notebooks execute clean (`nbconvert --execute`), < 30 s each.
  Datasets cached in `sources/datasets/`: `water.xyz`, `ethanol.xyz`,
  `benzene.xyz` (RDKit ETKDGv3+MMFF94, seed 0xC0FFEE), `alkanes.csv` (curated,
  TODO(verify)), `ir_spectrum_synthetic.csv` (generated, not measured).
  Slide figures exported to `slides/figures/`: `ir_spectrum.png`,
  `ir_absorbance_vs_transmittance.png`, `vdw_isotherms.png`,
  `beer_lambert_fit.png`. No new environment dependencies.
