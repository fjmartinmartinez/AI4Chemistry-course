# Changelog
- 2026-09-02: created from template.
- 2026-09-02: outline.md filled from approved syllabus (v0.1).
- 2026-09-02: built self-guided notebooks for sessions A and B
  (`week02_{a,b}_data-and-molecules_{solutions,student}.ipynb`), generated from
  `notebook/build_week02_{a,b}.py` via `lectures/_build/nbbuild.py`.
  Solutions notebooks execute clean (`nbconvert --execute`), < 10 s each.
  Dataset: `sources/datasets/esol_delaney.csv` (checksum-pinned download).
  Session B writes `sources/datasets/esol_clean.csv` (1117 rows, canonicalised,
  deduplicated) — the cleaned table Week 03 builds on.
  Slide figures exported to `slides/figures/`: `solubility_vs_psa.png`,
  `least_soluble_grid.png`, `solubility_vs_logp.png`, `chemical_space_pca.png`,
  `cluster_sizes.png`, `largest_cluster.png`.
  No new environment dependencies (rdkit, scikit-learn already in
  `env/environment.yml`).
