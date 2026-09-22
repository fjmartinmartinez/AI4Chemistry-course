# Prompt — Build the lecture notebook

---
Build the Jupyter notebook for week NN from the approved `outline.md`, following the notebook standards in `CLAUDE.md`. Requirements recap:
- Runs top-to-bottom in `env/environment.yml`, laptop CPU; seeds fixed; small dataset from `sources/datasets/` (or downloaded with checksum).
- Structure: objectives → setup → theory recap (LaTeX) → worked example → exercises (`# YOUR CODE HERE`) → extension → summary/further reading.
- Produce `weekNN_<topic>_solutions.ipynb` first, verify it executes cleanly, then derive `weekNN_<topic>_student.ipynb` by blanking the exercise cells.
- Save figures the slides will reuse to `slides/figures/` with descriptive names.
- Log the addition in the lecture `CHANGELOG.md` and add any new dependency to `env/environment.yml`.
