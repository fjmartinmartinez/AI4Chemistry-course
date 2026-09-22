# Changelog
- 2026-09-12: created from template (post-course-build addition, on
  instructor request — an optional, ungraded pre-course primer for students
  with no programming background).
- 2026-09-12: outline.md written; not part of the 10-week syllabus contact
  hours.
- 2026-09-12: built `week00_python-primer_{solutions,student}.ipynb` from
  `notebook/build_week00_primer.py` via `lectures/_build/nbbuild.py`.
  Solutions notebook executes clean (`nbconvert --execute`), instant runtime,
  standard-library only (no RDKit/NumPy/pandas) so it runs in any Python 3 +
  Jupyter install, not only the course environment. Single notebook (no A/B
  session split, no mini-challenge) covering: notebooks/cells/kernel, print
  and comments, variables and types, arithmetic, f-strings, and the most
  common beginner error messages (with a deliberately-triggered `NameError`).
  No new environment dependencies. No slides planned (self-study only).
