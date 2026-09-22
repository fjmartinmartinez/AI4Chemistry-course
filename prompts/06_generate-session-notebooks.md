# Prompt — Generate all session notebooks (paste into Claude Code, run from project root)

Read CLAUDE.md, admin/syllabus.md, admin/session-links.md, admin/learning-outcomes.md and sources/links.md in full before writing anything. Your task: produce the self-guided Jupyter notebooks for the whole course — 10 weeks x 2 sessions (A = 2.5 h interleaved lecture-lab, B = 2 h workshop) — following the notebook standards in CLAUDE.md.

## Process (strict — do not reorder)
1. Create admin/NOTEBOOK_PROGRESS.md: a checklist of all 20 sessions with status (pending / drafted / executed-clean / student-version-done). Update it after every step so an interrupted run can resume exactly where it stopped.
2. Set up the environment from env/environment.yml (create it with conda/mamba if absent) and verify imports before writing any notebook.
3. Work ONE WEEK AT A TIME, in order, starting at week 01. For each week:
   a. Re-read that week's syllabus row and its links in admin/session-links.md. WebFetch the CORE links (dmol.pub chapters and the listed lessons) and ground all content in them plus sources/. Do not invent facts, equations, data, or references not present in those sources; mark anything you cannot verify as TODO(verify).
   b. If lectures/week-NN_<topic>/ does not exist, copy lectures/_template/ to create it and fill outline.md concisely (the syllabus is already approved — the outline is a record, not a proposal).
   c. Write weekNN_a_<topic>_solutions.ipynb and weekNN_b_<topic>_solutions.ipynb in that week's notebook/ folder.
   d. Execute each solutions notebook top-to-bottom (jupyter nbconvert --to notebook --execute) in the course environment. Fix and re-run until clean. Budget: <10 min runtime per notebook on laptop CPU; fixed random seeds; datasets cached under sources/datasets/ (download once with a checksum check, then load locally).
   e. Derive weekNN_{a,b}_<topic>_student.ipynb from the executed solutions: replace each exercise solution body with a scaffold ("# YOUR CODE HERE" plus function signatures and hints), KEEP the assert-based checks, strip outputs.
   f. Export any figure the slides will reuse to slides/figures/; log the week in the lecture CHANGELOG.md; add any new dependency to env/environment.yml in the same change.
4. STOP after completing weeks 01–02 and report for review (structure, tone, difficulty calibration). Continue with weeks 03–10 only after approval; if running unattended, continue but list every judgement call in the final report.

## Self-guided documentation requirements (every notebook)
- Header cell: course name, week/session, time budget with a suggested per-section timing table, learning objectives mapped to admin/learning-outcomes.md, prerequisites ("before this notebook you should be able to..."), and a "how to use this notebook" paragraph for solo study.
- A markdown narrative cell BEFORE every code cell: what the cell does, why it matters chemically, and what to look for in the output. A student with no instructor must be able to follow the whole notebook. UK English, direct analytical tone, LaTeX for all equations.
- Code: comment every non-obvious line or block; no unexplained magic numbers; idiomatic numpy/pandas/sklearn/RDKit/PyTorch; cells <=25 lines unless justified.
- Checkpoints: after each major section, a "Check yourself" cell with asserts and a stated expected result, plus 1-2 conceptual questions (answers in the solutions version only).
- Exercises: 2-4 per session, graded difficulty with time estimates; hints in <details> blocks; session-B notebooks end with a mini-challenge.
- A "Common errors and troubleshooting" markdown box per major section (import errors, shape mismatches, RDKit parsing failures, etc.).
- Final cells: summary of what was learned, further-reading links taken ONLY from admin/session-links.md, and an attribution cell crediting adapted sources with links and licenses (dmol.pub, MolSSI, TeachOpenCADD, EPFL ai4chem, Pat Walters, SciCompforChemists). Adapt and rewrite — never paste long verbatim passages from sources.
- Weeks 08-09: notebooks must also run on Google Colab — include a guarded `if 'google.colab' in sys.modules:` pip-install cell and keep a CPU-only reduced-size fallback path so they still execute locally.
