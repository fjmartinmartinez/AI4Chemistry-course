---
name: build-notebook
description: Build the paired student/solutions Jupyter notebooks for a lecture of the AI-for-chemistry course. Use when asked to build, write, or revise a lecture notebook for this module.
---

# Build notebook

Follow `prompts/03_build-notebook.md` at the project root, with the notebook standards in `CLAUDE.md`. Non-negotiables:

- Requires an approved `outline.md` in the lecture folder; if absent, run the `lecture-outline` skill first and stop for approval.
- Solutions notebook first; execute it top-to-bottom in the `env/environment.yml` environment and fix failures before deriving the student version (blank exercise cells, keep scaffolding and asserts).
- Fixed seeds, laptop-CPU budget, datasets from `sources/datasets/` or checksum-verified download.
- Export slide figures to `slides/figures/`; update the lecture `CHANGELOG.md` and `env/environment.yml`.
- When revising an existing notebook, read it first and touch only what the task requires.
