# AI4Chemistry-Course

Working repository for developing the AI-in-chemistry module (KCL) with Claude as course-development collaborator.

## How to use this project

1. Drop raw material into `sources/` (papers → `papers/`, old decks → `legacy-slides/`, code → `code/`, data → `datasets/`, URLs → `links.md`).
2. Put the syllabus in `admin/syllabus.md` (any format is fine to start; ask Claude to normalise it).
3. Open a Claude session with this folder connected and say, e.g.:
   - "Outline week 3 on molecular representations from the sources" → uses the `lecture-outline` skill.
   - "Build the notebook for week 3" → uses `build-notebook`.
   - "Build the slides for week 3" → uses `build-slides`.
4. Review outputs in `lectures/week-NN_topic/`; Claude reads `CLAUDE.md` for all conventions.

`prompts/` contains the same instructions as copy-paste templates if you work outside this folder.
