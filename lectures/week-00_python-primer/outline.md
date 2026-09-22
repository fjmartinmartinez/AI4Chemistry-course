# Week 00 — Python primer (optional, pre-course)

_Not part of the 10-week syllabus contact hours. Added on instructor request
as a free-time, ungraded on-ramp for students with zero programming
background, so Week 01 Session A is not literally the first Python anyone
has ever typed._

## Learning objectives
By the end of this primer students can:
1. Open a Jupyter notebook, distinguish a code cell from a markdown cell, and
   run a cell. [pre-LO1]
2. Use `print()`, assign variables, and name the basic Python types
   (`int`, `float`, `str`). [LO1]
3. Perform simple arithmetic and build one f-string. [LO1]
4. Recognise the three or four most common beginner error messages and know
   the single most useful fix ("did you run the cells above this one, in
   order?"). [LO1]

These map to the same [LO1] as Week 01A/B in `admin/learning-outcomes.md`;
this primer exists only to make Week 01 easier, not to replace it.

## Prerequisites assumed
- None. A working computer and the course environment installed
  (`env/environment.yml`) or access to a Jupyter environment.

## Concept sequence
| # | Concept | Chemistry anchor | Where |
|---|---------|------------------|-------|
| 1 | Notebooks, cells, the kernel, running a cell | — | notebook |
| 2 | `print()`, comments | a first "hello, chemistry" message | notebook |
| 3 | Variables and types (`int`, `float`, `str`) | a compound's name and molar mass | notebook |
| 4 | Arithmetic, order of operations | density = mass / volume | notebook |
| 5 | f-strings | formatting a lab-style result line | notebook |
| 6 | Beginner error messages, and running cells in order | — | notebook |

## Notebook plan
- **One notebook**, not a session pair (`week00_python-primer_*`) — shorter
  than a normal session (45-60 min), no mini-challenge, 3 light exercises.
  No dataset, no third-party imports beyond the Python standard library —
  deliberately zero-friction to run anywhere, including outside the course
  environment.
- Compute budget: instant (no computation beyond arithmetic).

## Slides plan
- None. This primer is self-study only; it is not taught in a contact-hour
  session, so no deck is planned. `slides/figures/` is kept for structural
  consistency with every other lecture folder but is expected to stay empty.

## Assessment hooks
- None — ungraded, optional, not examinable in its own right (though the
  Python fluency it builds underlies everything that is examinable from
  Week 01 onward).

## Sources used
- `sources/links.md`'s own "Beginner on-ramps to dmol.pub" list: pointed to,
  not reproduced — pythoninchemistry.org's *Intro to Python for Chemistry*
  and MolSSI `python_scripting_cms` lesson 01, both already tagged
  "pre-course" in that file.

## Open questions for instructor
- Confirm this should be framed as fully optional/ungraded (as built) rather
  than a required pre-course step — the syllabus states Week 01 "assumes NO
  prior programming," so this primer is a convenience, not a prerequisite.
- Consider whether to advertise this primer's existence in any joining
  instructions / welcome email sent before Week 01, since a self-guided
  optional resource is only useful if students know it exists.
