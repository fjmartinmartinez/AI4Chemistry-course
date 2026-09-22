---
name: build-slides
description: Build or revise the PowerPoint deck for a lecture of the AI-for-chemistry course. Use when asked for slides, a deck, or a presentation for a week/lecture of this module.
---

# Build slides

Follow `prompts/04_build-slides.md` at the project root, with the slide standards in `CLAUDE.md`, using the pptx skill for file generation. Non-negotiables:

- Requires the approved `outline.md`; build after the notebook when one is planned, reusing its `slides/figures/`.
- 16:9 .pptx, one idea per slide, properly typeset equations, UK English.
- Speaker notes on every content slide: 2–5 sentences of lecture script in Fran's direct, analytical voice.
- Demo slides name the exact notebook section they point to.
- Missing figures become TODO(figure) placeholders — never fabricate data figures.
- Save to the lecture `slides/` folder; log in `CHANGELOG.md`.
