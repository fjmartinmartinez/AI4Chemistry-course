"""Shared notebook builder for the AI4Chemistry course.

One build script per session defines an ordered list of cells; this module
renders two synchronised notebooks from that single source so the solutions and
student versions can never diverge (CLAUDE.md notebook standards):

* ``<name>_solutions.ipynb`` — every exercise body filled in, conceptual-question
  answers shown.
* ``<name>_student.ipynb`` — exercise bodies replaced by a scaffold
  (``# YOUR CODE HERE`` + signature + hints), assert-based checks KEPT verbatim,
  answers hidden, all outputs stripped.

Usage
-----
    from nbbuild import md, code, exercise, checkpoint, solution_only, build

    CELLS = [
        md("# Title ..."),
        code("import numpy as np"),
        exercise(
            prompt="### Exercise 1 — ... (easy, ~10 min)\\n...",
            solution="def f(x):\\n    return 2 * x",
            scaffold="def f(x):\\n    # YOUR CODE HERE\\n    ...",
            check="assert f(3) == 6, 'expected 6'",
        ),
    ]
    build(__file__, "week01_a_python_bootcamp_i", CELLS)

Run the build script inside the ``ai4chem`` environment; it only needs
``nbformat``.
"""
from __future__ import annotations

import dataclasses
import pathlib
from typing import Iterable

import nbformat


@dataclasses.dataclass
class Cell:
    """One planned notebook cell.

    kind:     "markdown" or "code"
    sol:      source used in the solutions notebook
    stu:      source used in the student notebook (defaults to ``sol``)
    """

    kind: str
    sol: str
    stu: str | None = None

    def student_source(self) -> str:
        return self.sol if self.stu is None else self.stu


def _dedent(text: str) -> str:
    return text.strip("\n").rstrip() + "\n" if text.strip() else ""


def md(text: str) -> Cell:
    """A markdown cell shown identically in both notebooks."""
    return Cell("markdown", _dedent(text))


def code(src: str) -> Cell:
    """A code cell shown identically in both notebooks."""
    return Cell("code", _dedent(src))


def solution_only(text: str) -> Cell:
    """Markdown shown only in the solutions notebook.

    Used for conceptual-question answers. In the student notebook the cell
    collapses to a single reminder line.
    """
    return Cell(
        "markdown",
        _dedent(text),
        stu="> _Answers to the conceptual questions are in the solutions "
        "notebook._\n",
    )


def exercise(prompt: str, solution: str, scaffold: str, check: str) -> list[Cell]:
    """An exercise: prompt (md) -> work cell (code) -> check cell (code).

    * solutions notebook: work cell holds ``solution``.
    * student notebook:    work cell holds ``scaffold``.
    * the check cell (asserts) is identical in both and is never blanked.
    """
    return [
        md(prompt),
        Cell("code", _dedent(solution), stu=_dedent(scaffold)),
        code("# --- check your answer (do not edit) ---\n" + check.strip()),
    ]


def checkpoint(
    title: str,
    check: str,
    expected: str,
    questions: str,
    answers: str,
) -> list[Cell]:
    """A 'Check yourself' section: assert cell + expected result + Q&A.

    ``answers`` appears only in the solutions notebook.
    """
    header = f"### ✅ Check yourself — {title}\n\n{expected}"
    q_block = f"**Conceptual questions**\n\n{questions}"
    return [
        md(header),
        code("# --- checkpoint (do not edit) ---\n" + check.strip()),
        md(q_block),
        solution_only(f"**Answers**\n\n{answers}"),
    ]


def _to_nbnode(cells: Iterable[Cell], *, student: bool) -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3 (ai4chem)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        }
    )
    out = []
    for c in cells:
        src = c.student_source() if student else c.sol
        if not src.strip():
            continue
        if c.kind == "markdown":
            out.append(nbformat.v4.new_markdown_cell(src.rstrip() + "\n"))
        else:
            out.append(nbformat.v4.new_code_cell(src.rstrip() + "\n"))
    nb.cells = out
    _, nb = nbformat.validator.normalize(nb)
    return nb


def build(build_file: str, name: str, cells: list) -> None:
    """Write ``<name>_solutions.ipynb`` and ``<name>_student.ipynb``.

    Notebooks are written next to the build script (a session ``notebook/``
    folder). Flatten any nested lists returned by ``exercise``/``checkpoint``.
    """
    flat: list[Cell] = []
    for item in cells:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)

    here = pathlib.Path(build_file).resolve().parent
    for student, suffix in [(False, "solutions"), (True, "student")]:
        nb = _to_nbnode(flat, student=student)
        path = here / f"{name}_{suffix}.ipynb"
        nbformat.write(nb, str(path))
        print(f"wrote {path.relative_to(here.parents[3])}")
