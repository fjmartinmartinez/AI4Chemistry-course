"""Build the week 00 Python primer notebooks (optional, pre-course, ungraded).

    python lectures/week-00_python-primer/notebook/build_week00_primer.py
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "_build"))
from nbbuild import build, checkpoint, code, exercise, md  # noqa: E402

C = []

# ==========================================================================
C += [md(r'''
# AI for Chemistry — Week 00
## Python primer (optional, pre-course, ungraded)

**Course:** AI for Chemistry · **Session:** none — this is a free-time
on-ramp, not a contact-hour session · **Runtime:** instant (no computation
beyond arithmetic) · **Standard library only** — no RDKit, NumPy or pandas,
so this runs in *any* Python 3 + Jupyter install, not only the course
environment.

### Suggested timing (solo study, ~50 min)

| Section | Topic | Time |
|--------:|-------|-----:|
| 0 | What is a notebook? Cells, the kernel, running a cell | 10 min |
| 1 | `print()` and comments | 6 min |
| 2 | Variables and types | 10 min |
| 3 | Arithmetic | 8 min |
| 4 | f-strings | 6 min |
| 5 | Beginner error messages | 6 min |
| 6 | Exercises (3) | 14 min |

### Is this for you?
If you have **never written a line of code before**, start here, in your own
time, before Week 01. If you have programmed in *any* language before
(even a little), you can almost certainly skip straight to Week 01A — nothing
here is new to you.

### Learning objectives
1. Open a Jupyter notebook, tell a code cell from a markdown cell, and run a
   cell. *(pre-LO1)*
2. Use `print()`, assign variables, and name Python's basic types (`int`,
   `float`, `str`). *(LO1 — see `admin/learning-outcomes.md`)*
3. Perform simple arithmetic and build one f-string. *(LO1)*
4. Recognise the most common beginner error messages and know the single
   most useful fix. *(LO1)*

### Prerequisites — before this notebook you should be able to
- Nothing. Genuinely nothing. If you can open this file, you are ready.

### How to use this notebook (solo study)
This is **not** graded, timed, or checked by anyone. Work top to bottom,
slower than the timing table suggests if you like — there is no rush. Every
section still ends with a **✅ Check yourself** cell, exactly like every
other notebook in this course, so you get the same habit of "run it, see
that it passes, then move on" from day one. This is the **solutions**
version (everything filled in); a **student** version with blanks also
exists if you want to test yourself before Week 01.
''')]

# ==========================================================================
# 0. What is a notebook?
# ==========================================================================
C += [md(r'''
---
## 0. What is a notebook? Cells, the kernel, running a cell

A **Jupyter notebook** (this file) is a sequence of **cells**. Each cell is
one of two kinds:

- A **markdown cell** (like this one) holds formatted text — explanations,
  headings, equations. It is not code; nothing "runs".
- A **code cell** holds Python code. When you **run** it, Python executes
  that code and shows you the result underneath.

To run a cell: click on it, then press **Shift+Enter** (runs it and moves to
the next cell) or **Ctrl+Enter** (runs it and stays put). There is no other
way to make a code cell do anything — writing code and never running the
cell does nothing at all.

Behind the scenes, a running notebook is connected to a **kernel** — a live
Python process that remembers every variable and every cell you have run
**in order**, from the top. This one fact explains almost every confusing
error a beginner hits, and we come back to it explicitly in Section 5.

**What to try:** run the code cell immediately below this one now.
''')]

C += [code(r'''
print("If you can see this line appear below the cell, you just ran your first piece of Python.")
''')]

C += [md(r'''
> **Common errors — Section 0**
> - Nothing happens when you press Shift+Enter — make sure you actually
>   **clicked into** the cell first (it should show a highlighted border).
> - A cell shows `[*]` to its left for a long time — that means it is still
>   running; wait for it to become a number (e.g. `[1]`).
> - You see "No kernel" or "Kernel not found" — pick the course's Python
>   environment as the kernel (top-right of the notebook, e.g.
>   *Python 3 (ai4chem)*); ask a demonstrator if none is offered.
''')]

C += checkpoint(
    "Section 0",
    check=r'''
print("Section 0 OK")
''',
    expected="Prints `Section 0 OK`. There is nothing to get wrong here — "
    "if you can run a cell at all, you have passed this checkpoint.",
    questions=r'''
1. What is the difference between *writing* code in a cell and *running* it?
''',
    answers=r'''
1. Writing code only changes the text in the cell, exactly like typing in a
   text editor; nothing happens chemically or computationally until you
   **run** the cell (Shift+Enter), which sends that code to the kernel to be
   executed.
''',
)

# ==========================================================================
# 1. print() and comments
# ==========================================================================
C += [md(r'''
---
## 1. `print()` and comments

`print(...)` displays whatever is inside the parentheses. Text (a
**string**) must be wrapped in quotes, `"like this"` or `'like this'` —
either works, but be consistent.

A `#` starts a **comment**: everything after it on that line is ignored by
Python entirely. Comments are notes to a human reader (including your future
self), not instructions to the computer.

**What to look for:** two lines of output; the comment line produces no
output at all.
''')]

C += [code(r'''
print("Water boils at 100 degrees Celsius at 1 atmosphere.")
# This next line is a comment -- it does nothing when the cell runs.
print("Comments are for humans, not for Python.")
''')]

C += [md(r'''
> **Common errors — Section 1**
> - `SyntaxError` pointing at a quote mark — every opening quote needs a
>   matching closing quote of the **same** kind (`"..."` or `'...'`, not
>   `"...'`).
> - Forgetting the parentheses: `print "hello"` is not valid Python (this is
>   Python 3; some very old Python 2 tutorials show it without parentheses —
>   ignore those).
''')]

C += checkpoint(
    "Section 1",
    check=r'''
assert True   # Section 1 has no checkable state -- just confirm the cells above ran
print("Section 1 OK")
''',
    expected="Prints `Section 1 OK`.",
    questions=r'''
1. Why does the comment line in the code cell above produce no visible
   output?
''',
    answers=r'''
1. Python ignores everything from a `#` to the end of that line — a comment
   is never executed, so `print` is never actually called for that text; it
   only ever appears as text in the notebook itself, for a human reading the
   code.
''',
)

# ==========================================================================
# 2. Variables and types
# ==========================================================================
C += [md(r'''
---
## 2. Variables and types

A **variable** is a name you attach to a value with `=`. Once assigned, you
can use the name instead of retyping the value.

Python's three most common basic types:

| Type | Example | Meaning |
|------|---------|---------|
| `str` | `"ethanol"` | text |
| `int` | `46` | whole number |
| `float` | `46.07` | number with a decimal point |

`type(x)` tells you what type a variable currently holds.

**What to look for:** `compound_name` prints as text; `molar_mass` prints as
a number; `type(molar_mass)` reports `<class 'float'>`.
''')]

C += [code(r'''
compound_name = "ethanol"      # a string: text, in quotes
molar_mass = 46.07             # a float: a number with a decimal point
formula_weight_is_whole = 46   # an int: a whole number, no decimal point

print(compound_name)
print(molar_mass)
print(type(molar_mass))
print(type(formula_weight_is_whole))
''')]

C += [md(r'''
> **Common errors — Section 2**
> - `NameError: name '...' is not defined` — you used a variable before
>   running the cell that creates it (or you misspelled its name; Python is
>   case-sensitive, so `Molar_Mass` and `molar_mass` are different names).
> - Mixing up `=` (assignment: "store this value") with `==` (comparison:
>   "are these equal?", which you will meet properly in Week 01) — using `=`
>   where you meant `==` is one of the most common beginner slips in *any*
>   language.
''')]

C += checkpoint(
    "Section 2",
    check=r'''
assert compound_name == "ethanol"
assert isinstance(molar_mass, float)
assert isinstance(formula_weight_is_whole, int)
print("Section 2 OK")
''',
    expected="Prints `Section 2 OK`. `compound_name` is `\"ethanol\"`; "
    "`molar_mass` is a `float`; `formula_weight_is_whole` is an `int`.",
    questions=r'''
1. `46` and `46.0` look almost the same to a human. Why does Python treat
   them as different types?
''',
    answers=r'''
1. Python stores whole numbers (`int`) and decimal numbers (`float`)
   differently in memory, and some operations behave slightly differently
   depending on which type you have (Week 01B revisits this: `3 / 2` always
   gives a `float`, `1.5`, even though both `3` and `2` are `int`) — keeping
   the types distinct lets Python (and you) reason precisely about what a
   calculation will produce.
''',
)

# ==========================================================================
# 3. Arithmetic
# ==========================================================================
C += [md(r'''
---
## 3. Arithmetic

Python supports the arithmetic you already know: `+`, `-`, `*` (multiply),
`/` (divide), `**` (power). Normal order-of-operations rules apply
(multiplication/division before addition/subtraction; parentheses first).

Chemistry anchor: density $\rho = m/V$.

**What to look for:** `density` comes out around `0.79` g/mL — a
chemically sensible value for a light organic liquid like ethanol.
''')]

C += [code(r'''
mass_g = 39.5          # grams
volume_mL = 50.0       # millilitres

density = mass_g / volume_mL
print("density:", density, "g/mL")

# order of operations: multiplication happens before addition
result = 2 + 3 * 4
print("2 + 3 * 4 =", result)   # NOT (2+3)*4
''')]

C += [md(r'''
> **Common errors — Section 3**
> - Dividing by a variable that turns out to be `0` raises
>   `ZeroDivisionError` — always sanity-check a denominator (here, `volume_mL`
>   should never legitimately be zero for a real sample).
> - Forgetting Python's order of operations matches ordinary maths — if you
>   want addition to happen *first*, you need explicit parentheses:
>   `(2 + 3) * 4`, not `2 + 3 * 4`.
''')]

C += checkpoint(
    "Section 3",
    check=r'''
assert abs(density - 0.79) < 0.01
assert result == 14
print("Section 3 OK")
''',
    expected="Prints `Section 3 OK`. `density` is about `0.79` g/mL; "
    "`2 + 3 * 4` is `14`, not `20`.",
    questions=r'''
1. Why is `2 + 3 * 4` equal to `14` rather than `20`?
''',
    answers=r'''
1. Python follows the standard mathematical order of operations
   ("PEMDAS"/"BODMAS"): multiplication happens before addition, so
   `3 * 4 = 12` is computed first, then `2 + 12 = 14`. Getting `20` would
   require `(2 + 3) * 4`, with explicit parentheses forcing the addition
   first.
''',
)

# ==========================================================================
# 4. f-strings
# ==========================================================================
C += [md(r'''
---
## 4. f-strings

An **f-string** lets you drop a variable's value directly into a piece of
text: put `f` right before the opening quote, then wrap any variable (or
calculation) in `{curly braces}` inside the string. This is the tidiest way
to build a readable, lab-notebook-style output line, and you will see it
used everywhere from Week 01 onward.

**What to look for:** one neatly formatted sentence, with `compound_name` and
`density` inserted into the right places; `:.2f` rounds a float to 2 decimal
places for a cleaner-looking number.
''')]

C += [code(r'''
print(f"The density of {compound_name} was measured as {density:.2f} g/mL.")
''')]

C += [md(r'''
> **Common errors — Section 4**
> - Forgetting the `f` before the opening quote — without it, Python prints
>   the literal text `{compound_name}` instead of substituting the value.
> - Mismatched braces — every `{` inside an f-string needs a matching `}`.
''')]

C += checkpoint(
    "Section 4",
    check=r'''
message = f"The density of {compound_name} was measured as {density:.2f} g/mL."
assert "ethanol" in message
assert "0.79" in message
print("Section 4 OK")
''',
    expected="Prints `Section 4 OK`. The sentence contains both the "
    "compound name and the rounded density.",
    questions=r'''
1. What would `print("The density of {compound_name}...")` (no `f`) actually
   print, and why?
''',
    answers=r'''
1. It would print the literal characters `{compound_name}`, braces and all —
   without the `f` prefix, Python treats the string as plain text with no
   special meaning attached to curly braces, so nothing is substituted.
''',
)

# ==========================================================================
# 5. Beginner error messages
# ==========================================================================
C += [md(r'''
---
## 5. Beginner error messages — and the one fix that solves most of them

Almost every confusing error a beginner hits traces back to one of these:

| You see | It usually means |
|---|---|
| `NameError: name 'x' is not defined` | You have not **run** the cell that creates `x` yet, or you have run cells out of order, or you misspelled the name. |
| `SyntaxError` | A typo in the code's structure — a missing colon, an unmatched quote or bracket, `=` where `==` was meant. |
| `IndentationError` | Lines that should line up (e.g. inside a loop, from Week 01 onward) do not — Python uses indentation to mean "this belongs together", unlike most other languages. |
| `TypeError` | You tried to combine two things Python cannot combine that way (e.g. `"3" + 3` — text and a number cannot be added directly). |

**The single most useful habit**: if a notebook is behaving strangely and
you cannot see why, use the menu **Kernel → Restart Kernel and Run All
Cells**. This clears every variable and runs the whole notebook from the
top, in order — it fixes the single most common beginner problem (having
run cells out of order, so the kernel's memory no longer matches what you
see on the page) more often than any other single action.

**What to look for:** the code cell below is written to **fail on purpose**
so you see a real `NameError` at least once before Week 01, in a safe,
low-stakes place.
''')]

C += [code(r'''
try:
    print(this_variable_was_never_created)
except NameError as error:
    print("Caught the error on purpose:", error)
    print("This is exactly what a NameError looks like -- now you'll recognise it.")
''')]

C += [md(r'''
> **Common errors — Section 5**
> - Seeing a `NameError` and immediately assuming your code is "broken" —
>   first ask: *did I run every cell above this one, in order, in this
>   kernel session?* This single question resolves the majority of beginner
>   `NameError`s.
> - Restarting the kernel and then running only the cell you care about
>   (instead of *all* the cells above it) — a fresh kernel remembers nothing
>   until you re-run the cells that build up the state you need.
''')]

C += checkpoint(
    "Section 5",
    check=r'''
print("Section 5 OK")
''',
    expected="Prints `Section 5 OK`, right after a deliberately-caught "
    "`NameError` message above it.",
    questions=r'''
1. Why does restarting the kernel and running every cell from the top fix
   most "it was working a minute ago" problems?
''',
    answers=r'''
1. The kernel's memory only ever reflects the cells you have actually run,
   in the order you ran them — if you go back and change an early cell, or
   run cells out of sequence, the kernel's state can silently drift out of
   sync with what the notebook *looks* like on the page. Restarting and
   running everything from the top guarantees the kernel's memory exactly
   matches the notebook as written, top to bottom.
''',
)

# ==========================================================================
# 6. Exercises
# ==========================================================================
C += [md(r'''
---
## 6. Exercises

Three short, easy exercises. No mini-challenge this week — this primer is
meant to build confidence, not test it.
''')]

C += exercise(
    prompt=r'''
### Exercise 1 — your own variables *(easy, ~5 min)*

Create a variable `my_compound` (a string, any molecule name you like) and
`my_molar_mass` (a float, its approximate molar mass in g/mol — a rough
guess is fine). Print both.

<details><summary>Show hint</summary>

Exactly the pattern from Section 2: `my_compound = "..."` and
`my_molar_mass = ...` (a number with a decimal point).
</details>
''',
    solution=r'''
my_compound = "caffeine"
my_molar_mass = 194.19

print(my_compound)
print(my_molar_mass)
''',
    scaffold=r'''
my_compound = ...  # YOUR CODE HERE: a molecule name, as a string
my_molar_mass = ...  # YOUR CODE HERE: its approximate molar mass, as a float

print(my_compound)
print(my_molar_mass)
''',
    check=r'''
assert isinstance(my_compound, str) and len(my_compound) > 0
assert isinstance(my_molar_mass, float)
print("Exercise 1 OK")
''',
)

C += exercise(
    prompt=r'''
### Exercise 2 — moles from mass *(easy, ~5 min)*

Write `moles = mass_sample_g / my_molar_mass`, where `mass_sample_g = 10.0`.
This is $n = m/M$ — one of the most common calculations in a chemistry lab.

<details><summary>Show hint</summary>

One line, using the `/` operator from Section 3, and the `my_molar_mass` you
just defined in Exercise 1.
</details>
''',
    solution=r'''
mass_sample_g = 10.0
moles = mass_sample_g / my_molar_mass
print(f"{mass_sample_g} g of {my_compound} is {moles:.4f} mol")
''',
    scaffold=r'''
mass_sample_g = 10.0
moles = ...  # YOUR CODE HERE
print(f"{mass_sample_g} g of {my_compound} is {moles:.4f} mol")
''',
    check=r'''
assert abs(moles - mass_sample_g / my_molar_mass) < 1e-9
assert moles > 0
print("Exercise 2 OK")
''',
)

C += exercise(
    prompt=r'''
### Exercise 3 — fix the deliberately broken cell *(easy, ~5 min)*

The scaffold below has **two** mistakes: a missing closing quote, and an `f`
missing from an f-string. Fix both so it runs and prints a sensible
sentence.

<details><summary>Show hint</summary>

Compare carefully against Section 4's working f-string example. Both bugs
are single-character fixes.
</details>
''',
    solution=r'''
fixed_message = f"{my_compound} has a molar mass of {my_molar_mass} g/mol."
print(fixed_message)
''',
    scaffold=r'''
# YOUR CODE HERE: fix the two bugs in the line below, then run this cell
# fixed_message = "{my_compound} has a molar mass of {my_molar_mass} g/mol.
# print(fixed_message)
fixed_message = ...  # YOUR CODE HERE
print(fixed_message)
''',
    check=r'''
assert my_compound in fixed_message
assert str(my_molar_mass) in fixed_message
print("Exercise 3 OK")
''',
)

# ==========================================================================
C += [md(r'''
---
## Summary — what you learned

- Notebooks are made of **cells** (markdown or code); code only runs when
  you run the cell; the **kernel** remembers everything you have run, in
  order.
- `print()`, comments (`#`), **variables**, and the three basic types
  (`str`, `int`, `float`).
- Simple **arithmetic** and Python's order of operations.
- **f-strings** for building readable output.
- The most common beginner **error messages**, and the single most useful
  fix (restart the kernel, run everything from the top, in order).

You are ready for **Week 01, Session A** — the course's real Python
bootcamp, which starts from variables and lists and moves quickly into
reading real chemical data files. Nothing here is graded or checked again;
it exists purely so that Week 01 feels like a second exposure, not a first
one.
''')]

C += [md(r'''
## Further reading (optional, if you want more practice before Week 01)

- pythoninchemistry.org, *Intro to Python for Chemistry* (a full open
  textbook, no prior programming assumed):
  <https://pythoninchemistry.org/intro_python_chemists/intro.html>
- MolSSI *Python Scripting for Computational Molecular Science*, lesson 01
  (installing Python, running your first script):
  <https://education.molssi.org/python_scripting_cms/01-introduction/index.html>

Both are listed as pre-course resources in this project's own
`sources/links.md`.
''')]

C += [md(r'''
## Attribution

This primer is original material written for *AI for Chemistry*, at the
instructor's request, as a free-time on-ramp for students with no prior
programming background. It does not adapt content from any single external
source; the "further reading" resources above are pointed to, not
reproduced.
''')]

build(__file__, "week00_python-primer", C)
