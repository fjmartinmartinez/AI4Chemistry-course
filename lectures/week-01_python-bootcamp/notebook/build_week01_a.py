"""Build week 01 session A notebooks (Python bootcamp I).

    python lectures/week-01_python-bootcamp/notebook/build_week01_a.py
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "_build"))
from nbbuild import build, checkpoint, code, exercise, md  # noqa: E402

C = []

# ==========================================================================
# Header
# ==========================================================================
C += [md(r'''
# AI for Chemistry — Week 01, Session A
## Python bootcamp I: data, loops, functions, and reading chemical files

**Course:** AI for Chemistry (10-week module) · **Session:** 01A (interleaved
lecture + lab, 2.5 h) · **Notebook runtime:** < 30 s on a laptop, no GPU.

### Suggested timing (solo study, ~110 min)

| Section | Topic | Time |
|--------:|-------|-----:|
| 0 | Setup and how to use this notebook | 5 min |
| 1 | Variables, numbers, units | 15 min |
| 2 | Lists, indexing, loops, conditionals | 20 min |
| 3 | Functions | 25 min |
| 4 | Reading chemical data files | 20 min |
| 5 | Exercises (4) | 25 min |

### Learning objectives
By the end of this notebook you can:

1. Use variables and numeric types for a thermochemical calculation and convert
   between energy units. *(module outcome LO1)*
2. Build, index, slice and iterate over lists; filter chemical data with `if`. *(LO1)*
3. Write a documented function with arguments, defaults and a return value, and
   call one function from another. *(LO1)*
4. Read numbers out of `.xyz` and `.csv` files using only the Python standard
   library. *(LO1)*

### Prerequisites — before this notebook you should be able to
- Open this file in Jupyter and run a cell with **Shift+Enter**.
- Do school-level algebra (powers, logarithms, exponentials).
- *No previous programming is assumed.*

### How to use this notebook (solo study)
Work top to bottom. **Read the markdown cell before each code cell first** — it
says what the code does, why it matters chemically, and what to look for in the
output. Run the code cell, check the output against that description, then move
on. Every section ends with a **✅ Check yourself** cell: run it: if an
`assert` fails, re-read that section before continuing. The exercises at the end
have hints hidden in *"Show hint"* blocks. This is the **solutions** notebook —
the exercise answers and the conceptual-question answers are filled in. Use the
`_student` version if you want to try them cold first.
''')]

# ==========================================================================
# 0. Setup
# ==========================================================================
C += [md(r'''
---
## 0. Setup

The only thing this notebook needs from outside is three geometry files and one
table of alkane properties, all in `sources/datasets/`. The cell below locates
that folder relative to the notebook and checks the files are present. It uses
`pathlib.Path`, the modern way to handle file paths in Python — it works
identically on Windows, macOS and Linux.

**What to look for:** a line for each of the four data files ending in `OK`. If
you see `MISSING`, run `python lectures/_build/make_datasets_wk01_02.py` from the
project root first.
''')]

C += [code(r'''
import sys                         # interpreter info
import math                        # square roots, pi, etc. (standard library)
from pathlib import Path           # object-oriented filesystem paths

# The notebook lives in lectures/week-01_python-bootcamp/notebook/, so the
# project root is three folders up. ".resolve()" makes the path absolute.
ROOT = Path.cwd()
while not (ROOT / "sources").is_dir() and ROOT != ROOT.parent:
    ROOT = ROOT.parent             # walk up until we find the "sources" folder
DATA = ROOT / "sources" / "datasets"

for name in ["water.xyz", "ethanol.xyz", "benzene.xyz", "alkanes.csv"]:
    status = "OK" if (DATA / name).is_file() else "MISSING"
    print(f"{name:16s} {status}")

print("\nPython", sys.version.split()[0])
''')]

C += [md(r'''
> **Common errors — Setup**
> - `NameError` for `math`, `Path` or `sys` further down the notebook — you
>   skipped this cell. Every notebook must be run top to bottom.
> - `MISSING` for every file — your working directory is not inside the project.
>   In Jupyter, `Path.cwd()` is the folder you launched `jupyter lab` from.
> - On Windows a path printed with `\` is normal; `pathlib` handles it.
''')]

# ==========================================================================
# 1. Variables, numbers, units
# ==========================================================================
C += [md(r'''
---
## 1. Variables, numbers and units

A **variable** is a name bound to a value with `=`. Python has three numeric
"flavours" you will meet constantly:

| Type | Example | Meaning |
|------|---------|---------|
| `int` | `298` | whole number |
| `float` | `-541.5` | real number (has a decimal point) |
| `str` | `"ethanol"` | text ("string of characters") |

Chemistry anchor: the **Gibbs free energy change** of a reaction,

$$\Delta G = \Delta H - T\,\Delta S$$

tells us whether a reaction is thermodynamically favourable ($\Delta G < 0$).
The next cell computes $\Delta G$ at 298 K for a reaction with
$\Delta H = -541.5\ \text{kJ mol}^{-1}$ and
$\Delta S = 0.0104\ \text{kJ mol}^{-1}\,\text{K}^{-1}$.

**What to look for:** a negative number near `-544.6`, i.e. the reaction is
spontaneous at room temperature, and that `type(delta_G)` is `float`.
''')]

C += [code(r'''
delta_H = -541.5      # kJ / mol   (enthalpy change)
delta_S = 0.0104      # kJ / (mol K)  (entropy change; note: kJ, not J)
temp = 298            # K          (absolute temperature)

delta_G = delta_H - temp * delta_S   # kJ / mol
print("delta_G =", round(delta_G, 1), "kJ/mol")
print("spontaneous at 298 K?", delta_G < 0)
print("type of delta_G:", type(delta_G))
''')]

C += [md(r'''
Energies in chemistry are quoted in several units. To convert
**kilocalories to kilojoules** we multiply by the exact definition
$1\ \text{cal} = 4.184\ \text{J}$, so $1\ \text{kcal} = 4.184\ \text{kJ}$.
The number `4.184` is a defined constant, not a magic number.

**What to look for:** `-13.4 kcal/mol` becomes about `-56.1 kJ/mol`.
''')]

C += [code(r'''
KCAL_TO_KJ = 4.184                       # exact: 1 cal == 4.184 J (definition)

energy_kcal = -13.4                       # kcal / mol
energy_kj = energy_kcal * KCAL_TO_KJ     # kJ / mol
print(f"{energy_kcal} kcal/mol = {energy_kj:.1f} kJ/mol")
''')]

C += [md(r'''
> **Common errors — Section 1**
> - `TypeError: can't multiply sequence by non-int` — you multiplied a string by
>   a float. Check you did not put quotes around a number.
> - Integer vs float: in Python 3, `3 / 2` is `1.5` (always a float). `3 // 2` is
>   `1` (floor division). Mixing them up changes your answer silently.
> - `SyntaxError` pointing at `=` — you wrote `=` (assignment) where you meant
>   `==` (equality test), or vice versa.
''')]

C += checkpoint(
    "Section 1",
    check=r'''
assert isinstance(delta_G, float)
assert abs(delta_G - (-544.6)) < 0.1, "delta_G should be about -544.6 kJ/mol"
assert abs(-13.4 * KCAL_TO_KJ - (-56.07)) < 0.01
print("Section 1 OK")
''',
    expected="Running this cell prints `Section 1 OK` and raises no "
    "`AssertionError`. `delta_G` is about **-544.6 kJ/mol**.",
    questions=r'''
1. At what temperature would this reaction stop being spontaneous
   (assuming $\Delta H$, $\Delta S$ are temperature-independent)?
2. Why do we store `4.184` in a named constant instead of typing it inline
   every time?
''',
    answers=r'''
1. $\Delta G = 0$ when $T = \Delta H / \Delta S = -541.5 / -0.0104 \approx
   52\,000\ \text{K}$ — far above any real condition, so this reaction is
   spontaneous at all attainable temperatures. (For a reaction with
   $\Delta H<0$ and $\Delta S<0$ the crossover is at $T=\Delta H/\Delta S$;
   here both are negative so the ratio is large and positive.)
2. A named constant documents *what the number means*, guarantees you use the
   same value everywhere, and lets you change it in one place. Un-named
   numbers scattered through code ("magic numbers") are a classic source of
   bugs.
''',
)

# ==========================================================================
# 2. Lists, indexing, loops, conditionals
# ==========================================================================
C += [md(r'''
---
## 2. Lists, indexing, loops and conditionals

A **list** is an ordered collection written in square brackets. Here is a
(made-up) reaction coordinate: the relative energy of each step of a mechanism,
in kcal/mol.

- `energies[0]` is the **first** element (Python counts from 0).
- `energies[-1]` is the **last**.
- `energies[1:3]` is a **slice**: elements 1 and 2 (the end index is excluded).
- `len(energies)` is how many elements there are.

**What to look for:** the first energy is `0.0` (reactants, our reference), the
last is `-18.7` (products, downhill overall), and there are `5` steps.
''')]

C += [code(r'''
# Relative energies along a hypothetical 5-point reaction profile (kcal/mol).
energies = [0.0, 12.4, -3.1, 7.8, -18.7]

print("first step :", energies[0])
print("last step  :", energies[-1])
print("middle 1:3 :", energies[1:3])
print("how many   :", len(energies))
''')]

C += [md(r'''
A **`for` loop** repeats a block of code once per element. Below we convert the
whole profile to kJ/mol, building up a new list with `.append()`. The indented
lines are the loop body; indentation (4 spaces) is how Python groups code.

**What to look for:** every number scaled by 4.184, e.g. `12.4 → 51.9`.
''')]

C += [code(r'''
energies_kj = []                          # start with an empty list
for e in energies:                        # e takes each value in turn
    energies_kj.append(e * KCAL_TO_KJ)    # add the converted value

print([round(x, 1) for x in energies_kj])
''')]

C += [md(r'''
An **`if`** statement runs code only when a condition is true. Combined with a
loop, this lets us *filter* data. Which steps release energy (are exergonic,
energy $< 0$)?

**What to look for:** two values, `-3.1` and `-18.7`.
''')]

C += [code(r'''
exergonic = []
for e in energies:
    if e < 0:                # condition: negative relative energy
        exergonic.append(e)

print("exergonic steps:", exergonic)
print("count:", len(exergonic))
''')]

C += [md(r'''
> **Common errors — Section 2**
> - `IndexError: list index out of range` — a list of 5 items has valid indices
>   `0..4`; `energies[5]` fails. Use `energies[-1]` for the last.
> - `IndentationError` — the lines inside a loop or `if` must all be indented by
>   the same amount. Use spaces, not tabs (Jupyter does this for you).
> - Forgetting the colon `:` at the end of a `for`/`if` line.
> - Modifying a list while looping over it — build a *new* list instead, as above.
''')]

C += checkpoint(
    "Section 2",
    check=r'''
assert len(energies) == 5
assert energies[-1] == -18.7
assert energies_kj[1] == 12.4 * 4.184
assert exergonic == [-3.1, -18.7]
print("Section 2 OK")
''',
    expected="Prints `Section 2 OK`. The exergonic-steps list is "
    "`[-3.1, -18.7]`.",
    questions=r'''
1. `energies[1:3]` returned two items, not three. Why?
2. What would `energies[:]` give you, and when is that useful?
''',
    answers=r'''
1. A slice `a:b` runs from index `a` up to *but not including* `b`, so `1:3`
   is indices 1 and 2 — two items. This "half-open" convention means the
   length of a slice is simply `b - a`.
2. `energies[:]` is a *copy* of the whole list. Useful when you want to modify
   a list without changing the original (assigning `b = a` makes `b` another
   name for the *same* list).
''',
)

# ==========================================================================
# 3. Functions
# ==========================================================================
C += [md(r'''
---
## 3. Functions

A **function** packages a computation under a name so you can reuse it. Syntax:

```
def name(arg1, arg2):
    "docstring: one line saying what the function does"
    ...              # body (indented)
    return result
```

Chemistry anchor: the **distance between two atoms** given their Cartesian
coordinates $(x, y, z)$ in ångström,

$$d_{12} = \sqrt{(x_1-x_2)^2 + (y_1-y_2)^2 + (z_1-z_2)^2}.$$

**What to look for:** the O–H distance in water comes out at about `0.97 Å`,
a chemically sensible bond length.
''')]

C += [code(r'''
def calculate_distance(a, b):
    """Euclidean distance between two 3D points a=(x,y,z) and b=(x,y,z).

    Units follow the inputs (ångström in this notebook).
    """
    sq = (a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2
    return math.sqrt(sq)


# quick sanity check with hand coordinates (a water O-H, angstrom)
o = (0.000, 0.000, 0.119)
h = (0.000, 0.763, -0.477)
print("O-H distance:", round(calculate_distance(o, h), 3), "angstrom")
''')]

C += [md(r'''
Functions can have **default arguments** — values used when the caller does not
supply them. `bond_check` decides whether a distance corresponds to a bond: a
plausible covalent bond is longer than 0 and no more than about 1.5 Å for the
light elements in our examples.

**What to look for:** `1.4` is a bond (`True`), `1.6` is not (`False`), but
`bond_check(1.6, maximum_length=1.8)` is `True` because we widened the window.
''')]

C += [code(r'''
def bond_check(distance, minimum_length=0.4, maximum_length=1.5):
    """True if `distance` (angstrom) is within a covalent-bond window."""
    return minimum_length < distance <= maximum_length


print(bond_check(1.4))
print(bond_check(1.6))
print(bond_check(1.6, maximum_length=1.8))
''')]

C += [md(r'''
Now **compose** the two functions. `read_xyz` parses an `.xyz` file (line 1 =
atom count, line 2 = comment, then `symbol x y z` per atom). `print_bonds` loops
over every unique atom pair, and uses `calculate_distance` + `bond_check` to
report the bonds.

**What to look for:** water gives exactly two O–H bonds at `0.969 Å`; the
H···H distance (`1.527 Å`) is correctly rejected.
''')]

C += [code(r'''
def read_xyz(path):
    """Read an .xyz file -> (symbols list, coords list of (x,y,z) tuples)."""
    lines = Path(path).read_text().splitlines()
    n_atoms = int(lines[0])                 # first line: number of atoms
    symbols, coords = [], []
    for line in lines[2:2 + n_atoms]:       # skip the comment line (line 2)
        parts = line.split()               # split on whitespace
        symbols.append(parts[0])
        coords.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return symbols, coords


def print_bonds(symbols, coords):
    """Print every atom pair whose separation passes bond_check."""
    n = len(symbols)
    for i in range(n):
        for j in range(i + 1, n):          # j > i => each pair once
            d = calculate_distance(coords[i], coords[j])
            if bond_check(d):
                print(f"{symbols[i]}-{symbols[j]} : {d:.3f} angstrom")


sym_w, xyz_w = read_xyz(DATA / "water.xyz")
print("water has", len(sym_w), "atoms:", sym_w)
print_bonds(sym_w, xyz_w)
''')]

C += [md(r'''
> **Common errors — Section 3**
> - `IndexError` in `read_xyz` — a blank trailing line or wrong atom count in
>   the file. Print `lines[:3]` to inspect.
> - `ValueError: could not convert string to float` — you tried to `float()` the
>   atom symbol; check you are indexing `parts[1:4]`, not `parts[0]`.
> - Returning nothing: forgetting `return` makes the function yield `None`.
> - Calling `calculate_distance` before the `def` cell has been run.
''')]

C += checkpoint(
    "Section 3",
    check=r'''
bonds_w = []
for i in range(len(sym_w)):
    for j in range(i + 1, len(sym_w)):
        d = calculate_distance(xyz_w[i], xyz_w[j])
        if bond_check(d):
            bonds_w.append((sym_w[i], sym_w[j], round(d, 3)))

assert abs(calculate_distance(o, h) - 0.958) < 0.02   # ~ equilibrium O-H
assert bond_check(1.4) is True and bond_check(1.6) is False
assert len(bonds_w) == 2, "water should have exactly two O-H bonds"
assert all(b[2] == 0.969 for b in bonds_w)
print("Section 3 OK")
''',
    expected="Prints `Section 3 OK`. `bonds_w` holds two `('O','H',0.969)` "
    "tuples.",
    questions=r'''
1. In `print_bonds`, why does the inner loop start at `i + 1` rather than `0`?
2. `bond_check` uses `minimum_length=0.4`. What chemical/physical situation is
   that guarding against?
''',
    answers=r'''
1. Starting `j` at `i + 1` visits each unordered pair once (and never pairs an
   atom with itself). Starting at `0` would report every bond twice and also
   test `atom i` against `atom i` (distance 0).
2. Two atoms can never sit 0.1 Å apart. A near-zero distance usually means
   duplicated coordinates or a file-parsing error; the lower bound stops such
   artefacts being reported as bonds.
''',
)

# ==========================================================================
# 4. Reading chemical data files
# ==========================================================================
C += [md(r'''
---
## 4. Reading chemical data files

Most chemical data arrives as text: `.xyz` (done above), or **CSV**
(comma-separated values) from instruments, databases and spreadsheets.
`sources/datasets/alkanes.csv` lists the straight-chain alkanes C1–C10 with
their boiling and melting points (°C).

> The values are standard reference physical constants (CRC-Handbook style) and
> are marked `TODO(verify)` in the dataset generator pending a citation check.

We read it with the standard-library `csv` module — no pandas yet (that is
Session 02A). Each row becomes a dictionary keyed by the header names.

**What to look for:** 10 rows; methane boils at `-161.5 °C`, decane at
`174.1 °C`.
''')]

C += [code(r'''
import csv

alkanes = []
with open(DATA / "alkanes.csv", newline="") as fh:
    reader = csv.DictReader(fh)             # uses the header row as keys
    for row in reader:
        alkanes.append({
            "name": row["name"],
            "n_carbons": int(row["n_carbons"]),          # text -> int
            "bp_C": float(row["boiling_point_C"]),        # text -> float
        })

print(len(alkanes), "alkanes loaded")
print(alkanes[0])
print(alkanes[-1])
''')]

C += [md(r'''
A classic trend: each extra –CH$_2$– group raises the boiling point, but by a
*decreasing* amount. Let us quantify the average increment between neighbours,

$$\overline{\Delta T_b} = \frac{1}{N-1}\sum_{k=2}^{N}\bigl(T_b[k] - T_b[k-1]\bigr).$$

**What to look for:** an average step of roughly `37 °C` per carbon across
C1→C10, and the *first* step (methane→ethane) being much bigger than the
*last* (nonane→decane).
''')]

C += [code(r'''
bps = [a["bp_C"] for a in alkanes]                 # list of boiling points

steps = []
for k in range(1, len(bps)):
    steps.append(bps[k] - bps[k - 1])             # consecutive differences

mean_step = sum(steps) / len(steps)
print(f"mean boiling-point increment: {mean_step:.1f} C per CH2")
print(f"first step  (C1->C2): {steps[0]:.1f} C")
print(f"last step   (C9->C10): {steps[-1]:.1f} C")
''')]

C += [md(r'''
> **Common errors — Section 4**
> - `KeyError: 'boiling_point_C'` — the header name must match the file exactly,
>   spaces and all. Print `reader.fieldnames` to see them.
> - `ValueError` on `int(row[...])` — an empty cell or stray text. Real datasets
>   have missing values; we handle those properly in Week 02.
> - Forgetting `newline=""` in `open(...)` can corrupt CSV parsing on Windows.
> - The `with` block closes the file automatically; do not `open` without it.
''')]

C += checkpoint(
    "Section 4",
    check=r'''
assert len(alkanes) == 10
assert alkanes[0]["name"] == "methane" and alkanes[0]["bp_C"] == -161.5
assert alkanes[-1]["bp_C"] == 174.1
assert len(steps) == 9
assert abs(mean_step - 37.29) < 0.1
assert steps[0] > steps[-1], "increment should shrink down the series"
print("Section 4 OK")
''',
    expected="Prints `Section 4 OK`. Mean increment ≈ **37.3 °C** per CH₂; "
    "the first step (72.9 °C) is far larger than the last (23.3 °C).",
    questions=r'''
1. Physically, why does the boiling point rise as the chain gets longer?
2. Why does the *increment* shrink further down the series?
''',
    answers=r'''
1. Longer chains have more electrons and a larger molecular surface, so the
   instantaneous-dipole (London dispersion) forces between molecules are
   stronger and more thermal energy is needed to separate them.
2. Each added CH$_2$ is a smaller *fractional* increase in polarisable surface
   area as the molecule grows, and chain flexibility/coiling reduces the
   effective contact area, so the marginal effect on boiling point tapers off.
''',
)

# ==========================================================================
# 5. Exercises
# ==========================================================================
C += [md(r'''
---
## 5. Exercises

Four exercises, increasing in difficulty. Each has a hidden hint. In the
`_student` notebook the bodies are blank (`# YOUR CODE HERE`); the `assert`
cells are the same in both versions and must pass unchanged.
''')]

C += exercise(
    prompt=r'''
### Exercise 1 — unit conversion over a dataset *(easy, ~10 min)*

Write `celsius_to_kelvin(temp_c)` returning the absolute temperature
($T/\text{K} = t/^{\circ}\text{C} + 273.15$), then build `bps_kelvin`, the list
of alkane boiling points in kelvin, in the same C1→C10 order as `bps`.

<details><summary>Show hint</summary>

One-line function: `return temp_c + 273.15`. Then a list comprehension:
`[celsius_to_kelvin(t) for t in bps]`.
</details>
''',
    solution=r'''
def celsius_to_kelvin(temp_c):
    """Convert a Celsius temperature to kelvin."""
    return temp_c + 273.15

bps_kelvin = [celsius_to_kelvin(t) for t in bps]
print([round(t, 2) for t in bps_kelvin])
''',
    scaffold=r'''
def celsius_to_kelvin(temp_c):
    """Convert a Celsius temperature to kelvin."""
    # YOUR CODE HERE
    ...

bps_kelvin = []  # YOUR CODE HERE: list of boiling points in kelvin, C1 -> C10
''',
    check=r'''
assert abs(celsius_to_kelvin(0) - 273.15) < 1e-9
assert abs(celsius_to_kelvin(-161.5) - 111.65) < 1e-9
assert len(bps_kelvin) == 10
assert abs(bps_kelvin[0] - 111.65) < 1e-9
assert abs(bps_kelvin[-1] - 447.25) < 1e-9
print("Exercise 1 OK")
''',
)

C += exercise(
    prompt=r'''
### Exercise 2 — count bonds below a cutoff *(medium, ~15 min)*

Write `count_bonds_below(symbols, coords, cutoff)` that returns the **number**
of unique atom pairs closer than `cutoff` ångström. Use it on benzene with
`cutoff = 1.5`.

<details><summary>Show hint</summary>

Re-use the double loop from `print_bonds` (`for i ... for j in range(i+1, n)`),
call `calculate_distance`, and increment a counter when `d < cutoff`.
Benzene has 6 C–C and 6 C–H bonds.
</details>
''',
    solution=r'''
def count_bonds_below(symbols, coords, cutoff):
    """Count unique atom pairs separated by less than `cutoff` angstrom."""
    n = len(symbols)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if calculate_distance(coords[i], coords[j]) < cutoff:
                count += 1
    return count

sym_b, xyz_b = read_xyz(DATA / "benzene.xyz")
n_bonds_benzene = count_bonds_below(sym_b, xyz_b, 1.5)
print("benzene bonds below 1.5 A:", n_bonds_benzene)
''',
    scaffold=r'''
def count_bonds_below(symbols, coords, cutoff):
    """Count unique atom pairs separated by less than `cutoff` angstrom."""
    # YOUR CODE HERE
    ...

sym_b, xyz_b = read_xyz(DATA / "benzene.xyz")
n_bonds_benzene = ...  # YOUR CODE HERE
''',
    check=r'''
assert count_bonds_below(sym_w, xyz_w, 1.5) == 2
assert n_bonds_benzene == 12, "6 C-C + 6 C-H"
assert count_bonds_below(sym_b, xyz_b, 1.2) == 6, "only the 6 C-H bonds"
print("Exercise 2 OK")
''',
)

C += exercise(
    prompt=r'''
### Exercise 3 — bond inventory of ethanol *(medium, ~20 min)*

Write `bond_inventory(symbols, coords, max_length=1.6)` returning a **dict** that
maps a sorted element-pair string (e.g. `"C-H"`, `"C-O"`, `"H-O"`) to how many
such bonds there are. Count a pair as bonded when
`bond_check(d, maximum_length=max_length)` is `True`. Run it on `ethanol.xyz`.

Why `1.6` and not the `1.5` default? Ethanol's C–C bond is about `1.52 Å`; the
tighter window would miss it. Choosing a cutoff is a real modelling decision.

<details><summary>Show hint</summary>

Build the key with `"-".join(sorted([symbols[i], symbols[j]]))` so `O` then `H`
and `H` then `O` both give `"H-O"`. Use `inv[key] = inv.get(key, 0) + 1` to tally.
Ethanol (CH₃CH₂OH): 5 C–H, 1 C–C, 1 C–O, 1 O–H.
</details>
''',
    solution=r'''
def bond_inventory(symbols, coords, max_length=1.6):
    """Return {'C-H': n, ...} counting bonds by sorted element pair."""
    n = len(symbols)
    inv = {}
    for i in range(n):
        for j in range(i + 1, n):
            d = calculate_distance(coords[i], coords[j])
            if bond_check(d, maximum_length=max_length):
                key = "-".join(sorted([symbols[i], symbols[j]]))
                inv[key] = inv.get(key, 0) + 1
    return inv

sym_e, xyz_e = read_xyz(DATA / "ethanol.xyz")
ethanol_bonds = bond_inventory(sym_e, xyz_e)
print(ethanol_bonds)
''',
    scaffold=r'''
def bond_inventory(symbols, coords, max_length=1.6):
    """Return {'C-H': n, ...} counting bonds by sorted element pair."""
    # YOUR CODE HERE
    ...

sym_e, xyz_e = read_xyz(DATA / "ethanol.xyz")
ethanol_bonds = ...  # YOUR CODE HERE
''',
    check=r'''
assert ethanol_bonds == {"C-C": 1, "C-H": 5, "C-O": 1, "H-O": 1}
assert sum(ethanol_bonds.values()) == 8
# order independence: benzene gives 6 C-C and 6 C-H regardless of atom order
assert bond_inventory(sym_b, xyz_b) == {"C-C": 6, "C-H": 6}
print("Exercise 3 OK")
''',
)

C += exercise(
    prompt=r'''
### Exercise 4 — molar mass from a formula dict *(harder, ~20 min)*

Write `molar_mass(counts, masses)` where `counts` is like `{"C": 2, "H": 6, "O": 1}`
and `masses` is the `ATOMIC_MASS` table below. Return the molar mass in
g mol⁻¹. Then confirm water and ethanol.

<details><summary>Show hint</summary>

Loop over `counts.items()`, look each element up in `masses`, multiply by its
count, and add to a running total. Water = 18.015, ethanol = 46.069 g/mol.
</details>
''',
    solution=r'''
ATOMIC_MASS = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999}

def molar_mass(counts, masses):
    """Molar mass (g/mol) from an element-count dict and a mass table."""
    total = 0.0
    for element, n in counts.items():
        total += masses[element] * n
    return total

print("water  :", round(molar_mass({"H": 2, "O": 1}, ATOMIC_MASS), 3))
print("ethanol:", round(molar_mass({"C": 2, "H": 6, "O": 1}, ATOMIC_MASS), 3))
''',
    scaffold=r'''
ATOMIC_MASS = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999}

def molar_mass(counts, masses):
    """Molar mass (g/mol) from an element-count dict and a mass table."""
    # YOUR CODE HERE
    ...
''',
    check=r'''
assert abs(molar_mass({"H": 2, "O": 1}, ATOMIC_MASS) - 18.015) < 1e-6
assert abs(molar_mass({"C": 2, "H": 6, "O": 1}, ATOMIC_MASS) - 46.069) < 1e-6
assert abs(molar_mass({"C": 6, "H": 6}, ATOMIC_MASS) - 78.114) < 1e-6
print("Exercise 4 OK")
''',
)

# ==========================================================================
# Summary / further reading / attribution
# ==========================================================================
C += [md(r'''
---
## Summary — what you learned

- **Variables and types** (`int`, `float`, `str`); a thermochemical calculation
  ($\Delta G = \Delta H - T\Delta S$) and unit conversion with a named constant.
- **Lists**: indexing (`[0]`, `[-1]`), slicing (`[1:3]`), `len`, `.append`.
- **`for` loops** and **`if`** conditionals to transform and filter data.
- **Functions**: `def`, arguments, **default arguments**, `return`, docstrings,
  and composing small functions into a geometry-analysis pipeline.
- **Reading files** with only the standard library: `.xyz` by hand,
  `.csv` with `csv.DictReader`.

Next session (01B) replaces the hand-written loops with **NumPy arrays** and adds
**matplotlib** plotting.
''')]

C += [md(r'''
## Further reading (course source list only)

- MolSSI *Python Scripting for Computational Molecular Science* — lesson 01
  (introduction) and lesson 02 (file parsing):
  <https://education.molssi.org/python_scripting_cms/01-introduction/index.html>,
  <https://education.molssi.org/python_scripting_cms/02-file_parsing/index.html>
- MolSSI lesson 06 (functions):
  <https://education.molssi.org/python_scripting_cms/06-functions/index.html>
- SciCompforChemists, chapters 0–1 (Python basics):
  <https://weisscharlesj.github.io/SciCompforChemists/notebooks/chapter_00/chap_00_notebook.html>
- Colab alternative (EPFL *AI for Chemistry* 01a, Python crash course):
  <https://colab.research.google.com/github/schwallergroup/ai4chem_course/blob/main/notebooks/01%20-%20Basics/01a_python_crash_course.ipynb>
''')]

C += [md(r'''
## Attribution

This notebook is original teaching material for the *AI for Chemistry* module.
Its structure and worked examples are **adapted and rewritten** from:

- **MolSSI Python Scripting for Computational Molecular Science** — the
  geometry-analysis example (`calculate_distance`, `bond_check`,
  reading `.xyz`). CC-BY 4.0. <https://education.molssi.org/python_scripting_cms/>
- **Scientific Computing for Chemists with Python** (C. J. Weiss) — Python and
  NumPy pedagogy. CC-BY-NC-SA 4.0. <https://github.com/weisscharlesj/SciCompforChemists>
- **EPFL CH-457 *AI for Chemistry*** (Schwaller group) — course framing.
  MIT licence. <https://github.com/schwallergroup/ai4chem_course>

Datasets: `*.xyz` generated with RDKit (ETKDGv3 + MMFF94, fixed seed);
`alkanes.csv` curated standard reference values (`TODO(verify)` against CRC).
No verbatim passages are reproduced from the sources above.
''')]

build(__file__, "week01_a_python-bootcamp", C)
