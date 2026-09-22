# Week 01 — Python bootcamp (data, loops, functions, arrays, plotting)

_Record of the approved design (syllabus v0.1). Not a proposal._

## Learning objectives
By the end of week 01 students can:
1. Use Python variables, numeric types and unit conversions for simple
   thermochemical calculations. [LO1]
2. Build, index, slice and iterate over lists; use `for` loops and `if` to
   transform and filter chemical data. [LO1]
3. Write documented functions with arguments, defaults and return values, and
   compose them (geometry analysis from xyz coordinates). [LO1]
4. Read chemical data from plain-text files (`.xyz`, `.csv`) without a
   dataframe library. [LO1]
5. (Session B) Create and manipulate NumPy arrays; use vectorised maths,
   broadcasting, boolean masks and axis-wise aggregation on spectral and
   equation-of-state data. [LO1]
6. (Session B) Produce labelled matplotlib figures (line, scatter, subplots),
   fit a straight line with `np.polyfit`, and save a figure for reuse. [LO1]

## Prerequisites assumed
- None. First contact with programming. A scientific calculator level of maths
  (logs, exponentials, powers).

## Concept sequence
| # | Concept | Chemistry anchor | Where |
|---|---------|------------------|-------|
| 1 | Variables, numeric types, arithmetic | ΔG = ΔH − TΔS; kcal↔kJ | notebook A |
| 2 | Lists, indexing, slicing | reaction-step energies, atomic masses | notebook A |
| 3 | Loops + conditionals | convert/filter an energy list | notebook A |
| 4 | Functions, docstrings, composition | bond length + bond detection from xyz | notebook A |
| 5 | Reading text/CSV files by hand | water/ethanol/benzene `.xyz`; alkane table | notebook A |
| 6 | NumPy arrays, vectorised maths, broadcasting | Beer–Lambert A = εcl; wavenumber axis | notebook B |
| 7 | Boolean masking, aggregation along axes | peak picking in an IR spectrum; replicate spectra | notebook B |
| 8 | matplotlib figures; `np.polyfit` | plot IR spectrum; van der Waals isotherms; calibration line | notebook B |

## Notebook plan
- **Session A** (`week01_a_python-bootcamp_*`): datasets `alkanes.csv`,
  `water.xyz`, `ethanol.xyz`, `benzene.xyz` (all in `sources/datasets/`).
  Worked example: geometry analysis pipeline (`calculate_distance`,
  `bond_check`, `print_bonds`). 4 exercises. No plotting (deferred to B).
- **Session B** (`week01_b_python-bootcamp_*`): datasets `ir_spectrum_synthetic.csv`
  (generated, documented as synthetic), van der Waals constants inline (cited).
  Worked example: build + plot a synthetic IR spectrum, pick peaks.
  4 exercises + mini-challenge: Beer–Lambert calibration fit.
- Compute budget: both < 30 s on laptop CPU.

## Slides plan
- Session A ~8 slides: why programming for chemistry; variables/lists/loops;
  functions; file formats in chemistry. Demo pointers → notebook A §1–§5.
- Session B ~8 slides: arrays vs lists; vectorisation; broadcasting; plotting;
  line fitting. Demo pointers → notebook B §1–§5.
- Figures reused: `ir_spectrum.png`, `vdw_isotherms.png`, `beer_lambert_fit.png`.

## Assessment hooks
- Examinable: reading data files, writing a correct function with a docstring,
  vectorised vs loop thinking, interpreting a calibration fit (ε, R²).

## Sources used
- MolSSI `python_scripting_cms` lessons 01, 02, 04, 05, 06 — Python + file
  parsing + tabular + plotting, chemistry-flavoured examples.
- SciCompforChemists ch. 0–1 (Python), ch. 3 (matplotlib), ch. 4 (NumPy).
- EPFL ai4chem 01a (Colab alt for students).
- Chemistry constants: van der Waals a, b (cited, TODO(verify)); alkane bp/mp
  (curated `alkanes.csv`, TODO(verify)).

## Open questions for instructor
- Confirm the alkane and van der Waals reference values against CRC before the
  deck is used in an assessed setting.
- Session B mini-challenge currently uses synthetic Beer–Lambert data; swap for
  a real teaching-lab dataset if one is available in `sources/`.
