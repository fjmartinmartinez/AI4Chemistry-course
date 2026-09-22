"""Build week 06 session A notebooks (representations & inductive bias).

    python lectures/week-06_representations-inductive-bias/notebook/build_week06_a.py
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "_build"))
from nbbuild import build, checkpoint, code, exercise, md  # noqa: E402

C = []

# ==========================================================================
C += [md(r'''
# AI for Chemistry — Week 06, Session A
## Representations and inductive bias

**Course:** AI for Chemistry · **Session:** 06A (interleaved lecture + lab,
2.5 h) · **Runtime:** < 10 s, no GPU.

### Suggested timing (solo study, ~120 min)

| Section | Topic | Time |
|--------:|-------|-----:|
| 0 | Setup | 5 min |
| 1 | Why raw coordinates fail: translational variance | 20 min |
| 2 | Invariance and equivariance, formally | 22 min |
| 3 | Descriptors vs fingerprints vs learned features: a spectrum | 20 min |
| 4 | Equivariant neural networks (qualitative) | 10 min |
| 5 | Exercises (4) | 33 min |

### Learning objectives
1. Demonstrate, concretely, that raw atomic coordinates are not directly
   usable as ML input, using translational variance. *(LO6)*
2. State the formal definitions of permutation/translation/rotation
   invariance and equivariance. *(LO6)*
3. Build an invariant descriptor from a point cloud (pairwise distances,
   radius of gyration). *(LO6)*
4. Place descriptors, fingerprints and learned features on a spectrum from
   "fully hand-engineered" to "fully learned". *(LO6)*
5. State, qualitatively, what an equivariant neural network is and why 3D
   molecular ML needs SE(3) — no implementation. *(LO6)*

### Prerequisites — before this notebook you should be able to
- Week 02: descriptors, Morgan fingerprints.
- Week 05: MLPs and hidden-layer representations.
- Basic 3D vectors/matrices (dot products, rotation matrices).

### How to use this notebook (solo study)
Read, run, check each **✅** cell in order. This is the **solutions**
notebook.
''')]

# ==========================================================================
C += [md(r'''
---
## 0. Setup

We reuse `ethanol.xyz` from Week 01 (an RDKit-generated 3D conformer) as a
concrete point cloud, and rebuild the ESOL descriptor/fingerprint blocks from
Weeks 02-03 for Section 3's representation-spectrum discussion.

**What to look for:** `R` has shape `(9, 3)` — ethanol's 9 atoms, each an
$(x,y,z)$ position.
''')]

C += [code(r'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit import DataStructs
RDLogger.DisableLog("rdApp.*")

RNG = np.random.default_rng(0xC0FFEE)

ROOT = Path.cwd()
while not (ROOT / "sources").is_dir() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
DATA = ROOT / "sources" / "datasets"
FIGDIR = ROOT / "lectures" / "week-06_representations-inductive-bias" / "slides" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

def read_xyz(path):
    """Read an .xyz file -> (symbols list, (N,3) numpy array of positions)."""
    lines = Path(path).read_text().splitlines()
    n_atoms = int(lines[0])
    symbols, coords = [], []
    for line in lines[2:2 + n_atoms]:
        parts = line.split()
        symbols.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return symbols, np.array(coords)

symbols, R = read_xyz(DATA / "ethanol.xyz")
print("atoms:", symbols)
print("R shape:", R.shape)
''')]

C += [md(r'''
> **Common errors — Setup**
> - `FileNotFoundError` for `ethanol.xyz` — run
>   `python lectures/_build/make_datasets_wk01_02.py` first (Week 01 setup).
''')]

# ==========================================================================
# 1. Translational variance
# ==========================================================================
C += [md(r'''
---
## 1. Why raw coordinates fail: translational variance

Ethanol's chemistry — its energy, its solubility, everything about it — does
not depend on *where in space* you happen to have written down its
coordinates. But a naive "descriptor" built directly from raw coordinates,
such as **the z-coordinate of the first atom**, has no way of knowing that:
shift the whole molecule and the number changes, even though nothing
chemical happened. This is **translational variance**, and it is the opening
example of dmol.pub's *Input data and equivariances* chapter.

**What to look for:** `naive(R)` and `naive(R + t)` differ substantially for a
non-zero translation `t`, even though `R` and `R + t` describe the *same*
molecule in the *same* conformation.
''')]

C += [code(r'''
def naive_descriptor(coords):
    """A deliberately bad 'descriptor': the z-coordinate of the first atom."""
    return coords[0, 2]

t = RNG.normal(size=3) * 5.0          # an arbitrary shift, angstrom
R_translated = R + t                   # same molecule, same conformation, new origin

print("naive descriptor, original position :", round(naive_descriptor(R), 4))
print("naive descriptor, translated        :", round(naive_descriptor(R_translated), 4))
print("translation vector t (A):", np.round(t, 2))
''')]

C += [md(r'''
> **Common errors — Section 1**
> - Concluding "coordinates are therefore useless" — they are not; the fix is
>   to build **invariant functions of the coordinates** (Section 3), not to
>   abandon 3D information altogether. Modern GNNs on 3D structures (Week 07
>   onward, and the equivariant architectures in Section 4) use coordinates
>   directly, but through carefully designed, invariant/equivariant operations.
> - Applying a translation to *some* atoms but not others — that is not a
>   translation of the molecule, it silently changes the conformation.
''')]

C += checkpoint(
    "Section 1",
    check=r'''
assert R.shape == (9, 3)
assert abs(naive_descriptor(R_translated) - (naive_descriptor(R) + t[2])) < 1e-8
assert abs(naive_descriptor(R_translated) - naive_descriptor(R)) > 1.0
print("Section 1 OK")
''',
    expected="Prints `Section 1 OK`. The naive descriptor shifts by exactly "
    "`t[2]` under translation — a large, chemically meaningless change.",
    questions=r'''
1. Why does the naive descriptor shift by *exactly* `t[2]`, not some other
   amount?
2. A model trained on coordinates from simulations that all happen to start
   near the origin might "accidentally" work despite this problem. Why is
   that dangerous?
''',
    answers=r'''
1. `naive_descriptor` just reads off `coords[0, 2]`, the raw z-coordinate;
   adding `t` to every atom's coordinates adds `t[2]` to that one number
   directly — there is no chemistry in the calculation to cancel it out.
2. It creates a hidden dependency on an accident of the training data (where
   molecules happened to be centred); the model will fail unpredictably the
   moment it sees a molecule described in a different frame — exactly the
   kind of silent generalisation failure this course keeps returning to
   (Week 02B/03B's leakage, Week 04's imbalance blind spots).
''',
)

# ==========================================================================
# 2. Invariance and equivariance, formally
# ==========================================================================
C += [md(r'''
---
## 2. Invariance and equivariance, formally

Let $f$ be a model taking atomic positions $R$ (and features $X$, e.g.
element identity) to an output. Three symmetry operations matter for
molecules:

$$\textbf{Permutation:}\quad f(P[R,X]) = P[f(R,X)] \ \ (\text{equivariant}), \qquad
  f(P[R,X]) = f(R,X) \ \ (\text{invariant, for scalar output}),$$
$$\textbf{Translation:}\quad f(R+\vec t, X) = f(R,X) + \vec t \ \ (\text{equivariant}), \qquad
  f(R+\vec t, X) = f(R,X) \ \ (\text{invariant}),$$
$$\textbf{Rotation:}\quad f(\mathcal{R}R, X) = \mathcal{R}f(R,X) \ \ (\text{equivariant}), \qquad
  f(\mathcal{R}R, X) = f(R,X) \ \ (\text{invariant}),$$

where $P$ permutes atom order and $\mathcal{R}$ is a rotation matrix.
**Invariant** outputs (energy, solubility — a single number describing the
whole molecule) should not change under these symmetries at all.
**Equivariant** outputs (per-atom forces, atomic charges) should transform
*consistently* with the input — rotate the molecule, and the predicted force
vectors rotate the same way.

There is no "first" or "second" atom in a molecule — atom order is an
artefact of how the SMILES or file was written, not a chemical fact, which is
why **permutation invariance is not optional** for scalar molecular
properties.

**What to look for:** the naive descriptor fails all three symmetries; two
constructed invariants — the **sum of all pairwise distances** and the
**radius of gyration** $R_g=\sqrt{\langle(\vec r_i-\bar{\vec r})^2\rangle}$ —
are unchanged (to floating-point precision) under translation, rotation
*and* permutation.
''')]

C += [code(r'''
def pairwise_distance_sum(coords):
    """Sum of all pairwise atom-atom distances -- invariant to T/R/permutation."""
    diff = coords[:, None, :] - coords[None, :, :]     # (N,N,3), Week 05A broadcasting
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    return dist.sum()

def radius_of_gyration(coords):
    """R_g: RMS distance of atoms from the centre of mass (unweighted here)."""
    centre = coords.mean(axis=0)
    return np.sqrt(((coords - centre) ** 2).sum(axis=1).mean())

# a random rigid rotation matrix (orthonormal, right-handed) via QR decomposition
A = RNG.normal(size=(3, 3))
Qmat, _ = np.linalg.qr(A)
if np.linalg.det(Qmat) < 0:
    Qmat[:, 0] *= -1                                    # ensure a proper rotation

R_rotated = R @ Qmat.T
perm = RNG.permutation(len(R))
R_permuted = R[perm]

cases = [("original", R), ("translated", R_translated),
        ("rotated", R_rotated), ("permuted", R_permuted)]
for name, coords in cases:
    print(f"{name:10s}  naive={naive_descriptor(coords):8.3f}   "
          f"pdist_sum={pairwise_distance_sum(coords):8.3f}   "
          f"Rg={radius_of_gyration(coords):.4f}")
''')]

C += [code(r'''
labels = [name for name, _ in cases]
naive_vals = [naive_descriptor(c) for _, c in cases]
pdist_vals = [pairwise_distance_sum(c) for _, c in cases]

fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
axes[0].bar(labels, naive_vals, color="crimson")
axes[0].set_title("naive descriptor (VARIANT)"); axes[0].set_ylabel("value")
axes[1].bar(labels, pdist_vals, color="seagreen")
axes[1].set_title("pairwise distance sum (INVARIANT)")
axes[1].set_ylim(min(pdist_vals) - 5, max(pdist_vals) + 5)
for ax in axes:
    ax.tick_params(axis="x", rotation=20)
fig.suptitle("Ethanol under translation / rotation / permutation")
fig.tight_layout()
fig.savefig(FIGDIR / "invariant_vs_variant.png", dpi=200)
plt.show()
''')]

C += [md(r'''
> **Common errors — Section 2**
> - Building a rotation matrix that is not orthonormal (e.g. from arbitrary
>   random numbers without `qr` or a proper rotation-matrix construction) —
>   it would also scale/shear the molecule, not just rotate it, and the
>   "invariant" descriptors would then (correctly) change.
> - Forgetting that **invariance** and **equivariance** are properties of a
>   *function*, not of data alone — the same coordinates `R` can feed an
>   invariant function (radius of gyration) or a variant one (the naive
>   descriptor); the data did not change, the function did.
''')]

C += checkpoint(
    "Section 2",
    check=r'''
tol = 1e-6
for coords in [R_translated, R_rotated, R_permuted]:
    assert abs(pairwise_distance_sum(coords) - pairwise_distance_sum(R)) < tol
    assert abs(radius_of_gyration(coords) - radius_of_gyration(R)) < tol
assert abs(naive_descriptor(R_rotated) - naive_descriptor(R)) > 0.05
assert abs(naive_descriptor(R_permuted) - naive_descriptor(R)) > 0.05
assert (FIGDIR / "invariant_vs_variant.png").is_file()
print("Section 2 OK")
''',
    expected="Prints `Section 2 OK`. `pairwise_distance_sum` and "
    "`radius_of_gyration` are unchanged (to 1e-6) under all three symmetry "
    "operations; the naive descriptor changes under rotation and permutation.",
    questions=r'''
1. `radius_of_gyration` centres the coordinates before measuring spread. Why
   does that step alone make it translation-invariant?
2. Give a molecular property that should be **equivariant**, not invariant,
   and explain why invariance would be the wrong requirement for it.
''',
    answers=r'''
1. Subtracting the centre of mass $\bar{\vec r}$ removes any constant offset
   added to every atom — translating $R\to R+\vec t$ shifts $\bar{\vec r}$ by
   the same $\vec t$, so $R-\bar{\vec r}$ is completely unchanged, and
   everything computed from it (including $R_g$) inherits that invariance.
2. Per-atom **partial charges arranged spatially** (or the dipole moment
   vector, or per-atom forces): rotate the molecule and the physically
   correct answer is the *same* charges attached to the *rotated* atoms — an
   invariant model would incorrectly report the same numbers in the old,
   now-wrong orientation.
''',
)

# ==========================================================================
# 3. Descriptors vs fingerprints vs learned features
# ==========================================================================
C += [md(r'''
---
## 3. Descriptors vs fingerprints vs learned features: a spectrum

Every representation this course has used sits somewhere between "entirely
hand-engineered" and "entirely learned":

| Representation | Where it comes from | Learned? | Week |
|---|---|---|---|
| 7 physicochemical descriptors (MW, logP, TPSA, ...) | a chemist's formula for each number | No — fixed by definition | 02 |
| Morgan fingerprint (2048 bits) | a fixed hashing algorithm over local substructures | No — fixed algorithm, but combinatorial | 02 |
| MLP hidden-layer activations | `Wx+b` then $\sigma$, on top of the descriptors above | Partially — the *weights* are learned, but the *input* is still hand-engineered | 05 |
| Graph neural network node/graph embeddings | learned directly from the molecular graph, no hand-designed features required | Yes — the representation itself is learned | 07 |

This is the essence of **inductive bias**: a fixed representation *encodes*
prior chemical knowledge (which is powerful when that knowledge is right, and
limiting when it is not); a learned representation must *discover* useful
structure from data alone (flexible, but needs more data to do so reliably —
Week 05B's honest "MLP barely beats random forest" result on a 900-row
dataset was exactly this trade-off in action).

**What to look for:** roughly 74% of the descriptor block's entries are
non-zero (a zero here is still a meaningful value — e.g. "no rotatable
bonds" — not a missing one); the fingerprint is over 95% zero despite having
300× more columns. Both are entirely fixed, non-trainable transformations of
a molecule.
''')]

C += [code(r'''
TARGET = "measured log solubility in mols per litre"
esol = pd.read_csv(DATA / "esol_clean.csv")

DESCRIPTORS = {
    "MolWt": Descriptors.MolWt, "MolLogP": Descriptors.MolLogP,
    "TPSA": Descriptors.TPSA, "NumHDonors": Descriptors.NumHDonors,
    "NumHAcceptors": Descriptors.NumHAcceptors,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    "NumAromaticRings": Descriptors.NumAromaticRings,
}
mols = [Chem.MolFromSmiles(s) for s in esol["canonical_smiles"]]
Xdesc = np.array([[fn(m) for fn in DESCRIPTORS.values()] for m in mols], dtype=float)

mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
Xfp = np.zeros((len(mols), 2048), dtype=np.uint8)
for i, m in enumerate(mols):
    fp = mfpgen.GetFingerprint(m)
    DataStructs.ConvertToNumpyArray(fp, Xfp[i])

print(f"descriptor block:  shape {Xdesc.shape},  {(Xdesc != 0).mean()*100:.0f}% non-zero entries")
print(f"Morgan fingerprint: shape {Xfp.shape},  {(Xfp != 0).mean()*100:.1f}% non-zero entries")
''')]

C += [md(r'''
> **Common errors — Section 3**
> - Assuming "more hand-engineered" always means "worse" or vice versa — Week
>   05B showed descriptors *beating* raw MLP features on a small dataset;
>   inductive bias is a genuine trade-off, not a hierarchy.
> - Calling a fingerprint "learned" because it is high-dimensional — Morgan
>   fingerprints use a **fixed** hashing procedure; nothing about them is fit
>   to data (contrast with Week 07's GNN embeddings, which are).
''')]

C += checkpoint(
    "Section 3",
    check=r'''
assert Xdesc.shape[1] == 7
assert Xfp.shape[1] == 2048
assert (Xdesc != 0).mean() > 0.5        # descriptor block: mostly non-zero
assert (Xfp != 0).mean() < 0.05         # fingerprint is sparse
print("Section 3 OK")
''',
    expected="Prints `Section 3 OK`. The descriptor block is mostly non-zero "
    "(~74%); the fingerprint is sparse (<5% non-zero) despite having 300x "
    "more columns.",
    questions=r'''
1. Where would "distance to the nearest training-set neighbour by Tanimoto
   similarity" (Week 02B) sit on the hand-engineered/learned spectrum?
2. Why might a *learned* representation need substantially more training data
   than a well-chosen fixed one to reach the same accuracy?
''',
    answers=r'''
1. Fully hand-engineered — Tanimoto similarity is a fixed formula, and
   "nearest neighbour" is a fixed rule; no parameters are fit to data (unlike,
   say, a learned metric/embedding that *decides* what "similar" means).
2. A fixed representation already encodes the modeller's chemical prior (e.g.
   "logP matters for solubility"); a learned representation has to *discover*
   which combinations of raw information are useful entirely from examples,
   which needs enough data to distinguish real structure from noise — exactly
   Week 03's bias-variance trade-off, now applied to the representation
   itself rather than just the model.
''',
)

# ==========================================================================
# 4. Equivariant neural networks (qualitative)
# ==========================================================================
C += [md(r'''
---
## 4. Equivariant neural networks — qualitative only

*(Deliberately conceptual — `syllabus.md` scopes equivariant neural networks
as a survey topic for this course. No code here.)*

Rather than hand-building invariant descriptors (Section 2) and hoping they
capture everything relevant, **equivariant neural networks** bake
permutation/rotation/translation (in)variance directly into the model
architecture, so the network is *guaranteed* to respect these symmetries by
construction — no data augmentation, no hoping the model learns them.

The relevant symmetry group for a 3D molecule (ignoring reflections) is
**SE(3)**: 3D rotations (the group $SO(3)$) combined with translations.
Architectures such as **Tensor Field Networks**, **Cormorant** and
**E3NN**-based models represent internal features not as plain numbers but as
objects with well-defined rotational behaviour (built from **spherical
harmonics**, the natural basis functions for $SO(3)$), and combine them with
**tensor products** that are mathematically guaranteed to preserve
equivariance through every layer.

This matters directly for 3D molecular tasks — predicting forces, energies
from geometry, or generating 3D conformations — because it removes an entire
category of "wasted" learning capacity (Section 1's translational-variance
failure, at every layer of a deep network) and typically needs far less data
to reach the same accuracy as a generic network fed raw coordinates plus
heavy data augmentation.

We do not implement any of this in the course; `scipy`/RDKit-based invariant
descriptors (Section 2) and Week 07's *graph*-based (not point-cloud-based)
neural networks are the representations we build with hands-on.
''')]

# ==========================================================================
# 5. Exercises
# ==========================================================================
C += [md(r'''
---
## 5. Exercises
''')]

C += exercise(
    prompt=r'''
### Exercise 1 — classify a descriptor *(easy, ~10 min)*

Write `is_translation_invariant(descriptor_fn, coords, n_trials=5)` that
applies `n_trials` random translations to `coords`, evaluates
`descriptor_fn` on each, and returns `True` only if all results (and the
original) agree to within `1e-6`. Test it on `naive_descriptor` and
`pairwise_distance_sum`.

<details><summary>Show hint</summary>

Generate each translation with `RNG.normal(size=3) * 5.0` (as in Section 1);
compare every result to `descriptor_fn(coords)` with `abs(...) < 1e-6`.
</details>
''',
    solution=r'''
def is_translation_invariant(descriptor_fn, coords, n_trials=5):
    """True if descriptor_fn is unchanged (to 1e-6) under n_trials translations."""
    base = descriptor_fn(coords)
    for _ in range(n_trials):
        t = RNG.normal(size=3) * 5.0
        if abs(descriptor_fn(coords + t) - base) > 1e-6:
            return False
    return True

print("naive_descriptor:      ", is_translation_invariant(naive_descriptor, R))
print("pairwise_distance_sum: ", is_translation_invariant(pairwise_distance_sum, R))
''',
    scaffold=r'''
def is_translation_invariant(descriptor_fn, coords, n_trials=5):
    """True if descriptor_fn is unchanged (to 1e-6) under n_trials translations."""
    # YOUR CODE HERE
    ...

print("naive_descriptor:      ", is_translation_invariant(naive_descriptor, R))
print("pairwise_distance_sum: ", is_translation_invariant(pairwise_distance_sum, R))
''',
    check=r'''
assert is_translation_invariant(naive_descriptor, R) is False
assert is_translation_invariant(pairwise_distance_sum, R) is True
assert is_translation_invariant(radius_of_gyration, R) is True
print("Exercise 1 OK")
''',
)

C += exercise(
    prompt=r'''
### Exercise 2 — an invariant that is NOT sensitive to conformation *(medium, ~15 min)*

`pairwise_distance_sum` is invariant to T/R/permutation, but is it sensitive
to **conformation** (which is what we actually want — a real conformational
change *should* change the descriptor)? Write
`bond_stretch_test(coords, bond=(0, 1), delta=0.3)` that stretches the
distance between two given atoms by `delta` angstrom (moving both atoms
apart symmetrically along their connecting axis) and returns
`(before, after)` values of `pairwise_distance_sum`.

<details><summary>Show hint</summary>

Unit vector `u = (coords[b]-coords[a]) / norm(coords[b]-coords[a])`; move
atom `a` by `-delta/2 * u` and atom `b` by `+delta/2 * u`; recompute the
descriptor on the modified copy.
</details>
''',
    solution=r'''
def bond_stretch_test(coords, bond=(0, 1), delta=0.3):
    """Stretch the bond between two atoms by delta A; return (before, after)."""
    a, b = bond
    before = pairwise_distance_sum(coords)
    u = (coords[b] - coords[a])
    u = u / np.linalg.norm(u)
    stretched = coords.copy()
    stretched[a] -= 0.5 * delta * u
    stretched[b] += 0.5 * delta * u
    after = pairwise_distance_sum(stretched)
    return before, after

before, after = bond_stretch_test(R)
print(f"pairwise_distance_sum before={before:.3f}  after stretch={after:.3f}")
''',
    scaffold=r'''
def bond_stretch_test(coords, bond=(0, 1), delta=0.3):
    """Stretch the bond between two atoms by delta A; return (before, after)."""
    # YOUR CODE HERE
    ...

before, after = ...  # YOUR CODE HERE: bond_stretch_test(R)
''',
    check=r'''
assert after > before        # stretching a bond increases total pairwise distance
assert after - before < 5.0  # a single 0.3 A stretch is a small perturbation
before0, after0 = bond_stretch_test(R, bond=(0, 1), delta=0.0)
assert abs(after0 - before0) < 1e-8    # zero stretch changes nothing
print("Exercise 2 OK")
''',
)

C += exercise(
    prompt=r'''
### Exercise 3 — equivariant, not just invariant *(medium, ~15 min)*

Write `centre_of_mass(coords)` (equivariant: it should **move** correctly
under translation and rotation, unlike an invariant scalar). Confirm
$\text{centre}(R+\vec t) = \text{centre}(R) + \vec t$ and
$\text{centre}(\mathcal{R}R) = \mathcal{R}\,\text{centre}(R)$.

<details><summary>Show hint</summary>

`coords.mean(axis=0)`. For the rotation check, compare `centre(R @ Q.T)` to
`centre(R) @ Q.T` using the `Qmat` built in Section 2.
</details>
''',
    solution=r'''
def centre_of_mass(coords):
    """Mean position -- an EQUIVARIANT quantity (a vector, not a scalar)."""
    return coords.mean(axis=0)

translation_ok = np.allclose(centre_of_mass(R_translated), centre_of_mass(R) + t, atol=1e-6)
rotation_ok = np.allclose(centre_of_mass(R_rotated), centre_of_mass(R) @ Qmat.T, atol=1e-6)
print("translation equivariance holds:", translation_ok)
print("rotation equivariance holds:", rotation_ok)
''',
    scaffold=r'''
def centre_of_mass(coords):
    """Mean position -- an EQUIVARIANT quantity (a vector, not a scalar)."""
    # YOUR CODE HERE
    ...

translation_ok = ...  # YOUR CODE HERE
rotation_ok = ...  # YOUR CODE HERE
''',
    check=r'''
assert centre_of_mass(R).shape == (3,)
assert translation_ok is True
assert rotation_ok is True
print("Exercise 3 OK")
''',
)

C += exercise(
    prompt=r'''
### Exercise 4 — where does a fingerprint sit on the spectrum? *(harder, ~20 min)*

Write `representation_report(X, name)` returning a dict with `name`,
`n_dims`, `pct_nonzero`, and `is_binary` (`True` if every entry is 0 or 1).
Apply it to `Xdesc` and `Xfp` and use the results to argue (in a short
markdown answer, printed as a string) which one a linear model is likely to
struggle with more on a ~1000-row dataset, and why.

<details><summary>Show hint</summary>

`is_binary = set(np.unique(X)) <= {0, 1}`. `pct_nonzero = (X != 0).mean() *
100`. Revisit Week 03B's finding that raw linear regression on the 2048-bit
fingerprint had *negative* R² with 893 training rows.
</details>
''',
    solution=r'''
def representation_report(X, name):
    """Summary stats characterising a feature matrix's representation."""
    return {
        "name": name,
        "n_dims": X.shape[1],
        "pct_nonzero": round((X != 0).mean() * 100, 2),
        "is_binary": bool(set(np.unique(X).tolist()) <= {0, 1}),
    }

report_desc = representation_report(Xdesc, "descriptors")
report_fp = representation_report(Xfp, "Morgan-2048")
print(report_desc)
print(report_fp)
print("\nA plain linear model struggles more with the fingerprint: it has "
      "2048 columns for under 1000 training rows (Week 03B's p >> n "
      "overfitting regime), and each column carries very little information "
      "on its own -- the opposite of the dense, low-dimensional descriptor block.")
''',
    scaffold=r'''
def representation_report(X, name):
    """Summary stats characterising a feature matrix's representation."""
    # YOUR CODE HERE
    ...

report_desc = ...  # YOUR CODE HERE: representation_report(Xdesc, "descriptors")
report_fp = ...  # YOUR CODE HERE: representation_report(Xfp, "Morgan-2048")
''',
    check=r'''
assert report_desc["n_dims"] == 7
assert report_fp["n_dims"] == 2048
assert report_fp["is_binary"] is True
assert report_desc["pct_nonzero"] > report_fp["pct_nonzero"]
print("Exercise 4 OK")
''',
)

# ==========================================================================
C += [md(r'''
---
## Summary — what you learned

- Raw atomic coordinates are **translationally variant**: a concrete, minimal
  demonstration, not just an abstract warning.
- Formal definitions of **permutation/translation/rotation invariance and
  equivariance**, and how to build invariant descriptors (pairwise distances,
  radius of gyration) from a point cloud.
- A **spectrum** from hand-engineered (descriptors) through fixed-but-high-
  dimensional (fingerprints) to fully learned (Week 07's GNN embeddings) —
  and why "more learned" is not automatically "better" at small data scale.
- **Equivariant neural networks** (qualitative): SE(3), spherical harmonics,
  tensor products, named architectures — no implementation this week.

Week 06B puts the "descriptors vs fingerprints" question to an empirical test:
a representation shoot-out with the model held fixed — and assigns the
module's mini-project.
''')]

C += [md(r'''
## Further reading (course source list only)

- dmol.pub *Input data and equivariances*: <https://dmol.pub/dl/data.html>
- dmol.pub *Equivariant neural networks* (survey only):
  <https://dmol.pub/dl/Equivariant.html>
''')]

C += [md(r'''
## Attribution

Original teaching material for *AI for Chemistry*, adapted and rewritten from:

- **dmol.pub** (A. White) — the translational-variance opening example, the
  invariance/equivariance definitions, and the invariant-descriptor
  strategies (pairwise distances, radius-of-gyration-style reductions).
  CC-BY 4.0. <https://dmol.pub>

Datasets: `ethanol.xyz` (Week 01, RDKit-generated); ESOL descriptor and
Morgan-fingerprint blocks (Weeks 02-03). No verbatim text is reproduced from
the sources above.
''')]

build(__file__, "week06_a_representations-inductive-bias", C)
