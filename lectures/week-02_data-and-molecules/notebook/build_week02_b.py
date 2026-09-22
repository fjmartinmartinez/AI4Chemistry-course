"""Build week 02 session B notebooks (build a clean molecular dataset).

    python lectures/week-02_data-and-molecules/notebook/build_week02_b.py
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "_build"))
from nbbuild import build, checkpoint, code, exercise, md  # noqa: E402

C = []

# ==========================================================================
C += [md(r'''
# AI for Chemistry — Week 02, Session B
## Workshop: build a clean, featurised molecular dataset (ESOL)

**Course:** AI for Chemistry · **Session:** 02B (hands-on workshop, 2.0 h) ·
**Runtime:** < 60 s, no GPU.

### Suggested timing (solo study, ~115 min)

| Section | Topic | Time |
|--------:|-------|-----:|
| 0 | Setup | 3 min |
| 1 | Load and clean (the data-hygiene checklist) | 25 min |
| 2 | Featurise: descriptors and Morgan fingerprints | 20 min |
| 3 | Chemical space with PCA | 20 min |
| 4 | Similarity and Butina clustering | 20 min |
| 5 | Assemble a reusable `make_dataset()` | 12 min |
| 6 | Exercises (3) + mini-challenge | 25 min |

### Learning objectives
1. Turn a raw CSV of SMILES + property into a clean modelling table:
   canonicalise, drop unparseable/missing rows, deduplicate by structure,
   sanity-check ranges — and justify each step. *(LO2)*
2. Featurise molecules two ways — a descriptor block and Morgan fingerprints —
   and store them as NumPy arrays. *(LO2)*
3. Project chemical space to 2-D with PCA and read the component loadings. *(LO2, LO4)*
4. Measure structural similarity (Tanimoto) and cluster molecules (Butina);
   interpret the cluster-size distribution. *(LO2, LO4)*
5. Package the pipeline as one function returning `X`, `y` and a dataframe,
   ready for Week 03. *(LO2)*

### Prerequisites — before this notebook you should be able to
- Everything from Week 02A: pandas selection/filtering/`groupby`, SMILES ↔
  RDKit, computing descriptors.
- Recall what a **fingerprint** and the **Tanimoto coefficient** are (Week 02A
  further reading, TeachOpenCADD T005).

### How to use this notebook (solo study)
Read, run, check each **✅** cell. Section 5 writes
`sources/datasets/esol_clean.csv`, which Week 03 loads directly. This is the
**solutions** notebook.
''')]

# ==========================================================================
C += [md(r'''
---
## 0. Setup
''')]

C += [code(r'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, Draw, rdFingerprintGenerator
from rdkit.ML.Cluster import Butina
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
RDLogger.DisableLog("rdApp.*")

RNG = np.random.default_rng(0xC0FFEE)      # fixed course seed

ROOT = Path.cwd()
while not (ROOT / "sources").is_dir() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
DATA = ROOT / "sources" / "datasets"
FIGDIR = ROOT / "lectures" / "week-02_data-and-molecules" / "slides" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

TARGET = "measured log solubility in mols per litre"
raw = pd.read_csv(DATA / "esol_delaney.csv")
raw.columns = raw.columns.str.strip()
print("raw:", raw.shape)
''')]

C += [md(r'''
> **Common errors — Setup**
> - `ImportError` for `rdFingerprintGenerator` — very old RDKit. The course
>   environment is `2026.03`; select the `ai4chem` kernel.
''')]

# ==========================================================================
# 1. Load and clean
# ==========================================================================
C += [md(r'''
---
## 1. Load and clean — the data-hygiene checklist

Every ML result is only as trustworthy as the table it was trained on. For a
SMILES + property dataset the standard checklist is:

1. **Parse** every SMILES; drop rows RDKit cannot read.
2. **Canonicalise** — so identical structures have identical strings.
3. **Missing target** — drop rows with no measured value.
4. **Deduplicate** by canonical SMILES. If duplicates disagree on the target,
   that disagreement is a noise floor you cannot beat.
5. **Range sanity** — flag physically impossible values.

We apply these one at a time and print the row count after each, so the effect
of every decision is visible.

**What to look for:** ESOL loses nothing to parsing or missing targets, and
**11 rows** to deduplication, leaving **1117** molecules. The duplicated
structures disagree on log S by a few tenths — our irreducible error.
''')]

C += [code(r'''
def canonical_or_none(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None

df = raw.copy()
n0 = len(df)

df["canonical_smiles"] = df["smiles"].apply(canonical_or_none)
df = df[df["canonical_smiles"].notna()]
n1 = len(df)

df = df[df[TARGET].notna()]
n2 = len(df)

# quantify how badly duplicates disagree BEFORE collapsing them
dup_spread = (df.groupby("canonical_smiles")[TARGET]
                .agg(lambda s: s.max() - s.min()))
print(f"duplicated structures: {(dup_spread > 0).sum()}, "
      f"max log S disagreement: {dup_spread.max():.2f}")

df = df.drop_duplicates("canonical_smiles").reset_index(drop=True)
n3 = len(df)

print(f"rows: {n0} -> parse {n1} -> target {n2} -> dedupe {n3}")
''')]

C += [md(r'''
Range sanity for aqueous log S: values below about -13 (less soluble than the
most insoluble organic solids known) or above ~2 (miscible) would be suspicious.
ESOL is already clean, so this step should flag **nothing** — but the check
belongs in the pipeline regardless.

**What to look for:** `0 rows outside [-13, 2]`.
''')]

C += [code(r'''
out_of_range = df[(df[TARGET] < -13) | (df[TARGET] > 2)]
print(len(out_of_range), "rows outside [-13, 2] log S")
df_clean = df.copy()
''')]

C += [md(r'''
> **Common errors — Section 1**
> - Deduplicating on the **raw** `smiles` column misses `OCC` vs `CCO`
>   duplicates. Always canonicalise first.
> - `drop_duplicates` keeps the *first* occurrence silently — if duplicates
>   disagree, consider averaging the target instead (Exercise 1 territory).
> - Forgetting `.reset_index(drop=True)` leaves gaps in the index that bite
>   later when you `concat` feature matrices.
''')]

C += checkpoint(
    "Section 1",
    check=r'''
assert n1 == n2 == 1128            # ESOL: no parse failures, no missing targets
assert n3 == 1117                   # 11 duplicate structures removed
assert df_clean["canonical_smiles"].is_unique
assert df_clean[TARGET].between(-13, 2).all()
assert (df_clean.index == range(len(df_clean))).all()
print("Section 1 OK")
''',
    expected="Prints `Section 1 OK`. 1128 → 1117 rows; canonical SMILES unique; "
    "index is a clean 0..1116 range.",
    questions=r'''
1. Two records for the same structure report log S = -3.1 and -3.6. What is the
   best target value to keep, and what does the gap tell you?
2. Why deduplicate *before* splitting into train and test sets rather than
   after?
''',
    answers=r'''
1. Their mean (-3.35) is the least-biased single estimate; the 0.5 gap is
   experimental/curation noise, so no model should be expected to predict
   this structure to better than ~0.25 log units.
2. If the same molecule sits in both train and test, the test score is
   inflated by memorisation — a form of **data leakage**. Removing duplicates
   first guarantees train and test are disjoint in structure.
''',
)

# ==========================================================================
# 2. Featurise
# ==========================================================================
C += [md(r'''
---
## 2. Featurise — descriptors and fingerprints

Two complementary numeric representations:

* **Descriptor block** — a handful of interpretable physicochemical numbers
  (MW, logP, TPSA, HBD, HBA, rotatable bonds, aromatic ring count). Low
  dimensional, chemically meaningful, but lossy.
* **Morgan fingerprint** (ECFP-like), radius 2, 2048 bits — each bit marks the
  presence of a circular substructure. High dimensional, near-lossless for
  substructure, but not individually interpretable.

The course standardises on **Morgan radius 2, 2048 bits** (`TODO(verify)` this
is the agreed default). We build both as NumPy arrays aligned to `df_clean`.

**What to look for:** `Xdesc` has shape `(1117, 7)`, `Xfp` has shape
`(1117, 2048)` and is 0/1 only; roughly 20–30 bits set per molecule on average.
''')]

C += [code(r'''
mols = [Chem.MolFromSmiles(s) for s in df_clean["canonical_smiles"]]

DESCRIPTORS = {
    "MolWt": Descriptors.MolWt,
    "MolLogP": Descriptors.MolLogP,
    "TPSA": Descriptors.TPSA,
    "NumHDonors": Descriptors.NumHDonors,
    "NumHAcceptors": Descriptors.NumHAcceptors,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    "NumAromaticRings": Descriptors.NumAromaticRings,
}
Xdesc = np.array([[fn(m) for fn in DESCRIPTORS.values()] for m in mols],
                 dtype=float)
desc_names = list(DESCRIPTORS)
print("Xdesc:", Xdesc.shape)
''')]

C += [code(r'''
mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def fp_to_array(fp):
    """RDKit ExplicitBitVect -> length-2048 uint8 numpy row."""
    arr = np.zeros((2048,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

fps = [mfpgen.GetFingerprint(m) for m in mols]           # keep for Section 4
Xfp = np.vstack([fp_to_array(fp) for fp in fps])
y = df_clean[TARGET].to_numpy()

print("Xfp:", Xfp.shape, "| unique values:", np.unique(Xfp))
print("mean bits set per molecule:", round(Xfp.sum(axis=1).mean(), 1))
''')]

C += [md(r'''
> **Common errors — Section 2**
> - Building features from `df` after a filter without `reset_index` → the
>   feature array and the target end up misaligned by a few rows. Align to
>   `df_clean` (clean index) throughout.
> - `np.array(fp)` on a `ExplicitBitVect` works but is slow; use
>   `DataStructs.ConvertToNumpyArray` into a preallocated row.
> - Descriptors on very large molecules can be slow; ESOL is small so it is
>   instant here.
''')]

C += checkpoint(
    "Section 2",
    check=r'''
assert Xdesc.shape == (1117, 7)
assert Xfp.shape == (1117, 2048)
assert set(np.unique(Xfp)).issubset({0, 1})
assert y.shape == (1117,)
assert np.isfinite(Xdesc).all()
assert 10 < Xfp.sum(axis=1).mean() < 60
print("Section 2 OK")
''',
    expected="Prints `Section 2 OK`. Descriptor matrix (1117, 7), fingerprint "
    "matrix (1117, 2048) binary, ~24 bits set per molecule.",
    questions=r'''
1. The fingerprint has 2048 columns but each molecule sets only ~24. What kind
   of matrix is this, and why does that matter for storage and for linear
   models?
2. Name one molecular property the descriptor block captures that a
   substructure fingerprint does *not*.
''',
    answers=r'''
1. It is **sparse** (~99% zeros). Sparse storage saves memory, and linear
   models with L1/L2 regularisation cope well with many mostly-zero columns;
   distance metrics should be similarity-based (Tanimoto), not Euclidean.
2. Whole-molecule physicochemical quantities — e.g. logP or TPSA — which
   depend on global composition and 3-D-ish polarity, not on the presence of
   any single local fragment.
''',
)

# ==========================================================================
# 3. PCA chemical space
# ==========================================================================
C += [md(r'''
---
## 3. Chemical space with PCA

**Principal component analysis** finds the orthogonal directions of greatest
variance in a dataset. Projecting onto the first two gives a 2-D map where
nearby points have similar feature vectors.

We run PCA on the **standardised descriptor block** (each descriptor scaled to
mean 0, variance 1 — otherwise `MolWt` in the hundreds would dominate
`NumHDonors` in the units). The **loadings** tell us what each axis means
chemically.

**What to look for:** PC1 and PC2 together explain roughly 70% of descriptor
variance. Reading the loadings: **PC1** (~44%) loads positively on the polarity
descriptors (TPSA, H-bond acceptors and donors) — a "hydrogen-bonding capacity"
axis. **PC2** (~28%) loads on size and lipophilicity (MolWt, logP, aromatic
rings). It is **PC2** that lines up with the colour (measured log S): bigger,
greasier molecules are less soluble.
''')]

C += [code(r'''
scaler = StandardScaler().fit(Xdesc)
Xdesc_std = scaler.transform(Xdesc)

pca = PCA(n_components=2, random_state=0).fit(Xdesc_std)
Zdesc = pca.transform(Xdesc_std)
print("explained variance ratio:", pca.explained_variance_ratio_.round(3))

loadings = pd.DataFrame(pca.components_.T, index=desc_names, columns=["PC1", "PC2"])
print(loadings.round(2))
''')]

C += [code(r'''
fig, ax = plt.subplots(figsize=(6.5, 5))
sc = ax.scatter(Zdesc[:, 0], Zdesc[:, 1], c=y, s=14, cmap="coolwarm_r")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.0f}% var)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.0f}% var)")
ax.set_title("ESOL chemical space (descriptor PCA)")
fig.colorbar(sc, label="measured log S")
fig.tight_layout(); fig.savefig(FIGDIR / "chemical_space_pca.png", dpi=200)
plt.show()
''')]

C += [md(r'''
> **Common errors — Section 3**
> - PCA **without** standardising a descriptor block → PC1 is just "molecular
>   weight" and nothing else. Always `StandardScaler` first for mixed-unit
>   features.
> - Fingerprint PCA "looks bad" (first PCs explain only a few % each) — that is
>   expected for sparse binary data; use it for a rough map only, or prefer a
>   similarity-based embedding.
> - Sign of a principal component is arbitrary; do not over-interpret
>   left-vs-right, only the *pattern* of loadings.
''')]

C += checkpoint(
    "Section 3",
    check=r'''
assert Zdesc.shape == (1117, 2)
assert pca.explained_variance_ratio_.sum() > 0.65
logp = Xdesc[:, desc_names.index("MolLogP")]
# one of the first two PCs is the size/lipophilicity axis that tracks log S
corr_logp = [abs(np.corrcoef(Zdesc[:, k], logp)[0, 1]) for k in (0, 1)]
corr_y = [abs(np.corrcoef(Zdesc[:, k], y)[0, 1]) for k in (0, 1)]
assert max(corr_logp) > 0.8
assert max(corr_y) > 0.8
assert (FIGDIR / "chemical_space_pca.png").is_file()
print("Section 3 OK")
''',
    expected="Prints `Section 3 OK`. First two PCs explain > 65% of variance; "
    "the size/lipophilicity PC correlates > 0.8 (in magnitude) with logP and "
    "with log S.",
    questions=r'''
1. Why standardise the descriptors before PCA but *not* the fingerprint bits?
2. A point sits far from every other in this map. Give two different chemical
   reasons that could happen.
''',
    answers=r'''
1. Descriptors have wildly different units and spreads, so PCA (a
   variance-maximiser) would otherwise just track the largest-magnitude
   column. Fingerprint bits are already on the same 0/1 scale, and centring
   them destroys sparsity without adding information.
2. Either the molecule is genuinely unusual (a rare element, a very large or
   highly halogenated structure — a real outlier), or it has an extreme value
   of one descriptor (e.g. a huge flexible chain inflating rotatable bonds)
   that pushes it out along one axis without being chemically "weird".
''',
)

# ==========================================================================
# 4. Similarity and clustering
# ==========================================================================
C += [md(r'''
---
## 4. Structural similarity and Butina clustering

The **Tanimoto coefficient** between two fingerprints is
$T = \dfrac{|A \cap B|}{|A \cup B|}$ — shared bits over total bits, from 0
(nothing in common) to 1 (identical bit pattern). Distance is $1 - T$.

**Butina clustering**: pick the molecule with the most neighbours within a
distance `cutoff`, make it a cluster centre, remove it and its neighbours,
repeat. Every molecule ends up in exactly one cluster (possibly alone).

**What to look for:** at `cutoff = 0.4` ESOL splits into hundreds of clusters
with a **long tail of singletons** — ESOL deliberately spans many scaffolds, so
it is structurally diverse. The largest cluster is a family of close analogues.
''')]

C += [code(r'''
def tanimoto_distance_matrix(fp_list):
    """Lower-triangle condensed distance list for Butina (isDistData=True)."""
    dists = []
    for i in range(1, len(fp_list)):
        sims = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list[:i])
        dists.extend(1.0 - s for s in sims)
    return dists

CUTOFF = 0.4
dmat = tanimoto_distance_matrix(fps)
clusters = Butina.ClusterData(dmat, len(fps), CUTOFF, isDistData=True)
clusters = sorted(clusters, key=len, reverse=True)

sizes = np.array([len(c) for c in clusters])
print(f"{len(clusters)} clusters | largest {sizes[0]} | "
      f"singletons {(sizes == 1).sum()} ({(sizes == 1).mean()*100:.0f}%)")
''')]

C += [code(r'''
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.hist(sizes, bins=range(1, sizes.max() + 2), align="left", color="teal")
ax.set_xlabel("cluster size"); ax.set_ylabel("number of clusters")
ax.set_yscale("log"); ax.set_title(f"Butina cluster sizes (cutoff {CUTOFF})")
fig.tight_layout(); fig.savefig(FIGDIR / "cluster_sizes.png", dpi=200)
plt.show()
''')]

C += [md(r'''
Draw the largest cluster: these molecules should look like obvious analogues of
each other.

**What to look for:** a grid of visually similar structures (a common
scaffold with small substituent changes).
''')]

C += [code(r'''
biggest = list(clusters[0])
grid = Draw.MolsToGridImage([mols[i] for i in biggest[:12]],
                            legends=[df_clean["Compound ID"].iloc[i] for i in biggest[:12]],
                            molsPerRow=4, subImgSize=(200, 150), returnPNG=False)
grid.save(FIGDIR / "largest_cluster.png")
grid
''')]

C += [md(r'''
> **Common errors — Section 4**
> - `Butina.ClusterData` expects a **condensed** lower-triangle list when
>   `isDistData=True`, in the exact order produced above — not a square matrix.
> - Using Euclidean distance on fingerprints — wrong metric; use
>   `1 - Tanimoto`.
> - A tiny cutoff makes almost everything a singleton; a large one collapses
>   everything into one blob. There is no universal "right" value.
''')]

C += checkpoint(
    "Section 4",
    check=r'''
assert len(dmat) == 1117 * 1116 // 2
assert sum(len(c) for c in clusters) == 1117      # every molecule placed once
assert sizes[0] >= 5                               # a real analogue series exists
assert (sizes == 1).sum() > 300                    # ESOL is diverse
assert (FIGDIR / "cluster_sizes.png").is_file()
assert (FIGDIR / "largest_cluster.png").is_file()
print("Section 4 OK")
''',
    expected="Prints `Section 4 OK`. All 1117 molecules assigned; largest "
    "cluster ≥ 5; > 300 singletons.",
    questions=r'''
1. For making a train/test split that tests **generalisation to new
   scaffolds**, how would you use these clusters?
2. Why is Tanimoto, not a count of shared bits, the right similarity measure?
''',
    answers=r'''
1. Put whole clusters entirely in train or entirely in test (a "cluster
   split" / scaffold split). Then the test molecules have no close analogue in
   training and the score reflects extrapolation, not interpolation.
2. Raw shared-bit counts scale with molecule size — big molecules share more
   bits with everything. Dividing by the union size normalises for size, so a
   small molecule and a large one are not automatically "similar".
''',
)

# ==========================================================================
# 5. Assemble
# ==========================================================================
C += [md(r'''
---
## 5. Assemble a reusable `make_dataset()`

Bundle the whole pipeline into one function so Week 03 can call it in a single
line and get identical, reproducible arrays. It returns the feature matrix `X`,
target `y`, and the cleaned dataframe (for inspection / grouping / splitting).

We also write `sources/datasets/esol_clean.csv` — the canonical cleaned table
the rest of the course builds on.

**What to look for:** `make_dataset("descriptors")` → `X` shape `(1117, 7)`;
`make_dataset("fingerprints")` → `(1117, 2048)`; both share the same `y`; the
CSV is written.
''')]

C += [code(r'''
def make_dataset(representation="descriptors", csv_path=DATA / "esol_delaney.csv"):
    """Clean ESOL and featurise it. Returns (X, y, df_clean).

    representation: "descriptors" (7 physicochemical) or "fingerprints"
    (Morgan radius 2, 2048 bits). Fully deterministic.
    """
    tbl = pd.read_csv(csv_path)
    tbl.columns = tbl.columns.str.strip()
    tbl["canonical_smiles"] = tbl["smiles"].apply(canonical_or_none)
    tbl = tbl[tbl["canonical_smiles"].notna() & tbl[TARGET].notna()]
    tbl = tbl.drop_duplicates("canonical_smiles").reset_index(drop=True)
    tbl = tbl[tbl[TARGET].between(-13, 2)].reset_index(drop=True)

    ms = [Chem.MolFromSmiles(s) for s in tbl["canonical_smiles"]]
    if representation == "descriptors":
        X = np.array([[fn(m) for fn in DESCRIPTORS.values()] for m in ms], float)
    elif representation == "fingerprints":
        X = np.vstack([fp_to_array(mfpgen.GetFingerprint(m)) for m in ms])
    else:
        raise ValueError(f"unknown representation: {representation!r}")
    return X, tbl[TARGET].to_numpy(), tbl


Xd, yd, clean = make_dataset("descriptors")
Xf, yf, _ = make_dataset("fingerprints")
clean.to_csv(DATA / "esol_clean.csv", index=False)
print("descriptors:", Xd.shape, "| fingerprints:", Xf.shape,
      "| target:", yd.shape, "| same y:", np.array_equal(yd, yf))
print("wrote", (DATA / "esol_clean.csv").name)
''')]

C += [md(r'''
> **Common errors — Section 5**
> - A pipeline that is not deterministic (e.g. `set` ordering, unсeeded
>   sampling) gives different `X` each run — pin every random source.
> - Writing the clean CSV with the default index adds a junk `Unnamed: 0`
>   column on reload; pass `index=False`.
''')]

C += checkpoint(
    "Section 5",
    check=r'''
assert Xd.shape == (1117, 7) and Xf.shape == (1117, 2048)
assert np.array_equal(yd, yf) and yd.shape == (1117,)
# determinism
Xd2, yd2, _ = make_dataset("descriptors")
assert np.array_equal(Xd, Xd2) and np.array_equal(yd, yd2)
# round-trips through disk cleanly
reloaded = pd.read_csv(DATA / "esol_clean.csv")
assert len(reloaded) == 1117 and "Unnamed: 0" not in reloaded.columns
try:
    make_dataset("bogus"); assert False
except ValueError:
    pass
print("Section 5 OK")
''',
    expected="Prints `Section 5 OK`. Both representations align to the same "
    "1117-row target; re-running is bit-identical; `esol_clean.csv` reloads "
    "with 1117 rows and no junk column.",
    questions=r'''
1. Why return the cleaned dataframe as well as `X` and `y`?
2. `make_dataset` re-cleans from scratch every call. When would you cache the
   intermediate instead?
''',
    answers=r'''
1. Downstream code needs the SMILES, IDs and canonical structures — to make a
   scaffold split, to inspect errors molecule-by-molecule, or to join on extra
   data. `X` and `y` alone throw that away.
2. When cleaning is expensive (large dataset, 3-D embedding, external lookups)
   or must be frozen for a release — then compute once, write to disk with a
   checksum, and load the cached artefact.
''',
)

# ==========================================================================
# 6. Exercises
# ==========================================================================
C += [md(r'''
---
## 6. Exercises
''')]

C += exercise(
    prompt=r'''
### Exercise 1 — average, don't drop, disagreeing duplicates *(medium, ~15 min)*

`drop_duplicates` keeps the first of a set of duplicate structures. Write
`dedupe_by_mean(df, key="canonical_smiles", target=TARGET)` that instead groups
by `key` and returns one row per structure with the target set to the **mean**
of the duplicates (keep the first value of every other column). Apply it to
`base` — `raw` cleaned for parsing and missing target only (no dedupe yet),
built for you in the answer cell.

<details><summary>Show hint</summary>

```
base = raw.copy()
base["canonical_smiles"] = base["smiles"].apply(canonical_or_none)
base = base[base["canonical_smiles"].notna() & base[TARGET].notna()]
```
Then `agg = {c: "first" for c in base.columns}; agg[target] = "mean"` and
`base.groupby(key, as_index=False).agg(agg)`.
</details>
''',
    solution=r'''
def dedupe_by_mean(df, key="canonical_smiles", target=TARGET):
    """One row per `key`; `target` averaged over duplicates, others kept first."""
    agg = {c: "first" for c in df.columns if c != key}
    agg[target] = "mean"
    return df.groupby(key, as_index=False).agg(agg)

base = raw.copy()
base["canonical_smiles"] = base["smiles"].apply(canonical_or_none)
base = base[base["canonical_smiles"].notna() & base[TARGET].notna()]
averaged = dedupe_by_mean(base)
print(averaged.shape)
''',
    scaffold=r'''
def dedupe_by_mean(df, key="canonical_smiles", target=TARGET):
    """One row per `key`; `target` averaged over duplicates, others kept first."""
    # YOUR CODE HERE
    ...

base = raw.copy()
base["canonical_smiles"] = base["smiles"].apply(canonical_or_none)
base = base[base["canonical_smiles"].notna() & base[TARGET].notna()]
averaged = ...  # YOUR CODE HERE
''',
    check=r'''
assert len(averaged) == 1117
assert averaged["canonical_smiles"].is_unique
# the mean-merge must shift at least one target value vs first-wins dedupe
first_wins = base.drop_duplicates("canonical_smiles").set_index("canonical_smiles")[TARGET]
merged = averaged.set_index("canonical_smiles")[TARGET]
assert not np.allclose(first_wins.sort_index(), merged.sort_index())
print("Exercise 1 OK")
''',
)

C += exercise(
    prompt=r'''
### Exercise 2 — nearest neighbours by Tanimoto *(medium, ~15 min)*

Write `nearest_neighbours(query_smiles, k=5)` returning a list of
`(compound_id, tanimoto)` for the `k` molecules in `df_clean` most similar to
the query (exclude an exact match at T = 1.0). Use the precomputed `fps`.

<details><summary>Show hint</summary>

`q = mfpgen.GetFingerprint(Chem.MolFromSmiles(query_smiles))`;
`sims = DataStructs.BulkTanimotoSimilarity(q, fps)`; `argsort` descending, skip
any with `sim >= 0.999`, take `k`.
</details>
''',
    solution=r'''
def nearest_neighbours(query_smiles, k=5):
    """k most Tanimoto-similar molecules in df_clean to the query SMILES."""
    q = mfpgen.GetFingerprint(Chem.MolFromSmiles(query_smiles))
    sims = np.array(DataStructs.BulkTanimotoSimilarity(q, fps))
    order = np.argsort(sims)[::-1]
    hits = [(df_clean["Compound ID"].iloc[i], float(sims[i]))
            for i in order if sims[i] < 0.999][:k]
    return hits

nn_aspirin = nearest_neighbours("CC(=O)Oc1ccccc1C(=O)O", k=5)
for cid, t in nn_aspirin:
    print(f"{t:.3f}  {cid}")
''',
    scaffold=r'''
def nearest_neighbours(query_smiles, k=5):
    """k most Tanimoto-similar molecules in df_clean to the query SMILES."""
    # YOUR CODE HERE
    ...

nn_aspirin = ...  # YOUR CODE HERE: nearest 5 to aspirin
''',
    check=r'''
assert len(nn_aspirin) == 5
sims_only = [t for _, t in nn_aspirin]
assert sims_only == sorted(sims_only, reverse=True)   # descending
assert all(0.0 <= t < 0.999 for t in sims_only)
assert sims_only[0] > 0.3                             # aspirin has analogues in ESOL
print("Exercise 2 OK")
''',
)

C += exercise(
    prompt=r'''
### Exercise 3 — how many PCs for 90% variance? *(easy, ~10 min)*

Write `n_components_for(X, threshold=0.9)` returning the smallest number of
principal components whose **cumulative** explained-variance ratio reaches
`threshold`. Run it on the standardised descriptor block `Xdesc_std`.

<details><summary>Show hint</summary>

`p = PCA().fit(X)`; `cum = np.cumsum(p.explained_variance_ratio_)`;
`int(np.argmax(cum >= threshold) + 1)`.
</details>
''',
    solution=r'''
def n_components_for(X, threshold=0.9):
    """Smallest #PCs with cumulative explained variance >= threshold."""
    p = PCA(random_state=0).fit(X)
    cum = np.cumsum(p.explained_variance_ratio_)
    return int(np.argmax(cum >= threshold) + 1)

k90 = n_components_for(Xdesc_std, 0.9)
print("descriptor block: need", k90, "of 7 PCs for 90% variance")
''',
    scaffold=r'''
def n_components_for(X, threshold=0.9):
    """Smallest #PCs with cumulative explained variance >= threshold."""
    # YOUR CODE HERE
    ...

k90 = ...  # YOUR CODE HERE
''',
    check=r'''
assert 1 <= k90 <= 7
assert n_components_for(Xdesc_std, 0.99) >= k90
assert n_components_for(Xdesc_std, 0.0001) == 1
print("Exercise 3 OK")
''',
)

C += [md(r'''
### 🏁 Mini-challenge — a leakage-aware scaffold split *(~25 min)*

Data leakage from near-duplicate structures across the train/test boundary is
the number-one way beginners fool themselves. Build a split that avoids it.

Write `cluster_split(clusters, n_total, test_frac=0.2, seed=0)` that assigns
**whole Butina clusters** to test until at least `test_frac` of molecules are in
test, returning boolean arrays `(train_mask, test_mask)` of length `n_total`
(indices refer to positions in `df_clean` / `fps`).

Then verify the split is leakage-free: the **maximum Tanimoto similarity**
between any test molecule and its nearest *training* molecule should be clearly
below 1, and typically well below a random split would give.

<details><summary>Show hint</summary>

Shuffle the cluster list with `np.random.default_rng(seed)`, walk through
adding each cluster's member indices to the test set until the count target is
met. `train_mask = ~test_mask`. For the leakage check, for each test index take
`max(BulkTanimotoSimilarity(fps[i], [fps[j] for j in train_idx]))`.
</details>
''')]

C += exercise(
    prompt=r'''
Implement `cluster_split`, then run the leakage check below it.
''',
    solution=r'''
def cluster_split(clusters, n_total, test_frac=0.2, seed=0):
    """Assign whole clusters to test until >= test_frac molecules are held out."""
    rng = np.random.default_rng(seed)
    order = list(range(len(clusters)))
    rng.shuffle(order)
    test_mask = np.zeros(n_total, dtype=bool)
    target = int(np.ceil(test_frac * n_total))
    for c in order:
        if test_mask.sum() >= target:
            break
        for idx in clusters[c]:
            test_mask[idx] = True
    return ~test_mask, test_mask


train_mask, test_mask = cluster_split(clusters, len(fps), test_frac=0.2, seed=0)
train_idx = np.where(train_mask)[0]

# leakage check: nearest training neighbour of each test molecule
train_fps = [fps[j] for j in train_idx]
max_sim_to_train = np.array([
    max(DataStructs.BulkTanimotoSimilarity(fps[i], train_fps))
    for i in np.where(test_mask)[0]
])
print(f"test fraction: {test_mask.mean():.2f}")
print(f"nearest-train Tanimoto  mean {max_sim_to_train.mean():.2f}  "
      f"max {max_sim_to_train.max():.2f}")
''',
    scaffold=r'''
def cluster_split(clusters, n_total, test_frac=0.2, seed=0):
    """Assign whole clusters to test until >= test_frac molecules are held out."""
    # YOUR CODE HERE: shuffle cluster order with np.random.default_rng(seed),
    # add whole clusters to test_mask until test_mask.sum() >= test_frac*n_total
    ...

train_mask, test_mask = ...  # YOUR CODE HERE

# leakage check (given): nearest training neighbour of each test molecule
train_idx = np.where(train_mask)[0]
train_fps = [fps[j] for j in train_idx]
max_sim_to_train = np.array([
    max(DataStructs.BulkTanimotoSimilarity(fps[i], train_fps))
    for i in np.where(test_mask)[0]
])
print("nearest-train Tanimoto max:", round(max_sim_to_train.max(), 2))
''',
    check=r'''
# --- check the mini-challenge (do not edit) ---
assert train_mask.sum() + test_mask.sum() == 1117
assert not (train_mask & test_mask).any()               # disjoint
assert 0.18 <= test_mask.mean() <= 0.30
# no test molecule is a near-identical twin of a training molecule
assert max_sim_to_train.max() < 0.99
# whole clusters stay together: no cluster is split across the boundary
for c in clusters:
    members = np.array(list(c))
    assert test_mask[members].all() or (~test_mask[members]).all()
print("Mini-challenge OK")
''',
)

# ==========================================================================
C += [md(r'''
---
## Summary — what you learned

- The **data-hygiene checklist**: parse → canonicalise → drop missing target →
  deduplicate by structure → range-check, counting rows at each step.
- **Two featurisations**: an interpretable 7-descriptor block and a
  2048-bit Morgan fingerprint, both as aligned NumPy arrays.
- **PCA** on standardised descriptors as a chemical-space map, read via
  component loadings.
- **Tanimoto similarity** and **Butina clustering**; ESOL is structurally
  diverse (many singletons).
- A reproducible **`make_dataset()`** and the cleaned `esol_clean.csv` that
  Week 03 uses.
- **Leakage**: near-duplicate structures across a train/test split inflate
  scores; a cluster/scaffold split fixes it.

Week 03 takes `esol_clean.csv` and fits the first real predictive models.
''')]

C += [md(r'''
## Further reading (course source list only)

- TeachOpenCADD **T005** (fingerprints, Tanimoto, Butina clustering):
  <https://projects.volkamerlab.org/teachopencadd/talktorials/T005_compound_clustering.html>
- TeachOpenCADD **T001** (querying ChEMBL — the upstream step we skipped):
  <https://projects.volkamerlab.org/teachopencadd/talktorials/T001_query_chembl.html>
- EPFL *AI for Chemistry* 01c (plotting essentials):
  <https://colab.research.google.com/github/schwallergroup/ai4chem_course/blob/main/notebooks/01%20-%20Basics/01c_python_essentials_plotting.ipynb>
''')]

C += [md(r'''
## Attribution

Original teaching material for *AI for Chemistry*, adapted and rewritten from:

- **TeachOpenCADD** (Volkamer Lab) — talktorials T001/T002/T005: the
  cleaning → fingerprint → Tanimoto → Butina workflow and the
  `tanimoto_distance_matrix` helper pattern. CC-BY 4.0.
  <https://github.com/volkamerlab/teachopencadd>
- **Practical Cheminformatics** (P. Walters) — the "avoid leakage with a
  cluster split" idea (developed further in Week 10). MIT licence.
  <https://github.com/PatWalters/practical_cheminformatics_tutorials>
- **EPFL CH-457 *AI for Chemistry*** (Schwaller group). MIT licence.
  <https://github.com/schwallergroup/ai4chem_course>

Dataset: ESOL / Delaney (`esol_delaney.csv`, checksum-pinned). Cleaned output
`esol_clean.csv` is generated by this notebook. No verbatim text is reproduced
from the sources above.
''')]

build(__file__, "week02_b_data-and-molecules", C)
