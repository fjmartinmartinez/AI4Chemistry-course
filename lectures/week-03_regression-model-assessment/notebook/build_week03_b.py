"""Build week 03 session B notebooks (solubility-prediction leaderboard lab).

    python lectures/week-03_regression-model-assessment/notebook/build_week03_b.py
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "_build"))
from nbbuild import build, checkpoint, code, exercise, md  # noqa: E402

C = []

# ==========================================================================
C += [md(r'''
# AI for Chemistry — Week 03, Session B
## Workshop: solubility-prediction leaderboard

**Course:** AI for Chemistry · **Session:** 03B (hands-on workshop, 2.0 h) ·
**Runtime:** < 30 s, no GPU.

### Suggested timing (solo study, ~120 min)

| Section | Topic | Time |
|--------:|-------|-----:|
| 0 | Setup | 4 min |
| 1 | Baseline: predict the mean | 8 min |
| 2 | Linear and Ridge regression, two representations | 22 min |
| 3 | Random forest, two representations; feature importance | 20 min |
| 4 | The leaderboard | 12 min |
| 5 | Leakage: random split vs Butina cluster split | 22 min |
| 6 | Exercises (3) + mini-challenge | 32 min |

### Learning objectives
1. Establish a **baseline** and explain why every leaderboard needs one. *(LO4)*
2. Fit linear, ridge and random-forest regressors on two representations
   (descriptors, fingerprints) and read a leaderboard table honestly. *(LO4)*
3. Explain *why* plain linear regression fails on wide fingerprint features,
   and how regularisation fixes it. *(LO3, LO4)*
4. Read random-forest feature importances. *(LO4)*
5. Show, with numbers from this exact dataset, that a random split can
   overstate performance relative to a structure-aware (Butina cluster)
   split, and explain why. *(LO4, LO8)*

### Prerequisites — before this notebook you should be able to
- Week 03A: train/test split, cross-validation, MAE/RMSE/R², regularisation.
- Week 02B: Morgan fingerprints, Butina clustering (this notebook rebuilds
  both if `esol_clean.csv` is missing).

### How to use this notebook (solo study)
Read, run, check each **✅** cell in order. This is the **solutions** notebook.
''')]

# ==========================================================================
C += [md(r'''
---
## 0. Setup

Rebuild the Week 02/03 pipeline: clean ESOL, the 7-descriptor block, and
2048-bit Morgan fingerprints. We fix **one** train/test index split
(`random_state=42`, same as Week 03A) and reuse it for every model so the
leaderboard compares like with like.

**What to look for:** `n_train 893  n_test 224`; `Xdesc`/`Xfp` shapes match
Week 02B.
''')]

C += [code(r'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.ML.Cluster import Butina
RDLogger.DisableLog("rdApp.*")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RNG = np.random.default_rng(0xC0FFEE)

ROOT = Path.cwd()
while not (ROOT / "sources").is_dir() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
DATA = ROOT / "sources" / "datasets"
FIGDIR = ROOT / "lectures" / "week-03_regression-model-assessment" / "slides" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

TARGET = "measured log solubility in mols per litre"
DESCRIPTORS = {
    "MolWt": Descriptors.MolWt, "MolLogP": Descriptors.MolLogP,
    "TPSA": Descriptors.TPSA, "NumHDonors": Descriptors.NumHDonors,
    "NumHAcceptors": Descriptors.NumHAcceptors,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    "NumAromaticRings": Descriptors.NumAromaticRings,
}

def canonical_or_none(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None

clean_path = DATA / "esol_clean.csv"
if clean_path.is_file():
    esol = pd.read_csv(clean_path)
else:
    raw = pd.read_csv(DATA / "esol_delaney.csv")
    raw.columns = raw.columns.str.strip()
    raw["canonical_smiles"] = raw["smiles"].apply(canonical_or_none)
    raw = raw[raw["canonical_smiles"].notna() & raw[TARGET].notna()]
    esol = raw.drop_duplicates("canonical_smiles").reset_index(drop=True)
    esol = esol[esol[TARGET].between(-13, 2)].reset_index(drop=True)
    esol.to_csv(clean_path, index=False)

mols = [Chem.MolFromSmiles(s) for s in esol["canonical_smiles"]]
desc_names = list(DESCRIPTORS)
Xdesc = np.array([[fn(m) for fn in DESCRIPTORS.values()] for m in mols], dtype=float)

mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
fps = [mfpgen.GetFingerprint(m) for m in mols]
Xfp = np.zeros((len(mols), 2048), dtype=np.uint8)
for i, fp in enumerate(fps):
    DataStructs.ConvertToNumpyArray(fp, Xfp[i])

y = esol[TARGET].to_numpy()

idx = np.arange(len(mols))
idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42)
y_train, y_test = y[idx_train], y[idx_test]
print(f"n_train {len(idx_train)}  n_test {len(idx_test)}")
print("Xdesc:", Xdesc.shape, " Xfp:", Xfp.shape)
''')]

C += [md(r'''
> **Common errors — Setup**
> - Splitting `Xdesc` and `Xfp` with **separate** calls to `train_test_split`
>   — unless you fix the same `random_state` *and* pass arrays in the same
>   order, the two representations end up with different test molecules and
>   the leaderboard stops being a fair comparison. We split **indices** once
>   and reuse them for exactly this reason.
''')]

# ==========================================================================
# 1. Baseline
# ==========================================================================
C += [md(r'''
---
## 1. Baseline: predict the mean

Every leaderboard needs a floor. `DummyRegressor(strategy="mean")` predicts the
training mean for every molecule, regardless of its features — by
construction its R² on the training set is exactly 0. Any real model that
cannot beat this is worthless.

**What to look for:** R² close to 0 (it can be slightly negative on the *test*
set — the test mean is not exactly the train mean); RMSE close to the target's
standard deviation.
''')]

C += [code(r'''
def evaluate(model, X_train, y_train, X_test, y_test):
    """Fit `model`, return a dict of test-set MAE/RMSE/R2."""
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return {
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": mean_squared_error(y_test, pred) ** 0.5,
        "R2": r2_score(y_test, pred),
    }

results = {}
results[("Baseline (mean)", "-")] = evaluate(
    DummyRegressor(strategy="mean"), Xdesc[idx_train], y_train, Xdesc[idx_test], y_test)
print(results[("Baseline (mean)", "-")])
''')]

C += [md(r'''
> **Common errors — Section 1**
> - Skipping the baseline and reporting "R² = 0.77" as if that number means
>   something in isolation — it only means something *relative to* this floor
>   (and to other models).
''')]

C += checkpoint(
    "Section 1",
    check=r'''
assert abs(results[("Baseline (mean)", "-")]["R2"]) < 0.05
print("Section 1 OK")
''',
    expected="Prints `Section 1 OK`. Baseline R² is within 0.05 of zero.",
    questions=r'''
1. Why is the baseline's R² not *exactly* 0 on the test set?
2. What would it mean if a fitted model's R² came out *below* the baseline's?
''',
    answers=r'''
1. R² = 0 by definition when predictions equal the **training** mean and are
   evaluated on the **training** set. On a held-out test set, the test mean
   differs slightly from the training mean by chance, so the "predict the
   training mean" strategy is not exactly variance-neutral there.
2. The model is worse than not modelling anything — actively harmful,
   typically a sign of severe overfitting or a broken pipeline (see Section 2).
''',
)

# ==========================================================================
# 2. Linear and Ridge regression
# ==========================================================================
C += [md(r'''
---
## 2. Linear and Ridge regression, two representations

Fit plain `LinearRegression` on both representations. The **descriptor block**
has 7 features and 893 training molecules — a well-posed least-squares
problem. The **fingerprint** has 2048 features for the same 893 molecules —
more parameters than data points, exactly the overfitting regime from
Week 03A §2.

**What to look for:** descriptors give a respectable R² (~0.77); the raw
fingerprint linear fit is **worse than the baseline** (negative R²) — it has
memorised training noise. Adding an L2 penalty (`Ridge`) tames it back to a
usable, if not great, model.
''')]

C += [code(r'''
results[("Linear", "descriptors")] = evaluate(
    LinearRegression(), Xdesc[idx_train], y_train, Xdesc[idx_test], y_test)
results[("Linear", "fingerprints")] = evaluate(
    LinearRegression(), Xfp[idx_train], y_train, Xfp[idx_test], y_test)
results[("Ridge (alpha=1)", "fingerprints")] = evaluate(
    Ridge(alpha=1.0), Xfp[idx_train], y_train, Xfp[idx_test], y_test)

for key in [("Linear", "descriptors"), ("Linear", "fingerprints"),
            ("Ridge (alpha=1)", "fingerprints")]:
    print(key, results[key])
''')]

C += [md(r'''
> **Common errors — Section 2**
> - Concluding "fingerprints are a bad representation" from the raw linear
>   result — the representation is fine; the *model* (unregularised linear
>   regression with $p \gg n$) is not. Section 3's random forest handles the
>   same fingerprint just fine.
> - Using `Ridge` with `alpha=1.0` blindly — Week 03A Exercise 4 showed how to
>   pick `alpha` by cross-validation instead of guessing.
''')]

C += checkpoint(
    "Section 2",
    check=r'''
assert results[("Linear", "descriptors")]["R2"] > 0.6
assert results[("Linear", "fingerprints")]["R2"] < results[("Baseline (mean)", "-")]["R2"]
assert results[("Ridge (alpha=1)", "fingerprints")]["R2"] > results[("Linear", "fingerprints")]["R2"]
print("Section 2 OK")
''',
    expected="Prints `Section 2 OK`. Linear/descriptors beats the baseline "
    "comfortably; raw Linear/fingerprints is *worse* than the baseline; Ridge "
    "recovers a positive, usable R².",
    questions=r'''
1. In plain English, why does having more features than training rows make
   ordinary least squares behave badly?
2. Ridge fixed the fingerprint model without removing a single feature. What
   did it actually change?
''',
    answers=r'''
1. With more unknowns ($p=2048$) than equations ($n=893$), there are
   infinitely many weight vectors that fit the training data exactly (zero
   training error) — least squares picks one of them with no preference for
   "sensible", and it typically has huge, noise-fitting coefficients that
   generalise terribly.
2. It added a preference (the $\lambda\sum w_k^2$ penalty) for **small**
   coefficients among all the weight vectors that fit the training data
   reasonably well, picking a much less extreme, more stable solution.
''',
)

# ==========================================================================
# 3. Random forest and feature importance
# ==========================================================================
C += [md(r'''
---
## 3. Random forest, two representations; feature importance

A **random forest** averages many decision trees, each trained on a bootstrap
resample with a random subset of features considered at each split. It handles
high-dimensional, non-linear feature sets — including raw fingerprints —
without needing regularisation.

`feature_importances_` reports how much each descriptor reduced impurity
(variance) across the forest, on average. It is *not* a causal statement.

**What to look for:** random forest beats every linear model on both
representations; on the descriptor block, `MolLogP` dominates the importance
ranking by a wide margin (consistent with Week 02's correlation analysis).
''')]

C += [code(r'''
rf_desc = RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1)
rf_fp = RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1)

results[("Random forest", "descriptors")] = evaluate(
    rf_desc, Xdesc[idx_train], y_train, Xdesc[idx_test], y_test)
results[("Random forest", "fingerprints")] = evaluate(
    rf_fp, Xfp[idx_train], y_train, Xfp[idx_test], y_test)

for key in [("Random forest", "descriptors"), ("Random forest", "fingerprints")]:
    print(key, results[key])
''')]

C += [code(r'''
importance = pd.Series(rf_desc.feature_importances_, index=desc_names).sort_values()

fig, ax = plt.subplots(figsize=(5.5, 3.5))
importance.plot.barh(ax=ax, color="seagreen")
ax.set_xlabel("random-forest feature importance")
ax.set_title("Which descriptor drives the solubility model?")
fig.tight_layout(); fig.savefig(FIGDIR / "feature_importance.png", dpi=200)
plt.show()
print(importance.round(3))
''')]

C += [md(r'''
> **Common errors — Section 3**
> - Reading feature importance as "cause of low solubility" — it only says the
>   *model* relies on that feature to reduce error; correlated descriptors
>   (e.g. MolWt and TPSA both correlate with size) can trade importance
>   between them depending on the random seed.
> - Comparing importances across **different representations** (a fingerprint
>   bit vs a descriptor) — they are not on a comparable scale or meaning.
''')]

C += checkpoint(
    "Section 3",
    check=r'''
assert results[("Random forest", "descriptors")]["R2"] > results[("Linear", "descriptors")]["R2"]
assert results[("Random forest", "fingerprints")]["R2"] > results[("Ridge (alpha=1)", "fingerprints")]["R2"]
assert importance.idxmax() == "MolLogP"
assert importance["MolLogP"] > 0.5
assert (FIGDIR / "feature_importance.png").is_file()
print("Section 3 OK")
''',
    expected="Prints `Section 3 OK`. Random forest beats its linear/ridge "
    "counterpart on both representations; `MolLogP` is by far the most "
    "important descriptor.",
    questions=r'''
1. Random forest beats Ridge on fingerprints by a wide margin. What about
   tree-based models makes them well suited to sparse, high-dimensional binary
   features?
2. `NumHDonors` has low importance here even though Week 02 showed it
   correlates with solubility. Is that a contradiction?
''',
    answers=r'''
1. A single split in a tree ("is bit 743 set?") is a natural way to use a
   binary feature; trees also automatically capture interactions and are
   scale/units-free, so 2000+ mostly-irrelevant bits do not need to be
   individually down-weighted the way a linear model's coefficients do.
2. No — importance is about **relative** predictive contribution once other,
   more informative descriptors (MolLogP especially) are already available.
   A feature can be genuinely correlated with the target yet redundant given
   stronger correlated features, so the model "spends" little importance on it.
''',
)

# ==========================================================================
# 4. The leaderboard
# ==========================================================================
C += [md(r'''
---
## 4. The leaderboard

Collect every result so far into one sorted table — the standard way to report
a model comparison honestly (all models, same split, same metrics).

**What to look for:** `Random forest / descriptors` on top; the raw
`Linear / fingerprints` row visibly worse than the baseline.
''')]

C += [code(r'''
def leaderboard(results_dict):
    """{(model, representation): metrics} -> DataFrame sorted by RMSE."""
    rows = [{"model": m, "representation": r, **metrics}
            for (m, r), metrics in results_dict.items()]
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)

board = leaderboard(results)
board.round(3)
''')]

C += [code(r'''
fig, ax = plt.subplots(figsize=(7, 4))
labels = board["model"] + " / " + board["representation"]
ax.barh(labels[::-1], board["RMSE"][::-1], color="slateblue")
ax.set_xlabel("test RMSE (log S units, lower is better)")
ax.set_title("Week 03B leaderboard (random 80/20 split)")
fig.tight_layout(); fig.savefig(FIGDIR / "leaderboard.png", dpi=200)
plt.show()
''')]

C += [md(r'''
> **Common errors — Section 4**
> - Ranking by R² and by RMSE and getting different orders — for the *same*
>   test set they always agree (both are monotonic transforms of the sum of
>   squared errors), but comparing across *different* test sets can disagree.
>   Keep the split fixed, as we did in Setup.
''')]

C += checkpoint(
    "Section 4",
    check=r'''
assert len(board) == len(results)
assert board.iloc[0]["model"] == "Random forest" and board.iloc[0]["representation"] == "descriptors"
assert (board["RMSE"] > 0).all()
assert (FIGDIR / "leaderboard.png").is_file()
print("Section 4 OK")
''',
    expected="Prints `Section 4 OK`. `Random forest / descriptors` is row 0 "
    "(lowest RMSE).",
    questions=r'''
1. This leaderboard uses one fixed 80/20 split. What is the one-sentence
   objection a careful reviewer would raise, and how did Week 03A already give
   you the tool to answer it?
''',
    answers=r'''
1. "How much of this ranking is just luck of the split?" — re-run each model
   under k-fold cross-validation (Week 03A §3) and report mean ± std RMSE per
   model instead of a single number, so the ranking's uncertainty is visible.
''',
)

# ==========================================================================
# 5. Leakage: random split vs Butina cluster split
# ==========================================================================
C += [md(r'''
---
## 5. Leakage: random split vs Butina cluster split

A **random** split can place two near-identical molecules (an analogue series)
on opposite sides of the train/test boundary — the model "predicts" the test
molecule almost by looking up its close relative in training. Week 02B's
**Butina clustering** groups molecules by Tanimoto similarity; assigning whole
clusters to test removes that shortcut.

We rebuild the Week 02B clusters here (same recipe, cutoff 0.4) and compare the
same two random-forest models under both splitting strategies.

**What to look for:** `Random forest / descriptors` barely changes (physically
meaningful global properties do not "leak" much through near-duplicates);
`Random forest / fingerprints` drops noticeably more — it was partly relying on
near-neighbour lookup that the cluster split removes.
''')]

C += [code(r'''
def tanimoto_distance_matrix(fp_list):
    dists = []
    for i in range(1, len(fp_list)):
        sims = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list[:i])
        dists.extend(1.0 - s for s in sims)
    return dists

def cluster_split(clusters, n_total, test_frac=0.2, seed=0):
    """Assign whole Butina clusters to test until >= test_frac molecules held out."""
    rng = np.random.default_rng(seed)
    order = list(range(len(clusters)))
    rng.shuffle(order)
    test_mask = np.zeros(n_total, dtype=bool)
    target = int(np.ceil(test_frac * n_total))
    for c in order:
        if test_mask.sum() >= target:
            break
        for i in clusters[c]:
            test_mask[i] = True
    return ~test_mask, test_mask

CUTOFF = 0.4
dmat = tanimoto_distance_matrix(fps)
clusters = Butina.ClusterData(dmat, len(fps), CUTOFF, isDistData=True)
clusters = sorted(clusters, key=len, reverse=True)

train_mask, test_mask = cluster_split(clusters, len(mols), test_frac=0.2, seed=0)
print(f"{len(clusters)} clusters | cluster-split test fraction {test_mask.mean():.2f}")
''')]

C += [code(r'''
results_cluster = {}
results_cluster[("Random forest", "descriptors")] = evaluate(
    RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1),
    Xdesc[train_mask], y[train_mask], Xdesc[test_mask], y[test_mask])
results_cluster[("Random forest", "fingerprints")] = evaluate(
    RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1),
    Xfp[train_mask], y[train_mask], Xfp[test_mask], y[test_mask])

compare = pd.DataFrame({
    "random split RMSE": [results[("Random forest", "descriptors")]["RMSE"],
                           results[("Random forest", "fingerprints")]["RMSE"]],
    "cluster split RMSE": [results_cluster[("Random forest", "descriptors")]["RMSE"],
                            results_cluster[("Random forest", "fingerprints")]["RMSE"]],
}, index=["descriptors", "fingerprints"])
compare["increase (%)"] = 100 * (compare["cluster split RMSE"] / compare["random split RMSE"] - 1)
print(compare.round(3))
''')]

C += [code(r'''
fig, ax = plt.subplots(figsize=(5.5, 3.5))
x = np.arange(2)
ax.bar(x - 0.18, compare["random split RMSE"], width=0.36, label="random split")
ax.bar(x + 0.18, compare["cluster split RMSE"], width=0.36, label="cluster split")
ax.set_xticks(x); ax.set_xticklabels(["descriptors", "fingerprints"])
ax.set_ylabel("test RMSE"); ax.set_title("Random forest: does the split matter?")
ax.legend(); fig.tight_layout()
fig.savefig(FIGDIR / "leakage_random_vs_cluster_split.png", dpi=200)
plt.show()
''')]

C += [md(r'''
> **Common errors — Section 5**
> - Concluding "the cluster split is always worse, so it is pessimistic" — it
>   is not pessimistic, it is more **honest** about a model's ability to
>   extrapolate to genuinely new chemistry, which is what deployment usually
>   requires.
> - Re-tuning hyperparameters *after* seeing the cluster-split score and
>   reporting that as your original result — that reintroduces the leakage
>   you were trying to remove, just at the hyperparameter level.
''')]

C += checkpoint(
    "Section 5",
    check=r'''
assert compare.loc["fingerprints", "increase (%)"] > compare.loc["descriptors", "increase (%)"]
assert compare["cluster split RMSE"].min() > 0
assert (FIGDIR / "leakage_random_vs_cluster_split.png").is_file()
print("Section 5 OK")
''',
    expected="Prints `Section 5 OK`. The fingerprint model's RMSE increases "
    "more (in %) under the cluster split than the descriptor model's does.",
    questions=r'''
1. A random split and a cluster split can give the *same* answer to "which
   model is best" while giving different answers to "how good is the best
   model". Why might both of those questions matter to different people?
2. Would you expect this random-vs-cluster gap to be bigger or smaller on a
   *more* structurally diverse dataset than ESOL? Why?
''',
    answers=r'''
1. A method developer choosing between algorithms mainly needs the *ranking*
   (which stays fairly stable); someone deciding whether to trust a model's
   predictions on new, real compounds needs the *absolute* generalisation
   estimate, which is exactly what leaks under a random split.
2. Smaller — Week 02B showed ESOL is already structurally diverse (hundreds
   of Butina singletons), so a random split already separates many analogue
   pairs by chance. A dataset with more redundant near-duplicate structures
   (e.g. a medicinal-chemistry SAR series) would show a *larger* gap, because
   a random split would much more often leak near-neighbours into training.
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
### Exercise 1 — add k-nearest-neighbours to the leaderboard *(easy, ~10 min)*

Add `KNeighborsRegressor(n_neighbors=5)` on both representations to `results`
(same keys pattern as Section 2/3), then rebuild the leaderboard as `board2`.

<details><summary>Show hint</summary>

`results[("KNN (k=5)", "descriptors")] = evaluate(KNeighborsRegressor(n_neighbors=5), ...)`.
Fingerprint KNN uses Euclidean distance on 0/1 vectors by default — not ideal
(Week 02B used Tanimoto instead), but a reasonable first try.
</details>
''',
    solution=r'''
results[("KNN (k=5)", "descriptors")] = evaluate(
    KNeighborsRegressor(n_neighbors=5), Xdesc[idx_train], y_train, Xdesc[idx_test], y_test)
results[("KNN (k=5)", "fingerprints")] = evaluate(
    KNeighborsRegressor(n_neighbors=5), Xfp[idx_train], y_train, Xfp[idx_test], y_test)

board2 = leaderboard(results)
board2.round(3)
''',
    scaffold=r'''
# YOUR CODE HERE: add KNN (k=5) on descriptors and on fingerprints to `results`

board2 = leaderboard(results)
board2.round(3)
''',
    check=r'''
assert ("KNN (k=5)", "descriptors") in results
assert ("KNN (k=5)", "fingerprints") in results
assert len(board2) == len(results) == 8
assert (results[("KNN (k=5)", "descriptors")]["R2"] > 0
        and results[("KNN (k=5)", "fingerprints")]["R2"] > -1)
print("Exercise 1 OK")
''',
)

C += exercise(
    prompt=r'''
### Exercise 2 — how many trees does the forest need? *(medium, ~15 min)*

Write `rf_rmse_vs_trees(n_estimators_list)` returning a list of test RMSEs for
`RandomForestRegressor(n_estimators=n, random_state=0)` on **descriptors**, one
per value of `n`. Plot RMSE vs `n_estimators`.

<details><summary>Show hint</summary>

Loop the list, fit + evaluate exactly as in Section 3, collect `["RMSE"]`.
RMSE should drop quickly then flatten — diminishing returns past ~100-200
trees.
</details>
''',
    solution=r'''
def rf_rmse_vs_trees(n_estimators_list):
    """Test RMSE (descriptors) for each n_estimators value."""
    out = []
    for n in n_estimators_list:
        m = RandomForestRegressor(n_estimators=n, random_state=0, n_jobs=-1)
        out.append(evaluate(m, Xdesc[idx_train], y_train, Xdesc[idx_test], y_test)["RMSE"])
    return out

tree_counts = [5, 10, 25, 50, 100, 200, 400]
rmse_vs_trees = rf_rmse_vs_trees(tree_counts)

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(tree_counts, rmse_vs_trees, "o-")
ax.set_xlabel("n_estimators"); ax.set_ylabel("test RMSE")
ax.set_title("Random forest: RMSE vs forest size")
fig.tight_layout(); fig.savefig(FIGDIR / "rf_rmse_vs_trees.png", dpi=200)
plt.show()
''',
    scaffold=r'''
def rf_rmse_vs_trees(n_estimators_list):
    """Test RMSE (descriptors) for each n_estimators value."""
    # YOUR CODE HERE
    ...

tree_counts = [5, 10, 25, 50, 100, 200, 400]
rmse_vs_trees = ...  # YOUR CODE HERE
''',
    check=r'''
assert len(rmse_vs_trees) == len(tree_counts)
assert rmse_vs_trees[0] > rmse_vs_trees[-1] - 0.1   # more trees roughly helps or plateaus
assert all(r > 0 for r in rmse_vs_trees)
print("Exercise 2 OK")
''',
)

C += exercise(
    prompt=r'''
### Exercise 3 — which descriptor matters second-most? *(medium, ~12 min)*

Using `rf_desc.feature_importances_` (already fitted in Section 3), write
`top_k_descriptors(k)` returning the `k` descriptor names in decreasing order
of importance.

<details><summary>Show hint</summary>

`pd.Series(rf_desc.feature_importances_, index=desc_names).sort_values(ascending=False)`,
then `.index[:k].tolist()`.
</details>
''',
    solution=r'''
def top_k_descriptors(k):
    """The k most important descriptors for rf_desc, most important first."""
    ranked = pd.Series(rf_desc.feature_importances_, index=desc_names)
    return ranked.sort_values(ascending=False).index[:k].tolist()

print("top 3:", top_k_descriptors(3))
''',
    scaffold=r'''
def top_k_descriptors(k):
    """The k most important descriptors for rf_desc, most important first."""
    # YOUR CODE HERE
    ...

print("top 3:", top_k_descriptors(3))
''',
    check=r'''
assert top_k_descriptors(1) == ["MolLogP"]
assert len(top_k_descriptors(3)) == 3
assert set(top_k_descriptors(7)) == set(desc_names)
print("Exercise 3 OK")
''',
)

C += [md(r'''
### 🏁 Mini-challenge — the official (leakage-aware) leaderboard *(~25 min)*

The leaderboard in Section 4 used a random split; Section 5 showed that
overstates fingerprint models. Build the **official** leaderboard on the
**cluster split** (`train_mask`/`test_mask` from Section 5), covering every
model type used above, and check that random forest on descriptors still beats
the plain linear baseline by a healthy margin under this stricter test.

Write `official_leaderboard()` returning a `DataFrame` like Section 4's, but
fit and evaluated on the cluster split.

<details><summary>Show hint</summary>

Repeat the `evaluate(...)` calls from Sections 1-3 with
`Xdesc[train_mask], y[train_mask], Xdesc[test_mask], y[test_mask]` (and the
fingerprint equivalents) into a fresh `results_dict`, then call
`leaderboard(results_dict)`.
</details>
''')]

C += exercise(
    prompt=r'''
Implement `official_leaderboard()` below.
''',
    solution=r'''
def official_leaderboard():
    """Leaderboard (Section 4 style) evaluated on the Section-5 cluster split."""
    Xd_tr, Xd_te = Xdesc[train_mask], Xdesc[test_mask]
    Xf_tr, Xf_te = Xfp[train_mask], Xfp[test_mask]
    ytr_c, yte_c = y[train_mask], y[test_mask]

    res = {
        ("Baseline (mean)", "-"): evaluate(DummyRegressor(strategy="mean"), Xd_tr, ytr_c, Xd_te, yte_c),
        ("Linear", "descriptors"): evaluate(LinearRegression(), Xd_tr, ytr_c, Xd_te, yte_c),
        ("Ridge (alpha=1)", "fingerprints"): evaluate(Ridge(alpha=1.0), Xf_tr, ytr_c, Xf_te, yte_c),
        ("Random forest", "descriptors"): evaluate(
            RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1), Xd_tr, ytr_c, Xd_te, yte_c),
        ("Random forest", "fingerprints"): evaluate(
            RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1), Xf_tr, ytr_c, Xf_te, yte_c),
    }
    return leaderboard(res)

official = official_leaderboard()
official.round(3)
''',
    scaffold=r'''
def official_leaderboard():
    """Leaderboard (Section 4 style) evaluated on the Section-5 cluster split."""
    # YOUR CODE HERE: build a results dict with the same 5 (model, representation)
    # keys as Section 4, evaluated on train_mask/test_mask, then call leaderboard()
    ...

official = official_leaderboard()
official.round(3)
''',
    check=r'''
assert len(official) == 5
assert official.iloc[0]["model"] == "Random forest" and official.iloc[0]["representation"] == "descriptors"
baseline_rmse = official.loc[official["model"] == "Baseline (mean)", "RMSE"].iloc[0]
linear_rmse = official.loc[official["model"] == "Linear", "RMSE"].iloc[0]
best_rmse = official["RMSE"].min()
assert linear_rmse < 0.85 * baseline_rmse       # linear clearly beats the floor
assert best_rmse < 0.85 * linear_rmse            # best model clearly beats linear
print("Mini-challenge OK")
''',
)

# ==========================================================================
C += [md(r'''
---
## Summary — what you learned

- Always start a model comparison with a **baseline** (predict the mean).
- **Linear regression fails** on wide, low-sample-size feature sets ($p \gg
  n$); **Ridge** or a tree-based model recovers a usable fit.
- **Random forests** handle both interpretable descriptors and sparse
  fingerprints without manual regularisation, and expose **feature
  importances** (read with care — not causal).
- A **leaderboard** is only meaningful with a fixed split, the same metrics,
  and (Week 03A) an honest sense of the split's own uncertainty.
- **Leakage**: a random split can overstate performance, especially for models
  that can "memorise" near-duplicate structures; a **Butina cluster split**
  gives a stricter, more honest estimate — and the size of the gap between the
  two is itself diagnostic of *how* a model is succeeding.

This is the last week to use `sklearn`'s classical models as the whole story:
Week 04 adds classification and unsupervised learning; Week 05 begins deep
learning.
''')]

C += [md(r'''
## Further reading (course source list only)

- dmol.pub *Regression and model assessment* (bias-variance, regularisation,
  AqSolDB worked example): <https://dmol.pub/ml/regression.html>
- Pat Walters, `regression_model.ipynb` (linear → tree-based leaderboard
  pattern): <https://colab.research.google.com/github/PatWalters/practical_cheminformatics_tutorials/blob/main/ml_models/regression_model.ipynb>
- Pat Walters, `cross_validation.ipynb` (grouped/clustered splitting to avoid
  leakage): <https://colab.research.google.com/github/PatWalters/practical_cheminformatics_tutorials/blob/main/ml_models/cross_validation.ipynb>
''')]

C += [md(r'''
## Attribution

Original teaching material for *AI for Chemistry*, adapted and rewritten from:

- **Practical Cheminformatics** (P. Walters) — the linear-to-tree-based
  leaderboard pattern (`regression_model.ipynb`) and the grouped-split leakage
  study (`cross_validation.ipynb`), here simplified to reuse Week 02B's own
  Butina clusters rather than a fresh repeated grouped-holdout study, and using
  `RandomForestRegressor` in place of LightGBM (not in `env/environment.yml`).
  MIT licence. <https://github.com/PatWalters/practical_cheminformatics_tutorials>
- **dmol.pub** — regularisation and model-assessment framing carried over from
  Week 03A. CC-BY 4.0. <https://dmol.pub>

Continues the ESOL dataset and Butina clusters built in Week 02B. No verbatim
text is reproduced from the sources above.
''')]

build(__file__, "week03_b_regression-model-assessment", C)
