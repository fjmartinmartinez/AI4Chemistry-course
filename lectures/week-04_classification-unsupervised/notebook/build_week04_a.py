"""Build week 04 session A notebooks (classification + unsupervised learning).

    python lectures/week-04_classification-unsupervised/notebook/build_week04_a.py
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "_build"))
from nbbuild import build, checkpoint, code, exercise, md  # noqa: E402

C = []

# ==========================================================================
C += [md(r'''
# AI for Chemistry — Week 04, Session A
## Classification and unsupervised learning

**Course:** AI for Chemistry · **Session:** 04A (interleaved lecture + lab,
2.5 h) · **Runtime:** < 20 s, no GPU.

### Suggested timing (solo study, ~120 min)

| Section | Topic | Time |
|--------:|-------|-----:|
| 0 | Setup: load and clean BBBP (extends Week 02's checklist) | 15 min |
| 1 | Classification framing: sigmoid, log-odds, cross-entropy | 18 min |
| 2 | Metrics: confusion matrix, precision/recall/F1, ROC-AUC | 20 min |
| 3 | Class imbalance and `class_weight="balanced"` | 15 min |
| 4 | Unsupervised: PCA and k-means vs the true label | 18 min |
| 5 | The kernel idea (qualitative) | 8 min |
| 6 | Exercises (4) | 26 min |

### Learning objectives
1. Explain classification as sigmoid + log-odds + cross-entropy loss, and fit
   logistic regression. *(LO4)*
2. Compute a confusion matrix, accuracy, precision, recall, F1 and ROC-AUC,
   and explain why accuracy alone misleads under class imbalance. *(LO4)*
3. Use `class_weight="balanced"` and read its precision/recall trade-off. *(LO4)*
4. Run PCA and k-means on a labelled dataset and quantify how well the
   unsupervised clusters agree with the true label (adjusted Rand index). *(LO4)*
5. State, qualitatively, what a kernel is (no implementation this week). *(LO4)*

### Prerequisites — before this notebook you should be able to
- Week 03: train/test split, standardisation, random forest, regularisation.
- Week 02: RDKit descriptors, canonical SMILES, dataframe cleaning.

### How to use this notebook (solo study)
Read, run, check each **✅** cell in order. This is the **solutions**
notebook.

### Dataset — BBBP (blood-brain barrier penetration)
2050 compounds from MoleculeNet, each labelled `p_np`: **1** if the compound
is known to cross the blood-brain barrier, **0** if not. About 76% are
labelled permeable — our first real, moderately imbalanced classification
task.
''')]

# ==========================================================================
C += [md(r'''
---
## 0. Setup: load and clean BBBP

BBBP needs one cleaning step Week 02's ESOL did not: several entries are
**salts** — a counter-ion written as a separate SMILES fragment joined by a
dot, e.g. `"[Cl].CC(C)NCC(O)COc1cccc2ccccc12"`. Computing descriptors on the
whole thing would count the chloride's mass and charge as if they were part of
the drug. The fix is to **keep the largest fragment** ("desalting") before
canonicalising, then continue with Week 02's checklist unchanged: drop
unparseable rows, deduplicate by structure.

**What to look for:** row counts falling `2050 -> 2039 -> 1966` (parse
failures, then duplicate structures); `Xdesc: (1966, 7)`.
''')]

C += [code(r'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Draw
RDLogger.DisableLog("rdApp.*")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, roc_curve, confusion_matrix)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

RNG = np.random.default_rng(0xC0FFEE)

ROOT = Path.cwd()
while not (ROOT / "sources").is_dir() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
DATA = ROOT / "sources" / "datasets"
FIGDIR = ROOT / "lectures" / "week-04_classification-unsupervised" / "slides" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

def largest_fragment_smiles(smiles):
    """Canonical SMILES of the biggest disconnected fragment ('desalting')."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    biggest = max(frags, key=lambda m: m.GetNumAtoms())
    return Chem.MolToSmiles(biggest)

raw = pd.read_csv(DATA / "BBBP.csv")
n0 = len(raw)
raw["canonical_smiles"] = raw["smiles"].apply(largest_fragment_smiles)
bbbp = raw[raw["canonical_smiles"].notna()]
n1 = len(bbbp)
bbbp = bbbp.drop_duplicates("canonical_smiles").reset_index(drop=True)
n2 = len(bbbp)
print(f"rows: {n0} -> parse {n1} -> dedupe {n2}")
print("class balance (p_np):")
print(bbbp["p_np"].value_counts(normalize=True).round(3))
''')]

C += [code(r'''
DESCRIPTORS = {
    "MolWt": Descriptors.MolWt, "MolLogP": Descriptors.MolLogP,
    "TPSA": Descriptors.TPSA, "NumHDonors": Descriptors.NumHDonors,
    "NumHAcceptors": Descriptors.NumHAcceptors,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    "NumAromaticRings": Descriptors.NumAromaticRings,
}
desc_names = list(DESCRIPTORS)
mols = [Chem.MolFromSmiles(s) for s in bbbp["canonical_smiles"]]
Xdesc = np.array([[fn(m) for fn in DESCRIPTORS.values()] for m in mols], dtype=float)
y = bbbp["p_np"].to_numpy()
print("Xdesc:", Xdesc.shape, "y:", y.shape)
''')]

C += [md(r'''
> **Common errors — Setup**
> - Forgetting to desalt before deduplicating — `"[Na].CC(=O)O"` and
>   `"[Cl].CC(=O)O"` are the *same* drug with different counter-ions and would
>   otherwise be counted as two different structures.
> - `Chem.GetMolFrags(mol)` without `asMols=True` returns atom-index tuples,
>   not molecules — you then cannot call `.GetNumAtoms()` on them directly.
''')]

C += checkpoint(
    "Section 0",
    check=r'''
assert n0 == 2050 and n1 == 2039 and n2 == 1966
assert Xdesc.shape == (1966, 7)
assert 0.7 < y.mean() < 0.8
print("Section 0 OK")
''',
    expected="Prints `Section 0 OK`. 2050 -> 2039 -> 1966 rows; about 76% of "
    "molecules are labelled permeable.",
    questions=r'''
1. Why keep the *largest* fragment rather than, say, the first one listed in
   the SMILES?
2. This desalting step changes descriptor values but not the class label.
   Why is that the right choice here?
''',
    answers=r'''
1. In a salt, the pharmacologically active species is essentially always the
   larger organic fragment; the counter-ion (Cl⁻, Na⁺, ...) is small and would
   almost never be the biggest fragment by atom count — a simple, robust
   heuristic that avoids needing a curated list of "known counter-ions".
2. The BBB-permeability label describes the *drug*, not the salt form it was
   measured as; removing the counter-ion's contribution to MW/logP/etc. makes
   the descriptors describe the same entity the label refers to.
''',
)

# ==========================================================================
# 1. Classification framing
# ==========================================================================
C += [md(r'''
---
## 1. Classification framing: sigmoid, log-odds, cross-entropy

Regression predicts a real number; **classification** predicts a label
$y\in\{0,1\}$. Logistic regression still computes a linear score
$z = \vec w\cdot\vec x + b$ (exactly as in Week 03), then squashes it into a
probability with the **sigmoid function**

$$\hat y = \sigma(z) = \frac{1}{1+e^{-z}}, \qquad z = \vec w \cdot \vec x + b.$$

$z$ is called the **logit** (log-odds): the decision boundary is $z=0$
($\hat y = 0.5$); $z>0$ predicts class 1. Instead of MSE, we minimise
**cross-entropy (log-loss)**:

$$L = -\frac{1}{N}\sum_i \bigl[y_i\log\hat y_i + (1-y_i)\log(1-\hat y_i)\bigr].$$

We standardise the 7 descriptors (Week 03's lesson: comparable scales matter)
and fit on an 80/20 **stratified** split — stratifying keeps the same
class balance in train and test, important with imbalance.

**What to look for:** training accuracy noticeably above the ~76% "always
permeable" baseline.
''')]

C += [code(r'''
X_train, X_test, y_train, y_test = train_test_split(
    Xdesc, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler().fit(X_train)
X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

clf = LogisticRegression(max_iter=1000).fit(X_train_s, y_train)
pred = clf.predict(X_test_s)
prob = clf.predict_proba(X_test_s)[:, 1]     # P(class = 1) for each test molecule

print("test accuracy:", round(accuracy_score(y_test, pred), 3))
print("always-permeable baseline:", round(y_test.mean(), 3))
''')]

C += [md(r'''
Visualise the decision boundary using the two descriptors most relevant to
brain penetration — **MolLogP** (lipophilicity) and **TPSA** (polar surface
area): low TPSA and moderate logP are the classic CNS-drug profile. We refit a
2-feature model just for this picture (the full 7-feature model above is what
we evaluate).

**What to look for:** permeable (1) compounds cluster at low TPSA; the fitted
boundary roughly follows the low-TPSA / moderate-logP region.
''')]

C += [code(r'''
i_logp, i_tpsa = desc_names.index("MolLogP"), desc_names.index("TPSA")
X2 = Xdesc[:, [i_logp, i_tpsa]]
clf2 = LogisticRegression(max_iter=1000).fit(StandardScaler().fit_transform(X2), y)

# build a grid, standardising with the same scaler used to fit clf2
scaler2 = StandardScaler().fit(X2)
xx, yy = np.meshgrid(np.linspace(X2[:, 0].min() - 1, X2[:, 0].max() + 1, 200),
                     np.linspace(X2[:, 1].min() - 10, X2[:, 1].max() + 10, 200))
grid_s = scaler2.transform(np.column_stack([xx.ravel(), yy.ravel()]))
zz = clf2.predict_proba(grid_s)[:, 1].reshape(xx.shape)

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.contourf(xx, yy, zz, levels=20, cmap="RdBu_r", alpha=0.6)
sc = ax.scatter(X2[:, 0], X2[:, 1], c=y, cmap="RdBu_r", s=8, edgecolor="k", linewidth=0.2)
ax.set_xlabel("MolLogP"); ax.set_ylabel("TPSA / A^2")
ax.set_title("BBB permeability: decision surface (2 descriptors)")
fig.tight_layout(); fig.savefig(FIGDIR / "bbbp_decision_boundary.png", dpi=200)
plt.show()
''')]

C += [md(r'''
> **Common errors — Section 1**
> - Fitting the scaler on the *2-feature* grid but predicting with the
>   *7-feature* model (or vice versa) — a scaler and model must always be
>   paired with the exact feature set they were fit on.
> - Reading `predict_proba(...)[:, 0]` when you meant class 1 — column 0 is
>   $P(y{=}0)$, column 1 is $P(y{=}1)$; always check `clf.classes_`.
''')]

C += checkpoint(
    "Section 1",
    check=r'''
assert prob.shape == y_test.shape
assert ((prob >= 0) & (prob <= 1)).all()
assert accuracy_score(y_test, pred) > y_test.mean()   # beats the trivial baseline
assert list(clf.classes_) == [0, 1]
assert (FIGDIR / "bbbp_decision_boundary.png").is_file()
print("Section 1 OK")
''',
    expected="Prints `Section 1 OK`. Every predicted probability is in [0,1]; "
    "test accuracy beats the always-permeable baseline.",
    questions=r'''
1. Why can beating "always predict permeable" in **accuracy** still be a weak
   claim here?
2. The decision boundary in the plot is a straight-ish line even though the
   background colouring looks curved. Why is that consistent with logistic
   regression being a *linear* classifier?
''',
    answers=r'''
1. With ~76% of the data already in the majority class, beating that baseline
   by a few points of accuracy could still mean doing badly on the *minority*
   class specifically (Section 2/3 makes this precise).
2. The boundary $\vec w\cdot\vec x + b = 0$ is a straight line in the original
   2-feature space; the *colour* shading shows $\sigma(z)$, a smooth
   S-shaped function of the (linear) distance to that line, which looks
   curved even though the underlying boundary is not.
''',
)

# ==========================================================================
# 2. Metrics under class imbalance
# ==========================================================================
C += [md(r'''
---
## 2. Metrics: confusion matrix, precision/recall/F1, ROC-AUC

A **confusion matrix** cross-tabulates true vs predicted class. From its four
counts (TP, FP, TN, FN):

$$\text{accuracy}=\frac{TP+TN}{N}, \quad
  \text{precision}=\frac{TP}{TP+FP}, \quad
  \text{recall}=\frac{TP}{TP+FN}, \quad
  F_1=\frac{2\,\text{precision}\cdot\text{recall}}{\text{precision}+\text{recall}}.$$

The **ROC curve** plots true-positive rate against false-positive rate as the
decision threshold sweeps from 0 to 1; **AUC** is the area underneath —
threshold-independent, unlike accuracy/precision/recall/F1.

**What to look for:** recall for class **1** (permeable, the majority) is high;
recall for class **0** (non-permeable, the minority) is much lower — the model
is quietly bad at exactly the harder, rarer case, even though overall accuracy
looked fine.
''')]

C += [code(r'''
cm = confusion_matrix(y_test, pred)
print("confusion matrix [rows=true 0/1, cols=pred 0/1]:\n", cm)

for cls in [0, 1]:
    print(f"class {cls}:  precision {precision_score(y_test, pred, pos_label=cls):.3f}"
          f"  recall {recall_score(y_test, pred, pos_label=cls):.3f}")

print("F1 (class 1):", round(f1_score(y_test, pred), 3))
print("ROC-AUC:", round(roc_auc_score(y_test, prob), 3))
''')]

C += [code(r'''
fpr, tpr, thresholds = roc_curve(y_test, prob)
fig, ax = plt.subplots(figsize=(4.5, 4.5))
ax.plot(fpr, tpr, label=f"logistic regression (AUC={roc_auc_score(y_test, prob):.2f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="random guessing")
ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
ax.set_title("ROC curve — BBBP"); ax.legend()
ax.set_aspect("equal")
fig.tight_layout(); fig.savefig(FIGDIR / "roc_curve.png", dpi=200)
plt.show()
''')]

C += [md(r'''
> **Common errors — Section 2**
> - Quoting "precision" or "recall" without saying **for which class** — the
>   two classes almost always have different values under imbalance, as here.
> - `roc_auc_score` needs **probabilities** (`predict_proba`), not hard
>   predictions (`predict`) — passing 0/1 predictions gives a degenerate,
>   uninformative curve.
> - Confusion matrix row/column order: `sklearn` orders by sorted label
>   (`0` then `1`); always check, do not assume.
''')]

C += checkpoint(
    "Section 2",
    check=r'''
assert cm.sum() == len(y_test)
recall_1 = recall_score(y_test, pred, pos_label=1)
recall_0 = recall_score(y_test, pred, pos_label=0)
assert recall_1 > 0.85                    # majority class: high recall
assert recall_0 < recall_1 - 0.2          # minority class: markedly lower recall
assert 0.75 < roc_auc_score(y_test, prob) < 0.95
assert (FIGDIR / "roc_curve.png").is_file()
print("Section 2 OK")
''',
    expected="Prints `Section 2 OK`. Recall for the permeable class (1) is "
    "above 0.85; recall for the non-permeable class (0) is at least 0.2 "
    "lower — the imbalance gap made concrete.",
    questions=r'''
1. A colleague reports "91% accuracy" for this task and nothing else. What two
   numbers would you ask for before trusting the model?
2. Why is ROC-AUC unaffected by *which* threshold you eventually deploy at,
   while precision/recall/F1 are not?
''',
    answers=r'''
1. Per-class recall (or the full confusion matrix) — accuracy alone cannot
   distinguish "good at everything" from "good only at the majority class",
   which is exactly the failure mode we just measured.
2. ROC-AUC integrates performance over **every possible threshold**
   simultaneously (it is the probability a random positive ranks above a
   random negative); precision/recall/F1 are each evaluated at one specific
   threshold (0.5, by default), so changing the threshold changes them but
   not the AUC.
''',
)

# ==========================================================================
# 3. class_weight="balanced"
# ==========================================================================
C += [md(r'''
---
## 3. Fighting imbalance: `class_weight="balanced"`

`class_weight="balanced"` reweights the cross-entropy loss inversely to class
frequency, so mistakes on the rare class cost more during fitting. It does not
add or remove data; it changes what the optimiser is asked to prioritise.

**What to look for:** recall for the minority class (0) rises substantially;
recall for the majority class (1) falls somewhat — a **trade-off**, not a free
lunch. ROC-AUC (threshold-independent) barely moves, because the ranking of
predictions has not fundamentally changed, only where we currently cut it.
''')]

C += [code(r'''
clf_bal = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X_train_s, y_train)
pred_bal = clf_bal.predict(X_test_s)
prob_bal = clf_bal.predict_proba(X_test_s)[:, 1]

for name, p in [("plain", pred), ("balanced", pred_bal)]:
    r1 = recall_score(y_test, p, pos_label=1)
    r0 = recall_score(y_test, p, pos_label=0)
    print(f"{name:9s}: recall(1)={r1:.3f}  recall(0)={r0:.3f}  "
          f"AUC={roc_auc_score(y_test, prob if name=='plain' else prob_bal):.3f}")
''')]

C += [md(r'''
> **Common errors — Section 3**
> - Expecting `class_weight="balanced"` to improve *every* metric — it
>   reshapes the trade-off, it does not create information the model did not
>   have. Accuracy can even drop, deliberately.
> - Applying `class_weight` and *also* manually oversampling the minority
>   class without realising you are compounding the correction — pick one.
''')]

C += checkpoint(
    "Section 3",
    check=r'''
recall0_plain = recall_score(y_test, pred, pos_label=0)
recall0_bal = recall_score(y_test, pred_bal, pos_label=0)
recall1_plain = recall_score(y_test, pred, pos_label=1)
recall1_bal = recall_score(y_test, pred_bal, pos_label=1)
assert recall0_bal > recall0_plain + 0.1     # meaningful minority-class gain
assert recall1_bal < recall1_plain            # majority-class recall gives some back
assert abs(roc_auc_score(y_test, prob_bal) - roc_auc_score(y_test, prob)) < 0.05
print("Section 3 OK")
''',
    expected="Prints `Section 3 OK`. Balanced weighting raises recall(0) by "
    "more than 0.1 while lowering recall(1); AUC changes only slightly.",
    questions=r'''
1. For a real BBB-permeability screen, would you rather have high recall(1) or
   high recall(0)? Justify from the consequence of each kind of mistake.
2. Why does AUC change so little compared to the recall numbers?
''',
    answers=r'''
1. It depends on the use: if you are screening candidates *for* CNS drugs and
   want to avoid discarding a permeable lead, prioritise recall(1); if you are
   screening *against* unwanted CNS exposure (e.g. a peripheral-only drug),
   missing a permeable compound (low recall(1) tolerance) is dangerous, so you
   would prioritise catching class 1 instead — the "right" trade-off is a
   domain decision, not a purely statistical one.
2. `class_weight` changes the fitted coefficients only modestly (it reweights
   the loss, it does not add data), so the model's *ranking* of molecules by
   predicted probability — what AUC measures — is largely preserved; what
   shifts more is the effective operating point relative to the 0.5 cutoff.
''',
)

# ==========================================================================
# 4. Unsupervised: PCA and k-means vs the true label
# ==========================================================================
C += [md(r'''
---
## 4. Unsupervised learning: PCA and k-means vs the true label

Unlike Week 02's fingerprint PCA, our 7-descriptor block gives PCA a lot to
work with here (~78% of variance in 2 components). But PCA and k-means never
see the `p_np` label — a fair question is: **does the chemistry that
separates permeable from non-permeable compounds also happen to be the
dominant unsupervised structure in the data?**

We measure agreement between `KMeans(n_clusters=2)` and the true label with the
**adjusted Rand index (ARI)**: 1 = perfect agreement, 0 = no better than
chance, negative = worse than chance.

**What to look for:** PCA visually separates the two classes somewhat, but
ARI is well below 1 — unsupervised clustering finds *a* pattern in the data,
not necessarily *the* clinically relevant one.
''')]

C += [code(r'''
Xdesc_std = StandardScaler().fit_transform(Xdesc)
pca = PCA(n_components=2, random_state=0).fit(Xdesc_std)
Z = pca.transform(Xdesc_std)
print("explained variance (2 PCs):", pca.explained_variance_ratio_.sum().round(3))

km = KMeans(n_clusters=2, random_state=0, n_init=10).fit(Xdesc_std)
ari = adjusted_rand_score(y, km.labels_)
print("adjusted Rand index (k-means vs p_np):", round(ari, 3))
''')]

C += [code(r'''
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
axes[0].scatter(Z[:, 0], Z[:, 1], c=y, cmap="RdBu_r", s=8)
axes[0].set_title("coloured by true label (p_np)")
axes[1].scatter(Z[:, 0], Z[:, 1], c=km.labels_, cmap="Set2", s=8)
axes[1].set_title(f"coloured by k-means cluster (ARI={ari:.2f})")
for ax in axes:
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
fig.tight_layout(); fig.savefig(FIGDIR / "pca_chemical_space_bbbp.png", dpi=200)
plt.show()
''')]

C += [md(r'''
> **Common errors — Section 4**
> - Using plain accuracy to compare cluster labels to true labels — cluster
>   `0`/`1` numbering is arbitrary (k-means does not know which cluster
>   "means" permeable), so accuracy would be meaningless; ARI/adjusted mutual
>   information handle the label-permutation problem correctly.
> - Concluding "clustering does not work" from a modest ARI — it is working
>   exactly as designed: finding variance structure, which need not match a
>   specific external label.
''')]

C += checkpoint(
    "Section 4",
    check=r'''
assert Z.shape == (len(y), 2)
assert pca.explained_variance_ratio_.sum() > 0.6
assert -1 <= ari <= 1
assert ari < 0.6           # clusters are informative but far from a perfect match
assert (FIGDIR / "pca_chemical_space_bbbp.png").is_file()
print("Section 4 OK")
''',
    expected="Prints `Section 4 OK`. First two PCs explain > 60% of variance; "
    "ARI is positive but well short of 1.",
    questions=r'''
1. If ARI had come out near 0, what would that say about the *chemistry*
   underlying BBB permeability?
2. Suppose you ran k-means with `n_clusters=5` instead of 2. Would comparing
   it to the binary label with ARI still make sense?
''',
    answers=r'''
1. It would suggest permeability is not the dominant axis of variation among
   these 7 descriptors — i.e. whatever makes molecules differ most in
   MW/logP/TPSA/etc. space is not simply "permeable vs not", and a richer
   representation or different features would be needed to expose that
   structure unsupervised.
2. Yes — ARI compares *partitions*, not label values, so it handles a
   different number of clusters than labels gracefully (it would ask whether
   the 5 clusters, whatever their sizes, tend to be pure with respect to the
   2 true classes), though a 2-vs-5 comparison is naturally capped below 1.
''',
)

# ==========================================================================
# 5. The kernel idea (qualitative)
# ==========================================================================
C += [md(r'''
---
## 5. The kernel idea — qualitative only

*(This section is deliberately conceptual — `syllabus.md` scopes kernel
learning as qualitative for this course. No code to write or run here.)*

Every model so far has needed an explicit feature vector $\vec x$ (7
descriptors, or Week 02's 2048-bit fingerprint). A **kernel** $k(\vec x_i,
\vec x_j)$ instead measures the **similarity** between two data points
directly, without ever materialising $\vec x$. You already know one: the
**Tanimoto similarity** between two fingerprints (Week 02B) is exactly a
molecular kernel.

The "kernel trick" is that many models (support vector machines, Gaussian
processes, kernel ridge regression) can be written using **only** pairwise
similarities $k(\vec x_i,\vec x_j)$ between training points, never the raw
features. This matters when:

* the natural feature space is awkward or infinite-dimensional to write down
  explicitly (some graph or quantum-chemical similarity measures), but a
  pairwise similarity is easy to define and compute;
* you already have a domain-appropriate similarity measure (Tanimoto on
  fingerprints, or an RBF kernel $k(\vec x_i,\vec x_j)=\exp(-\gamma\lVert
  \vec x_i-\vec x_j\rVert^2)$ on continuous descriptors) and would rather use
  that directly than engineer a new feature space.

We will not implement kernel methods in this course — `scikit-learn`'s `SVC`
and `KernelRidge` accept a `kernel="precomputed"` similarity matrix (e.g. a
Tanimoto matrix, exactly like the one you built for Butina clustering in
Week 02B) if you want to explore this independently.
''')]

# ==========================================================================
# 6. Exercises
# ==========================================================================
C += [md(r'''
---
## 6. Exercises
''')]

C += exercise(
    prompt=r'''
### Exercise 1 — metrics from scratch *(easy, ~12 min)*

Write `classification_metrics(y_true, y_pred)` returning a dict with
`accuracy`, `precision`, `recall`, `f1` for the **positive (1)** class,
computed from a manually built confusion matrix (NumPy only, no
`sklearn.metrics`). Check against Section 2's `clf` predictions.

<details><summary>Show hint</summary>

`tp = ((y_pred==1)&(y_true==1)).sum()`, similarly for `fp`, `fn`, `tn`.
`precision = tp/(tp+fp)`, `recall = tp/(tp+fn)`,
`f1 = 2*precision*recall/(precision+recall)`.
</details>
''',
    solution=r'''
def classification_metrics(y_true, y_pred):
    """accuracy/precision/recall/F1 for class 1, from a manual confusion matrix."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

metrics_manual = classification_metrics(y_test, pred)
print(metrics_manual)
''',
    scaffold=r'''
def classification_metrics(y_true, y_pred):
    """accuracy/precision/recall/F1 for class 1, from a manual confusion matrix."""
    # YOUR CODE HERE
    ...

metrics_manual = ...  # YOUR CODE HERE
''',
    check=r'''
assert abs(metrics_manual["accuracy"] - accuracy_score(y_test, pred)) < 1e-9
assert abs(metrics_manual["precision"] - precision_score(y_test, pred)) < 1e-9
assert abs(metrics_manual["recall"] - recall_score(y_test, pred)) < 1e-9
assert abs(metrics_manual["f1"] - f1_score(y_test, pred)) < 1e-9
print("Exercise 1 OK")
''',
)

C += exercise(
    prompt=r'''
### Exercise 2 — sweep `class_weight` finely *(medium, ~15 min)*

`class_weight` also accepts an explicit dict `{0: w0, 1: w1}`. Write
`recall_gap(weight_ratio)` that fits `LogisticRegression(class_weight={0:
weight_ratio, 1: 1.0})` and returns `recall(0) - recall(1)`. Evaluate it for
`weight_ratio` in `[1, 2, 4, 8]` and confirm the gap moves from negative
(favouring class 1) towards positive (favouring class 0) as `weight_ratio`
grows.

<details><summary>Show hint</summary>

Reuse `X_train_s, y_train, X_test_s, y_test` from Setup/Section 1.
</details>
''',
    solution=r'''
def recall_gap(weight_ratio):
    """recall(0) - recall(1) for LogisticRegression with class_weight {0: r, 1: 1}."""
    m = LogisticRegression(max_iter=1000, class_weight={0: weight_ratio, 1: 1.0})
    m.fit(X_train_s, y_train)
    p = m.predict(X_test_s)
    return recall_score(y_test, p, pos_label=0) - recall_score(y_test, p, pos_label=1)

gaps = [recall_gap(w) for w in [1, 2, 4, 8]]
print("weight_ratio 1,2,4,8 -> recall(0)-recall(1):", np.round(gaps, 3))
''',
    scaffold=r'''
def recall_gap(weight_ratio):
    """recall(0) - recall(1) for LogisticRegression with class_weight {0: r, 1: 1}."""
    # YOUR CODE HERE
    ...

gaps = ...  # YOUR CODE HERE: [recall_gap(w) for w in [1, 2, 4, 8]]
''',
    check=r'''
assert len(gaps) == 4
assert gaps[0] < gaps[-1]                 # gap increases as class 0 is weighted more
assert gaps[-1] > 0                        # eventually favours the minority class
print("Exercise 2 OK")
''',
)

C += exercise(
    prompt=r'''
### Exercise 3 — a manual ROC point *(medium, ~12 min)*

Write `confusion_at_threshold(prob, y_true, threshold)` returning
`(fpr, tpr)` for a hand-chosen probability `threshold` (instead of the default
0.5). Confirm your `(fpr, tpr)` at `threshold=0.5` is close to the point on
`sklearn`'s `roc_curve` nearest to threshold 0.5.

<details><summary>Show hint</summary>

`pred_t = (prob >= threshold).astype(int)`; then
`tpr = recall_score(y_true, pred_t, pos_label=1)`;
`fpr = ((pred_t==1)&(y_true==0)).sum() / (y_true==0).sum()`.
</details>
''',
    solution=r'''
def confusion_at_threshold(prob, y_true, threshold):
    """(fpr, tpr) of thresholding `prob` at `threshold`."""
    pred_t = (prob >= threshold).astype(int)
    tpr = recall_score(y_true, pred_t, pos_label=1)
    neg = (y_true == 0)
    fpr = ((pred_t == 1) & neg).sum() / neg.sum()
    return fpr, tpr

fpr_05, tpr_05 = confusion_at_threshold(prob, y_test, 0.5)
print("at threshold 0.5:", round(fpr_05, 3), round(tpr_05, 3))
''',
    scaffold=r'''
def confusion_at_threshold(prob, y_true, threshold):
    """(fpr, tpr) of thresholding `prob` at `threshold`."""
    # YOUR CODE HERE
    ...

fpr_05, tpr_05 = ...  # YOUR CODE HERE: confusion_at_threshold(prob, y_test, 0.5)
''',
    check=r'''
nearest = np.argmin(np.abs(thresholds - 0.5))
assert abs(fpr_05 - fpr[nearest]) < 0.05
assert abs(tpr_05 - tpr[nearest]) < 0.05
assert confusion_at_threshold(prob, y_test, 0.0) == (1.0, 1.0)   # predict all positive
print("Exercise 3 OK")
''',
)

C += exercise(
    prompt=r'''
### Exercise 4 — how many clusters agree best with the label? *(harder, ~18 min)*

Write `best_k_by_ari(k_values)` that fits `KMeans(n_clusters=k)` for each `k`
and returns the `k` with the highest ARI against `y`. Is more clusters always
better?

<details><summary>Show hint</summary>

Loop `k_values`, fit `KMeans(n_clusters=k, random_state=0, n_init=10)` on
`Xdesc_std`, compute `adjusted_rand_score(y, km.labels_)`, track the max.
</details>
''',
    solution=r'''
def best_k_by_ari(k_values):
    """k (from k_values) whose KMeans clustering has the highest ARI vs y."""
    scores = []
    for k in k_values:
        km_k = KMeans(n_clusters=k, random_state=0, n_init=10).fit(Xdesc_std)
        scores.append(adjusted_rand_score(y, km_k.labels_))
    return k_values[int(np.argmax(scores))], scores

best_k, ari_scores = best_k_by_ari([2, 3, 4, 5, 8])
print("k -> ARI:", list(zip([2, 3, 4, 5, 8], np.round(ari_scores, 3))))
print("best k:", best_k)
''',
    scaffold=r'''
def best_k_by_ari(k_values):
    """k (from k_values) whose KMeans clustering has the highest ARI vs y."""
    # YOUR CODE HERE
    ...

best_k, ari_scores = ...  # YOUR CODE HERE: best_k_by_ari([2, 3, 4, 5, 8])
''',
    check=r'''
assert best_k in [2, 3, 4, 5, 8]
assert len(ari_scores) == 5
assert all(-1 <= s <= 1 for s in ari_scores)
print("Exercise 4 OK")
''',
)

# ==========================================================================
C += [md(r'''
---
## Summary — what you learned

- **Classification framing**: sigmoid, log-odds, cross-entropy — the same
  linear score $\vec w\cdot\vec x+b$ as regression, differently interpreted.
- **Metrics**: confusion matrix, precision/recall/F1 **per class**, ROC-AUC —
  and why accuracy alone hides imbalance.
- **`class_weight="balanced"`** and threshold choice trade recall between
  classes; there is no free lunch, only an explicit decision about which
  errors matter more.
- **PCA + k-means vs a true label**: unsupervised structure need not align
  with the label you care about; **ARI** quantifies the (mis)match honestly.
- **Kernels** (qualitative): similarity in place of features; Tanimoto is a
  molecular kernel you already use.
- New cleaning step: **desalting** (largest fragment) before canonicalising.

Week 04B runs a full classifier comparison on a genuinely imbalanced
toxicity task and tunes a threshold against a stated requirement.
''')]

C += [md(r'''
## Further reading (course source list only)

- dmol.pub *Classification* (sigmoid, cross-entropy, confusion matrix,
  ROC-AUC, class imbalance): <https://dmol.pub/ml/classification.html>
- dmol.pub *Kernel learning* (qualitative background for Section 5):
  <https://dmol.pub/ml/kernel.html>
- Alternative: ML-in-chemistry-101:
  <https://github.com/BingqingCheng/ML-in-chemistry-101>
''')]

C += [md(r'''
## Attribution

Original teaching material for *AI for Chemistry*, adapted and rewritten from:

- **dmol.pub** (A. White) — the sigmoid/cross-entropy/ROC framing and the
  discussion of class imbalance and reweighting; we substitute the BBBP
  dataset and the Week 02 7-descriptor block for dmol's own ClinTox + Mordred
  example to avoid a new dependency. CC-BY 4.0. <https://dmol.pub>
- **TeachOpenCADD** (Volkamer Lab), talktorial T007 — the
  precision/recall/specificity/AUC comparison pattern. CC-BY 4.0.
  <https://github.com/volkamerlab/teachopencadd>

Dataset: BBBP (MoleculeNet), downloaded from the same DeepChem mirror as
Week 02's ESOL, checksum-pinned. No verbatim text is reproduced from the
sources above.
''')]

build(__file__, "week04_a_classification-unsupervised", C)
