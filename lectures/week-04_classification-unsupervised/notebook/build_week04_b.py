"""Build week 04 session B notebooks (toxicity classification mini-competition).

    python lectures/week-04_classification-unsupervised/notebook/build_week04_b.py
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "_build"))
from nbbuild import build, checkpoint, code, exercise, md  # noqa: E402

C = []

# ==========================================================================
C += [md(r'''
# AI for Chemistry — Week 04, Session B
## Workshop: toxicity classification mini-competition

**Course:** AI for Chemistry · **Session:** 04B (hands-on workshop, 2.0 h) ·
**Runtime:** < 20 s, no GPU.

### Suggested timing (solo study, ~120 min)

| Section | Topic | Time |
|--------:|-------|-----:|
| 0 | Setup: load and clean Tox21, task SR-p53 | 15 min |
| 1 | Baseline: why 94% accuracy is worthless here | 12 min |
| 2 | The mini-competition: logistic regression vs random forest | 20 min |
| 3 | Precision-recall trade-off; tuning a threshold to a requirement | 22 min |
| 4 | Error analysis: what does the model miss? | 20 min |
| 5 | Exercises (3) + mini-challenge | 31 min |

### Learning objectives
1. Recognise a **severely** imbalanced classification task and explain why
   accuracy is actively misleading on it. *(LO4)*
2. Compare classifiers with threshold-independent metrics (ROC-AUC, average
   precision) *and* threshold-dependent ones (balanced accuracy, F1), and
   explain why they can disagree on which model is "best". *(LO4)*
3. Read a precision-recall curve and choose an operating threshold against a
   stated requirement (e.g. "catch 80% of toxic compounds"). *(LO4)*
4. Perform basic error analysis: inspect false negatives structurally rather
   than only quoting a metric. *(LO4, LO8)*

### Prerequisites — before this notebook you should be able to
- Week 04A: confusion matrix, precision/recall/F1, ROC-AUC, `class_weight`.
- Week 03: leaderboard pattern, random forest.

### How to use this notebook (solo study)
Read, run, check each **✅** cell in order. This is the **solutions**
notebook.

### Dataset — Tox21, task SR-p53
Tox21 is a 12-assay toxicology screen (MoleculeNet). We use one assay,
**SR-p53** (activation of the p53 stress-response pathway, a marker of
genotoxic stress): about **6%** of tested compounds are positive. This is a
much harder imbalance than Week 04A's BBBP (76/24) — exactly the regime real
toxicology screening lives in.
''')]

# ==========================================================================
C += [md(r'''
---
## 0. Setup: load and clean Tox21 (task SR-p53)

Tox21 is **multi-task**: 12 assay columns, most rows missing most labels (not
every compound was tested in every assay). We select one task and drop rows
where *that* task is missing — a different missingness pattern from Week 02's
"drop rows with no target", but the same principle. We reuse Week 04A's
desalting step (Tox21 also contains salts) before deduplicating.

**What to look for:** row counts falling as we go
`(rows with a SR-p53 label) -> (parseable) -> (deduplicated)`; a **positive
rate around 6%**.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              balanced_accuracy_score, f1_score,
                              precision_recall_curve, confusion_matrix)

RNG = np.random.default_rng(0xC0FFEE)
TASK = "SR-p53"

ROOT = Path.cwd()
while not (ROOT / "sources").is_dir() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
DATA = ROOT / "sources" / "datasets"
FIGDIR = ROOT / "lectures" / "week-04_classification-unsupervised" / "slides" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

def largest_fragment_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    return Chem.MolToSmiles(max(frags, key=lambda m: m.GetNumAtoms()))

raw = pd.read_csv(DATA / "tox21.csv.gz", compression="gzip")
labelled = raw[["smiles", TASK]].dropna(subset=[TASK]).reset_index(drop=True)
n0 = len(labelled)

labelled["canonical_smiles"] = labelled["smiles"].apply(largest_fragment_smiles)
tox = labelled[labelled["canonical_smiles"].notna()]
n1 = len(tox)
tox = tox.drop_duplicates("canonical_smiles").reset_index(drop=True)
n2 = len(tox)

print(f"rows with a {TASK} label: {n0} -> parse {n1} -> dedupe {n2}")
print(f"positive rate: {tox[TASK].mean():.3f}")
''')]

C += [code(r'''
DESCRIPTORS = {
    "MolWt": Descriptors.MolWt, "MolLogP": Descriptors.MolLogP,
    "TPSA": Descriptors.TPSA, "NumHDonors": Descriptors.NumHDonors,
    "NumHAcceptors": Descriptors.NumHAcceptors,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    "NumAromaticRings": Descriptors.NumAromaticRings,
}
mols = [Chem.MolFromSmiles(s) for s in tox["canonical_smiles"]]
Xdesc = np.array([[fn(m) for fn in DESCRIPTORS.values()] for m in mols], dtype=float)
y = tox[TASK].to_numpy().astype(int)

idx = np.arange(len(mols))
idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler().fit(Xdesc[idx_train])
X_train_s = scaler.transform(Xdesc[idx_train])
X_test_s = scaler.transform(Xdesc[idx_test])
y_train, y_test = y[idx_train], y[idx_test]
print("Xdesc:", Xdesc.shape, " n_train", len(idx_train), " n_test", len(idx_test))
''')]

C += [md(r'''
> **Common errors — Setup**
> - Dropping rows with **any** missing task before selecting `SR-p53` — that
>   throws away compounds that *do* have a SR-p53 label just because some
>   *other* assay is missing. Select the task column first, then drop only
>   where *that* column is missing.
> - Forgetting `stratify=y` in `train_test_split` — with 6% positives, an
>   unlucky unstratified split could leave the test set with very few (or
>   zero) positive examples.
''')]

C += checkpoint(
    "Section 0",
    check=r'''
assert 0.055 < tox[TASK].mean() < 0.075
assert Xdesc.shape[1] == 7
assert abs(y_train.mean() - y_test.mean()) < 0.02   # stratification worked
print("Section 0 OK")
''',
    expected="Prints `Section 0 OK`. Positive rate is roughly 6%; train and "
    "test have almost the same positive rate.",
    questions=r'''
1. Why must a severely imbalanced task be split with `stratify=y` far more
   carefully than Week 03's roughly-continuous solubility target?
2. `SR-p53` had 68% of Tox21's raw rows labelled. What does the *pattern* of
   which compounds got tested (rather than the test result itself) risk
   introducing?
''',
    answers=r'''
1. A continuous target's mean is fairly stable across any reasonably sized
   random split; a rare binary event (6%) can easily be over- or
   under-represented by chance in a small test set, making one unlucky split
   swing every metric — stratification removes that source of noise.
2. **Selection bias**: compounds are often chosen for a specific toxicology
   assay because of prior structural alerts or programme interest, not at
   random from "all chemical space" — so the model may learn "what medicinal
   chemists suspected", not toxicity in general.
''',
)

# ==========================================================================
# 1. Baseline
# ==========================================================================
C += [md(r'''
---
## 1. Baseline: why accuracy is worthless here

`DummyClassifier(strategy="most_frequent")` always predicts "not toxic". At a
6% positive rate that is right about **94% of the time** — an impressive-
looking accuracy that identifies **zero** toxic compounds.

**What to look for:** accuracy ≈ 0.94, but ROC-AUC = 0.5 (no better than
random ranking) and F1 = 0 (it never predicts the positive class at all).
''')]

C += [code(r'''
dummy = DummyClassifier(strategy="most_frequent").fit(X_train_s, y_train)
dummy_pred = dummy.predict(X_test_s)
dummy_prob = dummy.predict_proba(X_test_s)[:, 1]

print("accuracy:", round((dummy_pred == y_test).mean(), 3))
print("ROC-AUC :", round(roc_auc_score(y_test, dummy_prob), 3))
print("F1      :", round(f1_score(y_test, dummy_pred), 3))
print("average precision (AP):", round(average_precision_score(y_test, dummy_prob), 3))
''')]

C += [md(r'''
> **Common errors — Section 1**
> - Reporting accuracy as *the* headline number for an imbalanced task — by
>   the numbers above, a model that does *nothing useful* still looks 94%
>   "correct".
> - Average precision (AP) of a trivial/constant-score classifier equals the
>   **positive rate** — a useful sanity check, not a coincidence.
''')]

C += checkpoint(
    "Section 1",
    check=r'''
assert (dummy_pred == y_test).mean() > 0.9
assert abs(roc_auc_score(y_test, dummy_prob) - 0.5) < 1e-9
assert f1_score(y_test, dummy_pred) == 0.0
assert abs(average_precision_score(y_test, dummy_prob) - y_test.mean()) < 1e-9
print("Section 1 OK")
''',
    expected="Prints `Section 1 OK`. Baseline accuracy > 0.9 but ROC-AUC is "
    "exactly 0.5 and F1 is exactly 0; AP equals the positive rate.",
    questions=r'''
1. Why is the baseline's ROC-AUC *exactly* 0.5, not just "close to" 0.5?
2. If a screening team's only stated requirement were "maximise accuracy",
   what model would technically satisfy that, and why would you refuse to
   ship it?
''',
    answers=r'''
1. Every test molecule gets the *same* predicted probability, so there is no
   way to rank one as more likely positive than another — a tied ranking has
   AUC exactly 0.5 by the definition (probability a random positive outranks
   a random negative) regardless of the class balance.
2. The constant "always negative" classifier — it is the accuracy-maximiser
   whenever positives are the minority. Shipping it would mean the tool never
   flags a single toxic compound, which defeats the purpose of screening even
   though its accuracy number looks excellent.
''',
)

# ==========================================================================
# 2. The mini-competition
# ==========================================================================
C += [md(r'''
---
## 2. The mini-competition: logistic regression vs random forest

Both use `class_weight="balanced"` (Week 04A). We report **both** kinds of
metric: threshold-independent (ROC-AUC, average precision) and
threshold-dependent at the default cutoff 0.5 (balanced accuracy, F1).

**What to look for:** random forest has the **better** ROC-AUC and AP, but
**worse** balanced accuracy at threshold 0.5 than logistic regression — the
two kinds of metric can disagree about which model is "better", because they
answer different questions (ranking quality vs. one specific cutoff).
''')]

C += [code(r'''
def evaluate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)
    return {
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "ROC_AUC": roc_auc_score(y_test, prob),
        "AP": average_precision_score(y_test, prob),
        "F1": f1_score(y_test, pred),
    }, prob, pred

lr = LogisticRegression(max_iter=2000, class_weight="balanced")
rf = RandomForestClassifier(n_estimators=300, random_state=0,
                            class_weight="balanced", n_jobs=-1)

metrics_lr, prob_lr, pred_lr = evaluate(lr, X_train_s, y_train, X_test_s, y_test)
metrics_rf, prob_rf, pred_rf = evaluate(rf, X_train_s, y_train, X_test_s, y_test)

board = pd.DataFrame({"Logistic regression": metrics_lr, "Random forest": metrics_rf}).T
board.round(3)
''')]

C += [code(r'''
fig, ax = plt.subplots(figsize=(6, 3.5))
board[["ROC_AUC", "AP", "balanced_accuracy"]].plot.bar(ax=ax)
ax.set_ylabel("score"); ax.set_title("Tox21 SR-p53: model comparison")
ax.legend(loc="lower right"); ax.tick_params(axis="x", rotation=0)
fig.tight_layout(); fig.savefig(FIGDIR / "tox21_leaderboard.png", dpi=200)
plt.show()
''')]

C += [md(r'''
> **Common errors — Section 2**
> - Picking "the best model" from a single metric when the metrics disagree —
>   state *which* property you need (good ranking? good default-threshold
>   calls?) before picking.
> - Comparing `RandomForestClassifier`'s default threshold behaviour to
>   `LogisticRegression`'s as if 0.5 means the same thing for both — tree
>   ensembles' predicted probabilities are not calibrated the same way as a
>   logistic model's; Section 3 tunes the threshold explicitly instead of
>   trusting 0.5 for either.
''')]

C += checkpoint(
    "Section 2",
    check=r'''
assert metrics_rf["ROC_AUC"] > metrics_lr["ROC_AUC"]
assert metrics_rf["AP"] > metrics_lr["AP"]
assert metrics_rf["balanced_accuracy"] < metrics_lr["balanced_accuracy"]
assert (FIGDIR / "tox21_leaderboard.png").is_file()
print("Section 2 OK")
''',
    expected="Prints `Section 2 OK`. Random forest wins on ROC-AUC and AP; "
    "logistic regression wins on balanced accuracy at threshold 0.5.",
    questions=r'''
1. Given this pattern, what would you check before declaring random forest
   the winner?
2. Why might a tree ensemble's default 0.5 threshold be a particularly poor
   operating point after `class_weight="balanced"` reweighting?
''',
    answers=r'''
1. Whether random forest's *ranking* advantage (AUC/AP) survives at whatever
   threshold you will actually deploy at — Section 3 answers this directly by
   tuning both models' thresholds to a shared requirement rather than
   comparing them at an arbitrary default.
2. `class_weight` reweights the *training* loss/impurity criterion, not the
   output probability calibration; the forest's vote fractions do not
   automatically land at a threshold of 0.5 for the reweighted decision
   boundary, so 0.5 can be a poor default even though the underlying ranking
   (AUC/AP) is strong.
''',
)

# ==========================================================================
# 3. Precision-recall trade-off and threshold tuning
# ==========================================================================
C += [md(r'''
---
## 3. Precision-recall trade-off; tuning a threshold to a requirement

Under 6% positives, ROC curves look deceptively good (Section 2); the
**precision-recall curve** is the more honest picture for rare-event
screening: as recall rises, precision typically collapses.

**Screening requirement** (given, a realistic one): *catch at least 80% of
truly toxic compounds* (recall ≥ 0.8), accepting whatever precision that
costs. We find, for each model, the **highest** probability threshold that
still achieves recall ≥ 0.8 (the most precise point satisfying the
requirement).

**What to look for:** both models need a **low** threshold (well under 0.5) to
reach recall 0.8, and precision there is low (correctly flagging most toxic
compounds also means flagging many non-toxic ones) — the real cost of a
high-recall screen.
''')]

C += [code(r'''
def threshold_for_recall(y_true, prob, min_recall):
    """Highest probability threshold with recall >= min_recall (or None)."""
    precision, recall, thresh = precision_recall_curve(y_true, prob)
    ok = recall[:-1] >= min_recall           # thresh has one fewer entry than precision/recall
    if not ok.any():
        return None, None
    best = np.argmax(thresh[ok])              # highest threshold among those satisfying it
    return thresh[ok][best], precision[:-1][ok][best]

for name, prob in [("Logistic regression", prob_lr), ("Random forest", prob_rf)]:
    t, p = threshold_for_recall(y_test, prob, 0.8)
    print(f"{name:20s} threshold={t:.3f}  precision at recall>=0.8 = {p:.3f}")
''')]

C += [code(r'''
fig, ax = plt.subplots(figsize=(5.5, 4.5))
for name, prob in [("Logistic regression", prob_lr), ("Random forest", prob_rf)]:
    precision, recall, _ = precision_recall_curve(y_test, prob)
    ax.plot(recall, precision, label=f"{name} (AP={average_precision_score(y_test, prob):.2f})")
ax.axvline(0.8, color="k", ls="--", lw=1, label="requirement: recall >= 0.8")
ax.axhline(y_test.mean(), color="gray", ls=":", lw=1, label="random classifier")
ax.set_xlabel("recall"); ax.set_ylabel("precision")
ax.set_title("Tox21 SR-p53: precision-recall curve"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIGDIR / "tox21_pr_curve.png", dpi=200)
plt.show()
''')]

C += [md(r'''
> **Common errors — Section 3**
> - `precision_recall_curve` returns `precision`/`recall` with **one more**
>   entry than `thresholds` (the last point is the "threshold = +infinity"
>   corner); index them consistently (`[:-1]` on precision/recall), as above.
> - Treating "precision at recall 0.8" as fixed truth rather than a business
>   decision — a *different* recall requirement (0.5? 0.95?) gives a
>   different, equally valid operating point.
''')]

C += checkpoint(
    "Section 3",
    check=r'''
t_lr, p_lr = threshold_for_recall(y_test, prob_lr, 0.8)
t_rf, p_rf = threshold_for_recall(y_test, prob_rf, 0.8)
assert t_lr is not None and t_rf is not None
assert t_lr < 0.5 and t_rf < 0.5
assert 0.05 < p_lr < 0.3 and 0.05 < p_rf < 0.3
assert (FIGDIR / "tox21_pr_curve.png").is_file()
print("Section 3 OK")
''',
    expected="Prints `Section 3 OK`. Both models need thresholds well below "
    "0.5 to hit recall 0.8, with precision roughly 0.1-0.2 there.",
    questions=r'''
1. Precision of ~0.15 at the chosen threshold means roughly how many
   compounds must be tested/reviewed for every true toxic hit flagged?
2. Why does requiring **higher** recall inevitably push the threshold lower?
''',
    answers=r'''
1. Precision $\approx0.15$ means about 1 in 6-7 flagged compounds is truly
   toxic — a real, often acceptable cost in early screening, where a cheap
   confirmatory assay follows and missing true toxicity (Section 4) is far
   more costly than a follow-up test on a false alarm.
2. Lowering the threshold classifies more compounds as positive, which can
   only *keep or increase* the number of true positives caught (recall is
   monotonically non-decreasing as the threshold falls) — the price is that
   it also always keeps or increases false positives, which is exactly what
   drives precision down.
''',
)

# ==========================================================================
# 4. Error analysis
# ==========================================================================
C += [md(r'''
---
## 4. Error analysis: what does the model miss?

A leaderboard number never tells you *which* molecules are the problem. Using
random forest at the **default** threshold 0.5 (deliberately, to see the
model's "resting" behaviour before any tuning), we pull out the **false
negatives** — compounds the assay found genuinely toxic that the model called
safe — and look at them as structures, not just a count.

**What to look for:** a confusion matrix with a non-trivial false-negative
count; a grid of the missed structures — look for anything they visibly share
(reactive groups, quinones, halogenated rings) before concluding the model is
simply "wrong".
''')]

C += [code(r'''
cm = confusion_matrix(y_test, pred_rf)
print("confusion matrix [rows=true 0/1, cols=pred 0/1]:\n", cm)

false_negative_mask = (pred_rf == 0) & (y_test == 1)
false_positive_mask = (pred_rf == 1) & (y_test == 0)
print(f"false negatives (missed toxic compounds): {false_negative_mask.sum()}")
print(f"false positives (flagged but not toxic):  {false_positive_mask.sum()}")

fn_global_idx = idx_test[false_negative_mask]
fn_smiles = tox["canonical_smiles"].iloc[fn_global_idx].tolist()
''')]

C += [code(r'''
fn_mols = [Chem.MolFromSmiles(s) for s in fn_smiles[:8]]
grid = Draw.MolsToGridImage(fn_mols, molsPerRow=4, subImgSize=(200, 160), returnPNG=False)
grid.save(FIGDIR / "false_negatives_grid.png")
grid
''')]

C += [md(r'''
> **Common errors — Section 4**
> - Treating a handful of inspected structures as *proof* of a mechanism —
>   this is hypothesis generation ("maybe reactive electrophiles are
>   under-flagged"), not a validated structural alert; confirming that needs
>   a systematic substructure analysis over all false negatives, not 8 by eye.
> - Doing error analysis only on false negatives — false positives (here,
>   `false_positive_mask`) matter too, especially if the screen's cost model
>   penalises wasted follow-up assays.
''')]

C += checkpoint(
    "Section 4",
    check=r'''
assert cm.sum() == len(y_test)
assert false_negative_mask.sum() > 0
assert len(fn_smiles) == false_negative_mask.sum()
assert (FIGDIR / "false_negatives_grid.png").is_file()
print("Section 4 OK")
''',
    expected="Prints `Section 4 OK`. At least one false negative exists and a "
    "structure grid is saved.",
    questions=r'''
1. Why look at false negatives at the *default* threshold rather than the
   Section 3 high-recall threshold?
2. Propose one concrete next step (beyond eyeballing 8 structures) to turn
   this into real evidence about *why* these compounds are missed.
''',
    answers=r'''
1. The default threshold's false negatives show what the model does with **no
   deliberate recall correction** — a useful "before" picture; at the
   Section 3 threshold, recall is already forced to 0.8, so by construction
   there would be far fewer false negatives left to examine.
2. Compute a structural-alert or substructure-frequency comparison (e.g. via
   RDKit SMARTS matching) between the false-negative set and the correctly
   classified toxic compounds, to see whether specific fragments are
   statistically under-represented in what the model flags as risky.
''',
)

# ==========================================================================
# 5. Exercises
# ==========================================================================
C += [md(r'''
---
## 5. Exercises
''')]

C += exercise(
    prompt=r'''
### Exercise 1 — balanced accuracy from scratch *(easy, ~10 min)*

Write `balanced_accuracy_manual(y_true, y_pred)` computing
$\tfrac{1}{2}(\text{recall}_0+\text{recall}_1)$ directly (no
`sklearn.metrics.balanced_accuracy_score`). Check against `metrics_lr` /
`metrics_rf` from Section 2.

<details><summary>Show hint</summary>

`recall_c = ((y_pred==c)&(y_true==c)).sum() / (y_true==c).sum()` for
`c` in `{0, 1}`; average the two.
</details>
''',
    solution=r'''
def balanced_accuracy_manual(y_true, y_pred):
    """Mean of per-class recall, computed directly."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    recalls = []
    for c in (0, 1):
        mask = y_true == c
        recalls.append(((y_pred == c) & mask).sum() / mask.sum())
    return float(np.mean(recalls))

check_lr = balanced_accuracy_manual(y_test, pred_lr)
check_rf = balanced_accuracy_manual(y_test, pred_rf)
print(check_lr, check_rf)
''',
    scaffold=r'''
def balanced_accuracy_manual(y_true, y_pred):
    """Mean of per-class recall, computed directly."""
    # YOUR CODE HERE
    ...

check_lr = ...  # YOUR CODE HERE
check_rf = ...  # YOUR CODE HERE
''',
    check=r'''
assert abs(check_lr - metrics_lr["balanced_accuracy"]) < 1e-9
assert abs(check_rf - metrics_rf["balanced_accuracy"]) < 1e-9
assert balanced_accuracy_manual(np.array([0, 1]), np.array([0, 1])) == 1.0
print("Exercise 1 OK")
''',
)

C += exercise(
    prompt=r'''
### Exercise 2 — precision at a fixed recall, for either model *(medium, ~15 min)*

Generalise Section 3: write `precision_at_recall(y_true, prob, min_recall)`
(same logic as `threshold_for_recall`, but return just the precision, and
raise `ValueError` if no threshold reaches `min_recall`). Compare precision at
`min_recall = 0.5` vs `0.9` for the random forest.

<details><summary>Show hint</summary>

Reuse `threshold_for_recall`'s body; replace the "not found" `return None,
None` with `raise ValueError(...)`, and return only `precision`.
</details>
''',
    solution=r'''
def precision_at_recall(y_true, prob, min_recall):
    """Precision at the highest threshold achieving recall >= min_recall."""
    precision, recall, thresh = precision_recall_curve(y_true, prob)
    ok = recall[:-1] >= min_recall
    if not ok.any():
        raise ValueError(f"no threshold reaches recall >= {min_recall}")
    best = np.argmax(thresh[ok])
    return precision[:-1][ok][best]

p_50 = precision_at_recall(y_test, prob_rf, 0.5)
p_90 = precision_at_recall(y_test, prob_rf, 0.9)
print(f"precision at recall>=0.5: {p_50:.3f}   at recall>=0.9: {p_90:.3f}")
''',
    scaffold=r'''
def precision_at_recall(y_true, prob, min_recall):
    """Precision at the highest threshold achieving recall >= min_recall."""
    # YOUR CODE HERE
    ...

p_50 = ...  # YOUR CODE HERE: precision_at_recall(y_test, prob_rf, 0.5)
p_90 = ...  # YOUR CODE HERE: precision_at_recall(y_test, prob_rf, 0.9)
''',
    check=r'''
assert p_50 > p_90              # demanding higher recall costs precision
assert 0 < p_90 < 1
try:
    precision_at_recall(y_test, prob_rf, 1.5)      # impossible: recall cannot exceed 1
    assert False, "expected ValueError"
except ValueError:
    pass
print("Exercise 2 OK")
''',
)

C += exercise(
    prompt=r'''
### Exercise 3 — add an SVM to the competition *(medium, ~15 min)*

Fit `SVC(kernel="rbf", C=1.0, class_weight="balanced", random_state=0)` on
`X_train_s`/`y_train`. `SVC` has no `predict_proba` by default; use
`decision_function` (any monotonic score works for ROC-AUC/AP — ranking is all
that matters). Report ROC-AUC and AP alongside `metrics_lr`/`metrics_rf`.

<details><summary>Show hint</summary>

`score = svc.decision_function(X_test_s)`; then
`roc_auc_score(y_test, score)` and `average_precision_score(y_test, score)`
work exactly as with a probability.
</details>
''',
    solution=r'''
from sklearn.svm import SVC

svc = SVC(kernel="rbf", C=1.0, class_weight="balanced", random_state=0)
svc.fit(X_train_s, y_train)
score_svc = svc.decision_function(X_test_s)

svc_auc = roc_auc_score(y_test, score_svc)
svc_ap = average_precision_score(y_test, score_svc)
print(f"SVC: ROC-AUC={svc_auc:.3f}  AP={svc_ap:.3f}")
''',
    scaffold=r'''
from sklearn.svm import SVC

# YOUR CODE HERE: fit SVC(kernel="rbf", C=1.0, class_weight="balanced",
# random_state=0) on X_train_s/y_train, score X_test_s with decision_function

svc_auc = ...  # YOUR CODE HERE
svc_ap = ...  # YOUR CODE HERE
''',
    check=r'''
assert 0.5 < svc_auc <= 1.0
assert 0 < svc_ap <= 1.0
assert svc_auc > metrics_lr["ROC_AUC"] - 0.05   # competitive with logistic regression
print("Exercise 3 OK")
''',
)

C += [md(r'''
### 🏁 Mini-challenge — meet the screening requirement with the best precision *(~20 min)*

The lab has a firm requirement: **recall ≥ 0.8** on held-out data. Among the
three models fitted in this notebook (logistic regression, random forest,
SVC), find the one that meets the requirement with the **highest precision**,
and report its threshold.

Write `best_model_for_requirement(models_scores, min_recall)` where
`models_scores` is a dict `{name: (y_true-compatible score array)}`; return
`(best_name, threshold, precision)`.

<details><summary>Show hint</summary>

Loop the dict, call your Exercise 2 `precision_at_recall`-style logic
(catching the case a model cannot reach `min_recall` at all), and keep the
best precision seen.
</details>
''')]

C += exercise(
    prompt=r'''
Implement `best_model_for_requirement` below.
''',
    solution=r'''
def best_model_for_requirement(models_scores, min_recall):
    """(best_name, threshold, precision) meeting recall>=min_recall, max precision."""
    best_name, best_threshold, best_precision = None, None, -1.0
    for name, scores in models_scores.items():
        precision, recall, thresh = precision_recall_curve(y_test, scores)
        ok = recall[:-1] >= min_recall
        if not ok.any():
            continue
        idx = np.argmax(thresh[ok])
        p = precision[:-1][ok][idx]
        if p > best_precision:
            best_name, best_threshold, best_precision = name, thresh[ok][idx], p
    return best_name, best_threshold, best_precision

models_scores = {"Logistic regression": prob_lr, "Random forest": prob_rf, "SVC": score_svc}
winner, thr, prec = best_model_for_requirement(models_scores, 0.8)
print(f"winner: {winner}  threshold={thr:.3f}  precision={prec:.3f}")
''',
    scaffold=r'''
def best_model_for_requirement(models_scores, min_recall):
    """(best_name, threshold, precision) meeting recall>=min_recall, max precision."""
    # YOUR CODE HERE
    ...

models_scores = {"Logistic regression": prob_lr, "Random forest": prob_rf, "SVC": score_svc}
winner, thr, prec = ...  # YOUR CODE HERE: best_model_for_requirement(models_scores, 0.8)
''',
    check=r'''
assert winner in models_scores
assert thr is not None and prec > 0
# the winner must actually be able to reach recall 0.8 in isolation
_, r_check = threshold_for_recall(y_test, models_scores[winner], 0.8)
assert r_check is not None
print("Mini-challenge OK")
''',
)

# ==========================================================================
C += [md(r'''
---
## Summary — what you learned

- A **severely** imbalanced task (6% positive) makes accuracy actively
  misleading; ROC-AUC/AP and per-class recall are the honest metrics.
- Two models can disagree on ranking (ROC-AUC/AP) versus default-threshold
  behaviour (balanced accuracy/F1) — always be explicit about which you need.
- **Precision-recall curves** and explicit threshold tuning turn a vague
  "good model" question into a concrete, defensible operating point against a
  stated requirement.
- **Error analysis** — inspecting false negatives/positives as molecules, not
  just counting them — turns a metric into a hypothesis about *why* a model
  fails, which a metric alone never gives you.

Week 05 begins deep learning: tensors, MLPs and PyTorch, starting the
transition from `scikit-learn` classical models to learned representations.
''')]

C += [md(r'''
## Further reading (course source list only)

- Pat Walters, `classification_model.ipynb` and
  `comparing_classification_models.ipynb`:
  <https://colab.research.google.com/github/PatWalters/practical_cheminformatics_tutorials/blob/main/ml_models/classification_model.ipynb>,
  <https://colab.research.google.com/github/PatWalters/practical_cheminformatics_tutorials/blob/main/ml_models/comparing_classification_models.ipynb>
- TeachOpenCADD **T007** (ML for compound activity classification):
  <https://projects.volkamerlab.org/teachopencadd/talktorials/T007_compound_activity_machine_learning.html>
- Alternative: ML-in-chemistry-101:
  <https://github.com/BingqingCheng/ML-in-chemistry-101>
''')]

C += [md(r'''
## Attribution

Original teaching material for *AI for Chemistry*, adapted and rewritten from:

- **TeachOpenCADD** (Volkamer Lab), talktorial T007 — the classifier-comparison
  and sensitivity/specificity/AUC pattern, here extended with precision-recall
  threshold tuning for a much more imbalanced task than T007's own example.
  CC-BY 4.0. <https://github.com/volkamerlab/teachopencadd>
- **Practical Cheminformatics** (P. Walters) — the multi-model comparison
  pattern (`comparing_classification_models.ipynb`). MIT licence.
  <https://github.com/PatWalters/practical_cheminformatics_tutorials>

Dataset: Tox21 (MoleculeNet), downloaded from the same DeepChem mirror as
Week 02's ESOL and Week 04A's BBBP, checksum-pinned. No verbatim text is
reproduced from the sources above.
''')]

build(__file__, "week04_b_classification-unsupervised", C)
