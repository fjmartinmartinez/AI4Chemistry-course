"""Build week 03 session A notebooks (first ML: regression & model assessment).

    python lectures/week-03_regression-model-assessment/notebook/build_week03_a.py
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "_build"))
from nbbuild import build, checkpoint, code, exercise, md  # noqa: E402

C = []

# ==========================================================================
C += [md(r'''
# AI for Chemistry — Week 03, Session A
## First ML: regression and model assessment

**Course:** AI for Chemistry · **Session:** 03A (interleaved lecture + lab,
2.5 h) · **Runtime:** < 20 s, no GPU.

### Suggested timing (solo study, ~120 min)

| Section | Topic | Time |
|--------:|-------|-----:|
| 0 | Setup | 4 min |
| 1 | The supervised-learning framing; a one-feature fit | 18 min |
| 2 | Train/test split; overfitting and underfitting | 22 min |
| 3 | k-fold cross-validation | 18 min |
| 4 | Metrics: MAE, RMSE, R²; parity plot | 20 min |
| 5 | Regularisation (Ridge, Lasso) — brief | 15 min |
| 6 | Exercises (4) | 23 min |

### Learning objectives
1. State the supervised-learning framing (features $\vec{x}$, labels $y$, model
   $\hat f$, loss) and fit a linear model by minimising mean squared error. *(LO3)*
2. Explain why test error, not training error, measures generalisation, and
   recognise overfitting from a train/test gap. *(LO3)*
3. Run k-fold cross-validation and read a *spread* of scores, not one number. *(LO3)*
4. Compute MAE, RMSE and R², state what each is sensitive to, and read a
   parity plot. *(LO3)*
5. Explain, qualitatively, what L1 and L2 regularisation do to model
   coefficients. *(LO3)*

### Prerequisites — before this notebook you should be able to
- Weeks 01–02: NumPy arrays, pandas selection, RDKit descriptors, and ideally
  have run Week 02B (`esol_clean.csv`) — if that file is missing this notebook
  rebuilds it automatically from the same recipe.

### How to use this notebook (solo study)
Read, run, check each **✅** cell in order. This is the **solutions**
notebook; conceptual-question answers are filled in.
''')]

# ==========================================================================
C += [md(r'''
---
## 0. Setup

Load the clean ESOL table from Week 02. If you have not run Week 02B (or the
file was deleted), the cell rebuilds it with the identical recipe — canonical
SMILES → drop missing target → deduplicate → range check — so this notebook
never depends on another one having been executed first.

**What to look for:** `Xdesc: (1117, 7) y: (1117,)`.
''')]

C += [code(r'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
RDLogger.DisableLog("rdApp.*")

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RNG = np.random.default_rng(0xC0FFEE)          # course-wide fixed seed

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
    print("loaded cached esol_clean.csv:", esol.shape)
else:
    print("esol_clean.csv missing -> rebuilding (Week 02B recipe)")
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
y = esol[TARGET].to_numpy()
print("Xdesc:", Xdesc.shape, "y:", y.shape)
''')]

C += [md(r'''
> **Common errors — Setup**
> - `FileNotFoundError` for `esol_delaney.csv` — you have not run Week 01/02
>   dataset setup; run `python lectures/_build/make_datasets_wk01_02.py` first.
> - Different `Xdesc` values from Week 02 — check `rdkit.__version__`; Crippen
>   logP has changed slightly across RDKit releases in the past.
''')]

# ==========================================================================
# 1. The supervised-learning framing
# ==========================================================================
C += [md(r'''
---
## 1. The supervised-learning framing

Supervised ML fits a model $\hat f(\vec x)$ to **features** $\vec x_i$ and
**labels** $y_i$ drawn from some true (unknown) process
$y = f(\vec x) + \epsilon$, where $\epsilon$ is noise. A **linear model** with
one feature is

$$\hat y = w x + b,$$

fit by minimising the **mean squared error (MSE)** loss

$$L(w, b) = \frac{1}{N}\sum_{i=1}^N \bigl(y_i - (w x_i + b)\bigr)^2.$$

We already fit exactly this shape of equation in Week 01B
(`np.polyfit(conc, absorbance, 1)`, Beer–Lambert). Here the feature is
**MolLogP** and the label is **measured log S** — do lipophilic molecules
dissolve less?

**What to look for:** a negative slope $w$ (higher logP → lower solubility);
`sklearn` and `np.polyfit` agree to several decimal places, because both solve
the same least-squares problem.
''')]

C += [code(r'''
logp = Xdesc[:, desc_names.index("MolLogP")]

# two ways to solve the same least-squares problem
w_np, b_np = np.polyfit(logp, y, 1)
lr1 = LinearRegression().fit(logp.reshape(-1, 1), y)   # sklearn wants (N, D)

print(f"np.polyfit   : w={w_np:.4f}  b={b_np:.4f}")
print(f"sklearn      : w={lr1.coef_[0]:.4f}  b={lr1.intercept_:.4f}")

yhat = w_np * logp + b_np
mse = np.mean((y - yhat) ** 2)
print(f"training MSE = {mse:.3f}")
''')]

C += [md(r'''
> **Common errors — Section 1**
> - `ValueError: Expected 2D array, got 1D array` — `sklearn` estimators expect
>   `X` shaped `(n_samples, n_features)`; reshape a single feature with
>   `.reshape(-1, 1)`.
> - Confusing $w$ (slope, a *learned parameter*) with a *feature value*
>   $x_i$ — the loss is minimised over $w, b$, not over $x$.
''')]

C += checkpoint(
    "Section 1",
    check=r'''
assert w_np < 0, "logP should be negatively associated with solubility"
assert abs(w_np - lr1.coef_[0]) < 1e-6
assert abs(b_np - lr1.intercept_) < 1e-6
assert mse > 0
print("Section 1 OK")
''',
    expected="Prints `Section 1 OK`. `w` is negative and identical (to machine "
    "precision) between `np.polyfit` and `sklearn`.",
    questions=r'''
1. Why must $w<0$ be a *finding*, not something we assumed before fitting?
2. MSE uses squared error. What does squaring do to the effect of one badly
   mispredicted molecule?
''',
    answers=r'''
1. The model does not know any chemistry; it only minimises the loss. A
   negative slope emerges purely from the data — if the chemistry were
   different (e.g. inorganic salts, dominated by lattice energy rather than
   lipophilicity) the sign could come out differently or near zero.
2. Squaring makes MSE **very** sensitive to large errors — one prediction off
   by 4 log units contributes as much as sixteen predictions each off by 1.
   Outliers dominate an MSE-trained model.
''',
)

# ==========================================================================
# 2. Train/test split; overfitting and underfitting
# ==========================================================================
C += [md(r'''
---
## 2. Train/test split; overfitting and underfitting

A model that only ever sees the loss on the data it was *fit on* cannot tell
you how it will do on a new molecule. We hold out a **test set** the model
never touches during fitting, and compare training loss with test loss.

* **Underfitting**: both train and test error are high — the model is too
  simple, or lacks the right features.
* **Overfitting**: train error is low but test error is high — the model has
  learned the noise (and idiosyncrasies) of the training sample, not the
  general trend.

We reproduce this with a **synthetic** dataset where we control the truth:
$y = 2x_1 - 1.5x_2 + x_3 + \epsilon$ (only 3 of many available features
actually matter). Fitting with a **small training set (25 points) and many
irrelevant features (20)** should overfit badly; fitting with **only the 3
true features** should not.

**What to look for:** with 20 features the train RMSE is tiny but the test
RMSE is huge (the model memorised noise); with the correct 3 features, train
and test RMSE are close (both reflect the true noise level, $\sigma=1$).
''')]

C += [code(r'''
n_train, n_features = 25, 20
X_all = RNG.normal(size=(n_train + 200, n_features))    # more features than needed
true_w = np.zeros(n_features)
true_w[:3] = [2.0, -1.5, 1.0]                            # only 3 features matter
y_all = X_all @ true_w + RNG.normal(scale=1.0, size=X_all.shape[0])

X_tr, y_tr = X_all[:n_train], y_all[:n_train]
X_te, y_te = X_all[n_train:], y_all[n_train:]

def train_test_rmse(X_tr, y_tr, X_te, y_te):
    model = LinearRegression().fit(X_tr, y_tr)
    rmse_tr = mean_squared_error(y_tr, model.predict(X_tr)) ** 0.5
    rmse_te = mean_squared_error(y_te, model.predict(X_te)) ** 0.5
    return rmse_tr, rmse_te

rmse_tr_many, rmse_te_many = train_test_rmse(X_tr, y_tr, X_te, y_te)
rmse_tr_few, rmse_te_few = train_test_rmse(X_tr[:, :3], y_tr, X_te[:, :3], y_te)

print(f"20 features (overfit-prone): train {rmse_tr_many:.2f}  test {rmse_te_many:.2f}")
print(f"3 true features            : train {rmse_tr_few:.2f}  test {rmse_te_few:.2f}")
''')]

C += [code(r'''
fig, ax = plt.subplots(figsize=(5.5, 3.8))
labels = ["20 features\n(overfits)", "3 true features\n(well-specified)"]
x = np.arange(2)
ax.bar(x - 0.18, [rmse_tr_many, rmse_tr_few], width=0.36, label="train RMSE")
ax.bar(x + 0.18, [rmse_te_many, rmse_te_few], width=0.36, label="test RMSE")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("RMSE"); ax.set_title("Overfitting: train vs test error"); ax.legend()
fig.tight_layout(); fig.savefig(FIGDIR / "overfitting_train_test_gap.png", dpi=200)
plt.show()
''')]

C += [md(r'''
> **Common errors — Section 2**
> - Standardising or fitting anything using the **test** set's own mean/std —
>   this leaks test information into training. Always fit transforms on train
>   only (Week 02 raised this for deduplication; it applies here too).
> - Judging "overfitting" from training error alone — you need *both* numbers.
> - A gap can also come from too little data, not just too many features;
>   more training data would shrink the 20-feature gap too.
''')]

C += checkpoint(
    "Section 2",
    check=r'''
assert rmse_tr_many < 0.6                     # near-perfect fit to 25 noisy points
assert rmse_te_many > 3 * rmse_tr_many         # big overfitting gap
assert rmse_te_few < 1.5 * rmse_tr_few         # small, expected gap
assert (FIGDIR / "overfitting_train_test_gap.png").is_file()
print("Section 2 OK")
''',
    expected="Prints `Section 2 OK`. The 20-feature model has train RMSE well "
    "under test RMSE (a factor of >3); the 3-feature model's train and test "
    "RMSE are close.",
    questions=r'''
1. Both models see the *same* 25 training points. Why does one overfit and the
   other not?
2. Besides using fewer features, name one other way to fight the overfitting
   seen here.
''',
    answers=r'''
1. With 20 features and only 25 points, the model has enough free parameters
   to fit almost any pattern in the training noise — there is little
   information left over to constrain it. With 3 features matched to the true
   generating equation, the model cannot do better than fit the true signal
   plus irreducible noise.
2. Get more training data; or keep 20 features but add **regularisation**
   (Section 5), which penalises large coefficients and effectively reduces
   the model's usable complexity even without removing features.
''',
)

# ==========================================================================
# 3. Cross-validation
# ==========================================================================
C += [md(r'''
---
## 3. k-fold cross-validation

A single train/test split gives *one* estimate of test error — noisy if the
dataset is modest. **k-fold cross-validation** splits the data into $k$ equal
folds; each fold takes a turn as the test set while the model trains on the
remaining $k-1$ folds, giving $k$ test scores. Their **spread**, not just their
mean, tells you how much to trust the number.

We run 5-fold CV for linear regression on the ESOL descriptor block.

**What to look for:** five RMSE values, all in a similar range (roughly 0.9–1.1
log units) — a single split could have landed anywhere in that range by chance.
''')]

C += [code(r'''
kf = KFold(n_splits=5, shuffle=True, random_state=0)
cv_scores = -cross_val_score(LinearRegression(), Xdesc, y, cv=kf,
                              scoring="neg_root_mean_squared_error")
print("fold RMSEs:", np.round(cv_scores, 3))
print(f"mean {cv_scores.mean():.3f}  std {cv_scores.std():.3f}")
''')]

C += [code(r'''
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.bar(range(1, 6), cv_scores, color="steelblue")
ax.axhline(cv_scores.mean(), color="k", ls="--", label=f"mean = {cv_scores.mean():.2f}")
ax.set_xlabel("fold"); ax.set_ylabel("RMSE"); ax.set_title("5-fold CV, linear/descriptors")
ax.legend(); fig.tight_layout()
fig.savefig(FIGDIR / "cv_fold_scores.png", dpi=200)
plt.show()
''')]

C += [md(r'''
> **Common errors — Section 3**
> - Forgetting `shuffle=True` when the data has any ordering (ESOL is not
>   ordered by anything meaningful, but many real datasets are) — unshuffled
>   folds can be systematically different.
> - `cross_val_score`'s `neg_root_mean_squared_error` is **negative** RMSE
>   (sklearn convention: higher score = better); negate it back before
>   reporting.
> - Using the *same* `random_state` you tuned on to also report your final
>   number — pick the fold seed before looking at results, or better, average
>   over several seeds.
''')]

C += checkpoint(
    "Section 3",
    check=r'''
assert cv_scores.shape == (5,)
assert 0.85 < cv_scores.mean() < 1.10
assert cv_scores.std() < 0.15
assert (FIGDIR / "cv_fold_scores.png").is_file()
print("Section 3 OK")
''',
    expected="Prints `Section 3 OK`. Mean fold RMSE is roughly 0.9-1.0 with a "
    "small spread across folds.",
    questions=r'''
1. What does a **large** standard deviation across folds tell you that the
   mean alone does not?
2. LOOCV ($k=N$) uses almost all the data to train every fold. What is the
   price paid for that?
''',
    answers=r'''
1. That test performance depends heavily on *which* molecules happen to be
   held out — the dataset may be small, non-uniform, or contain a few
   hard-to-predict clusters. A single held-out score would then be
   unreliable evidence about real-world performance.
2. LOOCV requires training $N$ separate models (expensive for large datasets
   or slow models), and because consecutive training sets differ by only one
   point, the $N$ test errors are highly correlated — so you get many numbers
   but not much *new* information about variance.
''',
)

# ==========================================================================
# 4. Metrics and a parity plot
# ==========================================================================
C += [md(r'''
---
## 4. Metrics: MAE, RMSE, R²

Three standard regression metrics, all comparing predictions $\hat y_i$ to
labels $y_i$ over $N$ molecules:

$$\text{MAE} = \frac{1}{N}\sum_i |y_i - \hat y_i|, \qquad
  \text{RMSE} = \sqrt{\frac{1}{N}\sum_i (y_i - \hat y_i)^2}, \qquad
  R^2 = 1 - \frac{\sum_i (y_i - \hat y_i)^2}{\sum_i (y_i - \bar y)^2}.$$

* **MAE** — average absolute error, in the same units as $y$ (log S units);
  robust to outliers.
* **RMSE** — same units, but penalises large errors more (Section 1); always
  $\text{RMSE} \ge \text{MAE}$.
* **R²** — fraction of variance explained; 1 is perfect, 0 matches predicting
  the mean every time, and it *can go negative* for a model worse than that.

We fit on an 80/20 split (`random_state=42`, fixed for reproducibility) and
check our manual formulas against `sklearn.metrics`.

**What to look for:** manual and `sklearn` metrics agree to machine precision;
R² lands around 0.7–0.8; the parity plot hugs the diagonal with more scatter at
the extremes.
''')]

C += [code(r'''
Xd_tr, Xd_te, y_tr2, y_te2 = train_test_split(Xdesc, y, test_size=0.2, random_state=42)
lr_full = LinearRegression().fit(Xd_tr, y_tr2)
pred_te = lr_full.predict(Xd_te)

mae_manual = np.mean(np.abs(y_te2 - pred_te))
rmse_manual = np.sqrt(np.mean((y_te2 - pred_te) ** 2))
ss_res = np.sum((y_te2 - pred_te) ** 2)
ss_tot = np.sum((y_te2 - y_te2.mean()) ** 2)
r2_manual = 1 - ss_res / ss_tot

mae_skl = mean_absolute_error(y_te2, pred_te)
rmse_skl = mean_squared_error(y_te2, pred_te) ** 0.5
r2_skl = r2_score(y_te2, pred_te)

print(f"MAE   manual {mae_manual:.4f}  sklearn {mae_skl:.4f}")
print(f"RMSE  manual {rmse_manual:.4f}  sklearn {rmse_skl:.4f}")
print(f"R2    manual {r2_manual:.4f}  sklearn {r2_skl:.4f}")
''')]

C += [code(r'''
fig, ax = plt.subplots(figsize=(5, 5))
lims = [min(y_te2.min(), pred_te.min()) - 0.5, max(y_te2.max(), pred_te.max()) + 0.5]
ax.plot(lims, lims, "k--", lw=1, label="perfect prediction")
ax.scatter(y_te2, pred_te, s=14, alpha=0.6)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("measured log S"); ax.set_ylabel("predicted log S")
ax.set_title(f"Parity plot — linear/descriptors (R$^2$={r2_skl:.2f})")
ax.legend(); ax.set_aspect("equal")
fig.tight_layout(); fig.savefig(FIGDIR / "parity_plot_linear.png", dpi=200)
plt.show()
''')]

C += [md(r'''
> **Common errors — Section 4**
> - Computing R² on the **training** set and quoting it as model performance —
>   always report test-set metrics for an honest number.
> - Comparing RMSE values computed on different test sets (different $\bar y$,
>   different variance) as if they were on the same scale — R² partially
>   corrects for this, raw RMSE does not.
> - `ax.set_aspect("equal")` matters for a parity plot — without it, the
>   diagonal is not visually at 45°, which misleads the eye.
''')]

C += checkpoint(
    "Section 4",
    check=r'''
assert abs(mae_manual - mae_skl) < 1e-9
assert abs(rmse_manual - rmse_skl) < 1e-9
assert abs(r2_manual - r2_skl) < 1e-9
assert 0.6 < r2_skl < 0.9
assert rmse_skl >= mae_skl
assert (FIGDIR / "parity_plot_linear.png").is_file()
print("Section 4 OK")
''',
    expected="Prints `Section 4 OK`. Manual and sklearn metrics match exactly; "
    "R² is roughly 0.7-0.8; RMSE ≥ MAE as required algebraically.",
    questions=r'''
1. Prove to yourself in words why RMSE $\ge$ MAE always holds.
2. A colleague reports R² = -0.3 for their model. What does a negative R²
   mean, concretely?
''',
    answers=r'''
1. RMSE is the root of the mean of *squared* errors; MAE is the mean of
   *absolute* errors. By the QM-AM (or Jensen's) inequality, the root-mean-square
   of a set of non-negative numbers is always $\ge$ their mean — equality only
   when every error has exactly the same magnitude.
2. The model does **worse** than the trivial baseline of always predicting the
   training mean — it is actively misleading, e.g. badly miscalibrated, or fit
   on data that does not resemble the test distribution.
''',
)

# ==========================================================================
# 5. Regularisation — brief
# ==========================================================================
C += [md(r'''
---
## 5. Regularisation — brief

Adding a penalty on the size of the coefficients trades a little training fit
for a more stable, generalisable model:

$$L_{\text{ridge}} = \text{MSE} + \lambda \sum_k w_k^2 \qquad (\text{L2, "Ridge"})$$
$$L_{\text{lasso}} = \text{MSE} + \lambda \sum_k |w_k| \qquad (\text{L1, "Lasso"})$$

Larger $\lambda$ (`alpha` in `sklearn`) shrinks coefficients more. **L1 is
special**: it can drive coefficients to *exactly* zero, performing automatic
feature selection; L2 shrinks everything smoothly but rarely to exactly zero.

We standardise the 7 descriptors first (Section 2's lesson: fair comparison
needs comparable scales) and sweep `alpha`.

**What to look for:** the Ridge coefficient norm shrinks smoothly as `alpha`
grows; the number of *non-zero* Lasso coefficients drops from all 7 to 0.
''')]

C += [code(r'''
Xdesc_std = StandardScaler().fit_transform(Xdesc)

ridge_alphas = np.logspace(-2, 3, 12)
ridge_norms = [np.linalg.norm(Ridge(alpha=a).fit(Xdesc_std, y).coef_) for a in ridge_alphas]

lasso_alphas = np.logspace(-4, 1, 12)
lasso_nonzero = [int(np.sum(np.abs(Lasso(alpha=a, max_iter=5000)
                                    .fit(Xdesc_std, y).coef_) > 1e-6))
                 for a in lasso_alphas]

print("ridge alpha -> ||w||:", list(zip(np.round(ridge_alphas, 2), np.round(ridge_norms, 2))))
print("lasso alpha -> nonzero coefs:", list(zip(np.round(lasso_alphas, 4), lasso_nonzero)))
''')]

C += [code(r'''
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
axes[0].plot(ridge_alphas, ridge_norms, "o-")
axes[0].set_xscale("log"); axes[0].set_xlabel(r"Ridge $\alpha$")
axes[0].set_ylabel(r"$\|\vec w\|_2$"); axes[0].set_title("Ridge: smooth shrinkage")

axes[1].plot(lasso_alphas, lasso_nonzero, "o-", color="darkorange")
axes[1].set_xscale("log"); axes[1].set_xlabel(r"Lasso $\alpha$")
axes[1].set_ylabel("# non-zero coefficients"); axes[1].set_title("Lasso: sparsity")
fig.tight_layout(); fig.savefig(FIGDIR / "regularisation_shrinkage.png", dpi=200)
plt.show()
''')]

C += [md(r'''
> **Common errors — Section 5**
> - Regularising **unstandardised** features — a descriptor measured in the
>   hundreds (MolWt) gets penalised unfairly compared to one measured in units
>   (NumHDonors). Always standardise first.
> - Treating `alpha` as comparable between Ridge and Lasso — the two penalties
>   are on different scales ($w^2$ vs $|w|$); a "big" alpha for one can be tiny
>   for the other, as the differing grids above show.
> - Assuming regularisation always improves test error — with abundant, clean
>   data and few features it can do essentially nothing, or even underfit if
>   `alpha` is too large.
''')]

C += checkpoint(
    "Section 5",
    check=r'''
assert ridge_norms[0] > ridge_norms[-1]        # coefficients shrink overall
assert all(np.diff(ridge_norms) <= 1e-9)       # monotonically non-increasing
assert lasso_nonzero[0] == 7
assert lasso_nonzero[-1] == 0
assert all(np.diff(lasso_nonzero) <= 0)        # non-increasing sparsity
assert (FIGDIR / "regularisation_shrinkage.png").is_file()
print("Section 5 OK")
''',
    expected="Prints `Section 5 OK`. Ridge norm shrinks monotonically with "
    "`alpha`; Lasso's non-zero coefficient count falls from 7 to 0.",
    questions=r'''
1. Which regularisation would you reach for if you wanted to know *which one or
   two* descriptors matter most, and why?
2. At very large `alpha`, both penalties push every coefficient towards 0.
   What does the model predict in that limit, and is that under- or
   over-fitting?
''',
    answers=r'''
1. **Lasso (L1)** — its exact-zero property performs feature selection for
   you; Ridge shrinks everything but keeps all features "a little bit in
   play", which is less interpretable.
2. As every $w_k \to 0$, $\hat y \to b$, i.e. the model predicts (close to)
   the training mean for every molecule regardless of its features — the
   extreme of **underfitting**.
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
### Exercise 1 — metrics from scratch *(easy, ~10 min)*

Write `regression_metrics(y_true, y_pred)` returning a dict with keys
`mae`, `rmse`, `r2`, computed **without** calling `sklearn.metrics` (NumPy
only). Check it against `Section 4`'s fitted predictions.

<details><summary>Show hint</summary>

Reuse the three formulas from Section 4 verbatim, just wrapped in a function.
</details>
''',
    solution=r'''
def regression_metrics(y_true, y_pred):
    """MAE, RMSE, R2 computed directly from the definitions (no sklearn)."""
    err = y_true - y_pred
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return {"mae": mae, "rmse": rmse, "r2": r2}

metrics_check = regression_metrics(y_te2, pred_te)
print(metrics_check)
''',
    scaffold=r'''
def regression_metrics(y_true, y_pred):
    """MAE, RMSE, R2 computed directly from the definitions (no sklearn)."""
    # YOUR CODE HERE
    ...

metrics_check = ...  # YOUR CODE HERE
''',
    check=r'''
assert set(metrics_check) == {"mae", "rmse", "r2"}
assert abs(metrics_check["mae"] - mae_skl) < 1e-9
assert abs(metrics_check["rmse"] - rmse_skl) < 1e-9
assert abs(metrics_check["r2"] - r2_skl) < 1e-9
# a perfect prediction has mae=rmse=0 and r2=1
perfect = regression_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
assert perfect["mae"] == 0 and perfect["rmse"] == 0 and perfect["r2"] == 1
print("Exercise 1 OK")
''',
)

C += exercise(
    prompt=r'''
### Exercise 2 — cross-validate at a chosen k *(medium, ~12 min)*

Write `cv_rmse(model, X, y, k)` returning `(mean, std)` of RMSE over `k`-fold
CV (`shuffle=True, random_state=0`). Compare `k=3` and `k=10` for
`LinearRegression` on `Xdesc`.

<details><summary>Show hint</summary>

`KFold(n_splits=k, shuffle=True, random_state=0)`; reuse
`cross_val_score(..., scoring="neg_root_mean_squared_error")` and negate.
</details>
''',
    solution=r'''
def cv_rmse(model, X, y, k):
    """(mean, std) RMSE over k-fold CV."""
    kf = KFold(n_splits=k, shuffle=True, random_state=0)
    scores = -cross_val_score(model, X, y, cv=kf, scoring="neg_root_mean_squared_error")
    return scores.mean(), scores.std()

mean3, std3 = cv_rmse(LinearRegression(), Xdesc, y, 3)
mean10, std10 = cv_rmse(LinearRegression(), Xdesc, y, 10)
print(f"k=3:  {mean3:.3f} +/- {std3:.3f}")
print(f"k=10: {mean10:.3f} +/- {std10:.3f}")
''',
    scaffold=r'''
def cv_rmse(model, X, y, k):
    """(mean, std) RMSE over k-fold CV."""
    # YOUR CODE HERE
    ...

mean3, std3 = ...  # YOUR CODE HERE: k=3
mean10, std10 = ...  # YOUR CODE HERE: k=10
''',
    check=r'''
assert 0.85 < mean3 < 1.15 and 0.85 < mean10 < 1.15
assert std3 >= 0 and std10 >= 0
# fewer, larger folds (k=3) are typically noisier fold-to-fold than k=10
print("Exercise 2 OK")
''',
)

C += exercise(
    prompt=r'''
### Exercise 3 — measure the overfitting gap *(medium, ~15 min)*

Write `overfit_gap(n_train, n_features, seed=0)` that reproduces Section 2's
synthetic setup (same `true_w`, first 3 features matter) for a given
`n_train`/`n_features` and returns `test_rmse - train_rmse`. Show the gap
shrinks as `n_train` grows for a fixed 20 features.

<details><summary>Show hint</summary>

Copy Section 2's data-generation and `train_test_rmse` logic into the
function, parameterised by `n_train`, `n_features`, and a local
`np.random.default_rng(seed)`.
</details>
''',
    solution=r'''
def overfit_gap(n_train, n_features, seed=0):
    """test_rmse - train_rmse for the Section-2-style synthetic problem."""
    rng = np.random.default_rng(seed)
    X_all = rng.normal(size=(n_train + 200, n_features))
    true_w = np.zeros(n_features)
    true_w[:3] = [2.0, -1.5, 1.0]
    y_all = X_all @ true_w + rng.normal(scale=1.0, size=X_all.shape[0])
    X_tr, y_tr = X_all[:n_train], y_all[:n_train]
    X_te, y_te = X_all[n_train:], y_all[n_train:]
    model = LinearRegression().fit(X_tr, y_tr)
    rmse_tr = mean_squared_error(y_tr, model.predict(X_tr)) ** 0.5
    rmse_te = mean_squared_error(y_te, model.predict(X_te)) ** 0.5
    return rmse_te - rmse_tr

gaps = [overfit_gap(n, 20, seed=1) for n in [25, 50, 100, 400]]
print("gaps (n_train=25,50,100,400):", np.round(gaps, 2))
''',
    scaffold=r'''
def overfit_gap(n_train, n_features, seed=0):
    """test_rmse - train_rmse for the Section-2-style synthetic problem."""
    # YOUR CODE HERE
    ...

gaps = ...  # YOUR CODE HERE: [overfit_gap(n, 20, seed=1) for n in [25, 50, 100, 400]]
''',
    check=r'''
assert len(gaps) == 4
# with ample data (n_train=400 >> n_features) the gap can dip slightly
# negative from sampling noise alone, but never by much
assert all(g >= -0.2 for g in gaps)
assert gaps[0] > gaps[-1]                      # gap shrinks with more data
assert gaps[0] > 1.0                            # severe overfitting at n_train=25
print("Exercise 3 OK")
''',
)

C += exercise(
    prompt=r'''
### Exercise 4 — pick alpha by cross-validation *(harder, ~18 min)*

Write `best_ridge_alpha(X, y, alphas)` that, for each candidate `alpha`,
computes 5-fold CV RMSE (`cv_rmse` from Exercise 2, or inline) for
`Ridge(alpha=alpha)` on **standardised** `X`, and returns the alpha with the
lowest mean CV RMSE.

<details><summary>Show hint</summary>

Standardise once outside the loop; loop `alphas`, call `cv_rmse(Ridge(alpha=a),
Xstd, y, 5)`, track the minimum.
</details>
''',
    solution=r'''
def best_ridge_alpha(X, y, alphas):
    """Alpha (from `alphas`) with the lowest 5-fold CV RMSE for Ridge."""
    Xstd = StandardScaler().fit_transform(X)
    scores = [cv_rmse(Ridge(alpha=a), Xstd, y, 5)[0] for a in alphas]
    return alphas[int(np.argmin(scores))]

alpha_grid = np.logspace(-2, 3, 12)
best_alpha = best_ridge_alpha(Xdesc, y, alpha_grid)
print("best alpha:", best_alpha)
''',
    scaffold=r'''
def best_ridge_alpha(X, y, alphas):
    """Alpha (from `alphas`) with the lowest 5-fold CV RMSE for Ridge."""
    # YOUR CODE HERE
    ...

alpha_grid = np.logspace(-2, 3, 12)
best_alpha = ...  # YOUR CODE HERE
''',
    check=r'''
assert best_alpha in alpha_grid
# an essentially-unregularised alpha should be competitive on this clean,
# low-dimensional descriptor block (little to gain from shrinkage here)
assert best_alpha < 50
print("Exercise 4 OK")
''',
)

# ==========================================================================
C += [md(r'''
---
## Summary — what you learned

- The **supervised-learning framing**: features, labels, a model, and an MSE
  loss — the same shape of problem as Week 01's Beer–Lambert fit.
- **Train/test splitting** and how to *see* over/underfitting as a gap between
  train and test error.
- **k-fold cross-validation** and why the spread of scores matters as much as
  the mean.
- **MAE, RMSE, R²** — definitions, relationships, and a parity plot.
- **Ridge and Lasso** regularisation, and why Lasso's sparsity is special.

Week 03B turns this into a **leaderboard**: linear, ridge and random-forest
models, on two representations, with a leakage-aware split.
''')]

C += [md(r'''
## Further reading (course source list only)

- dmol.pub *Introduction to Machine Learning*:
  <https://dmol.pub/ml/introduction.html>
- dmol.pub *Regression and model assessment* (bias-variance, k-fold, LOOCV,
  L1/L2 regularisation, AqSolDB example):
  <https://dmol.pub/ml/regression.html>
- Gentler alternative: ML4chemArg, *A Gentle Introduction to Machine Learning
  for Chemists*: <https://github.com/ML4chemArg/Intro-to-Machine-Learning-in-Chemistry>
''')]

C += [md(r'''
## Attribution

Original teaching material for *AI for Chemistry*, adapted and rewritten from:

- **dmol.pub** (A. White, *Deep Learning for Molecules and Materials*) — the
  features/labels/model/loss framing, the bias-variance and k-fold
  presentation, and the choice to demonstrate overfitting with a synthetic
  example before real data. CC-BY 4.0. <https://dmol.pub>
- **ML4chemArg**, *A Gentle Introduction to Machine Learning for Chemists*
  (J. Chem. Educ. workshop material) — further-reading alternative.
  <https://github.com/ML4chemArg/Intro-to-Machine-Learning-in-Chemistry>

Continues the ESOL dataset built in Week 02 (`esol_clean.csv`). No verbatim
text is reproduced from the sources above.
''')]

build(__file__, "week03_a_regression-model-assessment", C)
