# Week 03 — First ML: regression and model assessment

_Record of the approved design (syllabus v0.1). Not a proposal._

## Learning objectives
By the end of week 03 students can:
1. State the supervised-learning framing (features, labels, model, loss) and
   fit a linear regression by minimising mean squared error. [LO3]
2. Explain train/test splitting, overfitting and underfitting, and diagnose
   them from a train-vs-test error gap. [LO3]
3. Apply k-fold cross-validation and interpret the resulting score
   distribution instead of a single number. [LO3]
4. Compute and interpret MAE, RMSE and R² and produce a parity plot. [LO3]
5. (Session B) Train and compare linear, ridge and random-forest regressors on
   two molecular representations (descriptors, fingerprints); build a
   leaderboard; recognise leakage from a non-scaffold split. [LO4]

## Prerequisites assumed
- Week 02: `esol_clean.csv`, descriptor block, Morgan fingerprints, Butina
  clusters — this week loads/rebuilds them and treats them as inputs.
- NumPy arrays and pandas (Weeks 01–02).

## Concept sequence
| # | Concept | Chemistry anchor | Where |
|---|---------|------------------|-------|
| 1 | ML framing: features/labels/model/loss (MSE) | log S from descriptors | notebook A |
| 2 | Train/test split; overfitting vs underfitting | synthetic + ESOL | notebook A |
| 3 | k-fold cross-validation | ESOL descriptors | notebook A |
| 4 | Metrics: MAE, RMSE, R²; parity plot | ESOL | notebook A |
| 5 | Regularisation (Ridge/Lasso), brief | coefficient shrinkage | notebook A |
| 6 | Baseline + linear + ridge + random forest, two representations | ESOL leaderboard | notebook B |
| 7 | Feature importance (RF) | which descriptor drives solubility | notebook B |
| 8 | Leakage: random split vs Butina cluster split | reuses wk-02B clusters | notebook B |

## Notebook plan
- **Session A** (`week03_a_regression-model-assessment_*`): loads
  `sources/datasets/esol_clean.csv` (rebuilds it if absent); descriptor block
  from Week 02. 4 exercises.
- **Session B** (`week03_b_regression-model-assessment_*`): full leaderboard
  (mean baseline, linear, ridge, random forest × {descriptors, fingerprints}),
  reuses the Week 02B Butina clustering for a leakage-aware split. 3 exercises
  + mini-challenge (beat the baseline under a cluster split).
- Compute budget: random forest with 300 trees on 1117×2048 fingerprints,
  < 5 s; entire notebook < 30 s.

## Slides plan
- Session A ~8 slides: features/labels/model/loss; bias-variance cartoon;
  cross-validation diagram; metrics. Demo → notebook A §1–§4.
- Session B ~8 slides: leaderboard table; feature importance bar; leakage
  before/after. Demo → notebook B §1, §4.
- Figures: `overfitting_train_test_gap.png`, `cv_fold_scores.png`,
  `parity_plot_linear.png`, `leaderboard.png`, `feature_importance.png`,
  `leakage_random_vs_cluster_split.png`.

## Assessment hooks
- Examinable: why test error, not train error, measures generalisation; how to
  read a cross-validation score spread; MAE vs RMSE vs R² trade-offs; why a
  cluster/scaffold split gives a more honest error estimate.

## Sources used
- dmol.pub *Introduction to ML* (features/labels/model/loss, SGD, standardisation).
- dmol.pub *Regression & model assessment* (train/test, k-fold, LOOCV,
  bias-variance decomposition, L1/L2 regularisation, AqSolDB example).
- Pat Walters `regression_model.ipynb` (linear→tree-based leaderboard pattern;
  we substitute `RandomForestRegressor` for LightGBM, which is not in
  `env/environment.yml`) and `cross_validation.ipynb` (grouped/clustered
  splitting to avoid leakage — we reuse Week 02B's Butina clusters directly
  instead of a fresh 5×5 repeated study, to keep runtime and scope
  laptop-appropriate for a first ML session).

## Open questions for instructor
- dmol's own worked example uses AqSolDB (~10k compounds); we continue with
  ESOL (1117, from Week 02) for continuity and speed — confirm this
  substitution is acceptable, or whether AqSolDB should be introduced here
  instead.
- Walters' notebooks use LightGBM; we use `RandomForestRegressor` (already
  in-scope via scikit-learn) to avoid a new heavy dependency. Confirm, or add
  `lightgbm` to `env/environment.yml`.
