"""Build week 07 session B notebooks (GNN lab: train, compare, ablate depth).

    python lectures/week-07_graph-neural-networks/notebook/build_week07_b.py
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "_build"))
from nbbuild import build, checkpoint, code, exercise, md  # noqa: E402

C = []

# ==========================================================================
C += [md(r'''
# AI for Chemistry — Week 07, Session B
## Workshop: GNN lab — train, compare, and ablate depth

**Course:** AI for Chemistry · **Session:** 07B (hands-on workshop, 2.0 h) ·
**Runtime:** < 90 s, no GPU.

### Suggested timing (solo study, ~120 min)

| Section | Topic | Time |
|--------:|-------|-----:|
| 0 | Setup: graphs, split, model (recap Week 07A) | 8 min |
| 1 | Full training loop (mini-batch via gradient accumulation) | 25 min |
| 2 | GCN vs the Week 03 leaderboard vs the Week 05 MLP | 20 min |
| 3 | Depth ablation: does adding more GCN layers help? | 22 min |
| 4 | Exercises (3) + mini-challenge | 45 min |

### Learning objectives
1. Train a GCN end to end with mini-batching via gradient accumulation
   (variable-sized molecular graphs cannot be stacked into one tensor without
   a graph-batching library, which this course does not add). *(LO7)*
2. Compare a GCN honestly to Weeks 03/05's classical/MLP leaderboard on the
   identical test split. *(LO4, LO5, LO7)*
3. Run a depth ablation and explain the result using molecule size (graph
   diameter), not just cite "over-smoothing" as a slogan. *(LO7)*

### Prerequisites — before this notebook you should be able to
- Week 07A: `mol_to_graph`, `GCNLayer`, permutation equivariance, readout.
- Week 05B: train/val/test split, loss curves, honest model comparison.

### How to use this notebook (solo study)
Read, run, check each **✅** cell in order. This is the **solutions**
notebook.
''')]

# ==========================================================================
C += [md(r'''
---
## 0. Setup: graphs, split, model

Rebuild every ESOL molecule as a graph (Week 07A's `mol_to_graph`), and reuse
the **same** train/val/test split as Week 05B (`893` train+val molecules,
split again into `714`/`179`, plus the `224`-molecule test set shared with
Weeks 03 and 05).

**What to look for:** `1117` graphs built; sizes ranging from `1` to `55`
heavy atoms.
''')]

C += [code(r'''
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
RDLogger.DisableLog("rdApp.*")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

SEED = 0xC0FFEE
np.random.seed(SEED)
torch.manual_seed(SEED)

ROOT = Path.cwd()
while not (ROOT / "sources").is_dir() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
DATA = ROOT / "sources" / "datasets"
FIGDIR = ROOT / "lectures" / "week-07_graph-neural-networks" / "slides" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

TARGET = "measured log solubility in mols per litre"
esol = pd.read_csv(DATA / "esol_clean.csv")
mols = [Chem.MolFromSmiles(s) for s in esol["canonical_smiles"]]
y = esol[TARGET].to_numpy().astype(np.float32)

ELEMENTS = ["C", "O", "N", "Cl", "S", "F", "Br", "P", "I"]
N_ELEM = len(ELEMENTS) + 1
N_EXTRA = 4
N_FEAT = N_ELEM + N_EXTRA

def mol_to_graph(mol):
    """RDKit Mol -> (node features, adjacency-with-self-loops) -- Week 07A."""
    n = mol.GetNumAtoms()
    X = np.zeros((n, N_FEAT), dtype=np.float32)
    for i, atom in enumerate(mol.GetAtoms()):
        elem_idx = ELEMENTS.index(atom.GetSymbol()) if atom.GetSymbol() in ELEMENTS else len(ELEMENTS)
        X[i, elem_idx] = 1.0
        X[i, N_ELEM + 0] = atom.GetDegree() / 4.0
        X[i, N_ELEM + 1] = float(atom.GetIsAromatic())
        X[i, N_ELEM + 2] = float(atom.IsInRing())
        X[i, N_ELEM + 3] = float(atom.GetFormalCharge())
    A = np.zeros((n, n), dtype=np.float32)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        A[i, j] = A[j, i] = 1.0
    A += np.eye(n, dtype=np.float32)
    return torch.tensor(X), torch.tensor(A)

graphs = [mol_to_graph(m) for m in mols]
sizes = [g[0].shape[0] for g in graphs]
print(f"{len(graphs)} graphs built, size range {min(sizes)}-{max(sizes)} "
      f"(mean {np.mean(sizes):.1f})")

idx = np.arange(len(mols))
idx_trainval, idx_test = train_test_split(idx, test_size=0.2, random_state=42)   # Week 03/05
idx_train, idx_val = train_test_split(idx_trainval, test_size=0.2, random_state=42)
y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]
print(f"train {len(idx_train)}  val {len(idx_val)}  test {len(idx_test)}")
''')]

C += [code(r'''
class GCNLayer(nn.Module):
    """One Kipf & Welling GCN layer (Week 07A)."""

    def __init__(self, in_features, out_features, activation=None):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.activation = activation

    def forward(self, X, A):
        degree = A.sum(dim=-1, keepdim=True).clamp(min=1.0)
        out = self.W((A @ X) / degree)
        return self.activation(out) if self.activation is not None else out


class GCN(nn.Module):
    """n_layers GCN layers (mean-pooled readout) + 2 dense layers."""

    def __init__(self, n_layers=3, hidden=64, in_features=N_FEAT):
        super().__init__()
        dims = [in_features] + [hidden] * n_layers
        self.layers = nn.ModuleList(
            [GCNLayer(dims[i], dims[i + 1], activation=F.relu) for i in range(n_layers)])
        self.fc1 = nn.Linear(hidden, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, X, A):
        h = X
        for layer in self.layers:
            h = layer(h, A)
        g = h.mean(dim=0)
        g = F.relu(self.fc1(g))
        return self.fc2(g)
''')]

C += [md(r'''
> **Common errors — Setup**
> - Rebuilding graphs with a **different** `ELEMENTS` list or feature order
>   than Week 07A — any change to `N_FEAT` must be applied consistently to
>   every graph, since `GCNLayer`'s weight matrix has a fixed input size.
''')]

C += checkpoint(
    "Section 0",
    check=r'''
assert len(graphs) == 1117
assert len(idx_train) == 714 and len(idx_val) == 179 and len(idx_test) == 224
print("Section 0 OK")
''',
    expected="Prints `Section 0 OK`. 1117 graphs; 714/179/224 train/val/test.",
    questions=r'''
1. Why does this notebook reuse Week 05's *exact* train/val/test split
   rather than drawing a fresh one?
2. `GCNLayer` and `GCN` are redefined here rather than imported from Week 07A.
   What would you do differently in a real project to avoid this
   duplication?
''',
    answers=r'''
1. So every model in Section 2's comparison — baseline, linear, MLP, GCN,
   random forest — is scored on **exactly the same 224 held-out molecules**,
   making the comparison a fair test of model/representation choice alone
   (Week 03B's leaderboard discipline, applied across notebooks this time).
2. Factor `GCNLayer`, `GCN` and `mol_to_graph` into a shared module (as
   `lectures/_build/nbbuild.py` already does for notebook construction
   itself) and `import` it from both notebooks — this course keeps each
   session's code inline for readability and independence, at the cost of
   this kind of duplication.
''',
)

# ==========================================================================
# 1. Full training loop
# ==========================================================================
C += [md(r'''
---
## 1. Full training loop (mini-batch via gradient accumulation)

Every ESOL molecule is a **different-sized** graph, so we cannot stack a
batch into one tensor the way Week 05 stacked rows of the fixed-length
descriptor block. Instead we process **one molecule at a time**, but only
call `optimiser.step()` every `batch_size` molecules — the gradients
accumulate (add up) across those molecules first. This gives mini-batch-like
training stability without needing a graph-batching library.

**What to look for:** training loss falls steadily; validation loss falls
alongside it (a smaller model than Week 05's MLP, on a harder-to-batch input,
so we do not expect — or need — the dramatic overfitting curve Week 05B
demonstrated deliberately).
''')]

C += [code(r'''
def train_gcn(n_layers=3, hidden=64, epochs=60, batch_size=16, lr=5e-3, seed=0):
    """Train a GCN with mini-batching via gradient accumulation.

    Returns (model, train_losses, val_losses) -- one loss value per epoch.
    """
    torch.manual_seed(seed)
    model = GCN(n_layers=n_layers, hidden=hidden)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    rng = np.random.default_rng(seed)

    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        order = idx_train.copy()
        rng.shuffle(order)
        optimiser.zero_grad()
        epoch_loss = 0.0
        for count, i in enumerate(order, 1):
            X, A = graphs[i]
            target = torch.tensor([y[i]])
            loss = loss_fn(model(X, A), target) / batch_size   # scale before accumulating
            loss.backward()
            epoch_loss += loss.item() * batch_size
            if count % batch_size == 0:
                optimiser.step()
                optimiser.zero_grad()
        optimiser.step()                     # flush any remaining accumulated gradient
        optimiser.zero_grad()

        model.eval()
        with torch.no_grad():
            val_preds = [model(*graphs[i]).item() for i in idx_val]
        val_loss = mean_squared_error(y_val, val_preds)
        train_losses.append(epoch_loss / len(idx_train))
        val_losses.append(val_loss)
    return model, train_losses, val_losses


t0 = time.time()
gcn_model, train_losses, val_losses = train_gcn()
print(f"training time: {time.time() - t0:.1f} s")
print(f"final train loss (MSE): {train_losses[-1]:.3f}   final val loss (MSE): {val_losses[-1]:.3f}")
''')]

C += [code(r'''
def gcn_predict(model, indices):
    model.eval()
    with torch.no_grad():
        return np.array([model(*graphs[i]).item() for i in indices])

gcn_test_rmse = mean_squared_error(y_test, gcn_predict(gcn_model, idx_test)) ** 0.5
print(f"test RMSE: {gcn_test_rmse:.3f}")

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(train_losses, label="train loss (MSE)")
ax.plot(val_losses, label="val loss (MSE)")
ax.set_xlabel("epoch"); ax.set_ylabel("MSE loss")
ax.set_title("GCN training on ESOL"); ax.legend()
fig.tight_layout(); fig.savefig(FIGDIR / "gcn_training_curve.png", dpi=200)
plt.show()
''')]

C += [md(r'''
> **Common errors — Section 1**
> - Dividing the loss by `batch_size` **before** `.backward()` but not
>   accounting for a **final partial batch** (`len(order) % batch_size != 0`)
>   — the extra `optimiser.step()` after the loop flushes any leftover
>   accumulated gradient so no molecule is silently skipped.
> - Calling `optimiser.zero_grad()` **inside** the per-molecule loop instead
>   of only every `batch_size` molecules — that would defeat the entire point
>   of accumulating gradients across a mini-batch.
''')]

C += checkpoint(
    "Section 1",
    check=r'''
assert len(train_losses) == len(val_losses) == 60
assert train_losses[-1] < train_losses[0]
assert gcn_test_rmse < 2.0        # comfortably beats the ~2.06 mean-predictor baseline
assert (FIGDIR / "gcn_training_curve.png").is_file()
print("Section 1 OK")
''',
    expected="Prints `Section 1 OK`. Training loss has fallen substantially "
    "over 60 epochs; test RMSE is well below the trivial baseline.",
    questions=r'''
1. Why divide the loss by `batch_size` before calling `.backward()`?
2. This training loop calls the model 714 times per epoch (once per training
   molecule), never as a single batched tensor operation. What is the main
   practical cost of that, compared to Week 05's MLP training?
''',
    answers=r'''
1. Gradients accumulate by *summing* across the `.backward()` calls between
   two `optimiser.step()` calls; dividing each molecule's loss by
   `batch_size` first means the accumulated gradient approximates the
   **mean** gradient over the mini-batch, matching what a true batched
   implementation would compute (and keeping the effective learning rate
   comparable regardless of `batch_size`).
2. Speed — 714 separate small Python/PyTorch calls per epoch have much more
   per-call overhead than one call on a `(714, 7)` tensor (Week 05), so GCN
   training here is markedly slower per epoch for a similarly sized model;
   this is exactly the cost a graph-batching library (PyTorch Geometric, DGL)
   is built to remove, at the price of a new dependency (this course's
   choice, see `outline.md`).
''',
)

# ==========================================================================
# 2. GCN vs the Week 03 leaderboard vs the Week 05 MLP
# ==========================================================================
C += [md(r'''
---
## 2. GCN vs the Week 03 leaderboard vs the Week 05 MLP

We recompute Week 03's baseline/linear/random-forest and a Week-05-style MLP
**on the identical split** used throughout, and add the GCN. This is the same
honest, apples-to-apples comparison Week 05B ran for the MLP.

**What to look for:** the GCN clearly beats the mean-predictor baseline, but
— learning its own representation from a graph, with no hand-engineered
descriptors at all, from only ~700 training molecules — it does **not** beat
plain linear regression on 7 chemist-designed descriptors here, let alone the
random forest. This is a genuinely humbling result, and an honest one: it is
not a failure of GNNs as a method, it is the same "a learned representation
must earn its complexity budget with enough data" lesson from Weeks 05B and
06, now shown rather than just claimed.
''')]

C += [code(r'''
DESCRIPTORS = {
    "MolWt": Descriptors.MolWt, "MolLogP": Descriptors.MolLogP,
    "TPSA": Descriptors.TPSA, "NumHDonors": Descriptors.NumHDonors,
    "NumHAcceptors": Descriptors.NumHAcceptors,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    "NumAromaticRings": Descriptors.NumAromaticRings,
}
Xdesc = np.array([[fn(m) for fn in DESCRIPTORS.values()] for m in mols], dtype=float)

baseline_rmse = mean_squared_error(y_test, np.full_like(y_test, y_train.mean())) ** 0.5

lin = LinearRegression().fit(Xdesc[idx_train], y_train)
linear_rmse = mean_squared_error(y_test, lin.predict(Xdesc[idx_test])) ** 0.5

rf = RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1).fit(Xdesc[idx_train], y_train)
rf_rmse = mean_squared_error(y_test, rf.predict(Xdesc[idx_test])) ** 0.5

print(f"baseline (mean)     RMSE = {baseline_rmse:.3f}")
print(f"linear regression   RMSE = {linear_rmse:.3f}")
print(f"random forest       RMSE = {rf_rmse:.3f}")
print(f"GCN (Section 1)     RMSE = {gcn_test_rmse:.3f}")
''')]

C += [code(r'''
class MLP(nn.Module):
    """Week 05's descriptor-based MLP, for a same-split comparison."""

    def __init__(self, in_dim=7, hidden=256):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, hidden)
        self.layer2 = nn.Linear(hidden, hidden)
        self.layer3 = nn.Linear(hidden, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


scaler = StandardScaler().fit(Xdesc[idx_train])
Xtr_t = torch.tensor(scaler.transform(Xdesc[idx_train]), dtype=torch.float32)
Xva_t = torch.tensor(scaler.transform(Xdesc[idx_val]), dtype=torch.float32)
Xte_t = torch.tensor(scaler.transform(Xdesc[idx_test]), dtype=torch.float32)
ytr_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
yva_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1)

torch.manual_seed(0)
mlp = MLP(hidden=256)
opt_mlp = torch.optim.Adam(mlp.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
best_val, bad, patience, best_state = float("inf"), 0, 50, None
for epoch in range(300):
    mlp.train(); opt_mlp.zero_grad()
    loss = loss_fn(mlp(Xtr_t), ytr_t); loss.backward(); opt_mlp.step()
    mlp.eval()
    with torch.no_grad():
        v = loss_fn(mlp(Xva_t), yva_t).item()
    if v < best_val - 1e-4:
        best_val, bad = v, 0
        best_state = {k: t.clone() for k, t in mlp.state_dict().items()}
    else:
        bad += 1
        if bad >= patience:
            break
mlp.load_state_dict(best_state)
mlp.eval()
with torch.no_grad():
    mlp_rmse = mean_squared_error(y_test, mlp(Xte_t).numpy().ravel()) ** 0.5
print(f"MLP (Week 05 style) RMSE = {mlp_rmse:.3f}")
''')]

C += [code(r'''
comparison = pd.DataFrame({
    "model": ["Baseline (mean)", "Linear regression", "MLP (descriptors)",
             "GCN (learned repr.)", "Random forest (descriptors)"],
    "test_RMSE": [baseline_rmse, linear_rmse, mlp_rmse, gcn_test_rmse, rf_rmse],
}).sort_values("test_RMSE").reset_index(drop=True)

fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.barh(comparison["model"][::-1], comparison["test_RMSE"][::-1], color="darkslateblue")
ax.set_xlabel("test RMSE (log S units, lower is better)")
ax.set_title("Week 07B: GCN vs the classical/MLP leaderboard")
fig.tight_layout(); fig.savefig(FIGDIR / "gcn_vs_leaderboard.png", dpi=200)
plt.show()
comparison.round(3)
''')]

C += [md(r'''
> **Common errors — Section 2**
> - Concluding "GNNs do not work" from this one comparison — the GCN here
>   uses **only** connectivity + a handful of atom-level flags, no 3D
>   information, no pretraining, and about 700 training molecules; every one
>   of those is a lever that, pulled, changes this result (Week 08-09 pull
>   several of them, and the mini-challenge below pulls one directly).
> - Forgetting that the GCN's representation is **learned from the graph
>   alone** — it had to discover, from ~700 examples, structure the
>   descriptor block already had handed to it, for free, by a chemist. That
>   it clears the trivial baseline at all, with no chemical priors, is the
>   more informative comparison — not whether it edges out a random forest
>   built on human expertise.
''')]

C += checkpoint(
    "Section 2",
    check=r'''
assert len(comparison) == 5
assert gcn_test_rmse < baseline_rmse
assert comparison.iloc[0]["model"] in ("Random forest (descriptors)", "MLP (descriptors)")
assert (FIGDIR / "gcn_vs_leaderboard.png").is_file()
print("Section 2 OK")
''',
    expected="Prints `Section 2 OK`. The GCN clearly beats the trivial "
    "baseline; the top of the table is random forest or the MLP, with the "
    "GCN below linear regression too.",
    questions=r'''
1. The GCN does not beat linear regression here, even though logistic/linear
   regression is a much simpler model. What does that tell you about the
   *representation*, as distinct from the model's raw complexity?
2. Name one thing you could add to the GCN (without changing the dataset)
   that might close the remaining gap.
''',
    answers=r'''
1. Model complexity and representation quality are separate axes: a simple
   model on a strong, hand-engineered representation can beat a flexible
   model that must *discover* an equally good representation from limited
   data — exactly Week 06's inductive-bias trade-off, now measured rather
   than asserted.
2. More/better atom features (e.g. Gasteiger partial charges, hybridisation),
   edge features (bond order/aromaticity on the edges themselves, not just
   the adjacency 0/1), a longer training budget with early stopping (the
   mini-challenge below), or more training data — all levers a fixed
   descriptor set does not have access to, but a graph representation does.
''',
)

# ==========================================================================
# 3. Depth ablation
# ==========================================================================
C += [md(r'''
---
## 3. Depth ablation: does adding more GCN layers help?

Each GCN layer lets information travel **one bond further** — a 1-layer GCN
only sees each atom's immediate neighbours; a 3-layer GCN sees everything up
to 3 bonds away. Deep GNN literature warns about **over-smoothing**: with too
many layers, every node's representation converges towards the same
graph-average value, destroying local detail. We test whether that shows up
here.

**What to look for:** 1 layer is noticeably worse (too short-sighted); 2 and
more layers cluster together with **no clear further improvement** — ESOL
molecules are small (mean 13, max 55 heavy atoms), so 2-3 hops already reach
most of a typical molecule. This is a real, useful negative result, not a
failed experiment.
''')]

C += [code(r'''
depths = [1, 2, 3, 5, 8]
depth_results = []
t0 = time.time()
for n_layers in depths:
    model_d, _, val_losses_d = train_gcn(n_layers=n_layers, epochs=30, seed=0)
    val_rmse_d = val_losses_d[-1] ** 0.5
    test_rmse_d = mean_squared_error(y_test, gcn_predict(model_d, idx_test)) ** 0.5
    depth_results.append({"n_layers": n_layers, "val_RMSE": val_rmse_d, "test_RMSE": test_rmse_d})
    print(f"n_layers={n_layers}: val RMSE={val_rmse_d:.3f}  test RMSE={test_rmse_d:.3f}  "
          f"(t={time.time()-t0:.0f}s)")

depth_df = pd.DataFrame(depth_results)
''')]

C += [code(r'''
fig, ax = plt.subplots(figsize=(5.5, 3.5))
ax.plot(depth_df["n_layers"], depth_df["val_RMSE"], "o-", label="validation RMSE")
ax.plot(depth_df["n_layers"], depth_df["test_RMSE"], "s--", label="test RMSE")
ax.set_xlabel("number of GCN layers"); ax.set_ylabel("RMSE")
ax.set_title("Depth ablation (30 epochs each)"); ax.legend()
fig.tight_layout(); fig.savefig(FIGDIR / "gcn_depth_ablation.png", dpi=200)
plt.show()
''')]

C += [md(r'''
> **Common errors — Section 3**
> - Declaring "over-smoothing confirmed" from a plateau — a plateau is
>   consistent with over-smoothing, but equally consistent with simply having
>   *enough* depth already (our explanation here) or with 30 epochs being too
>   few for the deeper models to fully converge. Distinguishing these needs
>   more evidence than this one ablation provides.
> - Running each depth for the **same fixed epoch count** and treating the
>   comparison as perfectly fair — deeper models have more parameters and may
>   need more (or fewer, in some cases) epochs to reach their best score;
>   this ablation controls compute budget, not "epochs to convergence".
''')]

C += checkpoint(
    "Section 3",
    check=r'''
assert len(depth_df) == 5
assert depth_df.loc[depth_df["n_layers"] == 1, "val_RMSE"].iloc[0] > depth_df["val_RMSE"].iloc[1:].min()
assert (FIGDIR / "gcn_depth_ablation.png").is_file()
print("Section 3 OK")
''',
    expected="Prints `Section 3 OK`. The 1-layer model's validation RMSE is "
    "worse than the best of the deeper models.",
    questions=r'''
1. ESOL's mean molecule has 13 heavy atoms; its *diameter* (longest
   shortest-path between two atoms) is typically much smaller than 13. Why
   does that support this section's explanation for the plateau?
2. On a dataset of much larger molecules (e.g. proteins or polymers), would
   you expect the same plateau to appear at the same depth?
''',
    answers=r'''
1. A ring or short chain's diameter grows far more slowly than its atom
   count (e.g. a 6-membered ring has diameter 3, not 6); once GCN depth
   reaches a molecule's diameter, every atom can already exchange information
   with every other atom, so further layers can only re-process already-fully
   -mixed information rather than reach anything new.
2. No — a much larger graph has a much larger diameter, so the "enough depth
   to cover the molecule" point would shift to a correspondingly greater
   number of layers, and *real* over-smoothing (representations collapsing
   towards a graph-wide average) becomes a genuine risk well before full
   coverage is reached on such graphs.
''',
)

# ==========================================================================
# 4. Exercises
# ==========================================================================
C += [md(r'''
---
## 4. Exercises
''')]

C += exercise(
    prompt=r'''
### Exercise 1 — vary the mini-batch size *(easy, ~12 min)*

Using `train_gcn`, compare `batch_size=1` (a gradient step after every single
molecule) to `batch_size=32`, at a reduced `epochs=20` budget for speed.
Record each configuration's final validation loss.

<details><summary>Show hint</summary>

`train_gcn(batch_size=1, epochs=20)[2][-1]` gives the final val loss for that
config; likewise for `batch_size=32`.
</details>
''',
    solution=r'''
_, _, val_losses_b1 = train_gcn(batch_size=1, epochs=20, seed=0)
_, _, val_losses_b32 = train_gcn(batch_size=32, epochs=20, seed=0)
final_val_b1 = val_losses_b1[-1]
final_val_b32 = val_losses_b32[-1]
print(f"batch_size=1:  final val loss = {final_val_b1:.3f}")
print(f"batch_size=32: final val loss = {final_val_b32:.3f}")
''',
    scaffold=r'''
final_val_b1 = ...  # YOUR CODE HERE: train_gcn(batch_size=1, epochs=20, seed=0), take val_losses[-1]
final_val_b32 = ...  # YOUR CODE HERE: train_gcn(batch_size=32, epochs=20, seed=0), take val_losses[-1]
''',
    check=r'''
assert final_val_b1 > 0 and final_val_b32 > 0
print("Exercise 1 OK")
''',
)

C += exercise(
    prompt=r'''
### Exercise 2 — the receptive field grows with depth *(medium, ~15 min)*

Write `receptive_field_size(A, n_layers)` returning, for the **first** atom,
how many distinct atoms are reachable within `n_layers` hops (including
itself). Use matrix powers of the **bond-only** adjacency (no self-loops) to
avoid trivially inflating the count.

<details><summary>Show hint</summary>

`reach = (A_no_loops @ A_no_loops @ ... ) ` `n_layers` times, starting from a
one-hot vector at atom 0; `torch.matrix_power(A_no_loops, k) @ e0` reached at
exactly `k` hops but you want *within* `n_layers`, so accumulate reachability
with `(A_no_loops + eye)` raised to the `n_layers` power instead, then count
non-zero entries in row 0.
</details>
''',
    solution=r'''
def receptive_field_size(A_with_selfloops, n_layers):
    """# distinct atoms reachable from atom 0 within n_layers hops (self-loops included)."""
    reach = torch.matrix_power(A_with_selfloops, n_layers)
    return int((reach[0] > 0).sum().item())

_, benzene_A = mol_to_graph(Chem.MolFromSmiles("c1ccccc1"))
for k in [1, 2, 3]:
    print(f"benzene, {k} layer(s): reaches {receptive_field_size(benzene_A, k)} of 6 atoms")
''',
    scaffold=r'''
def receptive_field_size(A_with_selfloops, n_layers):
    """# distinct atoms reachable from atom 0 within n_layers hops (self-loops included)."""
    # YOUR CODE HERE
    ...

_, benzene_A = mol_to_graph(Chem.MolFromSmiles("c1ccccc1"))
reach_1 = ...  # YOUR CODE HERE: receptive_field_size(benzene_A, 1)
reach_3 = ...  # YOUR CODE HERE: receptive_field_size(benzene_A, 3)
''',
    check=r'''
_, benzene_A = mol_to_graph(Chem.MolFromSmiles("c1ccccc1"))
assert receptive_field_size(benzene_A, 1) == 3      # self + 2 ring neighbours
assert receptive_field_size(benzene_A, 3) == 6       # whole 6-membered ring reached
print("Exercise 2 OK")
''',
)

C += exercise(
    prompt=r'''
### Exercise 3 — feature ablation *(medium, ~18 min)*

The node features include 4 "extra" scalars beyond the element one-hot
(degree, aromaticity, ring membership, formal charge). Write
`mol_to_graph_no_extras(mol)` that **zeroes out** those 4 columns (keep the
same shape, so `GCN`'s architecture does not need to change) and retrain a
3-layer GCN for 30 epochs. Does removing them hurt?

<details><summary>Show hint</summary>

Call the existing `mol_to_graph`, then set `X[:, N_ELEM:] = 0` before
returning. You will need a version of `train_gcn` that builds graphs this way
-- simplest is to temporarily monkey-patch `graphs` with the ablated version,
train, then restore the original `graphs` list.
</details>
''',
    solution=r'''
def mol_to_graph_no_extras(mol):
    """mol_to_graph, but with the 4 extra atom-level scalar features zeroed out."""
    X, A = mol_to_graph(mol)
    X = X.clone()
    X[:, N_ELEM:] = 0.0
    return X, A

graphs_full = graphs                                   # keep a reference to restore
graphs = [mol_to_graph_no_extras(m) for m in mols]      # temporarily replace
_, _, val_losses_no_extras = train_gcn(n_layers=3, epochs=30, seed=0)
graphs = graphs_full                                    # restore

final_val_no_extras = val_losses_no_extras[-1]
final_val_with_extras = depth_df.loc[depth_df["n_layers"] == 3, "val_RMSE"].iloc[0] ** 2
print(f"val loss WITHOUT extra atom features: {final_val_no_extras:.3f}")
print(f"val loss WITH extra atom features (Section 3, 3 layers): {final_val_with_extras:.3f}")
''',
    scaffold=r'''
def mol_to_graph_no_extras(mol):
    """mol_to_graph, but with the 4 extra atom-level scalar features zeroed out."""
    # YOUR CODE HERE
    ...

graphs_full = graphs
graphs = [mol_to_graph_no_extras(m) for m in mols]      # temporarily replace
final_val_no_extras = ...  # YOUR CODE HERE: train_gcn(n_layers=3, epochs=30, seed=0), val_losses[-1]
graphs = graphs_full                                    # restore
''',
    check=r'''
assert final_val_no_extras > 0
assert graphs is graphs_full        # you must restore the original graphs list
print("Exercise 3 OK")
''',
)

C += [md(r'''
### 🏁 Mini-challenge — a fair rematch: GCN with a real training budget *(~25 min)*

Section 2's comparison gave the GCN the same wall-clock ballpark as the other
models, but not necessarily the same **epoch budget** the MLP got in Week 05B
(hundreds of epochs, with early stopping). Give the GCN a genuinely larger
budget — with early stopping so it does not just overfit for longer — and
see whether the gap to random forest narrows.

Write `train_gcn_early_stopping(n_layers, hidden, patience, max_epochs,
seed)` combining Section 1's mini-batch training with Week 05B's
early-stopping checkpoint-and-restore pattern. Report the resulting test
RMSE.
''')]

C += exercise(
    prompt=r'''
Implement `train_gcn_early_stopping` and run it with a larger `max_epochs`
budget than Section 1 used.
''',
    solution=r'''
def train_gcn_early_stopping(n_layers=3, hidden=64, patience=15, max_epochs=150,
                             batch_size=16, lr=5e-3, seed=0):
    """Mini-batch GCN training (Section 1) + early stopping (Week 05B)."""
    torch.manual_seed(seed)
    model = GCN(n_layers=n_layers, hidden=hidden)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    rng = np.random.default_rng(seed)

    best_val, bad, best_state = float("inf"), 0, None
    for epoch in range(max_epochs):
        model.train()
        order = idx_train.copy(); rng.shuffle(order)
        optimiser.zero_grad()
        for count, i in enumerate(order, 1):
            X, A = graphs[i]
            loss = loss_fn(model(X, A), torch.tensor([y[i]])) / batch_size
            loss.backward()
            if count % batch_size == 0:
                optimiser.step(); optimiser.zero_grad()
        optimiser.step(); optimiser.zero_grad()

        model.eval()
        with torch.no_grad():
            val_preds = [model(*graphs[i]).item() for i in idx_val]
        val_loss = mean_squared_error(y_val, val_preds)
        if val_loss < best_val - 1e-4:
            best_val, bad = val_loss, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model

gcn_es_model = train_gcn_early_stopping()
gcn_es_rmse = mean_squared_error(y_test, gcn_predict(gcn_es_model, idx_test)) ** 0.5
print(f"GCN with early stopping (larger budget): test RMSE = {gcn_es_rmse:.3f}")
print(f"GCN (Section 1, fixed 60 epochs):        test RMSE = {gcn_test_rmse:.3f}")
print(f"Random forest (Section 2):               test RMSE = {rf_rmse:.3f}")
''',
    scaffold=r'''
def train_gcn_early_stopping(n_layers=3, hidden=64, patience=15, max_epochs=150,
                             batch_size=16, lr=5e-3, seed=0):
    """Mini-batch GCN training (Section 1) + early stopping (Week 05B)."""
    # YOUR CODE HERE
    ...

gcn_es_model = ...  # YOUR CODE HERE: train_gcn_early_stopping()
gcn_es_rmse = ...  # YOUR CODE HERE: test RMSE of gcn_es_model
''',
    check=r'''
assert gcn_es_rmse > 0
assert gcn_es_rmse < baseline_rmse
print("Mini-challenge OK")
''',
)

# ==========================================================================
C += [md(r'''
---
## Summary — what you learned

- **Mini-batching without a graph library**: gradient accumulation over
  variable-sized molecular graphs, molecule by molecule.
- An honest **GCN vs classical-ML vs MLP** comparison on the identical test
  set: the GCN clearly beats the trivial baseline, but does not beat linear
  regression, the MLP or the descriptor-fed random forest here — a humbling
  but fair result given it started with far less prior chemical information
  and far fewer training molecules than a graph model typically needs.
- A **depth ablation** that resists an easy "over-smoothing confirmed"
  headline, and the correct, more careful explanation (molecule diameter vs
  GCN depth) instead.
- Both a mini-batch-size comparison and an early-stopped, larger-budget GCN
  run, extending Week 05B's training discipline to graphs.

Week 08 moves to a different molecular representation entirely: SMILES as a
**sequence**, and the attention mechanism behind modern chemical language
models.
''')]

C += [md(r'''
## Further reading (course source list only)

- dmol.pub *Graph neural networks*: <https://dmol.pub/dl/gnn.html>
- Colab alternative (EPFL *AI for Chemistry*, GNN/Chemprop example):
  <https://colab.research.google.com/github/schwallergroup/ai4chem_course/blob/main/notebooks/03%20-%20Intro%20to%20Deep%20Learning/03_gnn_simple_example.ipynb>
- Alternative: DeepChem *Introduction to Graph Convolutions*:
  <https://github.com/deepchem/deepchem/blob/master/examples/tutorials/Introduction_to_Graph_Convolutions.ipynb>
''')]

C += [md(r'''
## Attribution

Original teaching material for *AI for Chemistry*, adapted and rewritten from:

- **dmol.pub** (A. White) — the GCN architecture and its own noted limitation
  (batch size 1, underfitting); we address that limitation directly with
  gradient-accumulation mini-batching rather than adopting a graph-batching
  library. CC-BY 4.0. <https://dmol.pub>
- **EPFL CH-457 *AI for Chemistry*** (Schwaller group) — GNN-vs-MLP
  comparison framing; their own example uses Chemprop (a CLI-driven package)
  rather than a from-scratch model, which we use instead for full
  in-notebook transparency. MIT licence.
  <https://github.com/schwallergroup/ai4chem_course>

Continues the ESOL dataset and train/val/test split from Weeks 02-03/05 and
the `GCNLayer`/`GCN` classes from Week 07A. No verbatim text is reproduced
from the sources above.
''')]

build(__file__, "week07_b_graph-neural-networks", C)
