"""Build week 07 session A notebooks (molecules as graphs, message passing,
GCN mechanics).

    python lectures/week-07_graph-neural-networks/notebook/build_week07_a.py
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "_build"))
from nbbuild import build, checkpoint, code, exercise, md  # noqa: E402

C = []

# ==========================================================================
C += [md(r'''
# AI for Chemistry — Week 07, Session A
## Graph neural networks: molecules as graphs, message passing, GCN

**Course:** AI for Chemistry · **Session:** 07A (interleaved lecture + lab,
2.5 h) · **Runtime:** < 10 s, no GPU.

### Suggested timing (solo study, ~120 min)

| Section | Topic | Time |
|--------:|-------|-----:|
| 0 | Setup | 8 min |
| 1 | Molecules as graphs | 20 min |
| 2 | Message passing and the GCN equation | 25 min |
| 3 | Graph-level readout: mean vs sum | 18 min |
| 4 | A GCN regression model; one manual step | 20 min |
| 5 | Exercises (4) | 29 min |

### Learning objectives
1. Represent a molecule as an adjacency matrix + node-feature matrix, and
   explain why atom order is arbitrary. *(LO7)*
2. State the message-passing steps (message, aggregate, update) and the
   Kipf & Welling GCN equation. *(LO7)*
3. Implement a `GCNLayer` in PyTorch and verify it is **permutation-
   equivariant** numerically. *(LO6, LO7)*
4. Explain mean vs sum readout and connect each to intensive vs extensive
   molecular properties. *(LO7)*

### Prerequisites — before this notebook you should be able to
- Week 06: the formal definitions of permutation invariance/equivariance.
- Week 05: `nn.Module`, tensors, one manual training step.
- Week 02: RDKit `Mol`, `Atom`, `Bond` objects.

### How to use this notebook (solo study)
Read, run, check each **✅** cell in order. This notebook is
**mechanics-only** — no full training run happens until Week 07B. This is
the **solutions** notebook.
''')]

# ==========================================================================
C += [md(r'''
---
## 0. Setup

We reuse the cleaned ESOL dataset (Weeks 02-03). No new dataset is needed —
we simply convert each molecule's RDKit `Mol` into a graph.

**What to look for:** version numbers print; `esol_clean.csv` reports `OK`.
''')]

C += [code(r'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

SEED = 0xC0FFEE
np.random.seed(SEED)
torch.manual_seed(SEED)

ROOT = Path.cwd()
while not (ROOT / "sources").is_dir() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
DATA = ROOT / "sources" / "datasets"
FIGDIR = ROOT / "lectures" / "week-07_graph-neural-networks" / "slides" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

print("torch", torch.__version__)
print("esol_clean.csv", "OK" if (DATA / "esol_clean.csv").is_file() else "MISSING")
''')]

C += [md(r'''
> **Common errors — Setup**
> - `MISSING` for `esol_clean.csv` — run Week 02B's notebook first, or accept
>   Week 03/05's automatic rebuild-from-`esol_delaney.csv` fallback (not
>   repeated here to keep this notebook focused on graph mechanics).
''')]

# ==========================================================================
# 1. Molecules as graphs
# ==========================================================================
C += [md(r'''
---
## 1. Molecules as graphs

A molecule is a graph: atoms are **nodes**, bonds are **edges**. We describe
it with two matrices:

* **Adjacency matrix** $E \in \{0,1\}^{N\times N}$: $E_{ij}=1$ if atoms $i$
  and $j$ are bonded (symmetric — an undirected graph; $E_{ii}=0$ before we
  add self-loops below).
* **Node-feature matrix** $X \in \mathbb{R}^{N\times D}$: row $i$ describes
  atom $i$ — here, a one-hot element indicator (C, O, N, Cl, S, F, Br, P, I,
  or "other") plus a few scalar atom properties (normalised degree,
  aromaticity, ring membership, formal charge).

Following Kipf & Welling's GCN, we add **self-loops** ($E \to E+I$) so a
node's own features are included when we aggregate its neighbourhood next
section.

**What to look for:** for ethanol (`CCO`, heavy atoms only — RDKit does not
add hydrogens by default), `N=3`; the adjacency matrix (before self-loops) has
exactly 2 ones above the diagonal (C-C, C-O), matching its 2 bonds.
''')]

C += [code(r'''
ELEMENTS = ["C", "O", "N", "Cl", "S", "F", "Br", "P", "I"]
N_ELEM = len(ELEMENTS) + 1     # +1 for "other" element
N_EXTRA = 4                    # degree, aromatic, in-ring, formal charge
N_FEAT = N_ELEM + N_EXTRA

def mol_to_graph(mol):
    """RDKit Mol -> (node features X [N,N_FEAT], adjacency A [N,N] with self-loops)."""
    n = mol.GetNumAtoms()
    X = np.zeros((n, N_FEAT), dtype=np.float32)
    for i, atom in enumerate(mol.GetAtoms()):
        elem_idx = ELEMENTS.index(atom.GetSymbol()) if atom.GetSymbol() in ELEMENTS else len(ELEMENTS)
        X[i, elem_idx] = 1.0
        X[i, N_ELEM + 0] = atom.GetDegree() / 4.0        # normalise roughly to [0,1]
        X[i, N_ELEM + 1] = float(atom.GetIsAromatic())
        X[i, N_ELEM + 2] = float(atom.IsInRing())
        X[i, N_ELEM + 3] = float(atom.GetFormalCharge())

    A = np.zeros((n, n), dtype=np.float32)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        A[i, j] = A[j, i] = 1.0
    n_bonds_before_selfloops = int(A.sum() / 2)
    A += np.eye(n, dtype=np.float32)                      # add self-loops

    return torch.tensor(X), torch.tensor(A), n_bonds_before_selfloops

ethanol = Chem.MolFromSmiles("CCO")
X_eth, A_eth, n_bonds = mol_to_graph(ethanol)
print("N (heavy atoms):", X_eth.shape[0], " feature dim:", X_eth.shape[1])
print("bonds (pre self-loop):", n_bonds)
print("adjacency (with self-loops):\n", A_eth.numpy().astype(int))
''')]

C += [md(r'''
> **Common errors — Section 1**
> - Forgetting `A[j, i] = 1.0` alongside `A[i, j] = 1.0` — bonds are
>   undirected; an asymmetric adjacency matrix silently breaks every
>   downstream calculation that assumes symmetry.
> - Adding self-loops **before** counting real bonds (as here we deliberately
>   count first) — otherwise every diagonal 1 gets double-counted as a "bond".
> - RDKit's default `Mol` has **no explicit hydrogens**; `mol.GetNumAtoms()`
>   counts heavy atoms only, matching this section's graphs.
''')]

C += checkpoint(
    "Section 1",
    check=r'''
assert X_eth.shape == (3, N_FEAT)
assert A_eth.shape == (3, 3)
assert n_bonds == 2
assert torch.allclose(A_eth, A_eth.T)          # symmetric
assert torch.all(A_eth.diagonal() == 1.0)       # self-loops present
print("Section 1 OK")
''',
    expected="Prints `Section 1 OK`. Ethanol: 3 heavy atoms, 2 bonds, "
    "symmetric adjacency with a self-loop diagonal.",
    questions=r'''
1. Why does the choice of which atom is "atom 0" not matter chemically, even
   though it changes the exact numbers in `X` and `A`?
2. What would change in `mol_to_graph` if we wanted to include hydrogens
   explicitly (`Chem.AddHs`)?
''',
    answers=r'''
1. Atom numbering is an artefact of how the SMILES/graph was traversed, not a
   chemical fact (Week 06's "there is no first or second atom") — relabelling
   atoms 0,1,2 as 1,2,0 describes the identical molecule, just with rows/
   columns of `X`/`A` permuted together, which Section 2 shows a correctly
   designed GNN layer handles consistently.
2. `mol.GetNumAtoms()` and the bond loop would include H atoms automatically
   after `Chem.AddHs(mol)`; graphs would become larger (roughly 2-3x for
   typical organic molecules) and `ELEMENTS` would need `"H"` added, at some
   extra compute cost for comparatively little chemical information for most
   properties (bond angles/lengths involving H are usually less
   discriminating than heavy-atom connectivity).
''',
)

# ==========================================================================
# 2. Message passing and the GCN equation
# ==========================================================================
C += [md(r'''
---
## 2. Message passing and the GCN equation

A message-passing layer updates every node's features from its neighbours in
three conceptual steps:

$$\textbf{message:}\ \ \bar e_{ij} = v_j W, \qquad
  \textbf{aggregate:}\ \ \bar e_i = \frac{1}{d_i}\sum_{j} \bar e_{ij}, \qquad
  \textbf{update:}\ \ v_i' = \sigma(\bar e_i),$$

where $v_j$ is neighbour $j$'s feature row, $W$ a trainable weight matrix,
$d_i$ node $i$'s degree (number of neighbours, **including** its self-loop),
and $\sigma$ an activation. Written as one matrix equation over the whole
graph (Kipf & Welling's GCN):

$$V' = \sigma\bigl(D^{-1}EVW\bigr),$$

with $D$ the diagonal degree matrix. Averaging by degree (not just summing)
keeps feature magnitudes from exploding as layers stack (Exercise 4 makes
this concrete).

**Why this is permutation-equivariant**: relabelling atoms with a permutation
$P$ turns $(V,E)$ into $(PV, PEP^\top)$; every step above only ever combines a
node with **its own neighbours**, indexed consistently, so the output simply
gets relabelled by the same $P$ — $f(PV,PEP^\top) = Pf(V,E)$.

**What to look for:** applying the **same** `GCNLayer` (same weights) to
ethanol's graph and to a **relabelled** copy gives node outputs that are
exactly each other's permutation — not identical row-by-row, but identical
*as a set*, correctly reordered.
''')]

C += [code(r'''
class GCNLayer(nn.Module):
    """One Kipf & Welling GCN layer: V' = activation(D^-1 E V W)."""

    def __init__(self, in_features, out_features, activation=None):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.activation = activation

    def forward(self, X, A):
        degree = A.sum(dim=-1, keepdim=True).clamp(min=1.0)   # (N,1)
        aggregated = (A @ X) / degree                          # mean over neighbours
        out = self.W(aggregated)
        return self.activation(out) if self.activation is not None else out

torch.manual_seed(0)
layer = GCNLayer(N_FEAT, 8, activation=F.relu)

out_original = layer(X_eth, A_eth)

perm = torch.tensor([2, 0, 1])            # relabel: new atom 0 = old atom 2, etc.
X_perm = X_eth[perm]
A_perm = A_eth[perm][:, perm]              # permute BOTH rows and columns
out_permuted = layer(X_perm, A_perm)

print("output for original order:\n", out_original.detach().numpy().round(3))
print("output for permuted order:\n", out_permuted.detach().numpy().round(3))
print("is out_permuted == out_original[perm]?",
      torch.allclose(out_permuted, out_original[perm], atol=1e-5))
''')]

C += [md(r'''
> **Common errors — Section 2**
> - Permuting `X` but forgetting to permute `A` **the same way on both axes**
>   (`A[perm][:, perm]`, not just `A[perm]`) — this describes a genuinely
>   different graph, and the equivariance check will (correctly) fail.
> - Using `bias=False` intentionally here so the layer is a pure linear map of
>   the aggregated neighbourhood — a bias term would still be equivariant
>   (it is added identically to every node), so this is a style choice, not a
>   requirement.
''')]

C += checkpoint(
    "Section 2",
    check=r'''
assert out_original.shape == (3, 8)
assert torch.allclose(out_permuted, out_original[perm], atol=1e-5)
print("Section 2 OK")
''',
    expected="Prints `Section 2 OK`. The permuted-input output equals the "
    "original output, reordered by the same permutation.",
    questions=r'''
1. Where, structurally, does the degree normalisation $1/d_i$ live in the
   `forward` method, and why does it need `clamp(min=1.0)`?
2. Is `GCNLayer` **invariant** or **equivariant**? Which would you need for
   predicting solubility (one number per molecule) instead?
''',
    answers=r'''
1. `degree = A.sum(dim=-1, keepdim=True).clamp(min=1.0)` computes each row's
   neighbour count (including the self-loop); clamping guards a node with no
   bonds and no self-loop from a division by zero (should not occur once
   self-loops are added, but is cheap insurance).
2. **Equivariant** — its output is one row per atom, and Section 2 showed
   those rows permute consistently with the input. Solubility is a
   *whole-molecule* scalar, so we need an **invariant** readout on top —
   exactly Section 3's mean/sum pooling, which collapses the equivariant
   per-atom output into a single, permutation-invariant number.
''',
)

# ==========================================================================
# 3. Graph-level readout: mean vs sum
# ==========================================================================
C += [md(r'''
---
## 3. Graph-level readout: mean vs sum

To predict a single molecular property we must collapse the (equivariant)
per-atom output into one (invariant) vector — the **readout**. Two standard
choices:

* **Mean pooling** — suits **intensive** properties (do not scale with
  molecule size): solubility, refractive index, most of what this course
  predicts.
* **Sum pooling** — suits **extensive** properties (scale with size):
  enthalpy of formation, total energy.

**What to look for:** mean-pooled readouts for ethanol and a much larger ESOL
molecule land in a **similar range**; sum-pooled readouts grow roughly with
molecule size — visibly different behaviour from the same underlying node
features.
''')]

C += [code(r'''
def mean_readout(node_features):
    return node_features.mean(dim=0)

def sum_readout(node_features):
    return node_features.sum(dim=0)

esol = pd.read_csv(DATA / "esol_clean.csv")
small_mol = Chem.MolFromSmiles(esol["canonical_smiles"].iloc[0])         # a small molecule
big_smiles = esol.loc[esol["canonical_smiles"].str.len().idxmax(), "canonical_smiles"]
big_mol = Chem.MolFromSmiles(big_smiles)                                  # the largest in ESOL

pool_report = []
for name, mol in [("small", small_mol), ("large", big_mol)]:
    Xg, Ag, _ = mol_to_graph(mol)
    h = layer(Xg, Ag)                      # reuse Section 2's fixed-weight layer
    pool_report.append((name, Xg.shape[0], mean_readout(h).norm().item(), sum_readout(h).norm().item()))
    print(f"{name:6s} (N={Xg.shape[0]:2d}):  "
          f"||mean_readout|| = {pool_report[-1][2]:.3f}   "
          f"||sum_readout|| = {pool_report[-1][3]:.3f}")
''')]

C += [code(r'''
labels = [f"{name}\n(N={n})" for name, n, _, _ in pool_report]
mean_norms = [m for _, _, m, _ in pool_report]
sum_norms = [s for _, _, _, s in pool_report]

fig, ax = plt.subplots(figsize=(5, 3.5))
x = np.arange(len(labels))
ax.bar(x - 0.18, mean_norms, width=0.36, label="mean readout (intensive)")
ax.bar(x + 0.18, sum_norms, width=0.36, label="sum readout (extensive)")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("readout vector norm"); ax.set_title("Pooling behaviour vs molecule size")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIGDIR / "readout_pooling_comparison.png", dpi=200)
plt.show()
''')]

C += [md(r'''
> **Common errors — Section 3**
> - Using sum pooling for an intensive property without realising it — a
>   model trained this way implicitly learns "bigger molecule -> bigger raw
>   pooled signal", which the final dense layers must then learn to undo.
> - Mean pooling an **extensive** property loses the size information
>   entirely (two molecules of very different size but similar per-atom
>   character would get near-identical readouts) — the wrong pooling choice
>   for the property genuinely loses information, not just accuracy.
''')]

C += checkpoint(
    "Section 3",
    check=r'''
Xs, As, _ = mol_to_graph(small_mol)
Xl, Al, _ = mol_to_graph(big_mol)
h_small = layer(Xs, As)
h_large = layer(Xl, Al)
assert Xl.shape[0] > Xs.shape[0]                                   # large mol really is bigger
assert sum_readout(h_large).norm() > sum_readout(h_small).norm()   # sum grows with size
assert (FIGDIR / "readout_pooling_comparison.png").is_file()
print("Section 3 OK")
''',
    expected="Prints `Section 3 OK`. The larger molecule's sum-pooled "
    "readout has a bigger norm than the smaller molecule's.",
    questions=r'''
1. Sketch (in words) what would happen to a sum-pooled model's predictions if
   you evaluated it on a molecule 10x larger than anything in its training
   set.
2. Could you build an intensive-property readout from sum pooling directly,
   without switching to mean pooling? How?
''',
    answers=r'''
1. Its pooled feature vector would have a magnitude far outside anything the
   dense readout layers saw during training, so predictions would likely
   extrapolate very badly — another instance of Week 03's train/test
   distribution-mismatch concern, here driven by molecule size specifically.
2. Yes — divide the sum-pooled vector by the number of atoms $N$
   afterwards; that is algebraically identical to mean pooling. The two are
   not fundamentally different operations, just a choice of where the
   size-normalisation happens.
''',
)

# ==========================================================================
# 4. A GCN regression model; one manual step
# ==========================================================================
C += [md(r'''
---
## 4. A GCN regression model; one manual training step

Stack three `GCNLayer`s, mean-pool (solubility is intensive), then two dense
layers down to a single number — exactly Week 05's MLP pattern, with the
first stage replaced by graph convolutions.

**What to look for:** the model runs on a single molecule end to end; after
one optimiser step, the loss on that same molecule is lower than before it
(Week 05A, Section 4's exact check, now for a graph).
''')]

C += [code(r'''
class GCN(nn.Module):
    """3 GCN layers (mean-pooled readout) + 2 dense layers -> one regression output."""

    def __init__(self, in_features=N_FEAT, hidden=64):
        super().__init__()
        self.gcn1 = GCNLayer(in_features, hidden, activation=F.relu)
        self.gcn2 = GCNLayer(hidden, hidden, activation=F.relu)
        self.gcn3 = GCNLayer(hidden, hidden, activation=F.relu)
        self.fc1 = nn.Linear(hidden, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, X, A):
        h = self.gcn1(X, A)
        h = self.gcn2(h, A)
        h = self.gcn3(h, A)
        g = h.mean(dim=0)                  # mean readout: solubility is intensive
        g = F.relu(self.fc1(g))
        return self.fc2(g)

torch.manual_seed(SEED)
model = GCN()
n_params = sum(p.numel() for p in model.parameters())
print("trainable parameters:", n_params)

optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()

TARGET = "measured log solubility in mols per litre"
Xg, Ag, _ = mol_to_graph(small_mol)
target = torch.tensor([esol[TARGET].iloc[0]], dtype=torch.float32)

loss_before = loss_fn(model(Xg, Ag), target).item()
optimiser.zero_grad()
loss = loss_fn(model(Xg, Ag), target)
loss.backward()
optimiser.step()
loss_after = loss_fn(model(Xg, Ag), target).item()

print(f"loss before step: {loss_before:.4f}   after step: {loss_after:.4f}")
''')]

C += [md(r'''
> **Common errors — Section 4**
> - Passing a **batch** of molecules with different `N` as one tensor —
>   `GCNLayer` as written handles one molecule (one `(N,N)` adjacency) at a
>   time; Week 07B's training loop calls the model once per molecule and
>   accumulates gradients, rather than stacking variable-sized graphs into a
>   single tensor.
> - Forgetting `model(Xg, Ag)` needs **both** arguments every call — unlike a
>   plain MLP, the "input" here is a pair (features, adjacency), not one
>   tensor.
''')]

C += checkpoint(
    "Section 4",
    check=r'''
assert n_params > 0
assert loss_after < loss_before
print("Section 4 OK")
''',
    expected="Prints `Section 4 OK`. Loss on the same molecule is lower after "
    "one optimiser step.",
    questions=r'''
1. Why does this model use **mean** readout but Section 3 showed sum readout
   is sometimes the right choice? What made the decision here?
2. If you swapped `nn.Linear(in_features, hidden)` for a layer with a bias
   term inside `GCNLayer`, would Section 2's equivariance check still pass?
''',
    answers=r'''
1. The target here is **measured log solubility** — an intensive property —
   so mean readout (Section 3) is the chemically appropriate choice; a
   different target (e.g. total formation enthalpy) would call for sum
   readout instead.
2. Yes — a bias is added identically to every node's output regardless of
   position, so it commutes with any permutation of the nodes; equivariance
   only breaks if some operation treats specific node *positions*
   differently (e.g. indexing `X[0]` specially), which nothing in `GCNLayer`
   does.
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
### Exercise 1 — graph-ify a new molecule *(easy, ~10 min)*

Write `graph_summary(smiles)` returning a dict with `n_atoms`, `n_bonds`
(before self-loops), and `is_symmetric` (whether the adjacency matrix, before
self-loops, equals its transpose). Apply it to benzene (`c1ccccc1`).

<details><summary>Show hint</summary>

Reuse `mol_to_graph`; recompute the adjacency **without** self-loops
separately (or subtract `torch.eye(n)`) to check symmetry cleanly.
</details>
''',
    solution=r'''
def graph_summary(smiles):
    """n_atoms, n_bonds and adjacency symmetry for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    X, A, n_bonds = mol_to_graph(mol)
    A_no_loops = A - torch.eye(A.shape[0])
    return {
        "n_atoms": X.shape[0],
        "n_bonds": n_bonds,
        "is_symmetric": bool(torch.allclose(A_no_loops, A_no_loops.T)),
    }

benzene_summary = graph_summary("c1ccccc1")
print(benzene_summary)
''',
    scaffold=r'''
def graph_summary(smiles):
    """n_atoms, n_bonds and adjacency symmetry for a SMILES string."""
    # YOUR CODE HERE
    ...

benzene_summary = ...  # YOUR CODE HERE: graph_summary("c1ccccc1")
''',
    check=r'''
assert benzene_summary == {"n_atoms": 6, "n_bonds": 6, "is_symmetric": True}
print("Exercise 1 OK")
''',
)

C += exercise(
    prompt=r'''
### Exercise 2 — equivariance on a bigger molecule *(medium, ~15 min)*

Write `check_equivariance(layer, mol, seed=0)` that builds `mol`'s graph,
applies a **random** permutation (`torch.randperm(n)`, seeded), and returns
`True` if `layer`'s output for the permuted graph equals the permuted
original output (to `atol=1e-5`). Test it on benzene with the Section 2/4
layers.

<details><summary>Show hint</summary>

Reuse Section 2's pattern exactly, but generate `perm` with
`torch.randperm(n)` under a seeded generator instead of a hand-written list.
</details>
''',
    solution=r'''
def check_equivariance(layer, mol, seed=0):
    """True if layer(X,A) is permutation-equivariant for this molecule's graph."""
    X, A, _ = mol_to_graph(mol)
    n = X.shape[0]
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator)
    out_original = layer(X, A)
    out_permuted = layer(X[perm], A[perm][:, perm])
    return torch.allclose(out_permuted, out_original[perm], atol=1e-5)

benzene = Chem.MolFromSmiles("c1ccccc1")
print("GCNLayer equivariant on benzene?", check_equivariance(layer, benzene))
''',
    scaffold=r'''
def check_equivariance(layer, mol, seed=0):
    """True if layer(X,A) is permutation-equivariant for this molecule's graph."""
    # YOUR CODE HERE
    ...

benzene = Chem.MolFromSmiles("c1ccccc1")
result_ex2 = ...  # YOUR CODE HERE: check_equivariance(layer, benzene)
''',
    check=r'''
assert check_equivariance(layer, benzene) is True
assert check_equivariance(layer, Chem.MolFromSmiles("CCO")) is True
print("Exercise 2 OK")
''',
)

C += exercise(
    prompt=r'''
### Exercise 3 — mean pooling is invariant, not just "similar" *(medium, ~15 min)*

Write `check_readout_invariance(readout_fn, layer, mol, seed=0)` confirming a
readout is **exactly** invariant (not just "in a similar range" as Section 3
showed informally) to a random permutation. Test `mean_readout` and
`sum_readout`.

<details><summary>Show hint</summary>

Compare `readout_fn(layer(X, A))` to
`readout_fn(layer(X[perm], A[perm][:, perm]))` with `torch.allclose`.
</details>
''',
    solution=r'''
def check_readout_invariance(readout_fn, layer, mol, seed=0):
    """True if readout_fn(layer(X,A)) is unchanged under a random permutation."""
    X, A, _ = mol_to_graph(mol)
    n = X.shape[0]
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator)
    original = readout_fn(layer(X, A))
    permuted = readout_fn(layer(X[perm], A[perm][:, perm]))
    return torch.allclose(original, permuted, atol=1e-5)

print("mean_readout invariant?", check_readout_invariance(mean_readout, layer, benzene))
print("sum_readout invariant? ", check_readout_invariance(sum_readout, layer, benzene))
''',
    scaffold=r'''
def check_readout_invariance(readout_fn, layer, mol, seed=0):
    """True if readout_fn(layer(X,A)) is unchanged under a random permutation."""
    # YOUR CODE HERE
    ...

mean_invariant = ...  # YOUR CODE HERE: check_readout_invariance(mean_readout, layer, benzene)
sum_invariant = ...  # YOUR CODE HERE: check_readout_invariance(sum_readout, layer, benzene)
''',
    check=r'''
assert check_readout_invariance(mean_readout, layer, benzene) is True
assert check_readout_invariance(sum_readout, layer, benzene) is True
print("Exercise 3 OK")
''',
)

C += exercise(
    prompt=r'''
### Exercise 4 — why degree-normalise? *(harder, ~20 min)*

Write `unnormalised_aggregate(X, A, n_layers)` that repeatedly applies **plain
summed** neighbour aggregation ($V \to AV$, **no** division by degree) for
`n_layers` steps and returns the final feature norm. Compare its growth over
1/2/3 layers to the same experiment using `GCNLayer`'s degree-normalised
aggregation, on a highly-connected molecule (benzene: every atom has degree
3).

<details><summary>Show hint</summary>

`X_next = A @ X_current`; track `X_current.norm()` after each layer. For the
normalised comparison, divide by `A.sum(dim=-1, keepdim=True)` at each step
too, exactly as `GCNLayer.forward` does.
</details>
''',
    solution=r'''
def unnormalised_aggregate(X, A, n_layers):
    """Repeated PLAIN sum aggregation (no degree normalisation); return final norm."""
    h = X.clone()
    for _ in range(n_layers):
        h = A @ h
    return h.norm().item()

def normalised_aggregate(X, A, n_layers):
    """Repeated degree-normalised aggregation (GCNLayer style); return final norm."""
    h = X.clone()
    degree = A.sum(dim=-1, keepdim=True).clamp(min=1.0)
    for _ in range(n_layers):
        h = (A @ h) / degree
    return h.norm().item()

Xb, Ab, _ = mol_to_graph(benzene)
for n_layers in [1, 2, 3]:
    un = unnormalised_aggregate(Xb, Ab, n_layers)
    norm = normalised_aggregate(Xb, Ab, n_layers)
    print(f"{n_layers} layer(s): unnormalised norm={un:.2f}   normalised norm={norm:.2f}")
''',
    scaffold=r'''
def unnormalised_aggregate(X, A, n_layers):
    """Repeated PLAIN sum aggregation (no degree normalisation); return final norm."""
    # YOUR CODE HERE
    ...

def normalised_aggregate(X, A, n_layers):
    """Repeated degree-normalised aggregation (GCNLayer style); return final norm."""
    # YOUR CODE HERE
    ...
''',
    check=r'''
Xb, Ab, _ = mol_to_graph(benzene)
norms_unnorm = [unnormalised_aggregate(Xb, Ab, n) for n in [1, 2, 3]]
norms_norm = [normalised_aggregate(Xb, Ab, n) for n in [1, 2, 3]]
assert norms_unnorm[2] > norms_unnorm[0]                 # unnormalised grows with depth
assert norms_unnorm[2] > norms_norm[2] * 1.5             # grows much faster than normalised
print("Exercise 4 OK")
''',
)

# ==========================================================================
C += [md(r'''
---
## Summary — what you learned

- A molecule as a **graph**: adjacency matrix + node-feature matrix, with
  self-loops added (Kipf & Welling's trick).
- **Message passing**: message → aggregate → update, and the closed-form GCN
  equation $V'=\sigma(D^{-1}EVW)$ — verified **numerically** to be
  permutation-equivariant, connecting Week 06's formal definitions to a real
  architecture.
- **Readout**: mean pooling for intensive properties, sum pooling for
  extensive ones, both exactly permutation-invariant.
- A 3-layer `GCN` regression model and the same four-line training-step
  anatomy from Week 05A, now operating on a graph.
- Why **degree normalisation** matters: unnormalised aggregation blows up
  with depth on well-connected molecules.

Week 07B trains this model properly — full loss curves, a fair comparison to
Week 03's leaderboard and Week 05's MLP, and a look at what happens when you
add too many GCN layers.
''')]

C += [md(r'''
## Further reading (course source list only)

- dmol.pub *Graph neural networks*: <https://dmol.pub/dl/gnn.html>
- Colab alternative (EPFL *AI for Chemistry*, graph NNs):
  <https://colab.research.google.com/github/schwallergroup/ai4chem_course/blob/main/notebooks/03%20-%20Intro%20to%20Deep%20Learning/02_graph_nns.ipynb>
- Alternative: DeepChem *Introduction to Graph Convolutions*:
  <https://github.com/deepchem/deepchem/blob/master/examples/tutorials/Introduction_to_Graph_Convolutions.ipynb>
''')]

C += [md(r'''
## Attribution

Original teaching material for *AI for Chemistry*, adapted and rewritten from:

- **dmol.pub** (A. White) — the message-passing equations, the Kipf & Welling
  GCN equation, the mean/sum readout distinction, and the choice to implement
  a minimal `GCNLayer` directly in PyTorch rather than a graph-learning
  library. CC-BY 4.0. <https://dmol.pub>
- **EPFL CH-457 *AI for Chemistry*** (Schwaller group) — graph-NN-on-ESOL
  framing. MIT licence. <https://github.com/schwallergroup/ai4chem_course>

Continues the ESOL dataset from Weeks 02-03/05 and the invariance/
equivariance definitions from Week 06A. No verbatim text is reproduced from
the sources above.
''')]

build(__file__, "week07_a_graph-neural-networks", C)
