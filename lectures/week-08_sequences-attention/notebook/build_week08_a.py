"""Build week 08 session A notebooks (sequences, attention, seq2vec mechanics).

    python lectures/week-08_sequences-attention/notebook/build_week08_a.py
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "_build"))
from nbbuild import build, checkpoint, code, exercise, md  # noqa: E402

C = []

# ==========================================================================
C += [md(r'''
# AI for Chemistry — Week 08, Session A
## Sequences and attention: SMILES language models

**Course:** AI for Chemistry · **Session:** 08A (interleaved lecture + lab,
2.5 h) · **Runtime:** < 10 s, no GPU.

### Suggested timing (solo study, ~120 min)

| Section | Topic | Time |
|--------:|-------|-----:|
| 0 | Setup | 8 min |
| 1 | SMILES as sequences: tokenisation | 18 min |
| 2 | Scaled dot-product attention | 25 min |
| 3 | Seq2vec: embedding + GRU for property prediction | 22 min |
| 4 | Causal masking for autoregressive prediction | 18 min |
| 5 | Transformers and pretraining (survey) | 8 min |
| 6 | Exercises (4) | 21 min |

### Learning objectives
1. Tokenise SMILES at character level and explain why token order is
   chemically meaningful. *(LO7)*
2. Implement scaled dot-product attention
   ($\mathrm{softmax}(\vec q\cdot K/\sqrt d)\cdot V$) from its query/key/value
   definitions. *(LO7)*
3. Build a minimal embedding+GRU seq2vec model for property prediction. *(LO7)*
4. Implement a causal attention mask for autoregressive next-token
   prediction. *(LO7)*
5. Describe, at survey level, how transformer blocks and pretraining combine
   into chemical language models. *(LO7)*

### Prerequisites — before this notebook you should be able to
- Week 07: `nn.Module`, tensors, one manual training step.
- Week 02: SMILES strings and why more than one string can encode the same
  molecule.

### How to use this notebook (solo study)
Read, run, check each **✅** cell in order. No pretrained model is used this
session — Week 08B loads a real one. This is the **solutions** notebook.
''')]

# ==========================================================================
C += [md(r'''
---
## 0. Setup

We reuse ESOL's SMILES strings (Weeks 02-03) purely as a source of sequences
— no property is predicted from them until Section 3.

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

SEED = 0xC0FFEE
np.random.seed(SEED)
torch.manual_seed(SEED)

ROOT = Path.cwd()
while not (ROOT / "sources").is_dir() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
DATA = ROOT / "sources" / "datasets"
FIGDIR = ROOT / "lectures" / "week-08_sequences-attention" / "slides" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

print("torch", torch.__version__)
print("esol_clean.csv", "OK" if (DATA / "esol_clean.csv").is_file() else "MISSING")
''')]

C += [md(r'''
> **Common errors — Setup**
> - `MISSING` for `esol_clean.csv` — run Week 02B's notebook, or Week 03's
>   automatic rebuild, first.
''')]

# ==========================================================================
# 1. SMILES as sequences
# ==========================================================================
C += [md(r'''
---
## 1. SMILES as sequences: tokenisation

A SMILES string is a **sequence** of characters, and — unlike a fingerprint's
bit positions, which have no inherent order — **the order of a SMILES string
is exactly what encodes the molecule**. `"CCO"` and `"OCC"` are the same
molecule (Week 02 taught you to canonicalise away that ambiguity); but
`"CCO"` and `"COC"` are *different* molecules (ethanol vs dimethyl ether) —
scrambling character order does not merely relabel the same structure the
way permuting a graph's atom order did in Week 07.

We tokenise at **character level**: build a vocabulary of every character
seen across the dataset, map each to an integer (reserving `0` for padding),
and pad every sequence to a common length so they can be batched.

**What to look for:** a vocabulary of roughly 25-30 distinct characters
(SMILES uses a fairly small alphabet: atoms, bonds, ring-closure digits,
brackets); `encode("CCO")` returns 3 integers, none of them `0`.
''')]

C += [code(r'''
esol = pd.read_csv(DATA / "esol_clean.csv")
all_smiles = esol["canonical_smiles"].tolist()

vocab_chars = sorted(set("".join(all_smiles)))
char_to_idx = {c: i + 1 for i, c in enumerate(vocab_chars)}   # 0 reserved for padding
idx_to_char = {i: c for c, i in char_to_idx.items()}
VOCAB_SIZE = len(char_to_idx) + 1                              # +1 for the padding index
MAX_LEN = max(len(s) for s in all_smiles)

def encode(smiles, max_len=MAX_LEN):
    """SMILES string -> fixed-length LongTensor of character indices (0 = pad)."""
    ids = [char_to_idx[c] for c in smiles]
    ids = ids[:max_len] + [0] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)

print("vocabulary size (incl. padding):", VOCAB_SIZE)
print("longest SMILES in ESOL:", MAX_LEN, "characters")
print("encode('CCO'):", encode("CCO", max_len=6))
''')]

C += [md(r'''
> **Common errors — Section 1**
> - Building the vocabulary from a **sample** of the data instead of the
>   whole dataset — a character seen only in a held-out molecule would then
>   have no entry in `char_to_idx`, raising a `KeyError` the first time it is
>   encountered. (For a train/test split done properly, the vocabulary should
>   still be built from *training* data only, to avoid a subtle test-set
>   leak; we use the whole dataset here purely for tokeniser mechanics, not
>   for training a leaderboard-comparable model.)
> - Confusing "0 reserved for padding" with "character index 0" — no real
>   character maps to `0`; every real token index is `>= 1`.
''')]

C += checkpoint(
    "Section 1",
    check=r'''
assert 20 <= VOCAB_SIZE <= 40
assert 0 not in [char_to_idx[c] for c in "CCO"]
assert torch.equal(encode("CCO", max_len=5), torch.tensor([char_to_idx["C"], char_to_idx["C"],
                                                            char_to_idx["O"], 0, 0]))
print("Section 1 OK")
''',
    expected="Prints `Section 1 OK`. Vocabulary size is in the 20s-30s; "
    "`encode('CCO', max_len=5)` is `[idx(C), idx(C), idx(O), 0, 0]`.",
    questions=r'''
1. Why must padding use an index that is **not** also a real character's
   index?
2. `"CCO"` and `"COC"` differ only in character order but describe different
   molecules. Contrast this with Week 07: does swapping two rows of a
   molecular graph's adjacency matrix change *which* molecule it represents?
''',
    answers=r'''
1. If padding reused a real character's index (e.g. `0` also meaning `"C"`),
   the model could not distinguish "this position really is a carbon" from
   "this position is empty padding" — every downstream computation over the
   sequence would be corrupted.
2. No — Week 07 showed permuting a graph's node order (with the adjacency
   matrix permuted consistently) leaves the underlying molecule unchanged;
   a GNN is built to be invariant/equivariant to that relabelling. A SMILES
   string has no such symmetry: character order **is** the information, not
   an arbitrary labelling artefact.
''',
)

# ==========================================================================
# 2. Scaled dot-product attention
# ==========================================================================
C += [md(r'''
---
## 2. Scaled dot-product attention

**Attention** computes a weighted average over a set of **values** $V$, where
the weights come from comparing a **query** $\vec q$ against a set of **keys**
$K$:

$$\vec b = \mathrm{softmax}\!\left(\frac{\vec q \cdot K}{\sqrt d}\right), \qquad
  \text{output} = \vec b \cdot V,$$

where $d$ is the query dimension (the $\sqrt d$ keeps the dot products from
growing too large as $d$ increases, which would make `softmax` too peaked).
When $\vec q$, $K$ and $V$ all come from the **same** sequence, this is
**self-attention** — every position asks "which other positions in *this
same sequence* are relevant to me?"

We compute self-attention over one SMILES string's character embeddings,
using **untrained, fixed-seed** projection matrices — the weights are
meaningless chemically (nothing has been trained), but the *mechanics*
(shapes, softmax normalisation) are exactly what a trained model would use.

**What to look for:** the attention matrix has shape `(L, L)` for a sequence
of length `L`; every **row** sums to 1 (a `softmax` property, not a
coincidence).
''')]

C += [code(r'''
def scaled_dot_product_attention(Q, K, V):
    """softmax(Q K^T / sqrt(d)) V -- Q:(Lq,d), K,V:(Lk,d) -> (output, weights)."""
    d = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / (d ** 0.5)       # (Lq, Lk)
    weights = F.softmax(scores, dim=-1)                  # normalise each row
    output = weights @ V                                 # (Lq, d)
    return output, weights

torch.manual_seed(SEED)
EMBED_DIM = 8
embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=0)
Wq, Wk, Wv = (nn.Linear(EMBED_DIM, EMBED_DIM, bias=False) for _ in range(3))

smiles_example = "CC(=O)Oc1ccccc1C(=O)O"        # aspirin
tokens = list(smiles_example)
ids = torch.tensor([char_to_idx[c] for c in tokens])
X = embedding(ids)                               # (L, EMBED_DIM)

Q, K, V = Wq(X), Wk(X), Wv(X)
attn_out, attn_weights = scaled_dot_product_attention(Q, K, V)

print("sequence length L:", len(tokens))
print("attention weights shape:", tuple(attn_weights.shape))
print("row sums (should all be 1):", attn_weights.sum(dim=-1).detach().numpy().round(4))
''')]

C += [code(r'''
fig, ax = plt.subplots(figsize=(6, 5.5))
im = ax.imshow(attn_weights.detach().numpy(), cmap="viridis")
ax.set_xticks(range(len(tokens))); ax.set_xticklabels(tokens, fontsize=8)
ax.set_yticks(range(len(tokens))); ax.set_yticklabels(tokens, fontsize=8)
ax.set_xlabel("key position"); ax.set_ylabel("query position")
ax.set_title("Self-attention weights (untrained -- mechanics only)")
fig.colorbar(im, label="attention weight")
fig.tight_layout(); fig.savefig(FIGDIR / "attention_weights_toy.png", dpi=200)
plt.show()
''')]

C += [md(r'''
> **Common errors — Section 2**
> - Forgetting the $1/\sqrt d$ scale factor — without it, larger embedding
>   dimensions push the dot products to extreme values, `softmax` saturates
>   towards a one-hot vector, and gradients through it vanish.
> - Applying `softmax` along the **wrong axis** (`dim=0` instead of `dim=-1`)
>   — each **query's** distribution over keys must sum to 1, which means
>   normalising across the *key* axis (the last one here), not the query axis.
> - Reading meaning into this section's specific attention pattern — the
>   projections are untrained; a real model's attention (Week 08B) reflects
>   what it *learned* to attend to, not a property of attention itself.
''')]

C += checkpoint(
    "Section 2",
    check=r'''
assert attn_weights.shape == (len(tokens), len(tokens))
assert torch.allclose(attn_weights.sum(dim=-1), torch.ones(len(tokens)), atol=1e-5)
assert attn_out.shape == (len(tokens), EMBED_DIM)
assert (FIGDIR / "attention_weights_toy.png").is_file()
print("Section 2 OK")
''',
    expected="Prints `Section 2 OK`. Attention weights form a square matrix "
    "whose rows each sum to 1.",
    questions=r'''
1. If `Q`, `K` and `V` were computed from **two different** SMILES strings
   instead of the same one, would this still be self-attention?
2. What happens to `attn_out` for a padding position (index 0), given
   `nn.Embedding(..., padding_idx=0)`?
''',
    answers=r'''
1. No — that would be **cross-attention** (Week 08's dmol.pub source gives
   the example of a query word attending across a *different* sentence's
   keys/values); self-attention specifically means the query, keys and
   values all derive from the same sequence.
2. `padding_idx=0` makes the embedding for index 0 a fixed all-zero vector
   (and excludes it from gradient updates), but attention **does not
   automatically ignore it** — a padding position still participates in the
   softmax like any other key unless it is explicitly masked out, which is
   exactly Section 4's job for a different kind of masking (causal, not
   padding).
''',
)

# ==========================================================================
# 3. Seq2vec: embedding + GRU for property prediction
# ==========================================================================
C += [md(r'''
---
## 3. Seq2vec: embedding + GRU for property prediction

To predict a single molecular property from a SMILES **sequence**, a
recurrent layer such as a **GRU** reads the sequence one token at a time,
carrying a **hidden state** forward: $h_t = f(h_{t-1}, x_t)$. After the last
token, the final hidden state is a fixed-length summary of the whole
sequence — a "seq2vec" encoding, playing the same role Week 07's mean-pooled
graph readout played for graphs.

We build the exact architecture dmol.pub uses for this task on this same
ESOL dataset: `Embedding -> GRU -> dense -> dense`. As in Weeks 05A/07A, this
section is **mechanics only** — one manual training step, not a full
training run.

**What to look for:** the model runs end to end on a padded batch; after one
optimiser step, the loss on that same batch is lower than before it.
''')]

C += [code(r'''
class Seq2VecGRU(nn.Module):
    """Embedding -> GRU -> dense -> dense, mirroring dmol.pub's ESOL example."""

    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=16, hidden=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden, batch_first=True)
        self.fc1 = nn.Linear(hidden, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, token_ids):
        x = self.embedding(token_ids)            # (B, L, embed_dim)
        _, h = self.gru(x)                        # h: (1, B, hidden) -- final hidden state
        h = h.squeeze(0)                          # (B, hidden)
        h = F.relu(self.fc1(h))
        return self.fc2(h)                        # (B, 1)

torch.manual_seed(SEED)
seq_model = Seq2VecGRU()
n_params = sum(p.numel() for p in seq_model.parameters())
print("trainable parameters:", n_params)

TARGET = "measured log solubility in mols per litre"
batch_smiles = esol["canonical_smiles"].iloc[:8].tolist()
batch_ids = torch.stack([encode(s) for s in batch_smiles])           # (8, MAX_LEN)
batch_y = torch.tensor(esol[TARGET].iloc[:8].to_numpy(), dtype=torch.float32).unsqueeze(-1)

optimiser = torch.optim.Adam(seq_model.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()

loss_before = loss_fn(seq_model(batch_ids), batch_y).item()
optimiser.zero_grad()
loss = loss_fn(seq_model(batch_ids), batch_y)
loss.backward()
optimiser.step()
loss_after = loss_fn(seq_model(batch_ids), batch_y).item()

print(f"loss before step: {loss_before:.4f}   after step: {loss_after:.4f}")
''')]

C += [md(r'''
> **Common errors — Section 3**
> - `nn.GRU` returns `(output, h)`: `output` is every timestep's hidden state
>   (`(B, L, hidden)`), `h` is only the **final** one (`(1, B, hidden)`) —
>   seq2vec wants `h`, not `output`, unless you plan to pool over timesteps
>   yourself (an alternative readout, analogous to Week 07's graph pooling).
> - Forgetting `batch_first=True` — `nn.GRU` defaults to `(L, B, *)` ordering,
>   the opposite of every other layer in this course; mismatching it silently
>   scrambles which axis is the batch.
> - dmol.pub's own reported result for this exact architecture is a
>   correlation of ~0.72 on AqSolDB, **below** simple linear regression on
>   descriptors — a genuine, published limitation of this minimal seq2vec
>   recipe, not a mistake to "fix" here.
''')]

C += checkpoint(
    "Section 3",
    check=r'''
assert n_params > 0
assert loss_after < loss_before
assert seq_model(batch_ids).shape == (8, 1)
print("Section 3 OK")
''',
    expected="Prints `Section 3 OK`. Loss on the same batch is lower after "
    "one optimiser step.",
    questions=r'''
1. Why does a GRU need to process a sequence **in order**, one token at a
   time, whereas Week 07's GCN could process an entire graph's nodes in one
   matrix operation?
2. dmol.pub reports this exact architecture underperforming linear
   regression on a similar solubility dataset. Give one plausible reason,
   referencing Week 06's representation spectrum.
''',
    answers=r'''
1. A GRU's hidden state at position $t$ is defined recursively from position
   $t-1$'s hidden state, so later positions cannot be computed before earlier
   ones finish — an inherently sequential dependency. A GCN layer instead
   combines each node with its neighbours using one shared matrix operation
   applied to the whole graph simultaneously; there is no such
   step-by-step ordering to respect.
2. A raw character sequence, with no chemical priors at all, sits at the
   **fully learned** end of Week 06's spectrum; with a modest dataset (ESOL
   has ~1000 molecules) the model must discover, from scratch, what a
   descriptor-based model was simply handed — exactly the same "learned
   representation needs enough data to earn its flexibility" pattern seen in
   Weeks 05B and 07B.
''',
)

# ==========================================================================
# 4. Causal masking for autoregressive prediction
# ==========================================================================
C += [md(r'''
---
## 4. Causal masking for autoregressive next-token prediction

A model that **generates** a SMILES string one character at a time (Week 09
will build one) predicts each token from only the tokens **before** it:

$$P(x_0, \ldots, x_L) = P(x_0)\, P(x_1\mid x_0)\, P(x_2\mid x_1, x_0) \cdots$$

If we implemented this with self-attention directly, a naive model could
"cheat" by attending to **future** tokens it is supposed to be predicting. A
**causal mask** prevents this: before the `softmax`, set every score at
position $(i,j)$ with $j>i$ (a *future* key) to $-\infty$, so its attention
weight becomes exactly 0.

**What to look for:** the masked attention matrix is **lower-triangular** —
every row sums to 1 as before, but now only over positions **up to and
including** the query's own position.
''')]

C += [code(r'''
def causal_attention(Q, K, V):
    """Scaled dot-product attention with a causal (no-peeking-ahead) mask."""
    d = Q.shape[-1]
    L = Q.shape[0]
    scores = Q @ K.transpose(-2, -1) / (d ** 0.5)
    mask = torch.triu(torch.ones(L, L), diagonal=1).bool()   # True above the diagonal
    scores = scores.masked_fill(mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return weights @ V, weights

causal_out, causal_weights = causal_attention(Q, K, V)          # reuse Section 2's Q,K,V

print("row sums (should all be 1):", causal_weights.sum(dim=-1).detach().numpy().round(4))
print("upper triangle max weight (should be 0):",
      causal_weights.triu(diagonal=1).max().item())
''')]

C += [code(r'''
fig, ax = plt.subplots(figsize=(6, 5.5))
im = ax.imshow(causal_weights.detach().numpy(), cmap="viridis")
ax.set_xticks(range(len(tokens))); ax.set_xticklabels(tokens, fontsize=8)
ax.set_yticks(range(len(tokens))); ax.set_yticklabels(tokens, fontsize=8)
ax.set_xlabel("key position"); ax.set_ylabel("query position")
ax.set_title("Causally-masked self-attention")
fig.colorbar(im, label="attention weight")
fig.tight_layout(); fig.savefig(FIGDIR / "causal_mask.png", dpi=200)
plt.show()
''')]

C += [md(r'''
> **Common errors — Section 4**
> - Masking with `0` instead of `-inf` **before** the softmax — a masked
>   score of 0 is not "no attention", it is "moderate attention" once
>   softmax exponentiates it; only $-\infty$ (which becomes exactly 0 after
>   `exp`) removes a position entirely.
> - `torch.triu(..., diagonal=1)` (excludes the diagonal) vs `diagonal=0`
>   (includes it) — a token **must** be allowed to attend to itself
>   (`diagonal=1` is correct here); excluding the diagonal too would leave a
>   query with no valid keys on its own row.
''')]

C += checkpoint(
    "Section 4",
    check=r'''
assert torch.allclose(causal_weights.sum(dim=-1), torch.ones(len(tokens)), atol=1e-5)
assert causal_weights.triu(diagonal=1).max().item() < 1e-6
assert causal_weights[0, 0].item() > 0.99     # first token can only attend to itself
assert (FIGDIR / "causal_mask.png").is_file()
print("Section 4 OK")
''',
    expected="Prints `Section 4 OK`. Every row still sums to 1; nothing "
    "above the diagonal; the first token attends entirely to itself.",
    questions=r'''
1. Why must the very first token's attention weight on itself be (almost)
   exactly 1?
2. A bidirectional model (Section 3's GRU processes left-to-right only in
   one direction, but a bidirectional GRU reads both ways) would be a poor
   choice for a *generative* SMILES model. Why?
''',
    answers=r'''
1. With a causal mask, the first token has **only one** valid key (itself —
   there is nothing before it); softmax over a single unmasked score always
   produces a weight of 1 for it.
2. A generative model must predict each token using only what would actually
   be available at generation time — the tokens already produced. A
   bidirectional model conditions on tokens from *both* directions, including
   ones that would not exist yet during generation; it can be an excellent
   property-predictor (it sees the whole finished string) but is
   architecturally unsuited to autoregressive generation.
''',
)

# ==========================================================================
# 5. Transformers and pretraining (survey)
# ==========================================================================
C += [md(r'''
---
## 5. Transformers and pretraining — survey

*(Deliberately survey-level — `syllabus.md` scopes "chemical LLMs" as survey
material this week; Session B *uses* one directly, which is different from
building one from scratch here.)*

A **transformer** is built almost entirely from the attention mechanism of
Sections 2 and 4, stacked and combined with a few supporting pieces:

* **Multi-head attention**: run several attention "heads" in parallel, each
  with its own $W_q, W_k, W_v$ projections, then concatenate their outputs —
  letting different heads specialise in different kinds of relationship
  (e.g. one head might learn to track ring-closure digit pairs, another
  local bonding patterns).
* **Positional encoding**: attention itself has no notion of token order (the
  softmax over keys does not care where they sit) — a positional signal must
  be added to each token's embedding so the model can still use sequence
  order, exactly the information Section 1 argued SMILES cannot do without.
* **Masked-language-model pretraining**: hide a random subset of tokens and
  train the model to predict them from context alone — entirely
  **self-supervised** (no property labels needed), which is why it can be
  done on huge unlabelled SMILES databases (millions of molecules) before any
  task-specific fine-tuning.

**Named chemical language models** built this way include **ChemBERTa**
(RoBERTa-style, pretrained on SMILES from ZINC) and **MolBERT**. Week 08B
loads ChemBERTa directly — using a real pretrained model, frozen, for
property prediction — rather than building a transformer from scratch, which
this course does not do.
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
### Exercise 1 — decode back to a string *(easy, ~10 min)*

Write `decode(ids)` — the inverse of `encode` — that turns a tensor of
indices back into a SMILES string, **stopping at the first padding index**
(`0`). Confirm `decode(encode(s))` round-trips for a few molecules.

<details><summary>Show hint</summary>

Loop the tensor; break as soon as you see `0`; look up `idx_to_char` for
everything before that.
</details>
''',
    solution=r'''
def decode(ids):
    """Inverse of encode(): index tensor -> SMILES string, stopping at padding."""
    chars = []
    for i in ids.tolist():
        if i == 0:
            break
        chars.append(idx_to_char[i])
    return "".join(chars)

for s in ["CCO", "c1ccccc1", "CC(=O)O"]:
    round_tripped = decode(encode(s))
    print(f"{s!r:15s} -> {round_tripped!r}   match: {round_tripped == s}")
''',
    scaffold=r'''
def decode(ids):
    """Inverse of encode(): index tensor -> SMILES string, stopping at padding."""
    # YOUR CODE HERE
    ...

for s in ["CCO", "c1ccccc1", "CC(=O)O"]:
    round_tripped = decode(encode(s))
    print(f"{s!r:15s} -> {round_tripped!r}   match: {round_tripped == s}")
''',
    check=r'''
assert decode(encode("CCO")) == "CCO"
assert decode(encode("c1ccccc1")) == "c1ccccc1"
assert decode(torch.tensor([0, 0, 0])) == ""
print("Exercise 1 OK")
''',
)

C += exercise(
    prompt=r'''
### Exercise 2 — attention is a weighted average, literally *(medium, ~15 min)*

Write `check_convex_combination(weights, V, output)` confirming that each row
of `output` really is a **weighted average** of the rows of `V` (a convex
combination: weights are non-negative and sum to 1, and reproducing `output`
from `weights @ V` matches to numerical precision). Test it on Section 2's
`attn_weights`, `V`, `attn_out`.

<details><summary>Show hint</summary>

Check `(weights >= 0).all()`, `torch.allclose(weights.sum(-1), ones)`, and
`torch.allclose(weights @ V, output)`.
</details>
''',
    solution=r'''
def check_convex_combination(weights, V, output):
    """True if output rows are genuinely convex combinations of V's rows."""
    nonneg = bool((weights >= 0).all())
    rows_sum_to_one = torch.allclose(weights.sum(dim=-1), torch.ones(weights.shape[0]), atol=1e-5)
    reproduces_output = torch.allclose(weights @ V, output, atol=1e-5)
    return nonneg and rows_sum_to_one and reproduces_output

print("Section 2 attention is a valid convex combination:",
      check_convex_combination(attn_weights, V, attn_out))
''',
    scaffold=r'''
def check_convex_combination(weights, V, output):
    """True if output rows are genuinely convex combinations of V's rows."""
    # YOUR CODE HERE
    ...

result_ex2 = ...  # YOUR CODE HERE: check_convex_combination(attn_weights, V, attn_out)
''',
    check=r'''
assert check_convex_combination(attn_weights, V, attn_out) is True
bad_weights = attn_weights.clone(); bad_weights[0, 0] += 1.0    # break the row-sum-to-1 property
assert check_convex_combination(bad_weights, V, attn_out) is False
print("Exercise 2 OK")
''',
)

C += exercise(
    prompt=r'''
### Exercise 3 — pooling instead of the final hidden state *(medium, ~15 min)*

`Seq2VecGRU` uses the GRU's **final** hidden state as the sequence summary.
Write `MeanPoolGRU`, a variant that instead **mean-pools** the GRU's
per-timestep outputs (properly excluding padded positions). Confirm it runs
end to end on `batch_ids`.

<details><summary>Show hint</summary>

`output, _ = self.gru(x)` gives `(B, L, hidden)`; build a mask from
`token_ids != 0`, then average only over unmasked timesteps:
`(output * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)`.
</details>
''',
    solution=r'''
class MeanPoolGRU(nn.Module):
    """Like Seq2VecGRU, but mean-pools over non-padding timesteps instead of
    using only the GRU's final hidden state."""

    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=16, hidden=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden, batch_first=True)
        self.fc1 = nn.Linear(hidden, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, token_ids):
        x = self.embedding(token_ids)
        output, _ = self.gru(x)                       # (B, L, hidden) -- every timestep
        mask = (token_ids != 0).unsqueeze(-1).float()  # (B, L, 1)
        pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        h = F.relu(self.fc1(pooled))
        return self.fc2(h)

torch.manual_seed(SEED)
meanpool_model = MeanPoolGRU()
meanpool_output = meanpool_model(batch_ids)
print("MeanPoolGRU output shape:", tuple(meanpool_output.shape))
''',
    scaffold=r'''
class MeanPoolGRU(nn.Module):
    """Like Seq2VecGRU, but mean-pools over non-padding timesteps instead of
    using only the GRU's final hidden state."""

    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=16, hidden=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden, batch_first=True)
        self.fc1 = nn.Linear(hidden, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, token_ids):
        # YOUR CODE HERE
        ...

torch.manual_seed(SEED)
meanpool_model = MeanPoolGRU()
meanpool_output = ...  # YOUR CODE HERE: meanpool_model(batch_ids)
''',
    check=r'''
assert meanpool_output.shape == (8, 1)
assert torch.isfinite(meanpool_output).all()
print("Exercise 3 OK")
''',
)

C += exercise(
    prompt=r'''
### Exercise 4 — a sliding window mask *(harder, ~20 min)*

Some models restrict attention to a **local window** (position $i$ may only
attend to keys within `window` positions of itself) rather than either the
full causal history or the whole sequence. Write
`windowed_attention(Q, K, V, window)` implementing this.

<details><summary>Show hint</summary>

Build a mask that is `True` (to be filled with `-inf`) wherever
`abs(i - j) > window`, using `torch.arange` and broadcasting, exactly like
Section 4's causal mask but symmetric instead of triangular.
</details>
''',
    solution=r'''
def windowed_attention(Q, K, V, window):
    """Self-attention restricted to keys within `window` positions of the query."""
    d = Q.shape[-1]
    L = Q.shape[0]
    scores = Q @ K.transpose(-2, -1) / (d ** 0.5)
    positions = torch.arange(L)
    distance = (positions[:, None] - positions[None, :]).abs()
    mask = distance > window
    scores = scores.masked_fill(mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return weights @ V, weights

windowed_out, windowed_weights = windowed_attention(Q, K, V, window=2)
print("row sums:", windowed_weights.sum(dim=-1).detach().numpy().round(4)[:5], "...")
print("weight[0, 5] (distance 5, should be 0):", windowed_weights[0, 5].item())
''',
    scaffold=r'''
def windowed_attention(Q, K, V, window):
    """Self-attention restricted to keys within `window` positions of the query."""
    # YOUR CODE HERE
    ...

windowed_out, windowed_weights = ...  # YOUR CODE HERE: windowed_attention(Q, K, V, window=2)
''',
    check=r'''
assert torch.allclose(windowed_weights.sum(dim=-1), torch.ones(len(tokens)), atol=1e-5)
assert windowed_weights[0, 5].item() < 1e-6     # position 5 is outside window=2 of position 0
assert windowed_weights[0, 1].item() > 0         # position 1 is inside the window
print("Exercise 4 OK")
''',
)

# ==========================================================================
C += [md(r'''
---
## Summary — what you learned

- SMILES as **sequences**: character tokenisation, padding, and why token
  order is chemically load-bearing (unlike a fingerprint or a graph's node
  order).
- **Scaled dot-product attention**: query/key/value, the $1/\sqrt d$ scale
  factor, and the "attention is a weighted average" view, verified as a
  genuine convex combination.
- **Seq2vec**: embedding + GRU for property prediction, and dmol.pub's own
  honest report that this minimal recipe underperforms linear regression on
  a similar dataset.
- **Causal masking** for autoregressive next-token prediction — the exact
  mechanism Week 09's generative model will use to sample new molecules.
- **Transformers and pretraining**, survey-level: multi-head attention,
  positional encoding, masked-language-model pretraining, and named chemical
  LLMs (ChemBERTa, MolBERT).

Week 08B loads a real pretrained chemical transformer and uses it, frozen,
for property prediction — the hands-on counterpart to this session's survey.
''')]

C += [md(r'''
## Further reading (course source list only)

- dmol.pub *Attention*: <https://dmol.pub/dl/attention.html>
- dmol.pub *Deep learning on sequences*: <https://dmol.pub/dl/NLP.html>
- dmol.pub *Pretraining* (survey): <https://dmol.pub/dl/pretraining.html>
''')]

C += [md(r'''
## Attribution

Original teaching material for *AI for Chemistry*, adapted and rewritten from:

- **dmol.pub** (A. White) — the Q/K/V attention formulation, the SMILES
  seq2vec architecture (embedding + GRU) and its honestly-reported
  under-linear-regression result, the autoregressive next-token-prediction
  framing, and the pretraining survey (ChemBERTa, MolBERT, masked-language
  modelling). CC-BY 4.0. <https://dmol.pub>

Continues the ESOL dataset from Weeks 02-03/05/07. No verbatim text is
reproduced from the sources above.
''')]

build(__file__, "week08_a_sequences-attention", C)
