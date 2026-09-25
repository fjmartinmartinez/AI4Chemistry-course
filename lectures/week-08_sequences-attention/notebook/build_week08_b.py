"""Build week 08 session B notebooks (a pretrained chemical transformer for
property prediction).

    python lectures/week-08_sequences-attention/notebook/build_week08_b.py
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "_build"))
from nbbuild import build, checkpoint, code, exercise, md  # noqa: E402

C = []

# ==========================================================================
C += [md(r'''
# AI for Chemistry — Week 08, Session B
## Workshop: a pretrained chemical transformer for property prediction

**Course:** AI for Chemistry · **Session:** 08B (hands-on workshop, 2.0 h) ·
**Runtime:** < 60 s on laptop CPU (`N_MAX=400` molecules).

This notebook downloads a ~180 MB pretrained model on first run (needs
network access). `transformers` is already installed in the course's
`ai4chem` environment (`env/environment.yml`). See **Section 0**.

### Suggested timing (solo study, ~120 min)

| Section | Topic | Time |
|--------:|-------|-----:|
| 0 | Setup: load ChemBERTa, reduced-size working set | 12 min |
| 1 | Subword vs character tokenisation | 15 min |
| 2 | Extracting frozen embeddings | 15 min |
| 3 | Chemical space from a pretrained embedding | 15 min |
| 4 | Property prediction: frozen embeddings vs the leaderboard | 20 min |
| 5 | Inspecting a real model's attention weights | 15 min |
| 6 | Exercises (3) + mini-challenge | 28 min |

### Learning objectives
1. Load a pretrained chemical language model (ChemBERTa) with `transformers`.
   *(LO7)*
2. Contrast subword (BPE) tokenisation with Week 08A's character-level
   tokenisation. *(LO7)*
3. Use a **frozen** pretrained model's embeddings for property prediction,
   and compare honestly to the classical/MLP/GNN leaderboard. *(LO4, LO7)*
4. Extract and interpret a real transformer's own attention weights. *(LO7)*

### Prerequisites — before this notebook you should be able to
- Week 08A: tokenisation, attention mechanics, seq2vec.
- Week 03/06/07: the leaderboard-comparison discipline.

### How to use this notebook (solo study)
Read, run, check each **✅** cell in order. This is the **solutions**
notebook.
''')]

# ==========================================================================
C += [md(r'''
---
## 0. Setup: load ChemBERTa, reduced-size working set

This notebook uses a **reduced-size working set (`N_MAX`)**: extracting
embeddings for every molecule from a transformer is far more expensive
per-molecule than computing an RDKit descriptor. Capping the working set
keeps Section 2's embedding extraction comfortably inside a laptop CPU's
session budget.

**What to look for:** `Using device: cpu`; `N_MAX = 400`.
''')]

C += [code(r'''
try:
    import transformers  # noqa: F401
    print("transformers", transformers.__version__)
except ImportError as exc:
    raise SystemExit(
        "transformers is not installed. Run "
        "`conda env update -f env/environment.yml` and restart the kernel."
    ) from exc
''')]

C += [code(r'''
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModel

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
RDLogger.DisableLog("rdApp.*")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

SEED = 0xC0FFEE
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_MAX = 400          # keeps embedding extraction inside a laptop CPU's session budget
print(f"Using device: {DEVICE}   N_MAX = {N_MAX}")

ROOT = Path.cwd()
while not (ROOT / "sources").is_dir() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
DATA = ROOT / "sources" / "datasets"
FIGDIR = ROOT / "lectures" / "week-08_sequences-attention" / "slides" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

TARGET = "measured log solubility in mols per litre"
esol = pd.read_csv(DATA / "esol_clean.csv")

rng = np.random.default_rng(SEED)
sub_idx = np.sort(rng.choice(len(esol), size=N_MAX, replace=False))
sub = esol.iloc[sub_idx].reset_index(drop=True)
y = sub[TARGET].to_numpy()
smiles_list = sub["canonical_smiles"].tolist()
print(f"working set: {len(sub)} molecules (of {len(esol)} total)")
''')]

C += [code(r'''
MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"        # RoBERTa-style, pretrained on ZINC SMILES

t0 = time.time()
chembert_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
chembert_model = AutoModel.from_pretrained(MODEL_NAME, output_attentions=True).to(DEVICE)
chembert_model.eval()                                  # frozen: no fine-tuning this session
print(f"loaded {MODEL_NAME} in {time.time() - t0:.1f} s "
      f"({sum(p.numel() for p in chembert_model.parameters()):,} parameters)")
''')]

C += [md(r'''
> **Common errors — Setup**
> - Running this notebook with no internet access on first use — both the
>   tokeniser and model weights download from the Hugging Face Hub the first
>   time; subsequent runs use a local cache (`~/.cache/huggingface`).
> - Calling `.train()` or computing gradients on `chembert_model` — we use it
>   strictly as a **frozen** feature extractor this week; fine-tuning a 44M
>   parameter model on a laptop CPU would breach the compute budget.
''')]

C += checkpoint(
    "Section 0",
    check=r'''
assert N_MAX == len(sub) == len(smiles_list)
assert chembert_model.training is False
print("Section 0 OK")
''',
    expected="Prints `Section 0 OK`. `N_MAX` matches the working subsample "
    "size, and the model is in evaluation (not training) mode.",
    questions=r'''
1. Why is `N_MAX` capped at 400 molecules rather than using the full
   1117-molecule ESOL set?
2. What would go wrong if you forgot to call `chembert_model.eval()`?
''',
    answers=r'''
1. Extracting a transformer embedding for every molecule means a full
   forward pass through a 44M-parameter model, far more expensive than
   computing a handful of closed-form RDKit descriptors. Capping the working
   set keeps Section 2's embedding extraction comfortably inside a laptop
   CPU's session budget without changing anything about how frozen-embedding
   property prediction works.
2. Layers that behave differently in train vs eval mode (dropout, batch
   normalisation — Week 05's *Standard layers* material) would use their
   *training-time* behaviour (e.g. randomly dropping activations), making
   "frozen" feature extraction non-deterministic and needlessly noisy.
''',
)

# ==========================================================================
# 1. Subword vs character tokenisation
# ==========================================================================
C += [md(r'''
---
## 1. Subword vs character tokenisation

Week 08A tokenised character by character. ChemBERTa instead uses **byte-pair
encoding (BPE)**: a fixed vocabulary of common multi-character *subwords*,
learned from a large corpus of SMILES so that frequent fragments (e.g.
`"ccccc"`, `"(="`) become single tokens instead of several characters each.

**What to look for:** aspirin's SMILES (22 characters) needs only about 13-15
BPE tokens (including the special `<s>`/`</s>` sequence markers) — noticeably
fewer than one token per character.
''')]

C += [code(r'''
aspirin_smiles = "CC(=O)Oc1ccccc1C(=O)O"
encoded = chembert_tokenizer(aspirin_smiles, return_tensors="pt")
bpe_tokens = chembert_tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])

print("SMILES length (characters):", len(aspirin_smiles))
print("BPE token count (incl. <s>/</s>):", len(bpe_tokens))
print("BPE tokens:", bpe_tokens)
''')]

C += [md(r'''
> **Common errors — Section 1**
> - Assuming BPE tokens correspond to chemically meaningful units (like
>   "functional groups") — they are chosen purely by **statistical
>   frequency** in the pretraining corpus, and can split a bond symbol from
>   an atom or merge unrelated-looking characters if that pairing was common.
> - Forgetting the special tokens `<s>` (start) and `</s>` (end) when
>   comparing token counts — they add 2 to every sequence regardless of
>   molecule size.
''')]

C += checkpoint(
    "Section 1",
    check=r'''
assert bpe_tokens[0] == "<s>" and bpe_tokens[-1] == "</s>"
assert len(bpe_tokens) < len(aspirin_smiles)
print("Section 1 OK")
''',
    expected="Prints `Section 1 OK`. Aspirin needs fewer BPE tokens than "
    "characters.",
    questions=r'''
1. Why might a smaller, more compressed token sequence be advantageous for a
   transformer specifically (beyond just "fewer numbers to process")?
2. Would you expect a novel, unusual SMILES fragment never seen in ZINC
   (the pretraining corpus) to tokenise as efficiently as a common one?
''',
    answers=r'''
1. Self-attention's compute cost grows **quadratically** with sequence
   length ($L\times L$ attention matrix); shorter sequences for the same
   information content are cheaper to process and let a fixed attention
   *window*/depth reach across more of the original molecule in fewer hops
   (echoing Week 07B's receptive-field argument, now for sequences).
2. No — BPE vocabularies are built from the pretraining corpus's actual
   statistics, so an unusual fragment likely gets split into more, smaller
   subword pieces (in the worst case, individual characters), since it never
   appeared often enough to earn its own token.
''',
)

# ==========================================================================
# 2. Extracting frozen embeddings
# ==========================================================================
C += [md(r'''
---
## 2. Extracting frozen embeddings

For each molecule, we run its SMILES through ChemBERTa and take the hidden
state at the `<s>` position (analogous to BERT's `[CLS]` token) as a
fixed-length, 768-dimensional summary of the whole molecule — computed once,
with **no gradient tracking** (`torch.no_grad()`), since we are not training
anything.

**What to look for:** one 768-dimensional vector per molecule; extraction
takes well under a minute for `N_MAX` molecules on a laptop CPU.
''')]

C += [code(r'''
def embed_smiles(smiles, tokenizer, model, device=DEVICE):
    """<s>-token hidden state (768-d) for one SMILES string, frozen inference."""
    encoded = tokenizer(smiles, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**encoded)
    return output.last_hidden_state[0, 0, :].cpu().numpy()   # position 0 = <s>

t0 = time.time()
chembert_embeddings = np.stack([embed_smiles(s, chembert_tokenizer, chembert_model)
                                for s in smiles_list])
print(f"extracted {chembert_embeddings.shape} embeddings in {time.time() - t0:.1f} s")
''')]

C += [md(r'''
> **Common errors — Section 2**
> - Omitting `torch.no_grad()` — gradients would still be tracked for every
>   forward pass, wasting memory and time for no benefit since nothing is
>   being trained.
> - Taking `last_hidden_state[0, -1, :]` (the *last* token) instead of
>   position 0 — for this model family the sequence-summary convention is the
>   *first* special token, not the last; always check a model's own
>   documentation rather than assuming BERT's convention applies universally.
''')]

C += checkpoint(
    "Section 2",
    check=r'''
assert chembert_embeddings.shape == (N_MAX, 768)
assert np.isfinite(chembert_embeddings).all()
print("Section 2 OK")
''',
    expected="Prints `Section 2 OK`. One finite 768-d embedding per molecule.",
    questions=r'''
1. Why does this section wrap the forward pass in `torch.no_grad()` given
   Section 0 already put the model in `.eval()` mode?
2. This embedding was produced by a model that never saw a single solubility
   label during pretraining. What *did* it learn to predict during
   pretraining that could plausibly make its embedding useful for solubility
   anyway?
''',
    answers=r'''
1. `.eval()` only changes the *behaviour* of mode-sensitive layers (dropout,
   batch norm); it does not stop gradient tracking, which has its own,
   separate cost in memory and time — the two settings address different
   things and are both needed for genuinely cheap frozen inference.
2. Masked-language-model pretraining (Week 08A, Section 5): predicting
   randomly hidden SMILES tokens from context forces the model to learn
   which atoms/fragments plausibly co-occur, which is correlated with real
   chemical properties (a hydroxyl next to a long alkyl chain "looks
   different" in context to the model than one on a small polar molecule) —
   without ever needing solubility labels directly.
''',
)

# ==========================================================================
# 3. Chemical space from a pretrained embedding
# ==========================================================================
C += [md(r'''
---
## 3. Chemical space from a pretrained embedding

Exactly Week 02B/06B's PCA recipe, now on ChemBERTa's embedding instead of
descriptors or fingerprints.

**What to look for:** at least one of the first two principal components
correlates noticeably with measured solubility — evidence that pretraining
alone (never having seen a solubility label) organised chemical space in a
way that is *not* independent of it.
''')]

C += [code(r'''
Xstd = StandardScaler().fit_transform(chembert_embeddings)
Z = PCA(n_components=2, random_state=0).fit_transform(Xstd)

corr_pc1 = np.corrcoef(Z[:, 0], y)[0, 1]
corr_pc2 = np.corrcoef(Z[:, 1], y)[0, 1]
print(f"corr(PC1, log S) = {corr_pc1:.3f}   corr(PC2, log S) = {corr_pc2:.3f}")

fig, ax = plt.subplots(figsize=(6, 4.5))
sc = ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap="coolwarm_r", s=10)
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
ax.set_title("ChemBERTa embedding, coloured by measured log S")
fig.colorbar(sc, label="measured log S")
fig.tight_layout(); fig.savefig(FIGDIR / "chembert_embedding_pca.png", dpi=200)
plt.show()
''')]

C += [md(r'''
> **Common errors — Section 3**
> - Expecting the correlation here to rival Week 02B's descriptor-PCA result
>   — ChemBERTa was never trained towards *any* downstream property, so this
>   is measuring an incidental (if genuine) alignment, not a designed one.
''')]

C += checkpoint(
    "Section 3",
    check=r'''
assert Z.shape == (N_MAX, 2)
assert max(abs(corr_pc1), abs(corr_pc2)) > 0.25
assert (FIGDIR / "chembert_embedding_pca.png").is_file()
print("Section 3 OK")
''',
    expected="Prints `Section 3 OK`. At least one of PC1/PC2 correlates with "
    "log S at |r| > 0.25.",
    questions=r'''
1. Why is it notable that *any* correlation with solubility appears here at
   all?
2. Would you expect this correlation to be stronger or weaker for a property
   very different from anything related to general "chemical similarity" —
   e.g. a specific enzyme's binding affinity to a narrow chemical series?
''',
    answers=r'''
1. The model's pretraining objective (predicting masked SMILES tokens on
   ZINC) never involved solubility, or any property label, at all — any
   alignment reflects that "chemically similar" molecules (by pretraining's
   implicit notion of similarity) also tend to be similarly soluble, a
   genuine, unforced finding rather than something engineered in.
2. Likely weaker — general pretraining captures broad chemical regularities,
   not the specific, often subtle structure-activity relationships that
   determine binding to one particular target; a property tightly coupled to
   one narrow mechanism is less likely to align with a generic embedding's
   dominant axes of variation.
''',
)

# ==========================================================================
# 4. Property prediction: frozen embeddings vs the leaderboard
# ==========================================================================
C += [md(r'''
---
## 4. Property prediction: frozen embeddings vs the leaderboard

We fit simple models on top of the frozen 768-d embedding (never touching
the transformer's own weights) and compare, **on this same `N_MAX`-molecule
subsample and split**, against RDKit descriptors — the same comparison
discipline as Weeks 03/06/07.

Note this uses a *different* (smaller) working set than Weeks 03/05/07's
1117-molecule leaderboard, because of Section 0's reduced-size fallback — so
treat this as a **self-contained** mini-study, not a direct number-for-number
continuation of the earlier leaderboard.

Ridge regression on a 768-dimensional embedding with only ~320 training rows
needs **much stronger regularisation** than Week 03's default `alpha=1` (the
same $p\gg n$ lesson from Week 03B, at a new scale).

**What to look for:** the frozen ChemBERTa embedding clearly beats the
mean-predictor baseline, but — just like Week 07B's from-scratch GCN — does
**not** beat simple linear regression or a random forest on the 7-descriptor
block here. A pretrained representation is a strong *prior* over chemical
space in general, not a guarantee of winning on every specific small
dataset.
''')]

C += [code(r'''
DESCRIPTORS = {
    "MolWt": Descriptors.MolWt, "MolLogP": Descriptors.MolLogP,
    "TPSA": Descriptors.TPSA, "NumHDonors": Descriptors.NumHDonors,
    "NumHAcceptors": Descriptors.NumHAcceptors,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    "NumAromaticRings": Descriptors.NumAromaticRings,
}
mols = [Chem.MolFromSmiles(s) for s in smiles_list]
Xdesc = np.array([[fn(m) for fn in DESCRIPTORS.values()] for m in mols], dtype=float)

idx = np.arange(N_MAX)
idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42)
y_train, y_test = y[idx_train], y[idx_test]

def fit_and_score(X, model_fn, standardise=True):
    X_train, X_test = X[idx_train], X[idx_test]
    if standardise:
        scaler = StandardScaler().fit(X_train)
        X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
    model = model_fn().fit(X_train, y_train)
    return mean_squared_error(y_test, model.predict(X_test)) ** 0.5

baseline_rmse = mean_squared_error(y_test, np.full_like(y_test, y_train.mean())) ** 0.5
linear_rmse = fit_and_score(Xdesc, LinearRegression)
rf_desc_rmse = fit_and_score(Xdesc, lambda: RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1),
                             standardise=False)
ridge_chembert_rmse = fit_and_score(chembert_embeddings, lambda: Ridge(alpha=100.0))
rf_chembert_rmse = fit_and_score(chembert_embeddings,
                                 lambda: RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1),
                                 standardise=False)

comparison = pd.DataFrame({
    "model": ["Baseline (mean)", "Linear (descriptors)", "Random forest (descriptors)",
             "Ridge, alpha=100 (ChemBERTa)", "Random forest (ChemBERTa)"],
    "test_RMSE": [baseline_rmse, linear_rmse, rf_desc_rmse, ridge_chembert_rmse, rf_chembert_rmse],
}).sort_values("test_RMSE").reset_index(drop=True)
comparison.round(3)
''')]

C += [code(r'''
fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.barh(comparison["model"][::-1], comparison["test_RMSE"][::-1], color="teal")
ax.set_xlabel(f"test RMSE (log S units, N_MAX={N_MAX} molecules)")
ax.set_title("Week 08B: frozen ChemBERTa vs the descriptor leaderboard")
fig.tight_layout(); fig.savefig(FIGDIR / "chembert_vs_leaderboard.png", dpi=200)
plt.show()
''')]

C += [md(r'''
> **Common errors — Section 4**
> - Using `alpha=1` on the embedding (as a first, unregularised attempt) and
>   concluding "ChemBERTa embeddings are useless" — check the *Common
>   errors* box would-be reader lesson from Week 03B: a poorly tuned Ridge on
>   a wide, correlated feature block can look far worse than it needs to.
> - Comparing this section's RMSEs directly to Week 03/05/07's numbers as if
>   on the same test set — they are not (different `N_MAX`, different
>   molecules); compare *rankings* here, not raw numbers across notebooks.
''')]

C += checkpoint(
    "Section 4",
    check=r'''
assert len(comparison) == 5
assert ridge_chembert_rmse < baseline_rmse
assert comparison.iloc[0]["model"] == "Random forest (descriptors)"
assert (FIGDIR / "chembert_vs_leaderboard.png").is_file()
print("Section 4 OK")
''',
    expected="Prints `Section 4 OK`. ChemBERTa-based models beat the "
    "baseline; random forest on descriptors still tops this table.",
    questions=r'''
1. dmol.pub's own GRU seq2vec (Week 08A) and this notebook's frozen ChemBERTa
   probe both underperform simple descriptors on ESOL-scale data. Is that a
   coincidence?
2. What would change about this comparison if instead of *freezing*
   ChemBERTa, you fine-tuned its own weights on the solubility task?
''',
    answers=r'''
1. No — both are instances of the same pattern this course keeps returning
   to (Weeks 05B, 06, 07B): a **learned** representation, whether from
   scratch or via general pretraining, must earn its flexibility with
   enough task-relevant data; ~1000 molecules (or fewer, with `N_MAX`) is
   comfortably enough for a 7-descriptor linear model but modest for
   representations with hundreds of learned dimensions.
2. Fine-tuning lets the model's own weights specialise directly for
   solubility rather than relying on a generic pretraining objective —
   typically closing much of this gap (this is exactly why fine-tuned
   chemical LLMs are common in practice) — at the cost of much more compute
   than this session's laptop-CPU, frozen-inference budget allows.
''',
)

# ==========================================================================
# 5. Inspecting a real model's attention weights
# ==========================================================================
C += [md(r'''
---
## 5. Inspecting a real model's attention weights

Week 08A's attention heatmap used **untrained, random** projections —
mechanically correct but chemically meaningless. ChemBERTa's attention
weights, in contrast, come from a model that has actually been trained (via
masked-language-model pretraining) — so, unlike Section A's toy example, a
real pattern may (or may not) be visible.

**What to look for:** an attention matrix over aspirin's BPE tokens from one
of ChemBERTa's later layers; rows still sum to 1 (attention is still exactly
the same softmax-normalised mechanism from Week 08A, just with **learned**
projections and **6 stacked layers x 12 heads** instead of one untrained
layer).
''')]

C += [code(r'''
encoded = chembert_tokenizer(aspirin_smiles, return_tensors="pt").to(DEVICE)
with torch.no_grad():
    output = chembert_model(**encoded)

# output.attentions: tuple of (1, n_heads, L, L), one per transformer layer
last_layer_attn = output.attentions[-1][0]        # (n_heads, L, L)
head0 = last_layer_attn[0].cpu().numpy()           # first head of the last layer
tokens_bpe = chembert_tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])

print("n layers:", len(output.attentions), " n heads per layer:", last_layer_attn.shape[0])
print("row sums (should all be 1):", head0.sum(axis=-1).round(4))
''')]

C += [code(r'''
fig, ax = plt.subplots(figsize=(6, 5.5))
im = ax.imshow(head0, cmap="viridis")
ax.set_xticks(range(len(tokens_bpe))); ax.set_xticklabels(tokens_bpe, rotation=90, fontsize=8)
ax.set_yticks(range(len(tokens_bpe))); ax.set_yticklabels(tokens_bpe, fontsize=8)
ax.set_title("ChemBERTa attention (last layer, head 0) -- aspirin")
fig.colorbar(im, label="attention weight")
fig.tight_layout(); fig.savefig(FIGDIR / "chembert_attention_heatmap.png", dpi=200)
plt.show()
''')]

C += [md(r'''
> **Common errors — Section 5**
> - Over-interpreting one head of one layer as "what the model is doing" —
>   with 6 layers x 12 heads, different heads specialise differently (and
>   some are famously close to uniform or diagonal); a rigorous
>   interpretability study (Week 09's XAI content) aggregates or selects
>   heads deliberately rather than reading one at random.
> - Forgetting the special tokens `<s>`/`</s>` are included in the attention
>   matrix — a head that appears to "attend mostly to position 0" may simply
>   be attending to the sequence-start marker, a known, well-documented
>   pattern in many transformer models, not a chemistry finding.
''')]

C += checkpoint(
    "Section 5",
    check=r'''
assert last_layer_attn.shape[-1] == len(tokens_bpe)
assert np.allclose(head0.sum(axis=-1), 1.0, atol=1e-4)
assert (FIGDIR / "chembert_attention_heatmap.png").is_file()
print("Section 5 OK")
''',
    expected="Prints `Section 5 OK`. Attention matrix size matches the "
    "token count; rows sum to 1, exactly as in Week 08A.",
    questions=r'''
1. What is exactly the same, mechanically, between this section's attention
   computation and Week 08A Section 2's, and what is different?
2. Why look at `output.attentions[-1]` (the **last** layer) rather than the
   first, if you wanted to see the most "semantically processed" attention
   pattern?
''',
    answers=r'''
1. Same: the scaled dot-product formula itself, and the row-sums-to-1
   softmax property. Different: the $W_q,W_k,W_v$ projections are **learned**
   (from masked-language-model pretraining on real SMILES) rather than
   random, and there are 6 layers x 12 heads combined, not one.
2. Early layers in a transformer tend to capture more local/surface
   patterns (adjacent tokens, syntax-like regularities); later layers build
   on those to represent more abstract, longer-range relationships — the
   same intuition as Week 07's depth discussion, applied to sequence models.
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
### Exercise 1 — tokenisation compression ratio *(easy, ~10 min)*

Write `compression_ratio(smiles, tokenizer)` returning
`len(smiles) / n_bpe_tokens_excluding_specials`. Compute it for 5 ESOL
molecules of varying length and check whether longer molecules compress more.

<details><summary>Show hint</summary>

`len(tokenizer(smiles)["input_ids"]) - 2` excludes `<s>`/`</s>`.
</details>
''',
    solution=r'''
def compression_ratio(smiles, tokenizer):
    """characters per BPE token (excluding <s>/</s>)."""
    n_tokens = len(tokenizer(smiles)["input_ids"]) - 2
    return len(smiles) / n_tokens

sample = sorted(smiles_list, key=len)[-5:]     # 5 of the longer molecules in the subsample
ratios = [compression_ratio(s, chembert_tokenizer) for s in sample]
for s, r in zip(sample, ratios):
    print(f"{r:.2f} chars/token  ({len(s):3d} chars)  {s}")
''',
    scaffold=r'''
def compression_ratio(smiles, tokenizer):
    """characters per BPE token (excluding <s>/</s>)."""
    # YOUR CODE HERE
    ...

sample = sorted(smiles_list, key=len)[-5:]
ratios = [compression_ratio(s, chembert_tokenizer) for s in sample]
''',
    check=r'''
assert all(r > 1.0 for r in ratios)          # BPE should compress, not expand
assert len(ratios) == 5
print("Exercise 1 OK")
''',
)

C += exercise(
    prompt=r'''
### Exercise 2 — pick alpha properly *(medium, ~15 min)*

Section 4 used `alpha=100` for `Ridge` on the ChemBERTa embedding without
justification. Write `best_ridge_alpha(X, y_train_idx, y, alphas)` that
5-fold cross-validates `Ridge(alpha=a)` **on the training set only** for each
candidate and returns the alpha with the lowest mean CV RMSE (Week 03A,
Exercise 4's pattern).

<details><summary>Show hint</summary>

`from sklearn.model_selection import cross_val_score`;
`-cross_val_score(Ridge(alpha=a), Xstd_train, y[idx_train], cv=5,
scoring="neg_root_mean_squared_error").mean()`.
</details>
''',
    solution=r'''
from sklearn.model_selection import cross_val_score

def best_ridge_alpha(X, idx_train, y, alphas):
    """Alpha (from `alphas`) with the lowest 5-fold CV RMSE on the training rows."""
    Xstd = StandardScaler().fit_transform(X[idx_train])
    scores = [-cross_val_score(Ridge(alpha=a), Xstd, y[idx_train], cv=5,
                                scoring="neg_root_mean_squared_error").mean()
             for a in alphas]
    return alphas[int(np.argmin(scores))]

alpha_grid = [1, 10, 30, 100, 300, 1000]
best_alpha = best_ridge_alpha(chembert_embeddings, idx_train, y, alpha_grid)
print("best alpha:", best_alpha)
''',
    scaffold=r'''
from sklearn.model_selection import cross_val_score

def best_ridge_alpha(X, idx_train, y, alphas):
    """Alpha (from `alphas`) with the lowest 5-fold CV RMSE on the training rows."""
    # YOUR CODE HERE
    ...

alpha_grid = [1, 10, 30, 100, 300, 1000]
best_alpha = ...  # YOUR CODE HERE: best_ridge_alpha(chembert_embeddings, idx_train, y, alpha_grid)
''',
    check=r'''
assert best_alpha in alpha_grid
assert best_alpha > 1          # alpha=1 was shown in Section 4 markdown to be far too weak
print("Exercise 2 OK")
''',
)

C += exercise(
    prompt=r'''
### Exercise 3 — mean-pool over tokens instead of `<s>` *(medium, ~15 min)*

Write `embed_smiles_meanpool(smiles, tokenizer, model)`, an alternative to
Section 2's `embed_smiles` that **mean-pools over every real token**
(excluding `<s>`/`</s>`) instead of reading off position 0. Extract embeddings
for the same `smiles_list` and refit `Ridge(alpha=100)`.

<details><summary>Show hint</summary>

`last_hidden_state[0, 1:-1, :].mean(dim=0)` slices off the first and last
positions before averaging.
</details>
''',
    solution=r'''
def embed_smiles_meanpool(smiles, tokenizer, model, device=DEVICE):
    """Mean-pooled hidden state over real tokens (excludes <s>/</s>)."""
    encoded = tokenizer(smiles, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**encoded)
    return output.last_hidden_state[0, 1:-1, :].mean(dim=0).cpu().numpy()

meanpool_embeddings = np.stack([embed_smiles_meanpool(s, chembert_tokenizer, chembert_model)
                                for s in smiles_list])
meanpool_rmse = fit_and_score(meanpool_embeddings, lambda: Ridge(alpha=100.0))
print(f"mean-pooled embedding + Ridge(alpha=100): RMSE = {meanpool_rmse:.3f}")
print(f"<s>-token embedding + Ridge(alpha=100):   RMSE = {ridge_chembert_rmse:.3f}")
''',
    scaffold=r'''
def embed_smiles_meanpool(smiles, tokenizer, model, device=DEVICE):
    """Mean-pooled hidden state over real tokens (excludes <s>/</s>)."""
    # YOUR CODE HERE
    ...

meanpool_embeddings = ...  # YOUR CODE HERE: stack embed_smiles_meanpool over smiles_list
meanpool_rmse = ...  # YOUR CODE HERE: fit_and_score(meanpool_embeddings, lambda: Ridge(alpha=100.0))
''',
    check=r'''
assert meanpool_embeddings.shape == (N_MAX, 768)
assert meanpool_rmse > 0
print("Exercise 3 OK")
''',
)

C += [md(r'''
### 🏁 Mini-challenge — is a bigger frozen model worth the wait? *(~20 min)*

`seyonec/ChemBERTa-zinc-base-v1` is a relatively small chemical transformer
(44M parameters). Rather than downloading a second model (network- and
time-costly), **simulate** the "does more capacity help a frozen probe"
question using the two representations you already have: the `<s>`-token
embedding (768-d, Section 2) and the mean-pooled embedding (768-d,
Exercise 3), **concatenated** into a 1536-d representation.

Write `evaluate_concat_embedding(emb_a, emb_b, alpha)` that concatenates two
embedding matrices and scores `Ridge(alpha=alpha)` on the result. Does giving
the probe *both* views help, hurt, or make no real difference versus the
better of the two alone?
''')]

C += exercise(
    prompt=r'''
Implement `evaluate_concat_embedding` and compare it to Section 4 and
Exercise 3's individual results.
''',
    solution=r'''
def evaluate_concat_embedding(emb_a, emb_b, alpha):
    """RMSE of Ridge(alpha) fit on [emb_a | emb_b] concatenated column-wise."""
    combined = np.hstack([emb_a, emb_b])
    return fit_and_score(combined, lambda: Ridge(alpha=alpha))

concat_rmse = evaluate_concat_embedding(chembert_embeddings, meanpool_embeddings, alpha=100.0)
print(f"<s>-token only:        RMSE = {ridge_chembert_rmse:.3f}")
print(f"mean-pooled only:      RMSE = {meanpool_rmse:.3f}")
print(f"concatenated (1536-d): RMSE = {concat_rmse:.3f}")
''',
    scaffold=r'''
def evaluate_concat_embedding(emb_a, emb_b, alpha):
    """RMSE of Ridge(alpha) fit on [emb_a | emb_b] concatenated column-wise."""
    # YOUR CODE HERE
    ...

concat_rmse = ...  # YOUR CODE HERE: evaluate_concat_embedding(chembert_embeddings, meanpool_embeddings, 100.0)
''',
    check=r'''
assert concat_rmse > 0
assert concat_rmse < baseline_rmse
print("Mini-challenge OK")
''',
)

# ==========================================================================
C += [md(r'''
---
## Summary — what you learned

- A **reduced-size working set (`N_MAX`)** pattern that keeps a notebook
  using a real pretrained model runnable comfortably on a laptop CPU.
- **Subword (BPE) tokenisation**, and how it compresses a SMILES string
  compared to Week 08A's character-level tokenisation.
- Extracting **frozen** embeddings from a pretrained chemical transformer,
  and using them for property prediction with proper regularisation
  (`alpha` tuned, not guessed).
- An **honest comparison**: the frozen ChemBERTa probe beats the trivial
  baseline but not simple descriptors on this dataset scale — the same
  "learned representations need enough data" pattern as Weeks 05B, 06 and
  07B, now shown for a *pretrained*, not from-scratch, representation.
- Inspecting a **real** transformer's own attention weights, contrasted with
  Week 08A's untrained toy example.

Week 09 moves from understanding molecules to **generating** them: VAEs,
generative RNNs, and a first look at explainability (XAI).
''')]

C += [md(r'''
## Further reading (course source list only)

- dmol.pub *Pretraining*: <https://dmol.pub/dl/pretraining.html>
- EPFL *AI for Chemistry*, reaction prediction (template-free) — the
  syllabus's original B-session core link (see `outline.md` for why this
  notebook uses property prediction instead):
  <https://colab.research.google.com/github/schwallergroup/ai4chem_course/blob/main/notebooks/07%20-%20Reaction%20Prediction/template_free.ipynb>
''')]

C += [md(r'''
## Attribution

Original teaching material for *AI for Chemistry*, adapted and rewritten from:

- **ChemBERTa** (Chithrananda, Grand & Ramsundar, 2020) via the
  `seyonec/ChemBERTa-zinc-base-v1` checkpoint on the Hugging Face Hub — the
  pretrained model used directly (frozen) in this notebook. Model and
  tokenizer distributed under their respective Hugging Face Hub licence
  terms; `TODO(verify)` the exact licence before any redistribution beyond
  classroom use.
- **dmol.pub** (A. White) — the pretraining/frozen-embedding framing carried
  over from Week 08A. CC-BY 4.0. <https://dmol.pub>

Continues the ESOL dataset from Weeks 02-03/05-07. No verbatim text is
reproduced from the sources above.
''')]

build(__file__, "week08_b_sequences-attention", C)
