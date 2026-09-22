"""Build week 09 session A notebooks (VAE core, flows/diffusion survey, XAI).

    python lectures/week-09_generative-models-xai/notebook/build_week09_a.py
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "_build"))
from nbbuild import build, checkpoint, code, exercise, md  # noqa: E402

C = []

# ==========================================================================
C += [md(r'''
# AI for Chemistry — Week 09, Session A
## Generative models: VAE core; explaining predictions

**Course:** AI for Chemistry · **Session:** 09A (interleaved lecture + lab,
2.5 h) · **Runtime:** < 10 s, no GPU.

### Suggested timing (solo study, ~120 min)

| Section | Topic | Time |
|--------:|-------|-----:|
| 0 | Setup | 6 min |
| 1 | The VAE: encoder, reparameterisation, ELBO | 28 min |
| 2 | Sampling the prior — and a real failure mode | 16 min |
| 3 | Flows and diffusion (survey) | 8 min |
| 4 | Explaining predictions: gradient saliency and integrated gradients | 28 min |
| 5 | SHAP, LIME, counterfactuals (survey) | 8 min |
| 6 | Exercises (4) | 26 min |

### Learning objectives
1. Explain the VAE architecture and implement the reparameterisation trick
   and the ELBO loss (reconstruction + KL). *(LO7)*
2. Compare a trained VAE's latent space to PCA and identify a concrete
   sampling failure mode. *(LO4, LO7)*
3. State, at survey level, what normalising flows and diffusion models are. *(LO7)*
4. Compute gradient saliency and integrated gradients for a trained model and
   verify integrated gradients' completeness property. *(LO7, LO8)*

### Prerequisites — before this notebook you should be able to
- Week 05: `nn.Module`, autograd, training loops.
- Week 02B/06B: PCA on a standardised feature block.

### How to use this notebook (solo study)
Read, run, check each **✅** cell in order. This is the **solutions**
notebook.
''')]

# ==========================================================================
C += [md(r'''
---
## 0. Setup

Rebuild the ESOL 7-descriptor block (Weeks 02-03), standardised — the input
to both this session's VAE (Section 1) and the small regression model used
for explainability (Section 4).

**What to look for:** `Xt` has shape `(1117, 7)`.
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
from rdkit.Chem import Descriptors
RDLogger.DisableLog("rdApp.*")

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

SEED = 0xC0FFEE
np.random.seed(SEED)
torch.manual_seed(SEED)

ROOT = Path.cwd()
while not (ROOT / "sources").is_dir() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
DATA = ROOT / "sources" / "datasets"
FIGDIR = ROOT / "lectures" / "week-09_generative-models-xai" / "slides" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

TARGET = "measured log solubility in mols per litre"
DESCRIPTORS = {
    "MolWt": Descriptors.MolWt, "MolLogP": Descriptors.MolLogP,
    "TPSA": Descriptors.TPSA, "NumHDonors": Descriptors.NumHDonors,
    "NumHAcceptors": Descriptors.NumHAcceptors,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    "NumAromaticRings": Descriptors.NumAromaticRings,
}
desc_names = list(DESCRIPTORS)
esol = pd.read_csv(DATA / "esol_clean.csv")
mols = [Chem.MolFromSmiles(s) for s in esol["canonical_smiles"]]
Xraw = np.array([[fn(m) for fn in DESCRIPTORS.values()] for m in mols], dtype=float)
y = esol[TARGET].to_numpy()

scaler = StandardScaler().fit(Xraw)
Xstd = scaler.transform(Xraw)
Xt = torch.tensor(Xstd, dtype=torch.float32)
yt = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
print("Xt:", tuple(Xt.shape))
''')]

C += [md(r'''
> **Common errors — Setup**
> - Fitting `StandardScaler` on data that already contains the VAE's own
>   decoded outputs later in the notebook — always fit the scaler once, here,
>   on the real data, and reuse `scaler.inverse_transform` to interpret
>   anything decoded (Section 2).
''')]

# ==========================================================================
# 1. The VAE: encoder, reparameterisation, ELBO
# ==========================================================================
C += [md(r'''
---
## 1. The VAE: encoder, reparameterisation, ELBO

A **variational autoencoder** learns a compressed, continuous **latent
space** $z$ that a **decoder** can turn back into (approximate)
reconstructions of the input. Unlike Week 05's plain autoencoder-style
compression, a VAE's **encoder** outputs the *parameters* of a distribution —
a mean $\mu$ and log-variance $\log\sigma^2$ — rather than a single point,
and $z$ is *sampled* from $\mathcal N(\mu,\sigma^2)$.

Sampling is not differentiable directly, so we use the
**reparameterisation trick**:

$$z = \mu + \sigma \odot \epsilon, \qquad \epsilon \sim \mathcal N(0,1),$$

which moves all the randomness into $\epsilon$ (which needs no gradient),
leaving $\mu,\sigma$ on a fully differentiable path. Training minimises the
**ELBO** loss — reconstruction error plus a KL-divergence term pulling the
encoder's distribution towards the prior $\mathcal N(0,1)$:

$$\mathcal L = \underbrace{-\log p_\theta(x\mid z)}_{\text{reconstruction}}
  + \beta\underbrace{\left(-\log\sigma - \tfrac12 + \tfrac{\sigma^2+\mu^2}{2}\right)}_{\text{KL to }\mathcal N(0,1)}.$$

$\beta<1$ favours faithful reconstruction; $\beta>1$ favours a latent space
that matches the prior more closely (dmol.pub's *beta-VAE* framing).

**What to look for:** reconstruction loss falls steadily over training; the
2D latent space, coloured by measured solubility, shows visible structure —
**learned without ever using the label** (the VAE only ever sees `Xt`).
''')]

C += [code(r'''
class VAE(nn.Module):
    """Minimal VAE: Linear encoder/decoder, 2D latent space."""

    def __init__(self, in_dim=7, hidden=32, latent_dim=2):
        super().__init__()
        self.enc_hidden = nn.Linear(in_dim, hidden)
        self.enc_mu = nn.Linear(hidden, latent_dim)
        self.enc_logvar = nn.Linear(hidden, latent_dim)
        self.dec_hidden = nn.Linear(latent_dim, hidden)
        self.dec_out = nn.Linear(hidden, in_dim)

    def encode(self, x):
        h = F.relu(self.enc_hidden(x))
        return self.enc_mu(h), self.enc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std                      # the reparameterisation trick

    def decode(self, z):
        h = F.relu(self.dec_hidden(z))
        return self.dec_out(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(x_hat, x, mu, logvar, beta=0.01):
    """Reconstruction MSE + beta * KL[N(mu,sigma^2) || N(0,1)] (closed form)."""
    reconstruction = F.mse_loss(x_hat, x, reduction="mean")
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean()
    return reconstruction + beta * kl, reconstruction, kl
''')]

C += [code(r'''
torch.manual_seed(SEED)
vae = VAE()
optimiser = torch.optim.Adam(vae.parameters(), lr=1e-3)

recon_losses, kl_losses = [], []
for epoch in range(300):
    vae.train()
    optimiser.zero_grad()
    x_hat, mu, logvar = vae(Xt)
    loss, recon, kl = vae_loss(x_hat, Xt, mu, logvar, beta=0.01)
    loss.backward()
    optimiser.step()
    recon_losses.append(recon.item())
    kl_losses.append(kl.item())

print(f"reconstruction loss: {recon_losses[0]:.3f} -> {recon_losses[-1]:.3f}")
print(f"KL term:             {kl_losses[0]:.3f} -> {kl_losses[-1]:.3f}")
''')]

C += [code(r'''
vae.eval()
with torch.no_grad():
    mu, _ = vae.encode(Xt)
Z_vae = mu.numpy()                          # use the mean, not a random sample, to visualise

Z_pca = PCA(n_components=2, random_state=0).fit_transform(Xstd)

corr_vae = [np.corrcoef(Z_vae[:, i], y)[0, 1] for i in range(2)]
corr_pca = [np.corrcoef(Z_pca[:, i], y)[0, 1] for i in range(2)]
print("VAE latent |corr| with log S:", np.round(np.abs(corr_vae), 3))
print("PCA latent |corr| with log S:", np.round(np.abs(corr_pca), 3))

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, Z, title in [(axes[0], Z_vae, "VAE latent space (learned)"),
                     (axes[1], Z_pca, "PCA latent space (linear)")]:
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap="coolwarm_r", s=8)
    ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2"); ax.set_title(title)
fig.colorbar(sc, ax=axes, label="measured log S", shrink=0.8)
fig.savefig(FIGDIR / "vae_vs_pca_latent.png", dpi=200)
plt.show()
''')]

C += [md(r'''
> **Common errors — Section 1**
> - Visualising a **random sample** of $z$ (via `reparameterize`) instead of
>   the encoder's **mean** $\mu$ — the mean is the natural "best estimate" of
>   where a point sits in latent space; a random sample adds noise that
>   makes the plot needlessly harder to read (still a valid latent point,
>   just a noisier one to visualise from).
> - Setting `beta` too high early in learning — an overly strong KL term can
>   collapse the latent space to match the prior *before* the decoder has
>   learned anything useful, a well-known VAE failure mode called "posterior
>   collapse".
''')]

C += checkpoint(
    "Section 1",
    check=r'''
assert recon_losses[-1] < recon_losses[0] * 0.5      # reconstruction improved substantially
assert Z_vae.shape == (1117, 2)
assert max(abs(c) for c in corr_vae) > 0.3            # the unlabelled latent space still tracks y
assert (FIGDIR / "vae_vs_pca_latent.png").is_file()
print("Section 1 OK")
''',
    expected="Prints `Section 1 OK`. Reconstruction loss more than halves "
    "over training; at least one VAE latent dimension correlates with log S "
    "at |r| > 0.3, despite never seeing that label.",
    questions=r'''
1. The VAE loss never mentions `y` (solubility). Why does its latent space
   end up correlating with it anyway?
2. What would `beta = 0` reduce this model to?
''',
    answers=r'''
1. Solubility is not independent of the very descriptors the VAE is trying
   to reconstruct (MolLogP, TPSA, etc. are known — Week 03 — to correlate
   with log S); a latent space that efficiently captures the *variance* in
   those descriptors will, incidentally, also capture the variance related
   to solubility, purely as a side-effect of compressing correlated inputs.
2. A plain (non-variational) autoencoder — with no KL pressure at all, the
   encoder is free to spread points however is most convenient for
   reconstruction, with no guarantee the latent space resembles $\mathcal
   N(0,1)$ anywhere, which is precisely what Section 2 needs to sample from.
''',
)

# ==========================================================================
# 2. Sampling the prior — and a real failure mode
# ==========================================================================
C += [md(r'''
---
## 2. Sampling the prior — and a real failure mode

The entire *point* of the KL term is to make $z\sim\mathcal N(0,1)$ a
reasonable stand-in for "a real encoded point" — so that we can **sample**
new points directly from the prior (no real input needed) and decode them
into new, hopefully plausible, data.

**What to look for:** decoding random prior samples produces vectors that,
after inverting the standardisation, land in a **chemically impossible**
range for at least one descriptor (e.g. a negative molecular weight, or a
negative TPSA) — a concrete, checkable illustration of a real generative-model
failure mode: nothing in this architecture *constrains* the decoder's output
to the space of valid molecular descriptors.
''')]

C += [code(r'''
torch.manual_seed(1)
n_samples = 500
z_samples = torch.randn(n_samples, 2)          # sample directly from the prior N(0,1)
with torch.no_grad():
    decoded_std = vae.decode(z_samples).numpy()
decoded_raw = scaler.inverse_transform(decoded_std)     # back to real descriptor units

molwt_col = desc_names.index("MolWt")
tpsa_col = desc_names.index("TPSA")
n_negative_molwt = int((decoded_raw[:, molwt_col] < 0).sum())
n_negative_tpsa = int((decoded_raw[:, tpsa_col] < 0).sum())

print(f"decoded {n_samples} prior samples")
print(f"samples with a NEGATIVE (impossible) molecular weight: {n_negative_molwt}")
print(f"samples with a NEGATIVE (impossible) TPSA:             {n_negative_tpsa}")
print("example decoded (raw units):", np.round(decoded_raw[0], 2))
''')]

C += [md(r'''
> **Common errors — Section 2**
> - Concluding "this VAE is broken" — it is doing exactly what it was
>   trained to do (minimise reconstruction error on the training
>   distribution); nothing in the architecture or loss *ever* told it that
>   molecular weight must be positive. This is a property of the modelling
>   choice, not a bug in the code.
> - Forgetting `scaler.inverse_transform` before judging plausibility —
>   standardised values are *supposed* to range roughly $-3$ to $3$;
>   checking for "negative" in standardised units would be meaningless.
''')]

C += checkpoint(
    "Section 2",
    check=r'''
assert decoded_raw.shape == (n_samples, 7)
assert n_negative_molwt > 0 or n_negative_tpsa > 0    # the failure mode is real, not hypothetical
print("Section 2 OK")
''',
    expected="Prints `Section 2 OK`. At least some decoded prior samples "
    "have a chemically impossible negative value.",
    questions=r'''
1. Propose one architectural change that would rule out a negative decoded
   molecular weight by construction (not just by hoping the loss discourages
   it).
2. Why might this failure mode be *less* obvious, but still present, in a
   VAE that decodes directly to SMILES characters instead of descriptors?
''',
    answers=r'''
1. Apply a positivity-enforcing activation (e.g. `softplus` or `exp`) to the
   decoder's output for any descriptor that is physically non-negative,
   instead of leaving it as an unconstrained linear output — this makes the
   constraint architectural, not merely statistical.
2. A SMILES-decoding VAE can output a syntactically-invalid **string**
   (unbalanced brackets, an impossible valence) rather than an obviously
   "wrong" number — the failure is just as real, but only visible once you
   try to *parse* the output (exactly what Week 09B's validity check does
   for its generative RNN), not by eyeballing a single out-of-range value.
''',
)

# ==========================================================================
# 3. Flows and diffusion (survey)
# ==========================================================================
C += [md(r'''
---
## 3. Normalising flows and diffusion models — survey

*(Deliberately survey-level — `syllabus.md` scopes flows as a "survey slide"
this week; no implementation.)*

A VAE compresses data into a *lower-dimensional* latent space and accepts an
approximate, lower-bound loss (the ELBO). Two alternative generative
families avoid parts of that trade-off:

* **Normalising flows** apply a sequence of **invertible**,
  volume-tracking transformations to a simple base distribution (e.g.
  $\mathcal N(0,1)$), so the exact data likelihood is computable (no ELBO
  approximation) — at the cost of requiring every layer to be exactly
  invertible, which restricts architecture choices.
* **Diffusion models** learn to reverse a gradual *noising* process:
  starting from pure noise, a trained network repeatedly denoises step by
  step until a sample emerges. They currently produce some of the highest-
  quality generated images and, increasingly, molecular conformations, at
  the cost of needing many sequential denoising steps to generate one
  sample (much slower than a VAE's single decoder pass).

Both are active areas of chemical machine learning (e.g. flow-based and
diffusion-based 3D conformer generators), but neither is implemented in this
course — Week 09B's hands-on generative model is the VAE/RNN family instead.
''')]

# ==========================================================================
# 4. Explaining predictions
# ==========================================================================
C += [md(r'''
---
## 4. Explaining predictions: gradient saliency and integrated gradients

A trained model can predict well without telling you **why**. The simplest
explanation method, **gradient saliency**, asks: "if I nudge each input
feature slightly, how much does the prediction change?" —
$\partial\hat y/\partial x_i$, evaluated at one specific input.

We first (briefly) train a small MLP regressor on the standardised
descriptors (Week 05's architecture), then compute saliency for one molecule.

**What to look for:** a bar chart of 7 gradient values, one per descriptor,
for a single molecule — the largest-magnitude bar is the descriptor the
model is *locally* most sensitive to, for **this** molecule specifically.
''')]

C += [code(r'''
class MLP(nn.Module):
    def __init__(self, in_dim=7, hidden=64):
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)

torch.manual_seed(0)
xai_model = MLP()
opt = torch.optim.Adam(xai_model.parameters(), lr=1e-3)
for _ in range(300):
    opt.zero_grad()
    loss = F.mse_loss(xai_model(Xt), yt)
    loss.backward()
    opt.step()
print("training MSE after 300 epochs:", round(loss.item(), 3))
''')]

C += [code(r'''
xai_model.eval()
example_idx = 0
x_example = Xt[example_idx:example_idx + 1].clone().requires_grad_(True)
pred = xai_model(x_example)
pred.backward()
saliency = x_example.grad[0].detach().numpy()

print(f"molecule: {esol['Compound ID'].iloc[example_idx]}  (predicted log S = {pred.item():.2f})")
for name, g in sorted(zip(desc_names, saliency), key=lambda kv: -abs(kv[1])):
    print(f"  {name:20s} {g:+.4f}")

fig, ax = plt.subplots(figsize=(5.5, 3.5))
order = np.argsort(np.abs(saliency))
ax.barh(np.array(desc_names)[order], saliency[order], color="darkorange")
ax.set_xlabel(r"$\partial \hat y / \partial x_i$"); ax.set_title("Gradient saliency (one molecule)")
fig.tight_layout(); fig.savefig(FIGDIR / "saliency_bar.png", dpi=200)
plt.show()
''')]

C += [md(r'''
Gradient saliency has a well-known weakness: it only looks at the *local*
slope at the input, which can be misleading for a non-linear model (e.g. in
a saturated region where the true effect of a feature is large but the local
gradient is near zero). **Integrated gradients** fixes this by averaging
gradients along a straight-line path from a **baseline** (here, the
all-zero, i.e. average-descriptor, point) to the actual input:

$$\mathrm{IG}_i(x) = (x_i - x_i') \int_0^1 \frac{\partial \hat f(x' + t(x-x'))}{\partial x_i}\, dt,$$

approximated by a Riemann sum over `steps` points. Integrated gradients
satisfies **completeness**: the attributions sum *exactly* (up to
integration error) to $\hat f(x) - \hat f(x')$ — a property plain saliency
does not have, and one we can check numerically.

**What to look for:** `IG.sum()` closely matches `f(x) - f(baseline)`.
''')]

C += [code(r'''
def integrated_gradients(model, x, baseline=None, steps=50):
    """Integrated gradients attribution for one input x (shape (1, D))."""
    if baseline is None:
        baseline = torch.zeros_like(x)
    alphas = torch.linspace(0, 1, steps).view(-1, 1)
    interpolated = (baseline + alphas * (x - baseline)).requires_grad_(True)
    preds = model(interpolated)
    grads = torch.autograd.grad(preds.sum(), interpolated)[0]
    avg_grad = grads.mean(dim=0)                    # Riemann-sum approximation of the integral
    return (x[0] - baseline[0]) * avg_grad

baseline = torch.zeros_like(x_example)
ig = integrated_gradients(xai_model, x_example.detach(), baseline).numpy()
f_x = xai_model(x_example.detach()).item()
f_baseline = xai_model(baseline).item()

print(f"IG attribution sum: {ig.sum():.4f}")
print(f"f(x) - f(baseline): {f_x - f_baseline:.4f}")

fig, ax = plt.subplots(figsize=(5.5, 3.5))
order = np.argsort(np.abs(ig))
ax.barh(np.array(desc_names)[order], ig[order], color="seagreen")
ax.set_xlabel("integrated gradient attribution"); ax.set_title("Integrated gradients (same molecule)")
fig.tight_layout(); fig.savefig(FIGDIR / "integrated_gradients_bar.png", dpi=200)
plt.show()
''')]

C += [md(r'''
> **Common errors — Section 4**
> - Choosing a baseline that is not chemically/statistically meaningful — the
>   all-zero point in **standardised** units corresponds to the *average*
>   molecule (mean descriptor values), a sensible reference; the all-zero
>   point in **raw** units (0 g/mol!) would not be.
> - Using too few `steps` in the Riemann sum — with very few interpolation
>   points, the completeness check can be noticeably off even though the
>   method is correct in principle; 50 is a reasonable default here.
> - Treating gradient saliency and integrated gradients as if they must
>   agree — they can rank features differently on the same input (they use
>   different amounts of the function's shape, not just its local slope).
''')]

C += checkpoint(
    "Section 4",
    check=r'''
assert saliency.shape == (7,)
assert abs(ig.sum() - (f_x - f_baseline)) < 0.15
assert (FIGDIR / "saliency_bar.png").is_file()
assert (FIGDIR / "integrated_gradients_bar.png").is_file()
print("Section 4 OK")
''',
    expected="Prints `Section 4 OK`. Integrated gradients' attribution sum "
    "matches `f(x) - f(baseline)` to within 0.15 log-S units.",
    questions=r'''
1. In your printed ranking, did gradient saliency and integrated gradients
   agree on the *most* important descriptor for this molecule? Does that
   matter?
2. Why is completeness a useful *sanity check* for an attribution method,
   even though it does not by itself prove the attributions are
   "chemically correct"?
''',
    answers=r'''
1. They often broadly agree but need not exactly agree — integrated
   gradients accounts for the function's curvature along the whole path to
   the baseline, while plain saliency only sees the slope exactly at $x$;
   for a strongly non-linear model these can genuinely differ, and neither is
   simply "wrong" if they do.
2. Completeness confirms the method is *implemented consistently* with the
   model's actual prediction (the numbers add up), which catches bugs and
   validates the Riemann-sum approximation — but it says nothing about
   whether the *reasons* the model relies on those features are chemically
   sound, only that the arithmetic behind the explanation is self-consistent.
''',
)

# ==========================================================================
# 5. SHAP, LIME, counterfactuals (survey)
# ==========================================================================
C += [md(r'''
---
## 5. SHAP, LIME, counterfactual explanations — survey

*(Survey-level, following dmol.pub's own chapter emphasis; not implemented.)*

* **SHAP** (SHapley Additive exPlanations) assigns each feature a
  contribution based on **Shapley values** from cooperative game theory:
  averaging a feature's marginal effect across every possible subset of the
  other features it could be combined with. This gives attributions with
  strong theoretical guarantees (they sum exactly to the prediction, like
  integrated gradients, but via a different, permutation-based derivation)
  at a computational cost that grows quickly with the number of features.
* **LIME** (Local Interpretable Model-agnostic Explanations) fits a simple,
  interpretable model (e.g. linear) **locally**, around one prediction, by
  perturbing the input and observing how the (possibly very complex) real
  model's output changes nearby — model-agnostic, since it never looks
  inside the model at all, only at input/output pairs.
* **Counterfactual explanations** ask a different question entirely: *not*
  "why did the model predict this?" but "what is the smallest change to this
  molecule that would flip the prediction?" — directly actionable for a
  chemist deciding what to modify next. dmol.pub's own example flips a
  hemolytic-peptide classifier's prediction via a single amino-acid
  substitution; the molecular analogue (e.g. the `exmol`/STONED approach)
  searches nearby chemical space for a structurally similar molecule with a
  different predicted class or property.
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
### Exercise 1 — the effect of beta *(easy, ~12 min)*

Write `train_vae(beta, epochs=300)` (Section 1's training loop, parameterised
by `beta`) returning the final `(reconstruction_loss, kl_loss)`. Compare
`beta=0.001` to `beta=0.5`.

<details><summary>Show hint</summary>

Copy Section 1's loop body into the function; call `vae_loss(..., beta=beta)`.
Higher `beta` should trade reconstruction quality for a KL term closer to 0.
</details>
''',
    solution=r'''
def train_vae(beta, epochs=300, seed=SEED):
    torch.manual_seed(seed)
    model = VAE()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        opt.zero_grad()
        x_hat, mu, logvar = model(Xt)
        loss, recon, kl = vae_loss(x_hat, Xt, mu, logvar, beta=beta)
        loss.backward()
        opt.step()
    return recon.item(), kl.item()

recon_small_beta, kl_small_beta = train_vae(beta=0.001)
recon_large_beta, kl_large_beta = train_vae(beta=0.5)
print(f"beta=0.001: recon={recon_small_beta:.3f}  kl={kl_small_beta:.3f}")
print(f"beta=0.5:   recon={recon_large_beta:.3f}  kl={kl_large_beta:.3f}")
''',
    scaffold=r'''
def train_vae(beta, epochs=300, seed=SEED):
    # YOUR CODE HERE
    ...

recon_small_beta, kl_small_beta = ...  # YOUR CODE HERE: train_vae(beta=0.001)
recon_large_beta, kl_large_beta = ...  # YOUR CODE HERE: train_vae(beta=0.5)
''',
    check=r'''
assert kl_large_beta < kl_small_beta          # bigger beta pulls the latent space towards the prior
assert recon_large_beta > recon_small_beta * 0.9   # generally at some cost to reconstruction
print("Exercise 1 OK")
''',
)

C += exercise(
    prompt=r'''
### Exercise 2 — a validity filter for decoded samples *(medium, ~15 min)*

Write `plausible_mask(decoded_raw, desc_names)` returning a boolean array
that is `True` only for rows where `MolWt > 0`, `TPSA >= 0`,
`NumHDonors >= 0`, `NumHAcceptors >= 0` and `NumRotatableBonds >= 0` (basic
non-negativity constraints real descriptors must satisfy). Apply it to
Section 2's `decoded_raw`.

<details><summary>Show hint</summary>

Index columns by `desc_names.index(...)`; combine conditions with `&`.
</details>
''',
    solution=r'''
def plausible_mask(decoded_raw, desc_names):
    """True where every non-negative-by-definition descriptor is >= 0 (MolWt > 0)."""
    i = lambda n: desc_names.index(n)
    mask = (
        (decoded_raw[:, i("MolWt")] > 0)
        & (decoded_raw[:, i("TPSA")] >= 0)
        & (decoded_raw[:, i("NumHDonors")] >= 0)
        & (decoded_raw[:, i("NumHAcceptors")] >= 0)
        & (decoded_raw[:, i("NumRotatableBonds")] >= 0)
    )
    return mask

mask = plausible_mask(decoded_raw, desc_names)
print(f"{mask.sum()} / {len(mask)} decoded samples pass the plausibility filter")
''',
    scaffold=r'''
def plausible_mask(decoded_raw, desc_names):
    """True where every non-negative-by-definition descriptor is >= 0 (MolWt > 0)."""
    # YOUR CODE HERE
    ...

mask = ...  # YOUR CODE HERE: plausible_mask(decoded_raw, desc_names)
''',
    check=r'''
assert mask.dtype == bool
assert mask.shape == (500,)
assert mask.sum() < 500        # Section 2 showed at least one implausible sample exists
print("Exercise 2 OK")
''',
)

C += exercise(
    prompt=r'''
### Exercise 3 — integrated gradients for a different molecule *(medium, ~15 min)*

Run Section 4's `integrated_gradients` for **molecule index 10** instead of
0, and write `top_feature(attributions, names)` returning the name of the
largest-|attribution| descriptor.

<details><summary>Show hint</summary>

Reuse `integrated_gradients(xai_model, Xt[10:11], torch.zeros_like(Xt[10:11]))`.
`names[np.argmax(np.abs(attributions))]`.
</details>
''',
    solution=r'''
def top_feature(attributions, names):
    """Name of the descriptor with the largest |attribution|."""
    return names[int(np.argmax(np.abs(attributions)))]

x10 = Xt[10:11]
ig_10 = integrated_gradients(xai_model, x10, torch.zeros_like(x10)).numpy()
print("IG for molecule 10:", dict(zip(desc_names, np.round(ig_10, 3))))
print("top feature:", top_feature(ig_10, desc_names))
''',
    scaffold=r'''
def top_feature(attributions, names):
    """Name of the descriptor with the largest |attribution|."""
    # YOUR CODE HERE
    ...

x10 = Xt[10:11]
ig_10 = integrated_gradients(xai_model, x10, torch.zeros_like(x10)).numpy()
top_feature_10 = ...  # YOUR CODE HERE: top_feature(ig_10, desc_names)
''',
    check=r'''
assert top_feature(ig_10, desc_names) in desc_names
assert isinstance(ig_10, np.ndarray) and ig_10.shape == (7,)
print("Exercise 3 OK")
''',
)

C += exercise(
    prompt=r'''
### Exercise 4 — how many Riemann-sum steps are enough? *(harder, ~18 min)*

Write `completeness_error(model, x, baseline, steps)` returning
`abs(IG.sum() - (f(x)-f(baseline)))` for a given number of `steps`. Compute
it for `steps` in `[2, 5, 10, 50, 200]` and confirm the error shrinks as
`steps` grows.

<details><summary>Show hint</summary>

Reuse `integrated_gradients` with the `steps` argument; the error should
decrease roughly monotonically (Riemann-sum approximation error).
</details>
''',
    solution=r'''
def completeness_error(model, x, baseline, steps):
    """|IG.sum() - (f(x) - f(baseline))| for a given Riemann-sum resolution."""
    ig = integrated_gradients(model, x, baseline, steps=steps)
    f_x = model(x).item()
    f_base = model(baseline).item()
    return abs(ig.sum().item() - (f_x - f_base))

step_counts = [2, 5, 10, 50, 200]
errors = [completeness_error(xai_model, x_example.detach(), baseline, s) for s in step_counts]
print(list(zip(step_counts, np.round(errors, 4))))
''',
    scaffold=r'''
def completeness_error(model, x, baseline, steps):
    """|IG.sum() - (f(x) - f(baseline))| for a given Riemann-sum resolution."""
    # YOUR CODE HERE
    ...

step_counts = [2, 5, 10, 50, 200]
errors = ...  # YOUR CODE HERE: [completeness_error(xai_model, x_example.detach(), baseline, s) for s in step_counts]
''',
    check=r'''
assert len(errors) == 5
assert errors[-1] <= errors[0] + 1e-6      # more steps should not make the approximation worse
print("Exercise 4 OK")
''',
)

# ==========================================================================
C += [md(r'''
---
## Summary — what you learned

- The **VAE**: encoder/decoder, the reparameterisation trick, and the ELBO
  loss (reconstruction + KL), trained on the ESOL descriptor block.
- A learned VAE latent space can correlate with a label it never saw, but
  **sampling its prior can decode to chemically impossible values** — a real,
  checkable failure mode, not a hypothetical warning.
- **Normalising flows** and **diffusion models**, survey-level: what they
  are, and how they trade off against a VAE's approximate likelihood.
- **Gradient saliency** and **integrated gradients** (with a verified
  completeness property) for explaining one prediction — and that local
  explanations need not agree with each other or with a model's global
  feature importance (Week 03B).
- **SHAP, LIME and counterfactual explanations**, survey-level.

Week 09B builds a hands-on generative model — a character-level RNN that
samples new SMILES strings — and assesses them the way Section 2's failure
mode demands: by actually checking validity.
''')]

C += [md(r'''
## Further reading (course source list only)

- dmol.pub *Variational autoencoders*: <https://dmol.pub/dl/VAE.html>
- dmol.pub *Explaining predictions*: <https://dmol.pub/dl/xai.html>
- dmol.pub *Normalizing flows* (survey): <https://dmol.pub/dl/flows.html>
''')]

C += [md(r'''
## Attribution

Original teaching material for *AI for Chemistry*, adapted and rewritten from:

- **dmol.pub** (A. White) — the VAE architecture, ELBO derivation and
  beta-VAE framing (applied here to the ESOL descriptor block rather than
  the source's own synthetic/polymer examples); the gradient-saliency,
  integrated-gradients, SHAP/LIME/counterfactual survey; the
  normalising-flows/diffusion survey. CC-BY 4.0. <https://dmol.pub>

Continues the ESOL dataset and descriptor block from Weeks 02-03/05-07. No
verbatim text is reproduced from the sources above.
''')]

build(__file__, "week09_a_generative-models-xai", C)
