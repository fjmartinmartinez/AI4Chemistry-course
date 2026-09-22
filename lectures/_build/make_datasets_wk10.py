"""Cache a small QM9 subset for week 10.

Downloads the full QM9 CSV (134k molecules, ~30 MB) from the same DeepChem
mirror used for ESOL/BBBP/Tox21, verifies its checksum, draws a fixed-seed
random subsample, and writes ONLY the subsample (small, committable) to
`sources/datasets/qm9_subset.csv`. The full download is not cached (too large
for this repo's "small curated dataset" convention); re-running this script
re-downloads and re-verifies it, but always produces the identical subsample.

    python lectures/_build/make_datasets_wk10.py
"""
from __future__ import annotations

import hashlib
import io
import pathlib
import urllib.request

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "sources" / "datasets"
DATA.mkdir(parents=True, exist_ok=True)

QM9_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"
QM9_FULL_SHA256 = "3e668f8c34e4bc392a90d417a50a5eed3b64b842a817a633024bdc054c68ccb4"
SUBSET_SIZE = 3000
SEED = 0xC0FFEE

HARTREE_TO_EV = 27.211386245988


def main() -> None:
    dest = DATA / "qm9_subset.csv"
    if dest.is_file():
        print(f"ok    {dest.name} (already cached, skipped)")
        return

    req = urllib.request.Request(QM9_URL, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=120).read()
    got = hashlib.sha256(raw).hexdigest()
    if got != QM9_FULL_SHA256:
        raise SystemExit(
            f"checksum mismatch for qm9.csv:\n  expected {QM9_FULL_SHA256}\n"
            f"  got      {got}"
        )

    full = pd.read_csv(io.BytesIO(raw))
    rng = np.random.default_rng(SEED)
    sub_idx = np.sort(rng.choice(len(full), size=SUBSET_SIZE, replace=False))
    subset = full.iloc[sub_idx][["mol_id", "smiles", "homo", "lumo", "gap", "h298"]].copy()
    subset["gap_eV"] = subset["gap"] * HARTREE_TO_EV     # Hartree -> eV, the usual reporting unit
    subset = subset.reset_index(drop=True)

    subset.to_csv(dest, index=False)
    print(f"wrote {dest.name}  ({len(subset)} of {len(full)} QM9 molecules, "
          f"seed {SEED:#x})  sha256(full qm9.csv)={got[:12]}")


if __name__ == "__main__":
    main()
