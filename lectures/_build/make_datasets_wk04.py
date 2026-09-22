"""Cache the classification datasets used by week 04.

    python lectures/_build/make_datasets_wk04.py

Downloads are checksum-verified; a file already present with the right hash is
left untouched. Both are standard MoleculeNet tasks, mirrored (uncompressed)
from the DeepChem S3 bucket, same source as `esol_delaney.csv` (Week 02).
"""
from __future__ import annotations

import hashlib
import pathlib
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "sources" / "datasets"
DATA.mkdir(parents=True, exist_ok=True)

FILES = {
    # blood-brain-barrier penetration (binary p_np), Week 04A
    "BBBP.csv": (
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
        "d07a38487aeac5cee5508413e468043ef3097451d2a112701c2d60be9ec6b662",
    ),
    # Tox21 12-task toxicity screen (we use one task, SR-p53), Week 04B
    "tox21.csv.gz": (
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
        "45d09792492ce049039dd24aa27b07fc79ce20c573187d4d90bcd178c0c0d360",
    ),
}


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fetch(name: str, url: str, expected_sha256: str) -> None:
    dest = DATA / name
    if dest.exists() and _sha256(dest.read_bytes()) == expected_sha256:
        print(f"ok    {name} (checksum matches, skipped)")
        return
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=60).read()
    got = _sha256(raw)
    if got != expected_sha256:
        raise SystemExit(
            f"checksum mismatch for {name}:\n  expected {expected_sha256}\n"
            f"  got      {got}"
        )
    dest.write_bytes(raw)
    print(f"wrote {name}  sha256={got}")


if __name__ == "__main__":
    for fname, (url, sha) in FILES.items():
        fetch(fname, url, sha)
