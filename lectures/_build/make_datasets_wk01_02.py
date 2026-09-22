"""Create / cache the datasets used by weeks 01-02.

Run once inside the ``ai4chem`` environment:

    python lectures/_build/make_datasets_wk01_02.py

Writes to ``sources/datasets/``. Downloads are checksum-verified; if a file
already exists with the right hash it is left untouched. Geometry files are
generated deterministically with RDKit (fixed seed) so they are reproducible
rather than "invented".
"""
from __future__ import annotations

import hashlib
import pathlib
import urllib.request

from rdkit import Chem
from rdkit.Chem import AllChem

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "sources" / "datasets"
DATA.mkdir(parents=True, exist_ok=True)

SEED = 0xC0FFEE  # fixed everywhere in the course


# --------------------------------------------------------------------------
# 1. Downloaded dataset: ESOL (Delaney) aqueous solubility
# --------------------------------------------------------------------------
ESOL_URL = (
    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/"
    "delaney-processed.csv"
)
ESOL_SHA256 = "8c06a76f0c6487d29ab0f903e6a7a7139f189ab3c1178f159c8be8964602f189"


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fetch_esol() -> None:
    dest = DATA / "esol_delaney.csv"
    if dest.exists() and _sha256(dest.read_bytes()) == ESOL_SHA256:
        print(f"ok    {dest.name} (checksum matches, skipped)")
        return
    req = urllib.request.Request(ESOL_URL, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=60).read()
    got = _sha256(raw)
    if got != ESOL_SHA256:
        raise SystemExit(
            f"checksum mismatch for ESOL download:\n  expected {ESOL_SHA256}\n"
            f"  got      {got}"
        )
    dest.write_bytes(raw)
    print(f"wrote {dest.name}  sha256={got}")


# --------------------------------------------------------------------------
# 2. Generated geometries: small molecules as .xyz (week 01 file parsing)
# --------------------------------------------------------------------------
GEOMETRIES = {
    "water": "O",
    "ethanol": "CCO",
    "benzene": "c1ccccc1",
}


def write_xyz() -> None:
    for name, smiles in GEOMETRIES.items():
        dest = DATA / f"{name}.xyz"
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        params = AllChem.ETKDGv3()
        params.randomSeed = SEED  # deterministic embedding
        AllChem.EmbedMolecule(mol, params)
        AllChem.MMFFOptimizeMolecule(mol)  # tidy the geometry with MMFF94
        conf = mol.GetConformer()
        lines = [str(mol.GetNumAtoms()), f"{name} (RDKit ETKDGv3 + MMFF94, seed {SEED:#x})"]
        for atom in mol.GetAtoms():
            p = conf.GetAtomPosition(atom.GetIdx())
            lines.append(f"{atom.GetSymbol():2s} {p.x:12.6f} {p.y:12.6f} {p.z:12.6f}")
        dest.write_text("\n".join(lines) + "\n")
        print(f"wrote {dest.name}  ({mol.GetNumAtoms()} atoms)")


# --------------------------------------------------------------------------
# 3. Curated table: straight-chain alkane properties (week 01)
# --------------------------------------------------------------------------
# Standard reference physical constants for n-alkanes C1-C10.
# Boiling / melting points at 1 atm, in degrees Celsius.
# Source: CRC Handbook of Chemistry and Physics-type reference values.
# TODO(verify): confirm each value against the current CRC Handbook edition
# before using in an assessed context.
ALKANES = [
    # name, formula, n_carbons, bp_C, mp_C
    ("methane", "CH4", 1, -161.5, -182.5),
    ("ethane", "C2H6", 2, -88.6, -182.8),
    ("propane", "C3H8", 3, -42.1, -187.7),
    ("butane", "C4H10", 4, -0.5, -138.3),
    ("pentane", "C5H12", 5, 36.1, -129.7),
    ("hexane", "C6H14", 6, 68.7, -95.3),
    ("heptane", "C7H16", 7, 98.4, -90.6),
    ("octane", "C8H18", 8, 125.7, -56.8),
    ("nonane", "C9H20", 9, 150.8, -53.5),
    ("decane", "C10H22", 10, 174.1, -29.7),
]


def write_alkanes() -> None:
    dest = DATA / "alkanes.csv"
    rows = ["name,formula,n_carbons,boiling_point_C,melting_point_C"]
    rows += [f"{n},{f},{c},{bp},{mp}" for (n, f, c, bp, mp) in ALKANES]
    dest.write_text("\n".join(rows) + "\n")
    print(f"wrote {dest.name}  ({len(ALKANES)} rows)")


if __name__ == "__main__":
    fetch_esol()
    write_xyz()
    write_alkanes()
    print("\nsources/datasets/ now contains:")
    for p in sorted(DATA.iterdir()):
        print(f"  {p.name:24s} {p.stat().st_size:>8d} bytes  "
              f"sha256={_sha256(p.read_bytes())[:12]}")
