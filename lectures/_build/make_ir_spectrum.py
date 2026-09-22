"""Generate a SYNTHETIC infrared spectrum for the Week 01B NumPy workshop.

The spectrum is a sum of Lorentzian bands at chemically plausible group
frequencies (documented below). It is *not* measured data — it exists so the
workshop has a realistic 1-D array to slice, mask and plot without a network
call. Deterministic (no randomness).

    python lectures/_build/make_ir_spectrum.py
"""
import pathlib
import numpy as np

DATA = pathlib.Path(__file__).resolve().parents[2] / "sources" / "datasets"
DATA.mkdir(parents=True, exist_ok=True)

# wavenumber grid: 400-4000 cm^-1 at 2 cm^-1 resolution (1801 points)
wavenumber = np.arange(400.0, 4001.0, 2.0)

# (centre cm^-1, peak height, half-width at half-maximum cm^-1, assignment)
BANDS = [
    (3350.0, 0.55, 90.0, "O-H stretch (broad, H-bonded)"),
    (2950.0, 0.80, 35.0, "C-H stretch"),
    (1714.0, 1.00, 18.0, "C=O stretch (carbonyl, ~1715)"),
    (1450.0, 0.45, 25.0, "CH2/CH3 bend"),
    (1250.0, 0.40, 20.0, "C-O stretch"),
    (1050.0, 0.60, 22.0, "C-O / C-C skeletal"),
]


def lorentzian(x, x0, height, hwhm):
    """Unit-height Lorentzian scaled to `height`."""
    return height * hwhm**2 / ((x - x0) ** 2 + hwhm**2)


absorbance = np.zeros_like(wavenumber)
for x0, h, w, _ in BANDS:
    absorbance += lorentzian(wavenumber, x0, h, w)

# small constant baseline so nothing is exactly zero
absorbance += 0.02

header = (
    "synthetic IR spectrum (sum of Lorentzians); NOT measured data\n"
    "bands: " + "; ".join(f"{c:.0f}:{a}" for c, _, _, a in BANDS) + "\n"
    "wavenumber_cm-1,absorbance"
)
out = DATA / "ir_spectrum_synthetic.csv"
np.savetxt(out, np.column_stack([wavenumber, absorbance]),
           delimiter=",", header=header, fmt="%.4f")
print(f"wrote {out}  ({wavenumber.size} points)")
