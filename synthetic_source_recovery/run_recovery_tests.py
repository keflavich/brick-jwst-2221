"""
Run a parameter sweep of the saturated-star PSF fitter on synthetic images.

For each (filter, magnitude) combination we test:
  - mask_buffer : [0, 1, 2, 3, 5]
  - bkg_annulus : [(10,20), (15,30), (20,40), (30,60)]
  - fit_shape   : [81, 121, 161, 201]

The default in the production code is mask_buffer=1, LocalBackground(15,30),
fit_shape=81.  We vary one parameter at a time (and also run all combinations
at the default background annulus) to identify where bias originates.

Outputs:
  results/recovery_results.ecsv  — full parameter sweep table
  results/best_params.ecsv       — recommended parameters per filter × mag bin

Usage:
    python run_recovery_tests.py          # full sweep, ~20-30 min
    python run_recovery_tests.py --quick  # 3 mags × defaults only, ~2 min
"""

import os
import sys
import itertools
import numpy as np
from astropy.table import Table, vstack
import argparse

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_psf, make_synthetic_image, SAT_MAG_VEGA
from fit_saturated import fit_saturated_source

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Experiment configuration
# ──────────────────────────────────────────────────────────────────────────────

FILTERS = {
    "F200W": {"fwhm_pix": 2.14, "detector": "nrca1"},
    "F182M": {"fwhm_pix": 1.99, "detector": "nrca1"},
    "F212N": {"fwhm_pix": 2.34, "detector": "nrca1"},
}

# Only test magnitudes where the star is actually saturated
# (2 mag brighter than SAT_MAG_VEGA down to ~4 mag brighter = deeply saturated)
MAG_RANGES = {
    "F200W": [10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0],
    "F182M": [10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5],
    "F212N": [ 7.0,  7.5,  8.0,  8.5,  9.0,  9.5, 10.0],
}

# Parameter grid
MASK_BUFFERS  = [0, 1, 2, 3, 5]
BKG_ANNULI    = [(15, 30), (25, 50)]
FIT_SHAPES    = [81, 161]

# Default (production) values
DEFAULT_MASK  = 1
DEFAULT_BKG   = (15, 30)
DEFAULT_SHAPE = 81

# N realisations per (filter, mag, params) — more = better statistics
N_REALISATIONS = 3

# The 5 key combos for the targeted sweep (covers all scientific hypotheses):
#   A: old default          mb=1, bkg=(15,30), fs=81, adaptive=False
#   B: new fixed baseline   mb=2, bkg=(15,30), fs=81, adaptive=False
#   C: wide bkg             mb=2, bkg=(25,50), fs=81, adaptive=False
#   D: adaptive (min=2)     mb=2, bkg=(15,30), fs=81, adaptive=True
#   E: adaptive+wide bkg    mb=2, bkg=(25,50), fs=81, adaptive=True
# Plus mb sweep at default bkg (for the bias-vs-buffer plot):
#   F-J: mb=0,1,2,3,5 at default bkg/shape
TARGETED_COMBOS = [
    dict(mb=1, bi=15, bo=30, fs=81, adaptive=False),   # A: old default
    dict(mb=2, bi=15, bo=30, fs=81, adaptive=False),   # B: new fixed
    dict(mb=2, bi=25, bo=50, fs=81, adaptive=False),   # C: wide bkg
    dict(mb=2, bi=15, bo=30, fs=81, adaptive=True),    # D: adaptive
    dict(mb=2, bi=25, bo=50, fs=81, adaptive=True),    # E: adaptive+wide bkg
    dict(mb=0, bi=15, bo=30, fs=81, adaptive=False),   # mb sweep
    dict(mb=3, bi=15, bo=30, fs=81, adaptive=False),
    dict(mb=5, bi=15, bo=30, fs=81, adaptive=False),
    dict(mb=1, bi=25, bo=50, fs=81, adaptive=False),   # bkg comparison at old mb
    dict(mb=2, bi=15, bo=30, fs=161, adaptive=False),  # fit_shape test
    dict(mb=2, bi=15, bo=30, fs=161, adaptive=True),
]


def run_one(filtername, mag, psf_model, fwhm_pix,
            mask_buffer, bkg_inner, bkg_outer, fit_shape,
            adaptive=False,
            n_real=N_REALISATIONS):
    """Run N_REALISATIONS of the fitter and return median recovery statistics."""
    flux_ratios = []
    qfits = []
    snrs = []
    eff_buffers = []

    for seed in range(n_real):
        rng = np.random.default_rng(seed * 1000 + int(mag * 10))
        sci, dq, true_flux, sat_rad = make_synthetic_image(
            mag, filtername, psf_model, imsize=300, rng=rng
        )
        res = fit_saturated_source(
            sci, dq, psf_model,
            mask_buffer=mask_buffer,
            adaptive=adaptive,
            bkg_inner=bkg_inner,
            bkg_outer=bkg_outer,
            fit_shape=fit_shape,
            fwhm_pix=fwhm_pix,
        )
        if res["success"]:
            flux_ratios.append(res["flux_fit"] / true_flux)
            qfits.append(res["qfit"])
            snrs.append(res["snr"])
            eff_buffers.append(res["effective_mask_buffer"])

    n_ok = len(flux_ratios)
    if n_ok == 0:
        return None

    return {
        "filtername":    filtername,
        "mag_vega":      mag,
        "sat_mag":       SAT_MAG_VEGA[filtername],
        "delta_mag":     SAT_MAG_VEGA[filtername] - mag,
        "mask_buffer":   mask_buffer,
        "adaptive":      adaptive,
        "eff_buffer_med": float(np.median(eff_buffers)) if eff_buffers else mask_buffer,
        "bkg_inner":     bkg_inner,
        "bkg_outer":     bkg_outer,
        "fit_shape":     fit_shape,
        "n_ok":          n_ok,
        "flux_ratio_med":   float(np.median(flux_ratios)),
        "flux_ratio_std":   float(np.std(flux_ratios)),
        "flux_ratio_bias":  float(np.median(flux_ratios)) - 1.0,
        "flux_ratio_16":    float(np.percentile(flux_ratios, 16)),
        "flux_ratio_84":    float(np.percentile(flux_ratios, 84)),
        "qfit_med":      float(np.nanmedian(qfits)),
        "snr_med":       float(np.nanmedian(snrs)),
        "is_default":    (not adaptive and mask_buffer == DEFAULT_MASK and
                          (bkg_inner, bkg_outer) == DEFAULT_BKG and
                          fit_shape == DEFAULT_SHAPE),
        "is_adaptive":   adaptive,
    }


def main(quick=False):
    all_rows = []

    for filtername, finfo in FILTERS.items():
        print(f"\n{'='*60}")
        print(f"Filter: {filtername}  (FWHM={finfo['fwhm_pix']:.2f} pix)")
        print(f"{'='*60}")

        psf_model = load_psf(filtername, detector=finfo["detector"])
        fwhm_pix  = finfo["fwhm_pix"]
        mags      = MAG_RANGES[filtername]

        if quick:
            mags = mags[::3][:3]   # 3 representative magnitudes

        for mag in mags:
            print(f"\n  mag={mag:.1f}  (Δmag from sat limit: "
                  f"{SAT_MAG_VEGA[filtername]-mag:+.1f})")

            if quick:
                param_combos = [
                    dict(mb=mb, bi=DEFAULT_BKG[0], bo=DEFAULT_BKG[1],
                         fs=DEFAULT_SHAPE, adaptive=False)
                    for mb in MASK_BUFFERS
                ]
            else:
                param_combos = TARGETED_COMBOS

            for p in param_combos:
                mb, bi, bo, fs, adap = (
                    p["mb"], p["bi"], p["bo"], p["fs"], p["adaptive"]
                )
                tag = f"mb={mb} bkg=({bi},{bo}) fs={fs} adap={adap}"
                print(f"    {tag} ... ", end="", flush=True)
                row = run_one(filtername, mag, psf_model, fwhm_pix,
                              mask_buffer=mb, bkg_inner=bi, bkg_outer=bo,
                              fit_shape=fs, adaptive=adap)
                if row:
                    bias_pct = row["flux_ratio_bias"] * 100
                    eff = row["eff_buffer_med"]
                    print(f"bias={bias_pct:+.1f}%  eff_mb={eff:.0f}  qfit={row['qfit_med']:.3f}")
                    all_rows.append(row)
                else:
                    print("FAILED")

    if not all_rows:
        print("No successful fits — check PSF model and image generation.")
        return

    tbl = Table(rows=all_rows)
    outfile = os.path.join(RESULTS_DIR, "recovery_results.ecsv")
    tbl.write(outfile, overwrite=True, format="ascii.ecsv")
    print(f"\nSaved {len(tbl)} rows → {outfile}")

    # ── Quick summary ────────────────────────────────────────────────────────
    print("\n── Default-parameter bias per filter / mag bin ──")
    defaults = tbl[tbl["is_default"]]
    for row in defaults:
        print(f"  {row['filtername']:6s}  mag={row['mag_vega']:.1f}"
              f"  Δmag={row['delta_mag']:+.1f}"
              f"  flux_ratio={row['flux_ratio_med']:.3f}"
              f"  (bias={row['flux_ratio_bias']*100:+.1f}%)")

    # ── Best parameters per (filter, mag) ────────────────────────────────────
    best_rows = []
    for filtername in FILTERS:
        fmask = tbl["filtername"] == filtername
        for mag in sorted(set(tbl["mag_vega"][fmask])):
            mmask = fmask & (tbl["mag_vega"] == mag)
            sub = tbl[mmask]
            # best = minimum |bias|
            best_idx = np.argmin(np.abs(sub["flux_ratio_bias"]))
            best = sub[best_idx]
            best_rows.append({k: best[k] for k in best.colnames})

    if best_rows:
        best_tbl = Table(rows=best_rows)
        best_file = os.path.join(RESULTS_DIR, "best_params.ecsv")
        best_tbl.write(best_file, overwrite=True, format="ascii.ecsv")
        print(f"\nBest-parameter table → {best_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Fast run: 3 mags × mask_buffer sweep only")
    args = parser.parse_args()
    main(quick=args.quick)
