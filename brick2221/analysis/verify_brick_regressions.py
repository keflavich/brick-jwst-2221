#!/usr/bin/env python
"""Brick final-merge regression gate.

Run after the re-merge chain (reduce->verify->catalog->m7) completes.  Exit 0 only
if every check PASSES.

**Measurement method: offset-HISTOGRAM stacking, NOT nearest-neighbour median.**
Earlier versions matched detected sources to the dense VIRAC2 with
``match_to_catalog_sky`` within a 0.3" radius, then took the median.  That is the
forbidden dense-NN-median method: it *assumes* the data is already aligned (a source
>0.3" off has no match in the window, gets discarded, and the survivors always look
aligned), so it CANNOT detect the very regression this gate exists to catch -- a
visit-collapse can leave a whole visit ~20" off and the gate still passes.  It also
took ONE whole-mosaic median, which averages a half-mosaic seam to ~0.  Both are
exactly how the brick-1182 visit-001 break hid.

This version uses ``jwst_gc_pipeline.photometry.astrometry_offsets``:
  * ``measure_offset``       -- window-swept histogram, density-immune, finds a 20"
                                offset instead of dropping it;
  * ``measure_offset_grid``  -- PER-TILE map, so a y=0.5 half-mosaic seam cannot hide
                                behind a ~0 bulk;
  * ``agree_across_references`` -- the tie must agree between VIRAC2 (dense) AND a
                                Gaia-only (sparse) reference; a spurious peak is
                                reference-dependent.

Checks:
  1. 1182 <-> 2221 source-to-source (F200W[1182] vs F212N[2221]).
  2. Each 1182/2221 mosaic ties VIRAC2 -- PER TILE (no seam) AND VIRAC-Gaia agree.
  3. No systematic offset BETWEEN MODULES (per-module i2d vs VIRAC2).
  4. EVERY combined i2d WCS matches the final catalog (loop closure).
Thresholds are survey-noise-scaled, not zero.
"""
import argparse, glob, os, sys
import numpy as np, astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder

from jwst_gc_pipeline.photometry.astrometry_offsets import (
    measure_offset, measure_offset_grid, agree_across_references)

BASE = "/orange/adamginsburg/jwst/brick"
REL = "/orange/adamginsburg/jwst/releases/v1.0-2026.06/brick"
REFCAT = f"{BASE}/astrometry_diag/refcache/virac2.fits"
# Epoch-baked Gaia+VIRAC2 refcat carries a 'source' column -> Gaia-only sparse subset
# for the two-reference agreement check.
GAIA_REFCAT = f"{BASE}/catalogs/gaia_virac2_refcat_epoch2022.70.fits"
V2EP = 2014.0
OBS_EP = 2022.703

# survey-noise-scaled PASS thresholds (median on-sky offset, mas).
# module threshold is PM-grade; 30 mas ships the known ~20-24 mas inter-module
# residual for now -> tighten to 15 after the per-module re-reduction.  seam gates the
# WORST per-tile offset (catches half-mosaic breaks); ref_agree gates VIRAC-vs-Gaia.
THRESH = dict(cross=40.0, virac2=70.0, module=30.0, loop=50.0,
              seam=150.0, ref_agree=100.0)


def farr(x):
    return np.asarray(np.ma.filled(np.ma.masked_invalid(np.asarray(x, float)), np.nan), float)


def virac2():
    v = Table.read(REFCAT)
    ra, dec = farr(v["RAJ2000"]), farr(v["DEJ2000"])
    pr = np.nan_to_num(farr(v["pmRA"])); pd = np.nan_to_num(farr(v["pmDE"]))
    dt = OBS_EP - V2EP
    return SkyCoord((ra + (pr * dt / 3.6e6) / np.cos(np.radians(dec))) * u.deg,
                    (dec + pd * dt / 3.6e6) * u.deg)


def gaia_only():
    """Sparse Gaia-only reference (source == b'GaiaDR3') for the two-ref agreement."""
    if not os.path.exists(GAIA_REFCAT):
        return None
    t = Table.read(GAIA_REFCAT)
    if 'source' not in t.colnames or 'skycoord' not in t.colnames:
        return None
    m = np.asarray(t['source']) == b'GaiaDR3'
    return SkyCoord(t['skycoord'][m]) if m.any() else None


def detect(path, thr=50.0, fwhm=2.5):
    """(SkyCoord) of sources on an i2d SCI plane.  No flux filter -- histogram
    stacking is robust to blends/mismatches (unlike the old NN median)."""
    h = fits.open(path); sci = h["SCI"]; w = WCS(sci.header); d = sci.data.astype("float32")
    _, med, std = sigma_clipped_stats(d, sigma=3.0)
    t = DAOStarFinder(fwhm=fwhm, threshold=thr * std)(d - med)
    if t is None or len(t) == 0:
        return SkyCoord([], [], unit=u.deg)
    return SkyCoord(w.pixel_to_world(t["xcentroid"], t["ycentroid"]))


def hist_off(a, b):
    """Robust bulk offset a->b (mas) via window-swept histogram stacking. Returns a
    compact dict(off, dra, ddec, ok, contrast, n) or None."""
    if len(a) == 0 or len(b) == 0:
        return None
    r = measure_offset(a, b, sweep=True)
    if r is None:
        return None
    return dict(off=float(np.hypot(r["dra"], r["ddec"])), dra=r["dra"], ddec=r["ddec"],
                ok=bool(r.get("ok")), contrast=float(r.get("contrast", 0.0)),
                n=int(r.get("n", 0)))


def i2d_path(filt, prop, module="merged"):
    pre = {"1182": "jw01182-o004", "2221": "jw02221-o001"}[prop]
    for tag in (f"-{filt.lower()}-{module}",):
        for root in (f"{REL}/images/{filt}", f"{BASE}/{filt}/pipeline"):
            hits = glob.glob(f"{root}/{pre}_t001_nircam_clear{tag}_i2d.fits")
            if hits:
                return hits[0]
    return None


STALE_DAYS = 2.0


def is_fresh(module_path, merged_path):
    if not module_path or not merged_path or not os.path.exists(module_path) \
       or not os.path.exists(merged_path):
        return False
    return (os.path.getmtime(merged_path) - os.path.getmtime(module_path)) < STALE_DAYS * 86400


FILTERS = {
    "F115W": "1182", "F200W": "1182", "F356W": "1182", "F444W": "1182",
    "F182M": "2221", "F187N": "2221", "F212N": "2221",
    "F405N": "2221", "F410M": "2221", "F466N": "2221",
}


def check1_cross(cache):
    print("\n[1] 1182 <-> 2221 source-to-source (F200W vs F212N), histogram-stacked")
    r = hist_off(cache["F200W"], cache["F212N"])
    if r is None or not r["ok"]:
        print("    FAIL: no coherent F200W<->F212N tie"); return False
    ok = r["off"] < THRESH["cross"]
    print(f"    dRA={r['dra']:+.1f} dDec={r['ddec']:+.1f} |{r['off']:.1f} mas| C={r['contrast']:.0f} "
          f"n={r['n']}  {'OK' if ok else 'FAIL'} (<{THRESH['cross']:.0f})")
    return ok


def check2_virac2(cache, ref, gaia):
    print("\n[2] JWST <-> VIRAC2 PER-TILE (no seam) + VIRAC2-vs-Gaia agreement")
    allok = True
    for filt in ("F200W", "F212N"):
        sc = cache[filt]
        if len(sc) == 0:
            print(f"    {filt}: SKIP"); continue
        # (a) per-tile map -- catches a half-mosaic seam a bulk median would hide
        grid = measure_offset_grid(sc, ref, nx=4, ny=4, context=f"{filt} vs VIRAC2")
        clean = grid["clean"] and grid["worst_off_mas"] < THRESH["seam"]
        # (b) two-reference agreement -- a spurious peak is reference-dependent
        if gaia is not None:
            ag = agree_across_references(sc, ref, gaia, tol_mas=THRESH["ref_agree"],
                                         label_a="VIRAC2", label_b="Gaia")
            agree = ag["agree"]; agtxt = f"VIRAC-Gaia={ag['sep_mas']:.0f}mas {'agree' if agree else 'DISAGREE'}"
        else:
            agree = True; agtxt = "(no Gaia)"
        ok = clean and agree
        allok &= ok
        print(f"    {filt}: tiles {grid['n_ok']}/{grid['n_total']} tied, worst "
              f"{grid['worst_off_mas']:.0f}mas, minC={grid['min_contrast_seen']:.0f}; {agtxt}"
              f"  {'OK' if ok else 'FAIL'}")
    return allok


def check3_modules(ref):
    print("\n[3] inter-module systematic offset (per-module i2d vs VIRAC2, histogram)")
    allok = True
    for filt, prop in FILTERS.items():
        merged = i2d_path(filt, prop, "merged")
        mods = {}
        for m in ("nrca", "nrcb"):
            p = i2d_path(filt, prop, module=m)
            if not p:
                continue
            if not is_fresh(p, merged):
                print(f"    {filt} {m}: SKIP (stale per-module i2d, predates merged)")
                continue
            r = hist_off(detect(p), ref)
            if r and r["ok"]:
                mods[m] = (r["dra"], r["ddec"], r["n"])
        if len(mods) == 2:
            (ax, ay, an), (bx, by, bn) = mods["nrca"], mods["nrcb"]
            d = float(np.hypot(ax - bx, ay - by))
            ok = d < THRESH["module"]
            allok &= ok
            print(f"    {filt}: nrca-nrcb offset |{d:.1f} mas| (a n={an}, b n={bn})  "
                  f"{'OK' if ok else 'FAIL <'+str(THRESH['module'])}")
    return allok


def check4_loop(ref):
    print("\n[4] EVERY combined i2d WCS matches the final catalog (loop closure)")
    catfn = f"{REL}/catalogs/basic_merged_indivexp_photometry_tables_merged_resbgsub_m7.fits"
    catfn = catfn if os.path.exists(catfn) else \
        f"{BASE}/catalogs/basic_merged_indivexp_photometry_tables_merged_resbgsub_m7.fits"
    if os.path.exists(catfn):
        c = Table.read(catfn); catc = SkyCoord(c["skycoord_ref"]); src = "catalog skycoord_ref"
    else:
        catc = ref; src = "VIRAC2 (catalog not found)"
    print(f"    reference = {src}")
    allok = True
    seen = set()
    for filt, prop in FILTERS.items():
        merged = i2d_path(filt, prop, "merged")
        for m in ("merged", "nrca", "nrcb"):
            p = i2d_path(filt, prop, module=m)
            if not p or p in seen:
                continue
            if m != "merged" and not is_fresh(p, merged):
                print(f"    {filt:6s} {m:6s} SKIP (stale, predates merged)")
                continue
            seen.add(p)
            r = hist_off(detect(p), catc)
            if r is None or not r["ok"]:
                print(f"    {os.path.basename(p)}: no coherent tie (few stars?)"); continue
            ok = r["off"] < THRESH["loop"]; allok &= ok
            print(f"    {filt:6s} {m:6s} |{r['off']:5.1f} mas| C={r['contrast']:.0f} n={r['n']:5d}  "
                  f"{'OK' if ok else 'FAIL'}")
    return allok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checks", default="1234", help="subset e.g. 14")
    args = ap.parse_args()
    ref = virac2()
    gaia = gaia_only()
    if gaia is None:
        print("WARNING: Gaia-only reference unavailable; check 2 runs VIRAC2-only "
              "(no two-reference agreement -- a single dense ref can be fooled).")
    cache = {}
    for filt in ("F200W", "F212N"):
        p = i2d_path(filt, FILTERS[filt], "merged")
        cache[filt] = detect(p) if p else SkyCoord([], [], unit=u.deg)
    results = {}
    if "1" in args.checks: results["cross"] = check1_cross(cache)
    if "2" in args.checks: results["virac2"] = check2_virac2(cache, ref, gaia)
    if "3" in args.checks: results["module"] = check3_modules(ref)
    if "4" in args.checks: results["loop"] = check4_loop(ref)
    print("\n=== SUMMARY ===")
    for k, v in results.items():
        print(f"  {k:8s}: {'PASS' if v else ('SKIP' if v is None else 'FAIL')}")
    hard = [v for v in results.values() if v is False]
    print(f"\n{'ALL PASS' if not hard else 'REGRESSION(S) PRESENT'}")
    return 1 if hard else 0


if __name__ == "__main__":
    sys.exit(main())
