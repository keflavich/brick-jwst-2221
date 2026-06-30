#!/usr/bin/env python
"""Verify (BY IMAGE, not table) that the re-drizzled brick i2d are tied to the
absolute VIRAC2 frame.  Success = each filter's drizzled mosaic, and the
1182-vs-2221 cross-filter offset, are < THRESH_MAS via a crowding-robust
cross-correlation (peak of the 2-D pair-separation histogram).  A 0.1" nearest-
neighbour match MUST NOT be used here -- in this dense field it returns chance
coincidences whose median is ~0 and hides a multi-arcsec offset.

Usage: verify_1182_astrometry.py [--release]   (--release checks the staged copies)
"""
import argparse, numpy as np, astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.coordinates import SkyCoord, search_around_sky
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats

BASE = "/orange/adamginsburg/jwst/brick"
REFCAT = f"{BASE}/catalogs/gaia_virac2_refcat_epoch2022.70.fits"
THRESH_MAS = 50.0
# (filter, proposal, pipeline-relative i2d path)
TARGETS = [
    ("F115W", 1182, "F115W/pipeline/jw01182-o004_t001_nircam_clear-f115w-merged_i2d.fits"),
    ("F200W", 1182, "F200W/pipeline/jw01182-o004_t001_nircam_clear-f200w-merged_i2d.fits"),
    ("F356W", 1182, "F356W/pipeline/jw01182-o004_t001_nircam_clear-f356w-merged_i2d.fits"),
    ("F444W", 1182, "F444W/pipeline/jw01182-o004_t001_nircam_clear-f444w-merged_i2d.fits"),
    ("F212N", 2221, "F212N/pipeline/jw02221-o001_t001_nircam_clear-f212n-merged_i2d.fits"),
    ("F182M", 2221, "F182M/pipeline/jw02221-o001_t001_nircam_clear-f182m-merged_i2d.fits"),
]


def detect(path):
    sci = fits.open(path)["SCI"]; w = WCS(sci.header); d = sci.data.astype("float32")
    _, med, std = sigma_clipped_stats(d, sigma=3.0)
    t = DAOStarFinder(fwhm=2.5, threshold=80 * std)(d - med)
    return SkyCoord(w.pixel_to_world(t["xcentroid"], t["ycentroid"]))


def xcorr(a, b, maxsep=2.5, binarc=0.04):
    ia, ib, _, _ = search_around_sky(a, b, maxsep * u.arcsec)
    dra = (a[ia].ra - b[ib].ra).to(u.arcsec).value * np.cos(a[ia].dec.radian)
    ddec = (a[ia].dec - b[ib].dec).to(u.arcsec).value
    bins = np.arange(-maxsep, maxsep + binarc, binarc)
    H, xe, ye = np.histogram2d(dra, ddec, bins=[bins, bins])
    i, j = np.unravel_index(H.argmax(), H.shape)
    return (xe[i] + xe[i + 1]) / 2, (ye[j] + ye[j + 1]) / 2, H.max() / max(np.median(H[H > 0]), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", action="store_true",
                    help="check the staged release copies instead of the pipeline products")
    args = ap.parse_args()
    refc = SkyCoord(Table.read(REFCAT)["RA"], Table.read(REFCAT)["DEC"], unit=u.deg)
    cats, ok = {}, True
    print(f"{'filter':7s}{'prop':6s}{'offset vs VIRAC2':>22s}{'peak/bg':>9s}  verdict")
    for filt, prop, rel in TARGETS:
        path = (f"/orange/adamginsburg/jwst/releases/v1.0-2026.06/brick/images/{filt}/"
                + rel.split('/')[-1]) if args.release else f"{BASE}/{rel}"
        sc = detect(path); cats[filt] = sc
        dr, dd, pk = xcorr(sc, refc)
        off = np.hypot(dr, dd) * 1000
        good = off < THRESH_MAS
        ok &= good
        print(f"{filt:7s}{prop:<6d}  dRA={dr:+.3f} dDec={dd:+.3f} |{off:5.0f}mas|{pk:8.0f}x  "
              f"{'OK' if good else 'FAIL'}")
    # 1182 vs 2221 cross-filter
    dr, dd, _ = xcorr(cats["F200W"], cats["F212N"])
    cross = np.hypot(dr, dd) * 1000
    ok &= cross < THRESH_MAS
    print(f"\n1182(F200W) vs 2221(F212N): |{cross:.0f} mas|  {'OK' if cross < THRESH_MAS else 'FAIL'}")
    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'} (threshold {THRESH_MAS:.0f} mas)")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
