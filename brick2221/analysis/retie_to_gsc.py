#!/usr/bin/env python
"""Re-tie a Brick merged catalog to the GSC 3.2 / Gaia DR3 frame at the observation epoch.

The merged catalogs are internally precise (~1.5 mas bright-star repeatability) but their
absolute frame is tied to VVV, which is ~20 mas off the Gaia/GSC frame that JWST FGS / NIRSpec
use. Rather than re-run the whole pipeline, this applies a great-circle affine correction
(shift + rotation + scale + shear) measured against GSC 3.2 (proper-motion-propagated to the
observation epoch) using bright, photometrically clean stars, then rewrites every position
column. Output is a NIRSpec-ready catalog plus a validation report.

GSC 3.2 is the current active JWST guide star catalog (Gaia-DR3-sourced). It is not on VizieR;
retrieved from the STScI VO CatalogSearch service.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy import stats
from astropy.table import Table

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))
from brick2221.analysis.brick_astrometry_diagnostics import (  # noqa: E402
    load_gsc32, gsc32_propagated_coord, compute_query_footprint, JWST_EPOCH,
)

K_FILT = ("f212n", "f210m", "f200w")


def jwst_kcol(cat):
    for f in K_FILT:
        if f"mag_ab_{f}" in cat.colnames:
            return f"mag_ab_{f}"
    return None


def tangent_xy(coord, ra0, dec0):
    """Great-circle tangent-plane coords in arcsec about (ra0, dec0)."""
    x = (coord.ra.deg - ra0) * np.cos(np.deg2rad(dec0)) * 3600.0
    y = (coord.dec.deg - dec0) * 3600.0
    return x, y


def fit_affine(x, y, dra, ddec, niter=5, nsigma=3.0):
    """Fit dra,ddec [arcsec] = affine(x,y). Returns (A,B) coeffs for [1,x,y], plus keep mask."""
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(dra) & np.isfinite(ddec)
    for _ in range(niter):
        D = np.vstack([np.ones(keep.sum()), x[keep], y[keep]]).T
        A, *_ = np.linalg.lstsq(D, dra[keep], rcond=None)
        B, *_ = np.linalg.lstsq(D, ddec[keep], rcond=None)
        rdra = dra - (A[0] + A[1] * x + A[2] * y)
        rddec = ddec - (B[0] + B[1] * x + B[2] * y)
        s = stats.mad_std(rdra[keep]), stats.mad_std(rddec[keep])
        new = keep & (np.abs(rdra) < nsigma * s[0]) & (np.abs(rddec) < nsigma * s[1])
        if new.sum() == keep.sum():
            keep = new
            break
        keep = new
    return A, B, keep


def apply_affine_to_coord(coord, ra0, dec0, A, B):
    """Subtract the modelled offset (JWST - ref) from a SkyCoord; returns corrected SkyCoord."""
    x, y = tangent_xy(coord, ra0, dec0)
    dra = A[0] + A[1] * x + A[2] * y   # great-circle arcsec
    ddec = B[0] + B[1] * x + B[2] * y
    ra_corr = coord.ra.deg - (dra / 3600.0) / np.cos(coord.dec.to_value(u.rad))
    dec_corr = coord.dec.deg - (ddec / 3600.0)
    return SkyCoord(ra_corr * u.deg, dec_corr * u.deg, frame=coord.frame)


def gc_offsets(a, b):
    """a - b great-circle (dra*cosdec, ddec) in mas, b transformed to a.frame."""
    b = b.transform_to(a.frame)
    dra = (a.ra - b.ra).to(u.mas).value * np.cos(a.dec.to_value(u.rad))
    ddec = (a.dec - b.dec).to(u.mas).value
    return dra, ddec


def match(cat_coord, ref_coord, max_sep):
    i, s, _ = cat_coord.match_to_catalog_sky(ref_coord)
    r, _, _ = ref_coord.match_to_catalog_sky(cat_coord)
    k = (r[i] == np.arange(len(i))) & (s <= max_sep)
    return np.where(k)[0], i[k]


def summarize(dra, ddec, tag):
    md, mdd = np.median(dra), np.median(ddec)
    return (f"{tag:34s} N={len(dra):5d} med=({md:7.2f},{mdd:7.2f}) "
            f"vec={np.hypot(md, mdd):6.2f} MAD=({stats.mad_std(dra):5.1f},{stats.mad_std(ddec):5.1f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--obs-epoch", type=float, default=JWST_EPOCH.jyear,
                    help="Observation epoch (Julian year) to propagate GSC3.2 to.")
    ap.add_argument("--max-sep-arcsec", type=float, default=0.15)
    ap.add_argument("--ks-bright", type=float, default=14.0,
                    help="Use GSC3.2 sources with 2MASS Ks brighter than this for the fit.")
    ap.add_argument("--mode", choices=["shift", "affine"], default="shift",
                    help="shift (robust, default) or affine (shift+rot+scale; can overfit at "
                         "this S/N and inject spurious position-dependent shifts).")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    cachedir = outdir / "refcache"; cachedir.mkdir(exist_ok=True)
    max_sep = args.max_sep_arcsec * u.arcsec
    to_epoch = Time(args.obs_epoch, format="jyear")

    cat = Table.read(args.catalog)
    label = Path(args.catalog).stem
    pos_cols = [c for c in cat.colnames
                if c.startswith("skycoord") and isinstance(cat[c], SkyCoord)]
    refpos = "skycoord_ref" if "skycoord_ref" in pos_cols else pos_cols[0]
    cat_coord = cat[refpos]
    kcol = jwst_kcol(cat)
    cat_k = np.asarray(np.ma.filled(cat[kcol], np.nan), float) if kcol else None
    center, width, height = compute_query_footprint(cat_coord)
    ra0, dec0 = center.ra.deg, center.dec.deg

    # GSC 3.2 PM-propagated to obs epoch.
    gsc = load_gsc32(center, width, height, cachedir / "gsc32.fits")
    gcoord = gsc32_propagated_coord(gsc, to_epoch)
    ks = np.asarray(gsc["tmassKsMag"], float)
    bright = np.isfinite(ks) & (ks < args.ks_bright)
    gb = gcoord[bright]; ksb = ks[bright]

    report = [f"# Re-tie {label} -> GSC3.2/Gaia at epoch {args.obs_epoch:.3f}",
              f"mode={args.mode}, match={args.max_sep_arcsec}\", GSC3.2 Ks<{args.ks_bright} for fit", ""]

    # Match bright clean stars; flux-clean vs JWST K.
    ci, ri = match(cat_coord, gb, max_sep)
    dra, ddec = gc_offsets(cat_coord[ci], gb[ri])  # mas
    if cat_k is not None:
        dmag = cat_k[ci] - ksb[ri]
        fin = np.isfinite(dmag)
        if fin.sum() > 10:
            m, s = np.nanmedian(dmag[fin]), stats.mad_std(dmag[fin])
            fc = fin & (np.abs(dmag - m) <= 3 * s)
            ci, ri, dra, ddec = ci[fc], ri[fc], dra[fc], ddec[fc]
    report.append(summarize(dra, ddec, "BEFORE re-tie (JWST vs GSC3.2)"))

    # Fit in arcsec.
    x, y = tangent_xy(cat_coord[ci], ra0, dec0)
    A, B, keep = fit_affine(x, y, dra / 1000.0, ddec / 1000.0)
    # Diagnostic decomposition of the (unregularized) affine linear part, in mas/arcsec.
    report += [
        f"affine diagnostic (NOT necessarily applied): fit N={keep.sum()} of {len(x)}",
        f"  shift = ({A[0]*1000:.2f}, {B[0]*1000:.2f}) mas",
        f"  linear dRA/dx={A[1]*1000:.3f} dRA/dy={A[2]*1000:.3f} dDec/dx={B[1]*1000:.3f} "
        f"dDec/dy={B[2]*1000:.3f} (mas/arcsec; x{(np.nanmax(x)-np.nanmin(x)):.0f}\" field => "
        f"up to {max(abs(A[1]),abs(A[2]),abs(B[1]),abs(B[2]))*1000*(np.nanmax(x)-np.nanmin(x)):.1f} mas)",
    ]
    if args.mode == "shift":
        # Robust constant shift = median of the (flux-cleaned) fit-sample offsets.
        A = np.array([np.median(dra[keep]) / 1000.0, 0.0, 0.0])
        B = np.array([np.median(ddec[keep]) / 1000.0, 0.0, 0.0])
    report += [f"APPLIED ({args.mode}): shift = ({A[0]*1000:.2f}, {B[0]*1000:.2f}) mas", ""]

    # Fit-sample residual after correction (median forced to 0 for shift mode; sanity only).
    rdra = dra - (A[0] + A[1] * x + A[2] * y) * 1000.0
    rddec = ddec - (B[0] + B[1] * x + B[2] * y) * 1000.0
    report.append(summarize(rdra[keep], rddec[keep], "fit-sample residual (same pairs)"))

    # Honest cross-validation: fit a shift on half the pairs, measure residual on the held-out
    # half using the SAME pairs (no re-matching -> immune to dense-catalog counterpart flips).
    half = np.arange(len(dra)) % 2 == 0
    if args.mode == "shift" and half.sum() > 10 and (~half).sum() > 10:
        sh_ra, sh_dd = np.median(dra[half]), np.median(ddec[half])
        ho_ra, ho_dd = dra[~half] - sh_ra, ddec[~half] - sh_dd
        report.append(summarize(ho_ra, ho_dd, "HELD-OUT residual (cross-val)"))

    # Apply to ALL position columns.
    for pc in pos_cols:
        cat[pc] = apply_affine_to_coord(cat[pc], ra0, dec0, A, B)
    # Update plain RA/Dec deg columns if present (from the reference position).
    if "ra" in cat.colnames and "dec" in cat.colnames:
        cat["ra"] = cat[refpos].ra.deg
        cat["dec"] = cat[refpos].dec.deg
    cat.meta["RETIED_TO"] = "GSC3.2/GaiaDR3"
    cat.meta["RETIE_EPOCH"] = args.obs_epoch
    cat.meta["RETIE_SHIFT_DRA_MAS"] = float(A[0] * 1000)
    cat.meta["RETIE_SHIFT_DDEC_MAS"] = float(B[0] * 1000)
    cat.meta["RETIE_MODE"] = args.mode

    outpath = outdir / (label + "_gsc32retied.fits")
    cat.write(outpath, overwrite=True)
    report.append(f"wrote {outpath}")

    # Validate: re-match corrected catalog to GSC3.2 (bright clean) and to VVV.
    cc = cat[refpos]
    ci, ri = match(cc, gb, max_sep)
    d1, d2 = gc_offsets(cc[ci], gb[ri])
    if cat_k is not None:
        dmag = cat_k[ci] - ksb[ri]; fin = np.isfinite(dmag)
        m, s = np.nanmedian(dmag[fin]), stats.mad_std(dmag[fin]); fc = fin & (np.abs(dmag - m) <= 3 * s)
        d1, d2 = d1[fc], d2[fc]
    report.append(summarize(d1, d2, "AFTER re-tie (GSC3.2 re-match*)"))
    report.append("  *re-match median is crowding-biased in the dense GSC3.2; trust HELD-OUT above.")

    vvvpath = cachedir / "vvv.fits"
    if vvvpath.exists():
        vvv = Table.read(vvvpath)
        ci, ri = match(cc, vvv["skycoord"], 0.3 * u.arcsec)
        d1, d2 = gc_offsets(cc[ci], vvv["skycoord"][ri])
        report.append(summarize(d1, d2, "AFTER re-tie (JWST vs VVV)"))

    (outdir / f"{label}_retie_report.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
