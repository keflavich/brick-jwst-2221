#!/usr/bin/env python
"""Deep astrometric diagnostics for the Brick NIRCam merged catalogs.

Goes beyond astrometry_analysis.py by adding:
  * proper-motion propagation of Gaia DR3, VIRAC2 and GSC2.4.2 to the JWST epoch,
  * flux-ratio-cleaned matching (the same selection the pipeline alignment uses),
  * internal cross-filter repeatability (reference-independent precision test),
  * spatial/module structure maps (detect frame discontinuities),
to explain the origin of the systematic offset and to assess NIRSpec readiness.

All offsets are great-circle (dRA includes cos(dec)) in mas unless noted.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy import stats
from astropy.table import Table

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))
from brick2221.analysis.make_reference_from_pipeline_catalogs import (  # noqa: E402
    compute_query_footprint,
    fetch_gaia_catalog,
    fetch_gns_catalog,
    fetch_vvv_catalog,
    DEFAULT_GNS_CATALOG,
)

# JWST Brick prop 2221 obs001 epoch (DATE-BEG of f182m mosaic = 2022-08-28).
JWST_EPOCH = Time("2022-08-28T02:38:56", scale="utc")
K_FILT = ("f212n", "f210m", "f200w")  # K-band-equivalent JWST filters for flux selection


def robust(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, 0
    return float(np.median(x)), float(stats.mad_std(x)), int(x.size)


def great_circle_offsets(cat: SkyCoord, ref: SkyCoord):
    """dRA*cos(dec), dDec in mas: cat minus ref, in cat frame."""
    ref = ref.transform_to(cat.frame)
    dra = (cat.ra - ref.ra).to(u.mas).value * np.cos(cat.dec.to_value(u.rad))
    ddec = (cat.dec - ref.dec).to(u.mas).value
    return dra, ddec


def mutual_match(cat: SkyCoord, ref: SkyCoord, max_sep):
    idx, sep, _ = cat.match_to_catalog_sky(ref)
    rev, _, _ = ref.match_to_catalog_sky(cat)
    keep = (rev[idx] == np.arange(len(idx))) & (sep <= max_sep)
    return np.where(keep)[0], idx[keep]


def flux_ratio_clean(jwst_k, ref_k, nsigma=4.0):
    """Boolean mask keeping sources whose (jwst_k - ref_k) is within nsigma MAD."""
    d = np.asarray(jwst_k, float) - np.asarray(ref_k, float)
    fin = np.isfinite(d)
    if fin.sum() < 5:
        return fin
    med = np.nanmedian(d[fin])
    mad = stats.mad_std(d[fin])
    if not np.isfinite(mad) or mad == 0:
        return fin
    return fin & (np.abs(d - med) <= nsigma * mad)


def pick_jwst_k(cat: Table):
    for f in K_FILT:
        if f"mag_ab_{f}" in cat.colnames:
            return f"mag_ab_{f}"
    return None


def propagate_pm(ref: Table, ra_col, dec_col, pmra_col, pmdec_col, ref_epoch: Time, to_epoch: Time):
    """Return SkyCoord propagated to to_epoch. pm columns in mas/yr (pmRA already *cos dec)."""
    ra = np.asarray(ref[ra_col], float)
    dec = np.asarray(ref[dec_col], float)
    dt = (to_epoch.jyear - ref_epoch.jyear)
    pmra = np.asarray(ref[pmra_col], float)   # mas/yr, *cos(dec)
    pmdec = np.asarray(ref[pmdec_col], float)  # mas/yr
    # fill non-finite PM with 0 (no correction) so those sources still match
    pmra = np.where(np.isfinite(pmra), pmra, 0.0)
    pmdec = np.where(np.isfinite(pmdec), pmdec, 0.0)
    ra_new = ra + (pmra * dt / 3.6e6) / np.cos(np.deg2rad(dec))
    dec_new = dec + (pmdec * dt / 3.6e6)
    return SkyCoord(ra_new * u.deg, dec_new * u.deg, frame="icrs"), dt


def compare(cat_coord, cat_k, ref_coord, ref_k, max_sep, label, do_flux=True):
    ci, ri = mutual_match(cat_coord, ref_coord, max_sep)
    if len(ci) < 5:
        return dict(label=label, n=len(ci), note="too few matches")
    dra, ddec = great_circle_offsets(cat_coord[ci], ref_coord[ri])
    out = {}
    md, sd, n = robust(dra); mdd, sdd, _ = robust(ddec)
    out["raw"] = dict(n=n, med_dra=md, med_ddec=mdd, vec=float(np.hypot(md, mdd)),
                      mad_dra=sd, mad_ddec=sdd)
    if do_flux and cat_k is not None and ref_k is not None:
        mask = flux_ratio_clean(np.asarray(cat_k)[ci], np.asarray(ref_k)[ri])
        if mask.sum() >= 5:
            md, sd, n = robust(dra[mask]); mdd, sdd, _ = robust(ddec[mask])
            out["flux"] = dict(n=n, med_dra=md, med_ddec=mdd, vec=float(np.hypot(md, mdd)),
                               mad_dra=sd, mad_ddec=sdd)
            out["flux"]["mask"] = mask
    out["label"] = label
    out["_dra"] = dra; out["_ddec"] = ddec
    out["_ra"] = cat_coord[ci].ra.deg; out["_dec"] = cat_coord[ci].dec.deg
    return out


def fmt(d):
    if "flux" in d:
        f = d["flux"]
        return (f"{d['label']:24s} Nflux={f['n']:5d} vec={f['vec']:6.2f}  "
                f"med=({f['med_dra']:7.2f},{f['med_ddec']:7.2f}) MAD=({f['mad_dra']:5.1f},{f['mad_ddec']:5.1f})"
                f"   [raw vec={d['raw']['vec']:.1f} N={d['raw']['n']}]")
    r = d.get("raw")
    if r is None:
        return f"{d['label']:24s} {d.get('note','')}"
    return (f"{d['label']:24s} N={r['n']:5d} vec={r['vec']:6.2f}  "
            f"med=({r['med_dra']:7.2f},{r['med_ddec']:7.2f}) MAD=({r['mad_dra']:5.1f},{r['mad_ddec']:5.1f})")


def spatial_map(ra, dec, dra, ddec, outpath, title, nbin=12):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, val, name in [(axes[0], dra, "dRA*cos(dec)"), (axes[1], ddec, "dDec")]:
        fin = np.isfinite(val)
        H, xe, ye = np.histogram2d(ra[fin], dec[fin], bins=nbin, weights=val[fin])
        C, _, _ = np.histogram2d(ra[fin], dec[fin], bins=[xe, ye])
        with np.errstate(invalid="ignore"):
            M = H / C
        im = ax.imshow(M.T, origin="lower", extent=[xe[0], xe[-1], ye[0], ye[-1]],
                       aspect="auto", cmap="RdBu_r", vmin=-40, vmax=40)
        ax.set_xlabel("RA [deg]"); ax.set_ylabel("Dec [deg]")
        ax.set_title(f"{name} median [mas]")
        ax.invert_xaxis()
        fig.colorbar(im, ax=ax)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dra_ddec(dra, ddec, ra, dec, outpath, match=None):
    fig = plt.figure(figsize=(10, 10))
    sp1 = plt.subplot(2, 2, 1)
    sp1.hist(dra, bins=50)
    sp1.axvline(np.median(dra), color="k", ls="--", lw=1)

    sp2 = plt.subplot(2, 2, 2)
    sp2.hist(ddec, bins=50)
    sp2.axvline(np.median(ddec), color="k", ls="--", lw=1)

    sp3 = plt.subplot(2, 2, 3)
    sp3.scatter(dra, ddec, s=1)
    sp3.scatter(np.median(dra), np.median(ddec), color="k", marker="x", s=50)

    sp4 = plt.subplot(2, 2, 4)
    sp4.scatter(ra, dec, s=1)

    if match is not None:
        sp1.hist(dra[match], bins=50)
        sp2.hist(ddec[match], bins=50)
        sp3.scatter(dra[match], ddec[match], s=1)
        sp4.scatter(ra[match], dec[match], s=1)
        sp1.axvline(np.median(dra[match]), color="b", ls="--", lw=1)
        sp2.axvline(np.median(ddec[match]), color="b", ls="--", lw=1)
        sp3.scatter(np.median(dra[match]), np.median(ddec[match]), color="b", marker="x", s=50)

    fig.savefig(outpath, dpi=150, bbox_inches="tight")


def internal_repeatability(cat: Table, outdir: Path, label: str):
    """Per-filter position scatter about skycoord_ref: reference-independent precision."""
    filt_cols = [c for c in cat.colnames if c.startswith("skycoord_f")]
    ref = cat["skycoord_ref"]
    lines = ["", f"## Internal cross-filter repeatability ({label})",
             "Per-star offset of each filter position from skycoord_ref (reference-independent).",
             "", "| filter | N | med_dRA | med_dDec | MAD_dRA | MAD_dDec |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for fc in sorted(filt_cols):
        coord = cat[fc]
        good = np.isfinite(coord.ra.deg) & np.isfinite(ref.ra.deg)
        # exclude rows where this filter IS the reference (offset would be exactly 0)
        good &= (np.abs((coord.ra.deg - ref.ra.deg)) + np.abs(coord.dec.deg - ref.dec.deg)) > 0
        if good.sum() < 20:
            lines.append(f"| {fc.replace('skycoord_','')} | {int(good.sum())} | - | - | - | - |")
            continue
        dra, ddec = great_circle_offsets(coord[good], ref[good])
        md, sd, n = robust(dra); mdd, sdd, _ = robust(ddec)
        lines.append(f"| {fc.replace('skycoord_','')} | {n} | {md:.2f} | {mdd:.2f} | {sd:.2f} | {sdd:.2f} |")
    return lines


def load_virac2(center, width, height, cache):
    from astroquery.vizier import Vizier
    if cache.exists():
        return Table.read(cache)
    viz = Vizier(columns=["*"], row_limit=-1)
    res = viz.query_region(center, width=width, height=height, catalog="II/387/virac2")
    t = res[0]
    t.write(cache, overwrite=True)
    return t


def load_gsc(center, width, height, cache):
    from astroquery.vizier import Vizier
    if cache.exists():
        return Table.read(cache)
    viz = Vizier(columns=["*"], row_limit=-1)
    res = viz.query_region(center, width=width, height=height, catalog="I/353")
    t = res[0]
    t.write(cache, overwrite=True)
    return t


def load_gsc32(center, width, height, cache):
    """GSC 3.2 (current JWST FGS catalog, Gaia DR3-sourced) via the STScI VO service.

    The VOTABLE response trips astropy's bit-field parser, so request CSV. The
    service caps rows per cone, so tile in Dec across the (elongated) footprint.
    Columns: ra, dec, epoch (per-source jyear), rapm/decpm (mas/yr, *cos dec),
    tmassKsMag, gaiaGmag, GAIAdr3sourceID.
    """
    import urllib.request
    from astropy.io import ascii as asciirw
    from astropy.table import vstack
    if cache.exists():
        return Table.read(cache)
    base = "https://gsss.stsci.edu/webservices/vo/CatalogSearch.aspx"
    ra0 = center.ra.to_value(u.deg)
    dec0 = center.dec.to_value(u.deg)
    half_h = height.to_value(u.deg) / 2.0
    decs = [dec0 - half_h * 0.6, dec0, dec0 + half_h * 0.6]
    sr = max(width.to_value(u.deg), half_h) * 0.8 + 0.02
    parts = []
    for dd in decs:
        url = f"{base}?RA={ra0:.5f}&DEC={dd:.5f}&SR={sr:.3f}&CAT=GSC32&FORMAT=CSV"
        data = urllib.request.urlopen(url, timeout=240).read().decode("utf-8", "replace")
        parts.append(asciirw.read(data, format="csv", comment="#", fast_reader=False))
    t = vstack(parts)
    _, uniq = np.unique(np.asarray(t["objID"]), return_index=True)
    t = t[uniq]
    t.write(cache, overwrite=True)
    return t


def gsc32_propagated_coord(g: Table, to_epoch: Time):
    """Propagate GSC3.2 positions to to_epoch using per-source epoch and rapm/decpm."""
    ra = np.asarray(g["ra"], float)
    dec = np.asarray(g["dec"], float)
    ep = np.asarray(g["epoch"], float)
    pmra = np.where(np.isfinite(np.asarray(g["rapm"], float)), np.asarray(g["rapm"], float), 0.0)
    pmde = np.where(np.isfinite(np.asarray(g["decpm"], float)), np.asarray(g["decpm"], float), 0.0)
    dt = to_epoch.jyear - ep
    ra_new = ra + (pmra * dt / 3.6e6) / np.cos(np.deg2rad(dec))
    dec_new = dec + (pmde * dt / 3.6e6)
    return SkyCoord(ra_new * u.deg, dec_new * u.deg, frame="icrs")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalog", required=True, help="merged JWST catalog (fits/ecsv)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max-sep-arcsec", type=float, default=0.2)
    ap.add_argument("--virac-ref-epoch", type=float, default=2014.0,
                    help="VIRAC2 position reference epoch (jyear).")
    ap.add_argument("--refresh-cache", action="store_true")
    args = ap.parse_args()

    outdir = Path(os.path.expanduser(args.outdir)); outdir.mkdir(parents=True, exist_ok=True)
    cachedir = outdir / "refcache"; cachedir.mkdir(exist_ok=True)
    max_sep = args.max_sep_arcsec * u.arcsec

    catalog_filepath = os.path.expanduser(args.catalog)

    cat = Table.read(catalog_filepath)
    label = Path(catalog_filepath).stem
    cat_coord = cat["skycoord_ref"]
    kcol = pick_jwst_k(cat)
    cat_k = np.asarray(np.ma.filled(cat[kcol], np.nan), float) if kcol else None
    center, width, height = compute_query_footprint(cat_coord)

    report = [f"# Brick astrometry diagnostics: {label}", "",
              f"JWST epoch = {JWST_EPOCH.isot} (jyear {JWST_EPOCH.jyear:.3f}); "
              f"match radius {args.max_sep_arcsec}\"; JWST K column = {kcol}", "",
              "Great-circle offsets (dRA includes cos dec), mas. `flux` rows are flux-ratio cleaned.",
              "", "## External reference comparison", ""]
    print(report[2])

    results = {}

    # --- VVV (Ks) ---
    vvv = fetch_vvv_catalog(center=center, width=width, height=height, vvv_catalog="II/376/vvv4",
                            cache_path=cachedir / "vvv.fits", refresh_cache=args.refresh_cache)
    r = compare(cat_coord, cat_k, vvv["skycoord"], vvv["Ks_refmag"], max_sep, "VVV (Ks, no PM)")
    results["vvv"] = r; report.append("    " + fmt(r)); print(fmt(r))

    # --- GNS (Ks, no PM available here) ---
    gns = fetch_gns_catalog(center=center, width=width, height=height, gns_catalog=DEFAULT_GNS_CATALOG,
                            cache_path=cachedir / "gns.fits", refresh_cache=args.refresh_cache)
    r = compare(cat_coord, cat_k, gns["skycoord"], gns["Ks_refmag"], max_sep, "GNS (Ks, no PM)")
    results["gns"] = r; report.append("    " + fmt(r)); print(fmt(r))

    # --- Gaia DR3: no PM and PM-propagated ---
    gaia = fetch_gaia_catalog(center=center, width=width, height=height, gaia_catalog="I/355/gaiadr3",
                              cache_path=cachedir / "gaia.fits", refresh_cache=args.refresh_cache)
    r = compare(cat_coord, None, gaia["skycoord"], None, max_sep, "Gaia DR3 (no PM)", do_flux=False)
    results["gaia"] = r; report.append("    " + fmt(r)); print(fmt(r))
    gcols = {c.lower(): c for c in gaia.colnames}
    pmra = next((gcols[c] for c in ("pmra",) if c in gcols), None)
    pmde = next((gcols[c] for c in ("pmde", "pmdec") if c in gcols), None)
    racol = next((gcols[c] for c in ("ra_icrs", "raj2000", "ra") if c in gcols), None)
    decol = next((gcols[c] for c in ("de_icrs", "dej2000", "dec", "de") if c in gcols), None)
    if pmra and pmde and racol and decol:
        gcoord, dt = propagate_pm(gaia, racol, decol, pmra, pmde, Time(2016.0, format="jyear"), JWST_EPOCH)
        r = compare(cat_coord, None, gcoord, None, max_sep, f"Gaia DR3 (PM->2022.66,dt={dt:.1f}yr)", do_flux=False)
        results["gaia_pm"] = r; report.append("    " + fmt(r)); print(fmt(r))

    # --- VIRAC2: PM-propagated (Ks) ---
    try:
        virac = load_virac2(center, width, height, cachedir / "virac2.fits")
        vcoord, dt = propagate_pm(virac, "RAJ2000", "DEJ2000", "pmRA", "pmDE",
                                  Time(args.virac_ref_epoch, format="jyear"), JWST_EPOCH)
        vk = np.asarray(virac["Ksmag"], float)
        r0 = compare(cat_coord, cat_k, SkyCoord(virac["RAJ2000"], virac["DEJ2000"], unit="deg"),
                     vk, max_sep, "VIRAC2 (Ks, no PM)")
        results["virac"] = r0; report.append("    " + fmt(r0)); print(fmt(r0))
        r = compare(cat_coord, cat_k, vcoord, vk, max_sep,
                    f"VIRAC2 (Ks, PM->2022.66,dt={dt:.1f}yr)")
        results["virac_pm"] = r; report.append("    " + fmt(r)); print(fmt(r))
    except Exception as e:
        report.append(f"    VIRAC2 FAILED: {e}"); print("VIRAC2 failed:", e)
        raise e

    # --- GSC 2.4.2 (proxy for GSC 3.1 / JWST FGS frame) ---
    try:
        gsc = load_gsc(center, width, height, cachedir / "gsc242.fits")
        gc = {c.lower(): c for c in gsc.colnames}
        gra = next((gc[c] for c in ("raj2000", "ra_icrs", "ra") if c in gc), None)
        gde = next((gc[c] for c in ("dej2000", "de_icrs", "dec", "de") if c in gc), None)
        r = compare(cat_coord, None, SkyCoord(gsc[gra], gsc[gde], unit="deg"), None, max_sep,
                    "GSC2.4.2 (no PM)", do_flux=False)
        results["gsc"] = r; report.append("    " + fmt(r)); print(fmt(r))
    except Exception as e:
        report.append(f"    GSC2.4.2 FAILED: {e}"); print("GSC2.4.2 failed:", e)
        raise e

    # --- GSC 3.2 (CURRENT active JWST FGS catalog, Gaia DR3-sourced) ---
    try:
        gsc32 = load_gsc32(center, width, height, cachedir / "gsc32.fits")
        ks = np.asarray(gsc32["tmassKsMag"], float)
        coord_nopm = SkyCoord(gsc32["ra"], gsc32["dec"], unit="deg")
        coord_pm = gsc32_propagated_coord(gsc32, JWST_EPOCH)
        # Full sample (crowding-inflated in the GC) and clean bright-NIR subset.
        r = compare(cat_coord, None, coord_pm, None, max_sep, "GSC3.2 (PM, all)", do_flux=False)
        results["gsc32"] = r; report.append("    " + fmt(r)); print(fmt(r))
        bright = np.isfinite(ks) & (ks < 14)
        if bright.sum() > 20:
            rb = compare(cat_coord, cat_k, coord_pm[bright], ks[bright], max_sep,
                         "GSC3.2 (PM, Ks<14 flux-clean)")
            results["gsc32_bright"] = rb; report.append("    " + fmt(rb)); print(fmt(rb))
    except Exception as e:
        report.append(f"    GSC3.2 FAILED: {e}"); print("GSC3.2 failed:", e)
        raise e

    # --- spatial maps vs VVV and vs GNS (module/frame structure) ---
    for key in ("vvv", "gns"):
        d = results.get(key)
        if d and "_dra" in d:
            spatial_map(d["_ra"], d["_dec"], d["_dra"], d["_ddec"],
                        outdir / f"spatial_{key}.png", f"{label} vs {key.upper()} (offset map)")
            report.append(f"    spatial map: spatial_{key}.png")

    for key in keys:
        d = results.get(key)
        if d and "_dra" in d:
            match = d.get("flux", {}).get("mask")
            plot_dra_ddec(d["_dra"], d["_ddec"], d["_ra"], d["_dec"],
                          outdir / f"dRA_dDec_{key}.png", match=match)
            report.append(f"    dRA/dDec scatter: dRA_dDec_{key}.png")

    # --- internal repeatability ---
    report += internal_repeatability(cat, outdir, label)

    (outdir / "diagnostics_report.md").write_text("\n".join(report) + "\n")
    print("\nWrote", outdir / "diagnostics_report.md")

    return results


if __name__ == "__main__":
    results = main()
