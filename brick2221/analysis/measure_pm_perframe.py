#!/usr/bin/env python
"""Per-frame PM measurement: stack per-exposure daophot catalogs by obs#,
cross-match between two epochs of the same pointing.

For gc2211 default: obs023 (MJD 60202) vs obs049 (MJD 60541), Δt≈0.93 yr.
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy import stats as astats
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table, vstack


OBS_PATTERN = re.compile(r"jw\d{5}(\d{3})\d{3}_")


def list_catalogs(catalog_dir: str, obs_codes: list[str]) -> list[Path]:
    out: list[Path] = []
    for f in sorted(Path(catalog_dir).glob("*_daophot_iterative.fits")):
        meta_filename = ""
        try:
            with fits.open(f) as h:
                meta_filename = h[1].header.get("FILENAME", "")
        except Exception:
            pass
        # if header FILENAME has obs, use that; else fall back to file path
        for code in obs_codes:
            if f"jw02211{code}" in meta_filename or f"jw02211{code}" in str(f):
                out.append(f)
                break
    return out


def list_catalogs_via_meta(catalog_dir: str, obs_codes: set[str]) -> dict[str, list[Path]]:
    """Group per-frame catalogs by obs# read from the catalog meta FILENAME."""
    groups: dict[str, list[Path]] = {c: [] for c in obs_codes}
    for f in sorted(Path(catalog_dir).glob("*_daophot_iterative.fits")):
        try:
            t = Table.read(f, format="fits", hdu=1)
            fn = str(t.meta.get("FILENAME", ""))
        except Exception:
            continue
        m = OBS_PATTERN.search(fn)
        if not m:
            continue
        obs = m.group(1)
        if obs in obs_codes:
            groups[obs].append(f)
    return groups


def load_obs_stack(paths: list[Path]) -> dict:
    """Concatenate ra, dec, flux, flux_err from a list of per-frame catalogs."""
    ras, decs, fluxes, eflux, qfits = [], [], [], [], []
    for p in paths:
        t = Table.read(p)
        # column names vary by fitter; try multiple
        ra = None
        for c in ("skycoord_centroid", ):
            if c in t.colnames:
                sc = t[c]
                ra = sc.ra.deg; dec = sc.dec.deg
                break
        if ra is None and "ra" in t.colnames and "dec" in t.colnames:
            ra = np.asarray(t["ra"], dtype=float)
            dec = np.asarray(t["dec"], dtype=float)
        if ra is None:
            continue
        # flux + err: prefer flux_jy / eflux_jy if present, else flux/flux_err
        for fc, ec in (("flux_jy", "eflux_jy"), ("flux_fit", "flux_err"), ("flux", "flux_err")):
            if fc in t.colnames and ec in t.colnames:
                f = np.asarray(t[fc], dtype=float)
                e = np.asarray(t[ec], dtype=float)
                break
        else:
            f = np.full(len(ra), np.nan); e = np.full(len(ra), np.nan)
        qf = np.asarray(t["qfit"], dtype=float) if "qfit" in t.colnames else np.full(len(ra), np.nan)
        ok = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(f) & np.isfinite(e) & (f > 0) & (e > 0)
        ras.append(ra[ok]); decs.append(dec[ok])
        fluxes.append(f[ok]); eflux.append(e[ok]); qfits.append(qf[ok])
    if not ras:
        return None
    return dict(
        ra=np.concatenate(ras),
        dec=np.concatenate(decs),
        flux=np.concatenate(fluxes),
        eflux=np.concatenate(eflux),
        qfit=np.concatenate(qfits),
    )


def median_position_dedup(d: dict, tol_arcsec: float = 0.10) -> dict:
    """Collapse stacked per-frame detections into one position per source via
    iterative nearest-neighbor clustering. Brute but works on ~1e5 sources."""
    sc = SkyCoord(d["ra"] * u.deg, d["dec"] * u.deg)
    # For each source, find all neighbors within tol; take median position + weighted flux
    n = len(sc)
    used = np.zeros(n, dtype=bool)
    out_ra, out_dec, out_flux, out_eflux, out_qfit, out_n = [], [], [], [], [], []
    idx, sep, _ = sc.match_to_catalog_sky(sc, nthneighbor=2)
    # build clusters using a simple "search around" approach via search_around_sky
    idx1, idx2, sep_pairs, _ = sc.search_around_sky(sc, tol_arcsec * u.arcsec)
    # adjacency dict
    adj = [[] for _ in range(n)]
    for a, b in zip(idx1, idx2):
        if a != b:
            adj[a].append(b)
    # connected components (BFS)
    for i in range(n):
        if used[i]:
            continue
        comp = [i]; used[i] = True
        stack = [i]
        while stack:
            cur = stack.pop()
            for nb in adj[cur]:
                if not used[nb]:
                    used[nb] = True
                    comp.append(nb); stack.append(nb)
        # median position over component, inverse-variance flux average
        out_ra.append(np.median(d["ra"][comp]))
        out_dec.append(np.median(d["dec"][comp]))
        w = 1.0 / (d["eflux"][comp] ** 2 + 1e-40)
        out_flux.append(np.sum(d["flux"][comp] * w) / np.sum(w))
        out_eflux.append(1.0 / np.sqrt(np.sum(w)))
        out_qfit.append(np.nanmedian(d["qfit"][comp]))
        out_n.append(len(comp))
    return dict(
        ra=np.asarray(out_ra), dec=np.asarray(out_dec),
        flux=np.asarray(out_flux), eflux=np.asarray(out_eflux),
        qfit=np.asarray(out_qfit), n_det=np.asarray(out_n),
    )


def crossmatch(a: dict, b: dict, max_sep_arcsec: float):
    sc_a = SkyCoord(a["ra"] * u.deg, a["dec"] * u.deg)
    sc_b = SkyCoord(b["ra"] * u.deg, b["dec"] * u.deg)
    idx, sep, _ = sc_a.match_to_catalog_sky(sc_b)
    keep = sep.to(u.arcsec).value < max_sep_arcsec
    # bidirectional
    idx_back, _, _ = sc_b.match_to_catalog_sky(sc_a)
    bi = np.arange(len(sc_a)) == idx_back[idx]
    keep &= bi
    return np.where(keep)[0], idx[keep], sep[keep].to(u.mas).value


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog-dir", required=True, help="Dir containing per-frame *_daophot_iterative.fits")
    p.add_argument("--early-obs", default="023")
    p.add_argument("--late-obs", default="049")
    p.add_argument("--early-mjd", type=float, required=True)
    p.add_argument("--late-mjd", type=float, required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--outdir", default="/blue/adamginsburg/adamginsburg/jwst/proper_motions")
    p.add_argument("--max-sep-arcsec", type=float, default=0.3)
    p.add_argument("--dedup-tol-arcsec", type=float, default=0.10)
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir) / args.label
    outdir.mkdir(parents=True, exist_ok=True)
    dyr = (args.late_mjd - args.early_mjd) / 365.25
    print(f"Δt = {(args.late_mjd-args.early_mjd):.1f} d = {dyr:.3f} yr")

    print("Indexing per-frame catalogs by obs#...")
    groups = list_catalogs_via_meta(args.catalog_dir, {args.early_obs, args.late_obs})
    for k, v in groups.items():
        print(f"  obs{k}: {len(v)} catalogs")

    if not groups[args.early_obs] or not groups[args.late_obs]:
        raise SystemExit("Missing per-frame catalogs for one of the obs codes.")

    print("Stacking early...")
    e = load_obs_stack(groups[args.early_obs])
    print(f"  {len(e['ra'])} raw detections in obs{args.early_obs}")
    print("Stacking late...")
    l = load_obs_stack(groups[args.late_obs])
    print(f"  {len(l['ra'])} raw detections in obs{args.late_obs}")

    print(f"Dedup within obs (tol {args.dedup_tol_arcsec}\")...")
    e = median_position_dedup(e, args.dedup_tol_arcsec)
    l = median_position_dedup(l, args.dedup_tol_arcsec)
    print(f"  dedup: early n={len(e['ra'])}; late n={len(l['ra'])}")

    print(f"Cross-match early↔late (max sep {args.max_sep_arcsec}\")...")
    ie, il, seps = crossmatch(e, l, args.max_sep_arcsec)
    print(f"  matched {len(ie)} pairs; median sep={np.median(seps):.1f} mas")

    if len(ie) < 50:
        raise SystemExit("Too few matches for PM measurement.")

    dec_ref = float(np.median(e["dec"][ie]))
    pm_ra = (l["ra"][il] - e["ra"][ie]) * 3.6e6 * np.cos(np.deg2rad(dec_ref)) / dyr
    pm_dec = (l["dec"][il] - e["dec"][ie]) * 3.6e6 / dyr
    pm_tot = np.hypot(pm_ra, pm_dec)
    snr_e = e["flux"][ie] / e["eflux"][ie]
    snr_l = l["flux"][il] / l["eflux"][il]
    snr_min = np.minimum(snr_e, snr_l)

    # summary by S/N cut
    rows = []
    for s in (3, 5, 10, 30, 100):
        m = snr_min >= s
        if m.sum() < 10:
            continue
        rows.append(dict(
            snr_min=s, n=int(m.sum()),
            med_pm_ra=float(np.median(pm_ra[m])),
            med_pm_dec=float(np.median(pm_dec[m])),
            sigma_pm_ra=float(astats.mad_std(pm_ra[m])),
            sigma_pm_dec=float(astats.mad_std(pm_dec[m])),
            med_pm_tot=float(np.median(pm_tot[m])),
        ))
    summary = Table(rows=rows)
    for c in ["med_pm_ra", "med_pm_dec", "sigma_pm_ra", "sigma_pm_dec", "med_pm_tot"]:
        summary[c].unit = "mas/yr"
    summary.write(outdir / f"pm_summary_{args.label}.ecsv", overwrite=True)
    print()
    print("PM summary by S/N threshold:")
    summary.pprint(max_lines=-1, max_width=-1)

    # plots
    snr_cuts = (3, 5, 10, 30)
    fig, axes = plt.subplots(1, len(snr_cuts), figsize=(4 * len(snr_cuts), 4), sharex=True, sharey=True)
    for ax, s in zip(np.atleast_1d(axes), snr_cuts):
        m = snr_min >= s
        if m.sum() < 5: continue
        ax.hexbin(pm_ra[m], pm_dec[m], gridsize=80, bins="log",
                  extent=[-50, 50, -50, 50], cmap="viridis")
        med_r = np.median(pm_ra[m]); med_d = np.median(pm_dec[m])
        sig_r = astats.mad_std(pm_ra[m]); sig_d = astats.mad_std(pm_dec[m])
        ax.axhline(0, color="w", lw=0.5); ax.axvline(0, color="w", lw=0.5)
        ax.set_title(f"S/N≥{s}  n={m.sum()}\nmed=({med_r:+.2f},{med_d:+.2f}) σ=({sig_r:.2f},{sig_d:.2f}) mas/yr")
        ax.set_xlabel(r"$\mu_\alpha \cos\delta$ (mas/yr)")
    axes[0].set_ylabel(r"$\mu_\delta$ (mas/yr)")
    fig.suptitle(f"{args.label} VPD")
    fig.tight_layout()
    fig.savefig(outdir / f"vpd_{args.label}.png", dpi=120)
    plt.close(fig)

    # vector field
    fig, ax = plt.subplots(figsize=(9, 7))
    m = snr_min >= 10
    if m.sum() > 4000:
        idx = np.random.default_rng(0).choice(np.where(m)[0], 4000, replace=False)
        sel = np.zeros(len(m), dtype=bool); sel[idx] = True
    else:
        sel = m
    ra_mid = 0.5 * (e["ra"][ie] + l["ra"][il])
    dec_mid = 0.5 * (e["dec"][ie] + l["dec"][il])
    ax.quiver(ra_mid[sel], dec_mid[sel], pm_ra[sel], pm_dec[sel], pm_tot[sel],
              cmap="plasma", scale=400, width=0.0015, clim=(0, 30))
    ax.set_xlabel("RA (deg)"); ax.set_ylabel("Dec (deg)"); ax.invert_xaxis()
    ax.set_title(f"{args.label} PM vectors (S/N≥10, n={sel.sum()})")
    fig.colorbar(ax.collections[0], ax=ax, label="|PM| (mas/yr)")
    fig.tight_layout()
    fig.savefig(outdir / f"pmvectors_{args.label}.png", dpi=120)
    plt.close(fig)

    # save matched table
    out_tbl = Table({
        "ra_early": e["ra"][ie], "dec_early": e["dec"][ie],
        "ra_late": l["ra"][il], "dec_late": l["dec"][il],
        "pm_ra": pm_ra, "pm_dec": pm_dec, "pm_tot": pm_tot,
        "snr_early": snr_e, "snr_late": snr_l, "snr_min": snr_min,
        "sep_mas": seps,
    })
    for c in ("pm_ra", "pm_dec", "pm_tot"): out_tbl[c].unit = "mas/yr"
    out_tbl["sep_mas"].unit = "mas"
    out_tbl.write(outdir / f"pm_matches_{args.label}.ecsv", overwrite=True)
    print(f"\nWrote to {outdir}")


if __name__ == "__main__":
    main()
