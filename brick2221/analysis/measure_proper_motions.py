#!/usr/bin/env python
"""Measure proper motions in spatial overlap regions between two iter3 catalogs.

Cross-matches sources by sky position in the overlap footprint, computes
PM = (pos_late - pos_early) / Δt, plots PM distributions + vector field at
several S/N thresholds.

Default: sgra (2022 epoch) vs gc2211 (2023-2025 epoch). Other field pairs
can be passed via --early/--late + their filter columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy import stats as astats
from astropy.coordinates import SkyCoord
from astropy.table import Table


def load_catalog(path: str, filter_name: str) -> dict:
    """Load merged iter3 catalog, extract per-filter position + flux columns."""
    t = Table.read(path)
    f = filter_name.lower()
    sc = t[f"skycoord_{f}"]
    flux = np.asarray(t[f"flux_jy_{f}"], dtype=float)
    eflux = np.asarray(t[f"eflux_jy_{f}"], dtype=float)
    qfit_col = f"qfit_{f}"
    qfit = np.asarray(t[qfit_col], dtype=float) if qfit_col in t.colnames else np.full(len(t), np.nan)
    ra = sc.ra.deg
    dec = sc.dec.deg
    ok = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(flux) & np.isfinite(eflux) & (flux > 0) & (eflux > 0)
    return dict(
        ra=ra[ok], dec=dec[ok], flux=flux[ok], eflux=eflux[ok],
        qfit=qfit[ok], snr=flux[ok] / eflux[ok],
    )


def crossmatch(early: dict, late: dict, max_sep_arcsec: float = 0.3):
    """One-way nearest neighbor crossmatch. Drop bidirectionally inconsistent matches."""
    sc_e = SkyCoord(early["ra"] * u.deg, early["dec"] * u.deg, frame="icrs")
    sc_l = SkyCoord(late["ra"] * u.deg, late["dec"] * u.deg, frame="icrs")
    idx_l, sep, _ = sc_e.match_to_catalog_sky(sc_l)
    keep = sep.to(u.arcsec).value < max_sep_arcsec
    # bidirectional
    idx_e2, sep2, _ = sc_l.match_to_catalog_sky(sc_e)
    bi = np.arange(len(sc_e)) == idx_e2[idx_l]
    keep &= bi
    return np.where(keep)[0], idx_l[keep], sep[keep].to(u.mas).value


def compute_pm(early: dict, late: dict, ie, il, delta_years: float, dec_ref_deg: float):
    """Return dict of arrays for matched pairs."""
    ra_e = early["ra"][ie]
    ra_l = late["ra"][il]
    dec_e = early["dec"][ie]
    dec_l = late["dec"][il]
    cos_dec = np.cos(np.deg2rad(dec_ref_deg))
    # mas/yr; multiply ra by cos(dec) for projection
    pm_ra_cosdec = (ra_l - ra_e) * 3.6e6 * cos_dec / delta_years
    pm_dec = (dec_l - dec_e) * 3.6e6 / delta_years
    pm_tot = np.hypot(pm_ra_cosdec, pm_dec)
    return dict(
        ra_mid=(ra_e + ra_l) / 2,
        dec_mid=(dec_e + dec_l) / 2,
        pm_ra=pm_ra_cosdec,
        pm_dec=pm_dec,
        pm_tot=pm_tot,
        snr_e=early["snr"][ie],
        snr_l=late["snr"][il],
        snr_min=np.minimum(early["snr"][ie], late["snr"][il]),
    )


def plot_pm(pm: dict, outdir: Path, label: str, snr_thresholds=(3, 5, 10, 30)):
    outdir.mkdir(parents=True, exist_ok=True)
    pmra, pmdec = pm["pm_ra"], pm["pm_dec"]
    snrmin = pm["snr_min"]

    # ----- PM scatter (vector point diagram) per S/N cut -----
    fig, axes = plt.subplots(1, len(snr_thresholds), figsize=(4 * len(snr_thresholds), 4),
                             sharex=True, sharey=True)
    for ax, s in zip(np.atleast_1d(axes), snr_thresholds):
        m = snrmin >= s
        ax.hexbin(pmra[m], pmdec[m], gridsize=80, bins="log", extent=[-50, 50, -50, 50], cmap="viridis")
        med_ra = np.nanmedian(pmra[m])
        med_dec = np.nanmedian(pmdec[m])
        sig_ra = astats.mad_std(pmra[m])
        sig_dec = astats.mad_std(pmdec[m])
        ax.axhline(0, color="w", lw=0.5)
        ax.axvline(0, color="w", lw=0.5)
        ax.set_title(f"S/N≥{s}  n={m.sum()}\nmed=({med_ra:+.2f},{med_dec:+.2f}) σ=({sig_ra:.2f},{sig_dec:.2f}) mas/yr")
        ax.set_xlabel(r"$\mu_\alpha \cos\delta$ (mas/yr)")
    axes[0].set_ylabel(r"$\mu_\delta$ (mas/yr)")
    fig.suptitle(f"{label} VPD")
    fig.tight_layout()
    fig.savefig(outdir / f"vpd_{label}.png", dpi=120)
    plt.close(fig)

    # ----- PM histogram -----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    bins = np.linspace(-30, 30, 121)
    for s in snr_thresholds:
        m = snrmin >= s
        axes[0].hist(pmra[m], bins=bins, histtype="step", label=f"S/N≥{s} (n={m.sum()})")
        axes[1].hist(pmdec[m], bins=bins, histtype="step", label=f"S/N≥{s} (n={m.sum()})")
    axes[0].set_xlabel(r"$\mu_\alpha \cos\delta$ (mas/yr)"); axes[0].set_ylabel("N")
    axes[1].set_xlabel(r"$\mu_\delta$ (mas/yr)")
    for ax in axes:
        ax.legend(fontsize=8); ax.axvline(0, color="k", lw=0.5)
    fig.suptitle(f"{label} PM histograms")
    fig.tight_layout()
    fig.savefig(outdir / f"pmhist_{label}.png", dpi=120)
    plt.close(fig)

    # ----- PM vector field on sky (S/N cut) -----
    fig, ax = plt.subplots(figsize=(9, 7))
    cut = max(snr_thresholds[:-1]) if len(snr_thresholds) > 1 else snr_thresholds[0]
    m = snrmin >= cut
    # subsample for plotting
    if m.sum() > 5000:
        idx = np.random.default_rng(0).choice(np.where(m)[0], 5000, replace=False)
        sel = np.zeros(len(m), dtype=bool); sel[idx] = True
    else:
        sel = m
    ax.quiver(pm["ra_mid"][sel], pm["dec_mid"][sel],
              pmra[sel], pmdec[sel], pm["pm_tot"][sel],
              cmap="plasma", scale=400, width=0.0015, clim=(0, 30))
    ax.set_xlabel("RA (deg)"); ax.set_ylabel("Dec (deg)")
    ax.set_title(f"{label} PM vectors (S/N≥{cut}, n={sel.sum()})")
    ax.invert_xaxis()
    fig.colorbar(ax.collections[0], ax=ax, label="|PM| (mas/yr)")
    fig.tight_layout()
    fig.savefig(outdir / f"pmvectors_{label}.png", dpi=120)
    plt.close(fig)


def summarize_pm(pm: dict, snr_thresholds=(3, 5, 10, 30)) -> Table:
    rows = []
    for s in snr_thresholds:
        m = pm["snr_min"] >= s
        if m.sum() < 10:
            continue
        rows.append(dict(
            snr_min=s,
            n=int(m.sum()),
            med_pm_ra=float(np.nanmedian(pm["pm_ra"][m])),
            med_pm_dec=float(np.nanmedian(pm["pm_dec"][m])),
            sigma_pm_ra=float(astats.mad_std(pm["pm_ra"][m])),
            sigma_pm_dec=float(astats.mad_std(pm["pm_dec"][m])),
            med_pm_tot=float(np.nanmedian(pm["pm_tot"][m])),
        ))
    out = Table(rows=rows)
    for c in ["med_pm_ra", "med_pm_dec", "sigma_pm_ra", "sigma_pm_dec", "med_pm_tot"]:
        out[c].unit = "mas/yr"
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--early-catalog", required=True)
    p.add_argument("--late-catalog", required=True)
    p.add_argument("--early-filter", required=True, help="Filter column suffix in early catalog (e.g. f212n)")
    p.add_argument("--late-filter", required=True)
    p.add_argument("--early-mjd", type=float, required=True)
    p.add_argument("--late-mjd", type=float, required=True)
    p.add_argument("--label", required=True, help="Output label for plots/tables")
    p.add_argument("--outdir", default="/blue/adamginsburg/adamginsburg/jwst/proper_motions")
    p.add_argument("--max-sep-arcsec", type=float, default=0.3)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir) / args.label
    delta_days = args.late_mjd - args.early_mjd
    delta_years = delta_days / 365.25
    print(f"Δt = {delta_days:.1f} d = {delta_years:.3f} yr")

    print(f"Loading early catalog: {args.early_catalog} [{args.early_filter}]")
    early = load_catalog(args.early_catalog, args.early_filter)
    print(f"  {len(early['ra'])} valid sources")
    print(f"Loading late catalog: {args.late_catalog} [{args.late_filter}]")
    late = load_catalog(args.late_catalog, args.late_filter)
    print(f"  {len(late['ra'])} valid sources")

    # crossmatch handles spatial overlap directly; just log bbox
    print(f"early RA [{early['ra'].min():.4f},{early['ra'].max():.4f}] Dec [{early['dec'].min():.4f},{early['dec'].max():.4f}]")
    print(f"late  RA [{late['ra'].min():.4f},{late['ra'].max():.4f}] Dec [{late['dec'].min():.4f},{late['dec'].max():.4f}]")

    ie, il, seps = crossmatch(early, late, args.max_sep_arcsec)
    print(f"Matched {len(ie)} pairs (max sep {args.max_sep_arcsec}\")  median sep={np.median(seps):.1f} mas")

    dec_ref = float(np.median(early["dec"][ie])) if len(ie) else 0.0
    pm = compute_pm(early, late, ie, il, delta_years, dec_ref)
    pm["sep_mas"] = seps

    outdir.mkdir(parents=True, exist_ok=True)
    pm_tbl = Table({k: v for k, v in pm.items()})
    for c in ["pm_ra", "pm_dec", "pm_tot"]:
        pm_tbl[c].unit = "mas/yr"
    pm_tbl["sep_mas"].unit = "mas"
    pm_tbl.write(outdir / f"pm_matches_{args.label}.ecsv", overwrite=True)

    summary = summarize_pm(pm)
    summary.write(outdir / f"pm_summary_{args.label}.ecsv", overwrite=True)
    print()
    print("PM summary by S/N threshold:")
    summary.pprint(max_lines=-1, max_width=-1)

    plot_pm(pm, outdir, args.label)
    print(f"\nWrote plots + tables to {outdir}")


if __name__ == "__main__":
    main()
