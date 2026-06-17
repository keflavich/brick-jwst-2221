#!/usr/bin/env python
"""Per-exposure, per-detector vetting of F115W vs VIRAC2 (Gaia-tied frame).

For each exposure (dither), one figure laying out the 8 NIRCam SW detectors
(nrca1-4, nrcb1-4), each panel showing the dRA/dDec residuals of the stars
finally INCLUDED for astrometric referencing (mutual NN match to VIRAC2 +
J-band flux match + separation clip).  A companion figure per exposure shows
the pooled dRA & dDec histograms with a Gaussian fit, demonstrating the
included-star residuals are ~Gaussian.

F115W is short-wave only, so there is no LW-detector figure here; the same code
makes the 2-LW-detector layout (nrcalong/nrcblong) for LW filters.

Output dir: astrometry_diag/f115w_perdetector_vetting/
"""
import os, glob, re, warnings
import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm

warnings.simplefilter('ignore')
BASE = '/orange/adamginsburg/jwst/brick'
OUT = f'{BASE}/astrometry_diag/f115w_perdetector_vetting'
os.makedirs(OUT, exist_ok=True)
EPOCH = 2022.70
FRAMEDIR = f'{BASE}/F115W'
SW_LAYOUT = [['nrcb1', 'nrcb2', 'nrca2', 'nrca1'],
             ['nrcb4', 'nrcb3', 'nrca3', 'nrca4']]  # approximate SW focal-plane layout
MATCH_SEP = 0.25 * u.arcsec
CLEAN = 120.0  # mas


def farr(x):
    return np.asarray(np.ma.filled(np.ma.masked_invalid(np.asarray(x, float)), np.nan), float)


def prop(ra, dec, pmra, pmde, dt):
    pmra = np.where(np.isfinite(pmra), pmra, 0.); pmde = np.where(np.isfinite(pmde), pmde, 0.)
    return ra + (pmra * dt / 3.6e6) / np.cos(np.radians(dec)), dec + (pmde * dt / 3.6e6)


def load_virac():
    v = Table.read(f'{BASE}/astrometry_diag/refcache/virac2.fits')
    ra, dec = prop(farr(v['RAJ2000']), farr(v['DEJ2000']), farr(v['pmRA']), farr(v['pmDE']), EPOCH - 2016.0)
    return SkyCoord(ra * u.deg, dec * u.deg), farr(v['Jmag'])


def included_resid(catfn, vsc, vJ):
    """Return dRA, dDec (mas) for the stars finally included (mutual NN + Jflux + sep clip)."""
    t = Table.read(catfn)
    sc = SkyCoord(t['skycoord_centroid']).icrs
    fl = farr(t['flux_fit']); ok = np.isfinite(sc.ra.deg) & np.isfinite(fl) & (fl > 0)
    sc, fl = sc[ok], fl[ok]; mi = -2.5 * np.log10(fl)
    ci, s, _ = sc.match_to_catalog_sky(vsc); ri, _, _ = vsc.match_to_catalog_sky(sc)
    k = (ri[ci] == np.arange(len(ci))) & (s <= MATCH_SEP)
    if k.sum() < 3:
        return np.array([]), np.array([])
    idx, rid = np.where(k)[0], ci[k]
    dra = (sc[idx].ra.deg - vsc[rid].ra.deg) * np.cos(np.radians(sc[idx].dec.deg)) * 3.6e6
    dd = (sc[idx].dec.deg - vsc[rid].dec.deg) * 3.6e6
    Jm = vJ[rid]; mi2 = mi[idx]; fin = np.isfinite(Jm) & np.isfinite(mi2)
    if fin.sum() >= 5:
        dm = mi2[fin] - Jm[fin]; med = np.nanmedian(dm); sg = max(stats.mad_std(dm, ignore_nan=True), 0.3)
        keep = fin.copy(); keep[fin] = np.abs(dm - med) <= 3 * sg
    else:
        keep = np.ones(len(idx), bool)
    dra, dd = dra[keep], dd[keep]
    m, md = np.nanmedian(dra), np.nanmedian(dd)
    cl = np.hypot(dra - m, dd - md) <= CLEAN
    return dra[cl], dd[cl]


def frame_for(det, exp):
    g = glob.glob(f'{FRAMEDIR}/f115w_{det}_visit*_exp{exp:05d}_daophot_basic.fits')
    g = [f for f in g if not any(t in f for t in ('_m1', '_m2', '_m3', '_m4', 'iter', 'resbgsub', '_group'))]
    return g[0] if g else None


def main():
    vsc, vJ = load_virac()
    exposures = sorted({int(re.search(r'_exp(\d+)_', os.path.basename(f)).group(1))
                        for f in glob.glob(f'{FRAMEDIR}/f115w_nrc*_visit*_exp*_daophot_basic.fits')
                        if not any(t in f for t in ('_m1', '_m2', '_m3', '_m4', 'iter', 'resbgsub', '_group'))})
    print(f"{len(exposures)} exposures; VIRAC2 {len(vsc)} ({np.isfinite(vJ).sum()} J)")
    lim = 100
    pooled_all = {'dra': [], 'ddec': []}
    for exp in exposures:
        fig, axes = plt.subplots(2, 4, figsize=(15, 8))
        exp_dra, exp_ddec = [], []
        for r in range(2):
            for c in range(4):
                det = SW_LAYOUT[r][c]; ax = axes[r][c]
                fn = frame_for(det, exp)
                if fn is None:
                    ax.text(0.5, 0.5, f'{det}\n(no frame)', ha='center', va='center', transform=ax.transAxes); ax.set_xticks([]); ax.set_yticks([]); continue
                dra, dd = included_resid(fn, vsc, vJ)
                exp_dra += list(dra); exp_ddec += list(dd)
                ax.scatter(dra, dd, s=6, alpha=0.4, color='k')
                ax.axhline(0, color='0.7', lw=0.8); ax.axvline(0, color='0.7', lw=0.8)
                if len(dra) >= 3:
                    ax.plot(np.median(dra), np.median(dd), 'r+', ms=14, mew=2)
                ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
                ax.set_title(f"{det}  N={len(dra)}\nmed=({np.median(dra):.1f},{np.median(dd):.1f})" if len(dra) else f"{det} N=0", fontsize=9)
                if r == 1: ax.set_xlabel('dRA [mas]')
                if c == 0: ax.set_ylabel('dDec [mas]')
        fig.suptitle(f'F115W exposure {exp:02d} vs VIRAC2 — included reference stars, per SW detector', fontsize=13)
        fig.tight_layout()
        fig.savefig(f'{OUT}/exp{exp:02d}_8SWdet_dradec.png', dpi=130, bbox_inches='tight')
        plt.close(fig)
        # per-exposure Gaussian histogram of pooled included residuals
        if len(exp_dra) > 10:
            fig, axs = plt.subplots(1, 2, figsize=(11, 4))
            for ax, arr, lab in [(axs[0], np.array(exp_dra), 'dRA'), (axs[1], np.array(exp_ddec), 'dDec')]:
                mu, sg = np.median(arr), stats.mad_std(arr)
                bins = np.linspace(-lim, lim, 50)
                ax.hist(arr, bins=bins, density=True, histtype='stepfilled', color='0.6', alpha=0.7)
                xs = np.linspace(-lim, lim, 200)
                ax.plot(xs, norm.pdf(xs, mu, sg), 'r-', lw=2, label=f'Gauss μ={mu:.1f} σ={sg:.1f}')
                ax.set_xlabel(f'{lab} [mas]'); ax.legend(fontsize=9); ax.set_title(f'exp{exp:02d} {lab}  N={len(arr)}')
            fig.tight_layout(); fig.savefig(f'{OUT}/exp{exp:02d}_gaussian.png', dpi=130, bbox_inches='tight'); plt.close(fig)
        pooled_all['dra'] += exp_dra; pooled_all['ddec'] += exp_ddec
        print(f"  exp{exp:02d}: {len(exp_dra)} included stars")

    # global Gaussian summary (all exposures, all SW detectors)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, arr, lab in [(axs[0], np.array(pooled_all['dra']), 'dRA'), (axs[1], np.array(pooled_all['ddec']), 'dDec')]:
        mu, sg = np.median(arr), stats.mad_std(arr)
        bins = np.linspace(-lim, lim, 60)
        ax.hist(arr, bins=bins, density=True, histtype='stepfilled', color='steelblue', alpha=0.6)
        xs = np.linspace(-lim, lim, 200)
        ax.plot(xs, norm.pdf(xs, mu, sg), 'r-', lw=2, label=f'Gauss μ={mu:.1f} σ={sg:.1f} mas')
        ax.axvline(mu, color='r', ls='--', lw=1)
        ax.set_xlabel(f'{lab} [mas]'); ax.legend(); ax.set_title(f'F115W ALL included ref stars: {lab}  N={len(arr)}')
    fig.suptitle('F115W vs VIRAC2 — pooled included-reference-star residuals (Gaussianity check)', fontsize=13)
    fig.tight_layout(); fig.savefig(f'{OUT}/ALL_gaussian_summary.png', dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f"\nWrote per-exposure + Gaussian plots to {OUT}")
    print(f"Pooled included stars: {len(pooled_all['dra'])}; "
          f"dRA med={np.median(pooled_all['dra']):.1f} σ={stats.mad_std(pooled_all['dra']):.1f}; "
          f"dDec med={np.median(pooled_all['ddec']):.1f} σ={stats.mad_std(pooled_all['ddec']):.1f} mas")


if __name__ == '__main__':
    main()
