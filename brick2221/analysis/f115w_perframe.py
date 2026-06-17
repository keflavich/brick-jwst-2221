#!/usr/bin/env python
"""Per-frame (per-exposure) F115W astrometry vs VVV-J and Gaia, clean matching.

Tests whether the non-Gaussian residual grouping comes from individual exposures
being offset from one another, and whether the offset is a per-detector (WCS /
distortion) effect or a per-dither (exposure) effect.  Clean matching = mutual
nearest neighbour + flux match (VVV-J) + separation clip, so the residuals are
real stars, not crowded-field mismatches.
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

warnings.simplefilter('ignore')
BASE = '/orange/adamginsburg/jwst/brick'
OUT = f'{BASE}/astrometry_diag/f115w_perframe'
os.makedirs(OUT, exist_ok=True)

F115W_EPOCH = 2022.70
DT = F115W_EPOCH - 2016.0
MATCH_SEP = 0.25 * u.arcsec     # generous for initial mutual match
CLEAN_SEP = 120.0               # mas; "true match" cut around per-frame median
FRAMEDIR = f'{BASE}/F115W'
VVV_CACHE = f'{BASE}/astrometry_analysis/reference_cache/basic_merged_photometry_tables_merged_vvv.fits'
GAIA_CACHE = f'{BASE}/astrometry_analysis/reference_cache/basic_merged_photometry_tables_merged_gaia.fits'


def farr(x):
    return np.asarray(np.ma.filled(np.ma.masked_invalid(np.asarray(x, dtype=float)), np.nan), dtype=float)


def load_refs():
    vvv = Table.read(VVV_CACHE)
    vsc = SkyCoord(vvv['skycoord']).icrs
    J = farr(vvv['J1ap3']); u2 = ~np.isfinite(J); J[u2] = farr(vvv['J2ap3'])[u2]
    g = Table.read(GAIA_CACHE)
    gra, gdec = farr(g['RA_ICRS']), farr(g['DE_ICRS'])
    pmra = np.where(np.isfinite(farr(g['pmRA'])), farr(g['pmRA']), 0.0)
    pmde = np.where(np.isfinite(farr(g['pmDE'])), farr(g['pmDE']), 0.0)
    gra_p = gra + (pmra * DT / 3.6e6) / np.cos(np.radians(gdec))
    gdec_p = gdec + (pmde * DT / 3.6e6)
    gsc = SkyCoord(gra_p * u.deg, gdec_p * u.deg, frame='icrs')
    return vsc, J, gsc, farr(g['Gmag'])


def mutual(csc, rsc, max_sep):
    ci, sep, _ = csc.match_to_catalog_sky(rsc)
    ri, _, _ = rsc.match_to_catalog_sky(csc)
    keep = (ri[ci] == np.arange(len(ci))) & (sep <= max_sep)
    idx = np.where(keep)[0]; rid = ci[keep]
    mc, mr = csc[idx], rsc[rid]
    dra = (mc.ra.deg - mr.ra.deg) * np.cos(np.radians(mc.dec.deg)) * 3.6e6
    ddec = (mc.dec.deg - mr.dec.deg) * 3.6e6
    return idx, rid, dra, ddec, sep[idx].to(u.mas).value


def per_frame(csc, m_inst, rsc, refmag=None, fluxmatch=False):
    """Return dict of clean per-frame offset stats + the clean dra/ddec arrays."""
    ci, ri, dra, ddec, sep = mutual(csc, rsc, MATCH_SEP)
    if len(ci) < 5:
        return None
    fin = np.ones(len(ci), bool)
    if fluxmatch and refmag is not None:
        rm = refmag[ri]; fin = np.isfinite(rm)
        if fin.sum() >= 5:
            dmag = m_inst[ci][fin] - rm[fin]
            med = np.nanmedian(dmag); s = max(stats.mad_std(dmag, ignore_nan=True), 0.3)
            keepmag = np.abs(dmag - med) <= 3 * s
            f2 = fin.copy(); f2[fin] = keepmag; fin = f2
    dra, ddec, sep = dra[fin], ddec[fin], sep[fin]
    if len(dra) < 5:
        return None
    # robust per-frame median, then sep-clip to true matches around it
    mdra0, mddec0 = np.nanmedian(dra), np.nanmedian(ddec)
    r = np.hypot(dra - mdra0, ddec - mddec0)
    clean = r <= CLEAN_SEP
    if clean.sum() < 4:
        clean = np.ones(len(dra), bool)
    dra, ddec = dra[clean], ddec[clean]
    return dict(n=len(dra), mdra=np.nanmedian(dra), mddec=np.nanmedian(ddec),
                madra=stats.mad_std(dra, ignore_nan=True), maddec=stats.mad_std(ddec, ignore_nan=True),
                dra=dra, ddec=ddec)


def main():
    vsc, J, gsc, Gmag = load_refs()
    frames = sorted(glob.glob(f'{FRAMEDIR}/f115w_*_visit*_exp*_daophot_basic.fits'))
    frames = [f for f in frames if not any(t in f for t in ('_m1','_m2','_m3','_m4','iter','resbgsub','_group'))]
    print(f"VVV {len(vsc)}({np.isfinite(J).sum()} J)  Gaia {len(gsc)}  frames {len(frames)}")

    rows = []
    pool = {'vvv': {'b': [], 'a': []}, 'gaia': {'b': [], 'a': []}}
    for fn in frames:
        t = Table.read(fn)
        sc = SkyCoord(t['skycoord_centroid']).icrs
        fl = farr(t['flux_fit'])
        ok = np.isfinite(sc.ra.deg) & np.isfinite(fl) & (fl > 0)
        sc, fl = sc[ok], fl[ok]
        mi = -2.5 * np.log10(fl)
        mod = t.meta['MODULE']; vis = t.meta['VISIT']
        exp = int(str(t.meta['EXPOSURE']).lstrip('_').lstrip('exp') or 0)
        row = dict(frame=os.path.basename(fn), module=mod, detector=mod, visit=vis, exposure=exp, nsrc=len(sc))
        v = per_frame(sc, mi, vsc, J, fluxmatch=True)
        if v:
            row.update(vvv_n=v['n'], vvv_dra=v['mdra'], vvv_ddec=v['mddec'], vvv_madra=v['madra'], vvv_maddec=v['maddec'])
            pool['vvv']['b'] += list(v['dra']); pool['vvv']['a'] += list(v['dra'] - v['mdra'])
        g = per_frame(sc, mi, gsc, Gmag, fluxmatch=False)
        if g:
            row.update(gaia_n=g['n'], gaia_dra=g['mdra'], gaia_ddec=g['mddec'], gaia_madra=g['madra'], gaia_maddec=g['maddec'])
            pool['gaia']['b'] += list(g['dra']); pool['gaia']['a'] += list(g['dra'] - g['mdra'])
        rows.append(row)
    tab = Table(rows)
    tab.write(f'{OUT}/per_frame_offsets.ecsv', overwrite=True)

    def mad(a):
        a = np.array(a, float); return stats.mad_std(a[np.isfinite(a)], ignore_nan=True)

    print("\n===== PER-FRAME OFFSET SPREAD (frame-to-frame, clean matches) =====")
    for lab, k in [('VVV-J', 'vvv'), ('Gaia', 'gaia')]:
        v = tab[~np.isnan(farr(tab[f'{k}_dra']))] if f'{k}_dra' in tab.colnames else tab[:0]
        if len(v) == 0: continue
        print(f"{lab}: {len(v)} frames | median |offset|=({np.nanmedian(farr(v[f'{k}_dra'])):.1f},{np.nanmedian(farr(v[f'{k}_ddec'])):.1f}) "
              f"| frame-to-frame spread MAD=({mad(v[f'{k}_dra']):.1f},{mad(v[f'{k}_ddec']):.1f}) "
              f"range=({np.nanmax(farr(v[f'{k}_dra']))-np.nanmin(farr(v[f'{k}_dra'])):.0f},"
              f"{np.nanmax(farr(v[f'{k}_ddec']))-np.nanmin(farr(v[f'{k}_ddec'])):.0f}) mas "
              f"| typ within-frame MAD=({np.nanmedian(farr(v[f'{k}_madra'])):.0f},{np.nanmedian(farr(v[f'{k}_maddec'])):.0f})")

    print("\n===== POOLED CLEAN RESIDUAL: before vs after per-frame shift =====")
    for lab, k in [('VVV-J', 'vvv'), ('Gaia', 'gaia')]:
        b, a = mad(pool[k]['b']), mad(pool[k]['a'])
        print(f"{lab}: N={len(pool[k]['b'])} | dRA MAD {b:.1f} -> {a:.1f} mas ({100*(1-a/b):.0f}% reduction)")

    print("\n===== OFFSET STRUCTURE: per-detector mean (averaged over dithers) =====")
    for lab, k in [('VVV-J', 'vvv')]:
        if f'{k}_dra' not in tab.colnames: continue
        for det in sorted(set(tab['detector'])):
            sub = tab[(tab['detector'] == det)]
            dr = farr(sub[f'{k}_dra']); dd = farr(sub[f'{k}_ddec'])
            m = np.isfinite(dr)
            if m.sum() == 0: continue
            print(f"  {det}: {m.sum()} dithers | mean offset=({np.nanmean(dr):.1f},{np.nanmean(dd):.1f}) "
                  f"| dither-scatter=({mad(dr):.1f},{mad(dd):.1f}) mas")

    # plots
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, (k, lab) in zip(axes, [('vvv', 'VVV-J'), ('gaia', 'Gaia')]):
        if f'{k}_dra' not in tab.colnames: continue
        dets = sorted(set(tab['detector']))
        cmap = plt.cm.tab10
        for i, det in enumerate(dets):
            sub = tab[tab['detector'] == det]
            ax.scatter(farr(sub[f'{k}_dra']), farr(sub[f'{k}_ddec']), s=25, color=cmap(i % 10), label=det, alpha=0.8)
        ax.axhline(0, color='0.6'); ax.axvline(0, color='0.6'); ax.set_aspect('equal')
        ax.set_xlabel('per-frame median dRA [mas]'); ax.set_ylabel('per-frame median dDec [mas]')
        ax.set_title(f'F115W per-frame offsets vs {lab} (color=detector)', fontsize=11)
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(f'{OUT}/per_frame_offset_by_detector.png', dpi=150, bbox_inches='tight')
    print(f"\nWrote {OUT}/per_frame_offsets.ecsv + per_frame_offset_by_detector.png")


if __name__ == '__main__':
    main()
