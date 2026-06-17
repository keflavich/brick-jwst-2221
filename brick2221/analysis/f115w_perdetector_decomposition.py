#!/usr/bin/env python
"""Per-detector vs per-exposure offset decomposition for F115W vs VIRAC2.

Question: does the CRDS distortion + alignment solution put all 8 SW detectors on
a common per-exposure offset, or is there a per-detector systematic?  Re-measured
on the NEW (seed-reduced) crf to see whether the ~40 mas Dec per-detector spread
seen on the old crf shrinks (=> it was a TweakReg residual) or persists
(=> distortion-calibration signal).

Writes a text report to astrometry_diag/f115w_perdetector_decomposition_newcrf.txt
"""
import glob, re, warnings
import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import stats

warnings.simplefilter('ignore')
BASE = '/orange/adamginsburg/jwst/brick'
EPOCH = 2022.70
FRAMEDIR = f'{BASE}/F115W'
OUT = f'{BASE}/astrometry_diag/f115w_perdetector_decomposition_newcrf.txt'


def farr(x):
    return np.asarray(np.ma.filled(np.ma.masked_invalid(np.asarray(x, float)), np.nan), float)


def prop(ra, dec, pmra, pmde, dt):
    pmra = np.where(np.isfinite(pmra), pmra, 0.); pmde = np.where(np.isfinite(pmde), pmde, 0.)
    return ra + (pmra * dt / 3.6e6) / np.cos(np.radians(dec)), dec + (pmde * dt / 3.6e6)


v = Table.read(f'{BASE}/astrometry_diag/refcache/virac2.fits')
ra, dec = prop(farr(v['RAJ2000']), farr(v['DEJ2000']), farr(v['pmRA']), farr(v['pmDE']), EPOCH - 2016.0)
virac = SkyCoord(ra * u.deg, dec * u.deg); vJ = farr(v['Jmag'])

frames = sorted(glob.glob(f'{FRAMEDIR}/f115w_*_visit*_exp*_daophot_basic.fits'))
frames = [f for f in frames if not any(t in f for t in ('_m1', '_m2', '_m3', '_m4', 'iter', 'resbgsub', '_group'))]

rows = []
for fn in frames:
    t = Table.read(fn)
    sc = SkyCoord(t['skycoord_centroid']).icrs
    fl = farr(t['flux_fit']); ok = np.isfinite(sc.ra.deg) & np.isfinite(fl) & (fl > 0)
    sc, fl = sc[ok], fl[ok]; mi = -2.5 * np.log10(fl)
    ci, s, _ = sc.match_to_catalog_sky(virac); ri, _, _ = virac.match_to_catalog_sky(sc)
    k = (ri[ci] == np.arange(len(ci))) & (s <= 0.25 * u.arcsec)
    if k.sum() < 8:
        continue
    idx, rid = np.where(k)[0], ci[k]
    dra = (sc[idx].ra.deg - virac[rid].ra.deg) * np.cos(np.radians(sc[idx].dec.deg)) * 3.6e6
    dd = (sc[idx].dec.deg - virac[rid].dec.deg) * 3.6e6
    Jm = vJ[rid]; fin = np.isfinite(Jm) & np.isfinite(mi[idx])
    if fin.sum() >= 5:
        dm = mi[idx][fin] - Jm[fin]; med = np.nanmedian(dm); sg = max(stats.mad_std(dm, ignore_nan=True), 0.3)
        keep = fin.copy(); keep[fin] = np.abs(dm - med) <= 3 * sg
    else:
        keep = np.ones(len(idx), bool)
    dra, dd = dra[keep], dd[keep]
    m, md = np.nanmedian(dra), np.nanmedian(dd); cl = np.hypot(dra - m, dd - md) <= 120
    dra, dd = dra[cl], dd[cl]; n = len(dra)
    if n < 5: continue
    mr = re.search(r'(nrc[ab]\d)_visit(\d+)_vgroup\d+_exp(\d+)', fn)
    rows.append(dict(det=mr.group(1), vis=int(mr.group(2)), exp=int(mr.group(3)), n=n,
                     dra=float(np.nanmedian(dra)), ddec=float(np.nanmedian(dd)),
                     sem_ra=float(stats.mad_std(dra) / np.sqrt(n)), sem_dec=float(stats.mad_std(dd) / np.sqrt(n)),
                     scat_ra=float(stats.mad_std(dra)), scat_dec=float(stats.mad_std(dd))))

t = Table(rows)
det = np.array(t['det']); dra = np.array(t['dra']); dd = np.array(t['ddec'])
expid = np.array([f"{v}_{e}" for v, e in zip(t['vis'], t['exp'])])
dets = sorted(set(det))

L = []
L.append("F115W per-detector vs per-exposure offset decomposition (NEW seed-reduced crf), vs VIRAC2")
L.append(f"frames used: {len(t)}; detectors: {len(dets)}; exposures: {len(set(expid))}")
L.append(f"per-frame measurement SEM: median dRA={np.median(t['sem_ra']):.2f} dDec={np.median(t['sem_dec']):.2f} mas; "
         f"median within-frame scatter ({np.median(t['scat_ra']):.0f},{np.median(t['scat_dec']):.0f}) mas; median Nmatch={np.median(t['n']):.0f}")
wra = [np.nanstd(dra[expid == e]) for e in set(expid) if (expid == e).sum() >= 4]
wdd = [np.nanstd(dd[expid == e]) for e in set(expid) if (expid == e).sum() >= 4]
L.append(f"\n[within-exposure across-detector spread] median std: dRA={np.median(wra):.1f} dDec={np.median(wdd):.1f} mas")
L.append("\n[per-detector mean offset (avg over exposures)]  +/- scatter over exposures:")
pdra, pddd = {}, {}
for d in dets:
    m = det == d; pdra[d] = np.nanmean(dra[m]); pddd[d] = np.nanmean(dd[m])
    L.append(f"  {d}: ({pdra[d]:6.1f},{pddd[d]:6.1f})  scatter ({np.nanstd(dra[m]):.1f},{np.nanstd(dd[m]):.1f})  nexp={m.sum()}")
ar, ad = np.array(list(pdra.values())), np.array(list(pddd.values()))
L.append(f"  -> detector-to-detector spread of means: dRA std={np.std(ar):.1f} (range {ar.max()-ar.min():.0f}), "
         f"dDec std={np.std(ad):.1f} (range {ad.max()-ad.min():.0f}) mas")
era = [np.nanmean(dra[expid == e]) for e in set(expid)]
edd = [np.nanmean(dd[expid == e]) for e in set(expid)]
L.append(f"\n[per-exposure mean (avg over detectors)] spread: dRA std={np.nanstd(era):.1f} dDec std={np.nanstd(edd):.1f} mas")
L.append("\nVERDICT: compare detector-to-detector spread vs per-exposure spread vs measurement SEM.")
L.append("  - if det-to-det >> SEM and >> per-exposure -> real per-detector systematic (distortion/placement); keep per-detector.")
L.append("  - if det-to-det ~ SEM -> detectors agree; pool 8 (SW)/2 (LW) per exposure for S/N.")
L.append("  vs OLD crf (for comparison): within-exp (7.9,14.0); det-to-det range (23,42) mas; per-exp (4.5,4.7).")
open(OUT, 'w').write("\n".join(L) + "\n")
print("\n".join(L))
print(f"\nWrote {OUT}")
