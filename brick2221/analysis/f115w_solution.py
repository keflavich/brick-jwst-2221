#!/usr/bin/env python
"""Build & verify the F115W Gaia-frame astrometric solution.

(1) Saturated-star anchors: are JWST F115W saturated-star fits astrometrically
    good (cross-frame repeatability, absolute tie to Gaia/VIRAC2)?
(2) Per-frame alignment to VIRAC2 (dense, Gaia frame): per-frame shift precision.
(3) Rigid re-anchor of the merged catalog to the Gaia frame; verify vs Gaia,
    GSC242, VIRAC2 -- the accuracy budget for NIRSpec pointing.
"""
import os, glob, warnings
import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import stats

warnings.simplefilter('ignore')
BASE = '/orange/adamginsburg/jwst/brick'
EPOCH = 2022.70
FRAMEDIR = f'{BASE}/F115W'
PIPE = f'{BASE}/F115W/pipeline'
MERGED = f'{BASE}/catalogs/f115w_merged_indivexp_merged_dao_basic.fits'


def farr(x):
    return np.asarray(np.ma.filled(np.ma.masked_invalid(np.asarray(x, float)), np.nan), float)


def prop(ra, dec, pmra, pmde, dt):
    pmra = np.where(np.isfinite(pmra), pmra, 0.0); pmde = np.where(np.isfinite(pmde), pmde, 0.0)
    return ra + (pmra * dt / 3.6e6) / np.cos(np.radians(dec)), dec + (pmde * dt / 3.6e6)


def load():
    v = Table.read(f'{BASE}/astrometry_diag/refcache/virac2.fits')
    ra, dec = prop(farr(v['RAJ2000']), farr(v['DEJ2000']), farr(v['pmRA']), farr(v['pmDE']), EPOCH - 2016.0)
    virac = SkyCoord(ra * u.deg, dec * u.deg)
    g = Table.read(f'{BASE}/astrometry_analysis/reference_cache/basic_merged_photometry_tables_merged_gaia.fits')
    ra, dec = prop(farr(g['RA_ICRS']), farr(g['DE_ICRS']), farr(g['pmRA']), farr(g['pmDE']), EPOCH - 2016.0)
    gaia = SkyCoord(ra * u.deg, dec * u.deg)
    gsc = Table.read(f'{BASE}/astrometry_diag/refcache/gsc242.fits')
    ep = farr(gsc['Epoch']); ep = np.where(np.isfinite(ep), ep, 2016.0)
    ra, dec = prop(farr(gsc['RA_ICRS']), farr(gsc['DE_ICRS']), farr(gsc['pmRA']), farr(gsc['pmDE']), EPOCH - ep)
    gscsc = SkyCoord(ra * u.deg, dec * u.deg)
    return virac, gaia, gscsc


def off(a, b, sep=0.3 * u.arcsec, clip=150):
    ci, s, _ = a.match_to_catalog_sky(b); ri, _, _ = b.match_to_catalog_sky(a)
    k = (ri[ci] == np.arange(len(ci))) & (s <= sep)
    if k.sum() < 5: return None
    mc, mr = a[np.where(k)[0]], b[ci[k]]
    dra = (mc.ra.deg - mr.ra.deg) * np.cos(np.radians(mc.dec.deg)) * 3.6e6
    dd = (mc.dec.deg - mr.dec.deg) * 3.6e6
    m, md = np.nanmedian(dra), np.nanmedian(dd)
    cl = np.hypot(dra - m, dd - md) <= clip
    if cl.sum() >= 5: dra, dd = dra[cl], dd[cl]
    n = len(dra)
    return dict(N=n, dRA=float(np.nanmedian(dra)), dDec=float(np.nanmedian(dd)),
                madRA=float(stats.mad_std(dra, ignore_nan=True)), madDec=float(stats.mad_std(dd, ignore_nan=True)),
                semRA=float(stats.mad_std(dra, ignore_nan=True)/np.sqrt(n)),
                semDec=float(stats.mad_std(dd, ignore_nan=True)/np.sqrt(n)))


def main():
    virac, gaia, gsc = load()

    # ---------- (1) SATURATED STARS ----------
    print("===== (1) SATURATED-STAR ANCHORS =====")
    satfns = sorted(glob.glob(f'{PIPE}/*_destreak_o004_crf_satstar_catalog.fits'))
    sat_sc, sat_key = [], []
    for fn in satfns:
        t = Table.read(fn)
        if 'skycoord_fit' not in t.colnames or len(t) == 0: continue
        sc = SkyCoord(t['skycoord_fit']).icrs
        good = np.isfinite(sc.ra.deg)
        # flags==0 preferred
        if 'flags' in t.colnames:
            good &= (farr(t['flags']) == 0) | ~np.isfinite(farr(t['flags']))
        for s in sc[good]:
            sat_sc.append(s)
    if sat_sc:
        sat = SkyCoord(sat_sc)
        print(f"  {len(satfns)} frames, {len(sat)} saturated-star detections (flags ok)")
        # cross-frame repeatability: cluster detections of the same star (<0.15")
        # measure scatter within clusters
        idx, sep, _ = sat.match_to_catalog_sky(sat, nthneighbor=2)
        print(f"  nearest-other-detection sep median={np.nanmedian(sep.to(u.mas).value):.1f} mas "
              f"(same bright star seen in overlapping frames)")
        for rk, r in [('VIRAC2', virac), ('Gaia', gaia), ('GSC242', gsc)]:
            o = off(sat, r, sep=0.4*u.arcsec)
            if o: print(f"  satstar vs {rk}: N={o['N']} med=({o['dRA']:.1f},{o['dDec']:.1f}) "
                        f"MAD=({o['madRA']:.1f},{o['madDec']:.1f}) mas")

    # ---------- (2) PER-FRAME ALIGNMENT TO VIRAC2 ----------
    print("\n===== (2) PER-FRAME F115W ALIGNMENT TO VIRAC2 (dense, Gaia frame) =====")
    frames = sorted(glob.glob(f'{FRAMEDIR}/f115w_*_visit*_exp*_daophot_basic.fits'))
    frames = [f for f in frames if not any(t in f for t in ('_m1','_m2','_m3','_m4','iter','resbgsub','_group'))]
    shifts = []
    for fn in frames:
        t = Table.read(fn)
        sc = SkyCoord(t['skycoord_centroid']).icrs
        fl = farr(t['flux_fit']); ok = np.isfinite(sc.ra.deg) & np.isfinite(fl) & (fl > 0)
        o = off(sc[ok], virac, sep=0.25*u.arcsec)
        if o:
            shifts.append((o['dRA'], o['dDec'], o['N'], o['semRA'], o['semDec'], t.meta['MODULE']))
    sh = np.array([(a, b, n, sr, sd) for a, b, n, sr, sd, m in shifts])
    print(f"  {len(shifts)} frames aligned | per-frame match count median={np.median(sh[:,2]):.0f}")
    print(f"  per-frame shift precision (SEM) median=({np.median(sh[:,3]):.1f},{np.median(sh[:,4]):.1f}) mas")
    print(f"  per-frame shift spread (frame-to-frame) MAD=({stats.mad_std(sh[:,0]):.1f},{stats.mad_std(sh[:,1]):.1f}) mas")
    print(f"  mean per-frame shift to VIRAC2=({np.mean(sh[:,0]):.1f},{np.mean(sh[:,1]):.1f}) mas")

    # ---------- (3) RIGID RE-ANCHOR OF MERGED CATALOG ----------
    print("\n===== (3) MERGED CATALOG: current vs re-anchored to Gaia frame =====")
    t = Table.read(MERGED)
    sc = SkyCoord(t['skycoord']).icrs
    o_v = off(sc, virac)
    print(f"  current F115W vs VIRAC2: med=({o_v['dRA']:.1f},{o_v['dDec']:.1f}) sem=({o_v['semRA']:.2f},{o_v['semDec']:.2f}) mas (N={o_v['N']})")
    # apply rigid shift = -median(vs VIRAC2)
    dra_fix, ddec_fix = o_v['dRA'], o_v['dDec']
    ra2 = sc.ra.deg - (dra_fix / 3.6e6) / np.cos(np.radians(sc.dec.deg))
    dec2 = sc.dec.deg - ddec_fix / 3.6e6
    sc2 = SkyCoord(ra2 * u.deg, dec2 * u.deg)
    print(f"  applying rigid shift (-{dra_fix:.1f}, -{ddec_fix:.1f}) mas; verify:")
    for rk, r in [('VIRAC2', virac), ('Gaia', gaia), ('GSC242', gsc)]:
        o = off(sc2, r)
        if o: print(f"    re-anchored vs {rk}: med=({o['dRA']:.1f},{o['dDec']:.1f}) |{np.hypot(o['dRA'],o['dDec']):.1f}| "
                    f"sem=({o['semRA']:.2f},{o['semDec']:.2f}) MAD=({o['madRA']:.1f},{o['madDec']:.1f}) N={o['N']}")
    # internal consistency unchanged by rigid shift
    multi = farr(t['nmatch']) >= 3
    print(f"  internal frame-to-frame repeatability (std_ra/std_dec, nmatch>=3): "
          f"({np.nanmedian(farr(t['std_ra'])[multi])*3.6e6:.1f},{np.nanmedian(farr(t['std_dec'])[multi])*3.6e6:.1f}) mas")


if __name__ == '__main__':
    main()
