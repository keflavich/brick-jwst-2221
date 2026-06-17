#!/usr/bin/env python
"""Definitive F115W -> Gaia tie test using per-frame VIRAC2 alignment + saturated stars.

For each frame: derive the rigid shift to VIRAC2 (Gaia frame) from the dense faint
DAO sources, then apply that SAME shift to (a) the faint sources and (b) the
saturated-star fits of that frame.  Compare both to Gaia.  This tests whether
per-frame VIRAC2 alignment plus saturated-star astrometry yields a good Gaia tie,
and whether bright (saturated) stars carry a magnitude-dependent residual.
"""
import os, glob, re, warnings
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


def farr(x):
    return np.asarray(np.ma.filled(np.ma.masked_invalid(np.asarray(x, float)), np.nan), float)


def prop(ra, dec, pmra, pmde, dt):
    pmra = np.where(np.isfinite(pmra), pmra, 0.); pmde = np.where(np.isfinite(pmde), pmde, 0.)
    return ra + (pmra * dt / 3.6e6) / np.cos(np.radians(dec)), dec + (pmde * dt / 3.6e6)


def load():
    v = Table.read(f'{BASE}/astrometry_diag/refcache/virac2.fits')
    ra, dec = prop(farr(v['RAJ2000']), farr(v['DEJ2000']), farr(v['pmRA']), farr(v['pmDE']), EPOCH - 2016.0)
    virac = SkyCoord(ra * u.deg, dec * u.deg)
    g = Table.read(f'{BASE}/astrometry_analysis/reference_cache/basic_merged_photometry_tables_merged_gaia.fits')
    ra, dec = prop(farr(g['RA_ICRS']), farr(g['DE_ICRS']), farr(g['pmRA']), farr(g['pmDE']), EPOCH - 2016.0)
    gaia = SkyCoord(ra * u.deg, dec * u.deg)
    return virac, gaia


def shift_to(sc, ref, sep=0.25*u.arcsec, clip=120):
    ci, s, _ = sc.match_to_catalog_sky(ref); ri, _, _ = ref.match_to_catalog_sky(sc)
    k = (ri[ci] == np.arange(len(ci))) & (s <= sep)
    if k.sum() < 8: return None
    mc, mr = sc[np.where(k)[0]], ref[ci[k]]
    dra = (mc.ra.deg - mr.ra.deg) * np.cos(np.radians(mc.dec.deg)) * 3.6e6
    dd = (mc.dec.deg - mr.dec.deg) * 3.6e6
    m, md = np.nanmedian(dra), np.nanmedian(dd)
    cl = np.hypot(dra - m, dd - md) <= clip
    return float(np.nanmedian(dra[cl])), float(np.nanmedian(dd[cl])), int(cl.sum())


def apply_shift(sc, dra_mas, ddec_mas):
    ra = sc.ra.deg - (dra_mas / 3.6e6) / np.cos(np.radians(sc.dec.deg))
    return SkyCoord((ra) * u.deg, (sc.dec.deg - ddec_mas / 3.6e6) * u.deg)


def vsref(sc, ref, sep=0.3*u.arcsec, clip=150):
    ci, s, _ = sc.match_to_catalog_sky(ref); ri, _, _ = ref.match_to_catalog_sky(sc)
    k = (ri[ci] == np.arange(len(ci))) & (s <= sep)
    if k.sum() < 3: return None
    mc, mr = sc[np.where(k)[0]], ref[ci[k]]
    dra = (mc.ra.deg - mr.ra.deg) * np.cos(np.radians(mc.dec.deg)) * 3.6e6
    dd = (mc.dec.deg - mr.dec.deg) * 3.6e6
    m, md = np.nanmedian(dra), np.nanmedian(dd)
    cl = np.hypot(dra - m, dd - md) <= clip
    if cl.sum() >= 3: dra, dd = dra[cl], dd[cl]
    return dra, dd


def frame_key(fn):
    m = re.search(r'(nrc[ab]\d)_visit(\d+)_vgroup\d+_exp(\d+)', fn)
    return (m.group(1), m.group(2), m.group(3)) if m else None


def satstar_path(det, visit, exp):
    # jw01182004001_02101_<exp>_<det>_destreak_o004_crf_satstar_catalog.fits
    pats = glob.glob(f'{PIPE}/jw01182004{int(visit):03d}_*_{int(exp):05d}_{det}_destreak_o004_crf_satstar_catalog.fits')
    return pats[0] if pats else None


def main():
    virac, gaia = load()
    frames = sorted(glob.glob(f'{FRAMEDIR}/f115w_*_visit*_exp*_daophot_basic.fits'))
    frames = [f for f in frames if not any(t in f for t in ('_m1','_m2','_m3','_m4','iter','resbgsub','_group'))]

    faint_raw_g, faint_corr_g = [], []      # faint sources vs Gaia, raw and per-frame-corrected
    sat_raw_g, sat_corr_g = [], []          # saturated stars vs Gaia, raw and corrected
    sat_corr_v = []                         # saturated stars vs VIRAC2 after correction
    nsat = 0
    for fn in frames:
        t = Table.read(fn)
        sc = SkyCoord(t['skycoord_centroid']).icrs
        fl = farr(t['flux_fit']); ok = np.isfinite(sc.ra.deg) & np.isfinite(fl) & (fl > 0)
        sc = sc[ok]
        sh = shift_to(sc, virac)
        if sh is None: continue
        dra_s, ddec_s, _ = sh
        # faint vs Gaia raw and corrected
        r = vsref(sc, gaia)
        if r: faint_raw_g += list(np.hypot(r[0], r[1]) * 0 + r[0]); faint_raw_g_dec = r[1]
        sc_c = apply_shift(sc, dra_s, ddec_s)
        r = vsref(sc_c, gaia)
        if r: faint_corr_g.append((np.nanmedian(r[0]), np.nanmedian(r[1]), len(r[0])))
        # saturated stars for this frame
        key = frame_key(fn)
        if key:
            sp = satstar_path(*key)
            if sp:
                ts = Table.read(sp)
                if 'skycoord_fit' in ts.colnames and len(ts):
                    ssc = SkyCoord(ts['skycoord_fit']).icrs
                    ssc = ssc[np.isfinite(ssc.ra.deg)]
                    nsat += len(ssc)
                    rr = vsref(ssc, gaia, sep=0.4*u.arcsec)
                    if rr: sat_raw_g.append((np.nanmedian(rr[0]), np.nanmedian(rr[1]), len(rr[0])))
                    ssc_c = apply_shift(ssc, dra_s, ddec_s)
                    rr = vsref(ssc_c, gaia, sep=0.4*u.arcsec)
                    if rr: sat_corr_g.append((np.nanmedian(rr[0]), np.nanmedian(rr[1]), len(rr[0])))
                    rr = vsref(ssc_c, virac, sep=0.4*u.arcsec)
                    if rr: sat_corr_v.append((np.nanmedian(rr[0]), np.nanmedian(rr[1]), len(rr[0])))

    def agg(lst, label):
        if not lst:
            print(f"  {label}: (none)"); return
        a = np.array(lst)
        w = a[:, 2]
        mdra = np.average(a[:, 0], weights=w); mddec = np.average(a[:, 1], weights=w)
        print(f"  {label}: frames={len(a)} totmatch={int(w.sum())} "
              f"wmean offset=({mdra:.1f},{mddec:.1f}) |{np.hypot(mdra,mddec):.1f}| "
              f"frame-spread=({stats.mad_std(a[:,0]):.1f},{stats.mad_std(a[:,1]):.1f}) mas")

    print(f"Saturated-star detections used: {nsat}")
    print("\n=== FAINT DAO sources vs Gaia ===")
    agg(faint_corr_g, "after per-frame VIRAC2 shift")
    print("\n=== SATURATED stars vs Gaia ===")
    agg(sat_raw_g, "raw frame WCS")
    agg(sat_corr_g, "after per-frame VIRAC2 shift")
    print("\n=== SATURATED stars vs VIRAC2 (after shift) ===")
    agg(sat_corr_v, "after per-frame VIRAC2 shift")


if __name__ == '__main__':
    main()
