#!/usr/bin/env python
"""Absolute astrometric frame for F115W / F182M / F200W merged first-iteration catalogs.

The reference frame that matters for NIRSpec pointing is Gaia DR3 / GSC 3.x.
VIRAC2 is the dense NIR proxy on the Gaia frame; VVV DR4 (II/376) is 2MASS-tied
and ~24 mas off Gaia (kept only to document the discrepancy).  All references are
propagated to the JWST F115W epoch (2022.70) with their own proper motions.

For each filter we measure the merged-catalog absolute offset vs each reference,
derive the rigid shift that re-anchors it to the Gaia frame, and test whether the
three filters agree once shifted.
"""
import os, warnings
import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import stats

warnings.simplefilter('ignore')
BASE = '/orange/adamginsburg/jwst/brick'
EPOCH = 2022.70
REFCACHE = f'{BASE}/astrometry_diag/refcache'
GAIA_CACHE = f'{BASE}/astrometry_analysis/reference_cache/basic_merged_photometry_tables_merged_gaia.fits'
VVV_CACHE = f'{BASE}/astrometry_analysis/reference_cache/basic_merged_photometry_tables_merged_vvv.fits'

CATS = {
    'F115W': f'{BASE}/catalogs/f115w_merged_indivexp_merged_dao_basic.fits',
    'F182M': f'{BASE}/catalogs/f182m_merged_indivexp_merged_dao_basic.fits',
    'F200W': f'{BASE}/catalogs/f200w_merged_indivexp_merged_dao_basic.fits',
}
# nearest VIRAC2/VVV band for flux matching
NEARBAND = {'F115W': 'Jmag', 'F182M': 'Hmag', 'F200W': 'Ksmag'}


def farr(x):
    return np.asarray(np.ma.filled(np.ma.masked_invalid(np.asarray(x, float)), np.nan), float)


def prop(ra, dec, pmra, pmde, dt):
    pmra = np.where(np.isfinite(pmra), pmra, 0.0)
    pmde = np.where(np.isfinite(pmde), pmde, 0.0)
    ra2 = ra + (pmra * dt / 3.6e6) / np.cos(np.radians(dec))
    return ra2, dec + (pmde * dt / 3.6e6)


def load_refs():
    refs = {}
    # Gaia (epoch 2016.0)
    g = Table.read(GAIA_CACHE)
    ra, dec = prop(farr(g['RA_ICRS']), farr(g['DE_ICRS']), farr(g['pmRA']), farr(g['pmDE']), EPOCH - 2016.0)
    refs['Gaia'] = dict(sc=SkyCoord(ra * u.deg, dec * u.deg), mag=farr(g['Gmag']), band=None)
    # VIRAC2 (epoch 2016.0, Gaia frame)
    v = Table.read(f'{REFCACHE}/virac2.fits')
    ra, dec = prop(farr(v['RAJ2000']), farr(v['DEJ2000']), farr(v['pmRA']), farr(v['pmDE']), EPOCH - 2016.0)
    refs['VIRAC2'] = dict(sc=SkyCoord(ra * u.deg, dec * u.deg), tbl=v)
    # GSC 2.4.2 (~Gaia); propagate from per-source Epoch
    gsc = Table.read(f'{REFCACHE}/gsc242.fits')
    ep = farr(gsc['Epoch']); ep = np.where(np.isfinite(ep), ep, 2016.0)
    ra, dec = prop(farr(gsc['RA_ICRS']), farr(gsc['DE_ICRS']), farr(gsc['pmRA']), farr(gsc['pmDE']), EPOCH - ep)
    refs['GSC242'] = dict(sc=SkyCoord(ra * u.deg, dec * u.deg), mag=farr(gsc['Gmag']), band=None)
    # VVV DR4 (2MASS frame, no PM)
    vv = Table.read(VVV_CACHE)
    refs['VVV_DR4'] = dict(sc=SkyCoord(vv['skycoord']).icrs, mag=None, band=None)
    return refs


def virac_band(v, band):
    return farr(v[band]) if band in v.colnames else None


def measure(csc, m_inst, refsc, refmag=None, fluxmatch=False, max_sep=0.30 * u.arcsec, clean=150.0):
    ci, sep, _ = csc.match_to_catalog_sky(refsc)
    ri, _, _ = refsc.match_to_catalog_sky(csc)
    keep = (ri[ci] == np.arange(len(ci))) & (sep <= max_sep)
    idx, rid = np.where(keep)[0], ci[keep]
    if len(idx) < 5:
        return None
    mc, mr = csc[idx], refsc[rid]
    dra = (mc.ra.deg - mr.ra.deg) * np.cos(np.radians(mc.dec.deg)) * 3.6e6
    ddec = (mc.dec.deg - mr.dec.deg) * 3.6e6
    fin = np.ones(len(idx), bool)
    if fluxmatch and refmag is not None:
        rm = refmag[rid]; fin = np.isfinite(rm) & np.isfinite(m_inst[idx])
        if fin.sum() >= 5:
            dm = m_inst[idx][fin] - rm[fin]; md = np.nanmedian(dm); s = max(stats.mad_std(dm, ignore_nan=True), 0.3)
            f2 = fin.copy(); f2[fin] = np.abs(dm - md) <= 3 * s; fin = f2
    dra, ddec = dra[fin], ddec[fin]
    md, mdd = np.nanmedian(dra), np.nanmedian(ddec)
    r = np.hypot(dra - md, ddec - mdd); cl = r <= clean
    if cl.sum() >= 5:
        dra, ddec = dra[cl], ddec[cl]
    n = len(dra)
    return dict(N=n, dRA=float(np.nanmedian(dra)), dDec=float(np.nanmedian(ddec)),
                madRA=float(stats.mad_std(dra, ignore_nan=True)), madDec=float(stats.mad_std(ddec, ignore_nan=True)),
                semRA=float(stats.mad_std(dra, ignore_nan=True) / np.sqrt(n)),
                semDec=float(stats.mad_std(ddec, ignore_nan=True) / np.sqrt(n)))


def main():
    refs = load_refs()
    print(f"References (propagated to {EPOCH}):")
    for k, r in refs.items():
        print(f"  {k}: {len(r['sc'])}")

    shifts = {}
    print("\n===== MERGED CATALOG ABSOLUTE OFFSETS (median JWST - ref, mas) =====")
    print(f"{'filter':6} {'ref':8} {'N':>6} {'dRA':>8} {'dDec':>8} {'|vec|':>7} {'sem':>6} {'MAD_RA':>7} {'MAD_Dec':>7}")
    for filt, fn in CATS.items():
        t = Table.read(fn)
        sc = SkyCoord(t['skycoord']).icrs
        fl = farr(t['flux']); ok = np.isfinite(sc.ra.deg) & np.isfinite(fl) & (fl > 0)
        sc, fl = sc[ok], fl[ok]; mi = -2.5 * np.log10(fl)
        for rk, r in refs.items():
            if rk == 'VIRAC2':
                rmag = virac_band(r['tbl'], NEARBAND[filt]); fm = True
            elif rk == 'VVV_DR4':
                rmag = None; fm = False
            else:
                rmag = r['mag']; fm = False
            res = measure(sc, mi, r['sc'], refmag=rmag, fluxmatch=fm)
            if res is None:
                print(f"{filt:6} {rk:8} {'--':>6}"); continue
            vec = np.hypot(res['dRA'], res['dDec']); sem = np.hypot(res['semRA'], res['semDec'])
            print(f"{filt:6} {rk:8} {res['N']:6d} {res['dRA']:8.2f} {res['dDec']:8.2f} {vec:7.2f} {sem:6.2f} {res['madRA']:7.1f} {res['madDec']:7.1f}")
            if rk == 'VIRAC2':
                shifts[filt] = (res['dRA'], res['dDec'])

    print("\n===== IMPLIED Gaia-frame re-anchor shift (from VIRAC2) and inter-filter agreement =====")
    for filt, (dr, dd) in shifts.items():
        print(f"  {filt}: subtract ({dr:.1f}, {dd:.1f}) mas to land on VIRAC2/Gaia frame")
    if len(shifts) > 1:
        arr = np.array(list(shifts.values()))
        print(f"  inter-filter shift spread: dRA {arr[:,0].max()-arr[:,0].min():.1f}, dDec {arr[:,1].max()-arr[:,1].min():.1f} mas "
              f"(how well the 3 filters already agree on the absolute frame)")


if __name__ == '__main__':
    main()
