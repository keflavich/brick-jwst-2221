#!/usr/bin/env python
"""Astrometric matching-quality review for the Brick, using the ROBUST
offset-histogram (2D mode) method, NOT nearest-neighbour mean/median.

Why: in a confusion-limited field, NN matching pairs many sources to a random
neighbour.  True matches pile up at the real systematic offset; mispairs form a
broad ~uniform background.  The 2D MODE of the offset distribution recovers the
systematic offset robustly; the mean/median is pulled by the mispair background.
For each pair we report the mode, the concentration (fraction of matches within
20 mas of the mode = reliability), the core scatter (MAD near the mode = the
per-star matching floor), AND the naive NN-median for contrast.
"""
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
import os, warnings; warnings.filterwarnings('ignore')

C = '/orange/adamginsburg/jwst/brick/catalogs/'
VIRAC = '/orange/adamginsburg/jwst/brick/astrometry_retie_qualcuts_20251211/refcache/virac2.fits'
DT = 8.70

def load_jwst(band):
    t = Table.read(C + f'{band}_merged_indivexp_merged_resbgsub_m7_dao_basic_vetted.fits')
    sc = t['skycoord']
    ok = np.isfinite(sc.ra.deg) & np.isfinite(sc.dec.deg)
    return SkyCoord(sc.ra.deg[ok]*u.deg, sc.dec.deg[ok]*u.deg)

def load_virac():
    v = Table.read(VIRAC)
    ra = np.asarray(v['RAJ2000'], float); de = np.asarray(v['DEJ2000'], float)
    pmra = np.nan_to_num(np.asarray(v['pmRA'], float)); pmde = np.nan_to_num(np.asarray(v['pmDE'], float))
    cosd = np.cos(np.radians(de))
    return SkyCoord((ra + pmra*DT/3.6e6/cosd)*u.deg, (de + pmde*DT/3.6e6)*u.deg)

def offset_mode(A, B, radius=0.5, half=100, binsz=3.0):
    """NN match A->B within radius(arcsec); 2D-mode of (dRA_onsky, dDec) in mas."""
    idx, sep, _ = A.match_to_catalog_sky(B)
    m = sep < radius*u.arcsec
    dra = (A.ra.deg[m] - B.ra.deg[idx[m]])*np.cos(np.radians(A.dec.deg[m]))*3.6e6
    dde = (A.dec.deg[m] - B.dec.deg[idx[m]])*3.6e6
    edges = np.arange(-half, half+binsz, binsz)
    H, xe, ye = np.histogram2d(dra, dde, bins=[edges, edges])
    iy, ix = np.unravel_index(np.argmax(H.T), H.T.shape)  # peak bin
    px = 0.5*(xe[ix]+xe[ix+1]); py = 0.5*(ye[iy]+ye[iy+1])
    # refine mode: flux-weighted centroid in a 15 mas window
    w = (np.abs(dra-px) < 15) & (np.abs(dde-py) < 15)
    if w.sum() > 5:
        px, py = np.mean(dra[w]), np.mean(dde[w])
    near = np.hypot(dra-px, dde-py) < 20
    conc = near.sum()/m.sum() if m.sum() else 0
    core = np.hypot(dra-px, dde-py)[near]
    mad = np.median(np.abs(core - np.median(core))) if near.sum() else np.nan
    return dict(n=int(m.sum()), mode=(px, py), conc=conc, coremad=mad,
                nnmed=(np.median(dra), np.median(dde)))

print('Loading catalogs...')
cat = {b: load_jwst(b) for b in ['f115w', 'f200w', 'f356w', 'f444w', 'f212n', 'f182m']}
vir = load_virac()
print(f'  sizes: ' + ', '.join(f'{k}={len(v):,}' for k, v in cat.items()) + f', virac={len(vir):,}\n')

PAIRS = [
    ('JWST<->VIRAC   F115W vs VIRAC2', cat['f115w'], vir),
    ('JWST<->VIRAC   F200W vs VIRAC2', cat['f200w'], vir),
    ('JWST<->VIRAC   F212N vs VIRAC2', cat['f212n'], vir),
    ('same-obs 1182  F115W vs F200W ', cat['f115w'], cat['f200w']),
    ('same-obs 1182  F200W vs F356W ', cat['f200w'], cat['f356w']),
    ('same-obs 2221  F182M vs F212N ', cat['f182m'], cat['f212n']),
    ('CROSS-OBS      F200W vs F212N ', cat['f200w'], cat['f212n']),  # the cross-tie pair
    ('CROSS-OBS      F115W vs F212N ', cat['f115w'], cat['f212n']),
    ('CROSS-OBS      F444W vs F466N ', cat['f444w'], vir),           # placeholder guard below
]
print(f'{"pair":32s} {"N":>7} {"mode(dRA,dDec) mas":>20} {"conc":>6} {"coreMAD":>8} {"NNmedian":>18}')
for name, A, B in PAIRS[:-1]:
    r = offset_mode(A, B)
    print(f'{name:32s} {r["n"]:7d} ({r["mode"][0]:+6.1f},{r["mode"][1]:+6.1f})     '
          f'{r["conc"]*100:5.0f}% {r["coremad"]:7.1f}  ({r["nnmed"][0]:+6.1f},{r["nnmed"][1]:+6.1f})')

# ---- F115W inter-module proxy: N-S offset-mode split vs VIRAC2 ----------------
print('\nF115W spatial-half offset-mode vs VIRAC2 (proxy for module structure):')
A = cat['f115w']
for lab, sub in [('South half', A[A.dec.deg < -28.712]), ('North half', A[A.dec.deg > -28.712])]:
    r = offset_mode(sub, vir)
    print(f'  {lab}: N={r["n"]:6d}  mode=({r["mode"][0]:+6.1f},{r["mode"][1]:+6.1f}) mas  conc={r["conc"]*100:.0f}%')
