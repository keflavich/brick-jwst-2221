#!/usr/bin/env python
"""Build the absolute-frame per-exposure offsets table for fix_alignment (UPSTREAM FOLD).

Problem this solves
-------------------
`fix_alignment()` (PipelineRerunNIRCAM-LONG.py) ties each per-exposure crf GWCS to a
per-frame offsets table via `jwst.tweakreg.utils.adjust_wcs`.  For brick-1182 it reads
`Offsets_JWST_Brick1182_F200ref_average.csv` (refnames['1182']='F200ref'), an Aug-2024
*relative* solution.  Meanwhile the catalog merge ties to the *absolute* VIRAC2-2014.0
frame.  Two tables -> crf/_i2d/model/residual end up ~97 mas off the catalog & the
realigned mosaic.  This script folds the absolute VIRAC2 zero-point into the
fix_alignment table so every product shares ONE frame and realign_to_catalog ~= 0.

Recipe (validated conventions)
------------------------------
- `adjust_wcs(delta_ra=x)` shifts the sky by +x in RA-COORDINATE (Δα, no cosδ); +sign.
  (verified empirically 2026-06-18: +0.1" -> +100 mas Δα = +87.7 mas on-sky.)
- The current crf sit on the F200ref frame (old_dra applied to pristine _cal).
  Their residual to VIRAC2-2014.0 is  obs = median(α_det - α_ref) [Δα], median(δ_det - δ_ref).
- To land on VIRAC2 from pristine _cal:
      new_dra(arcsec)  = old_F200ref_dra  - obs_dra_coord
      new_ddec(arcsec) = old_F200ref_ddec - obs_ddec
  per (Visit, Module), averaged over F115W exposures (use_average=True semantics).

Output: offsets/Offsets_JWST_Brick1182_VIRAC2_average.csv
  = full copy of F200ref_average, with ONLY F115W rows folded to absolute.  Non-F115W
  rows are left identical to F200ref (to be folded per-filter as each filter is
  processed).  Set refnames['1182']='VIRAC2' to use it.

Run AFTER the per-frame F115W catalogs are stable (recat finished), so detections are
not being rewritten underneath the measurement.
"""
import glob
import re
import warnings
import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import stats

warnings.simplefilter('ignore')
BASE = '/orange/adamginsburg/jwst/brick'
EPOCH = 2022.70
VIRAC2_EPOCH = 2014.0
OLD = f'{BASE}/offsets/Offsets_JWST_Brick1182_F200ref_average.csv'
OUT = f'{BASE}/offsets/Offsets_JWST_Brick1182_VIRAC2_average.csv'
FRAMEDIR = f'{BASE}/F115W'
MATCH_RADIUS = 0.25 * u.arcsec
CLIP_MAS = 120.0


def farr(x):
    return np.asarray(np.ma.filled(np.ma.masked_invalid(np.asarray(x, float)), np.nan), float)


def prop(ra, dec, pmra, pmde, dt):
    pmra = np.where(np.isfinite(pmra), pmra, 0.); pmde = np.where(np.isfinite(pmde), pmde, 0.)
    return ra + (pmra * dt / 3.6e6) / np.cos(np.radians(dec)), dec + (pmde * dt / 3.6e6)


# --- VIRAC2 reference, propagated to obs epoch on the corrected 2014.0 baseline ---
v = Table.read(f'{BASE}/astrometry_diag/refcache/virac2.fits')
vra, vdec = prop(farr(v['RAJ2000']), farr(v['DEJ2000']), farr(v['pmRA']), farr(v['pmDE']),
                 EPOCH - VIRAC2_EPOCH)
virac = SkyCoord(vra * u.deg, vdec * u.deg)
vJ = farr(v['Jmag'])


def frame_files(visit, module):
    pat = f'{FRAMEDIR}/f115w_{module}_visit{visit}_*exp*_daophot_basic.fits'
    fs = sorted(glob.glob(pat))
    return [f for f in fs if not any(t in f for t in ('_m1', '_m2', '_m3', '_m4', '_m5', '_m6',
                                                      'iter', 'resbgsub', '_group'))]


def measure_residual(visit, module):
    """median (α_det-α_ref) [Δα arcsec], (δ_det-δ_ref) [arcsec] over all exposures; + nmatch."""
    dra_all, dd_all = [], []
    for fn in frame_files(visit, module):
        t = Table.read(fn)
        sc = SkyCoord(t['skycoord_centroid']).icrs
        fl = farr(t['flux_fit']); ok = np.isfinite(sc.ra.deg) & np.isfinite(fl) & (fl > 0)
        sc, fl = sc[ok], fl[ok]
        if len(sc) < 8:
            continue
        mi = -2.5 * np.log10(fl)
        ci, s, _ = sc.match_to_catalog_sky(virac)
        ri, _, _ = virac.match_to_catalog_sky(sc)
        k = (ri[ci] == np.arange(len(ci))) & (s <= MATCH_RADIUS)
        if k.sum() < 8:
            continue
        idx, rid = np.where(k)[0], ci[k]
        # flux/J clip to drop mismatches
        Jm = vJ[rid]; fin = np.isfinite(Jm) & np.isfinite(mi[idx])
        keep = np.ones(len(idx), bool)
        if fin.sum() >= 5:
            dm = mi[idx][fin] - Jm[fin]; med = np.nanmedian(dm)
            sg = max(stats.mad_std(dm, ignore_nan=True), 0.3)
            keep[fin] = np.abs(dm - med) <= 3 * sg
        i2, r2 = idx[keep], rid[keep]
        dra = (sc[i2].ra.deg - virac[r2].ra.deg) * 3600.0          # Δα coordinate, arcsec
        dd = (sc[i2].dec.deg - virac[r2].dec.deg) * 3600.0
        m, md = np.nanmedian(dra), np.nanmedian(dd)
        cl = np.hypot((dra - m) * np.cos(np.radians(np.nanmedian(sc[i2].dec.deg))),
                      dd - md) * 1000.0 <= CLIP_MAS
        dra_all.append(dra[cl]); dd_all.append(dd[cl])
    if not dra_all:
        return None
    dra_all = np.concatenate(dra_all); dd_all = np.concatenate(dd_all)
    return (float(np.nanmedian(dra_all)), float(np.nanmedian(dd_all)), int(len(dra_all)),
            float(stats.mad_std(dra_all) * 3.6e6 / max(len(dra_all), 1) ** 0.5))


old = Table.read(OLD)
new = old.copy()
is_f115 = np.array([str(x) == 'F115W' for x in new['Filter']])
print(f"folding {is_f115.sum()} F115W rows to absolute VIRAC2-{VIRAC2_EPOCH} frame")
log = []
for i in np.where(is_f115)[0]:
    visit = str(new['Visit'][i]).replace('jw01182', '')  # e.g. '004001' -> visit token
    # frame filenames use visitNNN where NNN is the visit number (001/002)
    vtok = str(new['Visit'][i])[-3:]
    module = str(new['Module'][i])
    res = measure_residual(vtok, module)
    if res is None:
        print(f"  WARN no matches for visit{vtok} {module}; leaving F200ref value")
        log.append((vtok, module, None))
        continue
    obs_dra, obs_dd, n, sem = res
    old_dra = float(new['dra (arcsec)'][i]); old_dd = float(new['ddec (arcsec)'][i])
    nd = old_dra - obs_dra; ndd = old_dd - obs_dd
    for col in ('dra', 'dra (arcsec)'):
        if col in new.colnames:
            new[col][i] = nd
    for col in ('ddec', 'ddec (arcsec)'):
        if col in new.colnames:
            new[col][i] = ndd
    if 'nmatch' in new.colnames:
        new['nmatch'][i] = n
    log.append((vtok, module, (obs_dra * 1000, obs_dd * 1000, n)))
    print(f"  visit{vtok} {module}: obs_resid=({obs_dra*1000:+.1f},{obs_dd*1000:+.1f})mas n={n} "
          f"=> dra {old_dra:.4f}->{nd:.4f}, ddec {old_dd:.4f}->{ndd:.4f}")

new.meta['NOTE'] = ('VIRAC2-2014.0 absolute fix_alignment table. F115W rows folded to '
                    'absolute (old F200ref - residual_to_VIRAC2); other filters = F200ref copy.')
new.write(OUT, overwrite=True)
print(f"\nWrote {OUT}")
print("Residual magnitudes folded (mas):",
      [(vt, mo, None if r is None else (round(r[0], 1), round(r[1], 1))) for vt, mo, r in log])
