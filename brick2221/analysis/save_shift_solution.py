#!/usr/bin/env python
"""Save the per-frame F115W -> VIRAC2(Gaia frame) alignment solution.

Writes a table (one row per frame) of the rigid shift that places each F115W
exposure on the Gaia/VIRAC2 frame, derived from the dense faint DAO sources.
Columns are compatible in spirit with merge_catalogs.shift_individual_catalog's
offsets_table (Visit, Exposure, Module, Filter, dra, ddec in arcsec) so the merge
can be re-run on the Gaia frame.  Sign/units should be confirmed against the
pipeline's RAOFFSET convention before a production re-merge.
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
OUTFN = f'{BASE}/astrometry_diag/f115w_virac2_perframe_shifts.ecsv'


def farr(x):
    return np.asarray(np.ma.filled(np.ma.masked_invalid(np.asarray(x, float)), np.nan), float)


def prop(ra, dec, pmra, pmde, dt):
    pmra = np.where(np.isfinite(pmra), pmra, 0.); pmde = np.where(np.isfinite(pmde), pmde, 0.)
    return ra + (pmra * dt / 3.6e6) / np.cos(np.radians(dec)), dec + (pmde * dt / 3.6e6)


v = Table.read(f'{BASE}/astrometry_diag/refcache/virac2.fits')
ra, dec = prop(farr(v['RAJ2000']), farr(v['DEJ2000']), farr(v['pmRA']), farr(v['pmDE']), EPOCH - 2016.0)
virac = SkyCoord(ra * u.deg, dec * u.deg)

frames = sorted(glob.glob(f'{BASE}/F115W/f115w_*_visit*_exp*_daophot_basic.fits'))
frames = [f for f in frames if not any(t in f for t in ('_m1','_m2','_m3','_m4','iter','resbgsub','_group'))]

rows = []
for fn in frames:
    t = Table.read(fn)
    sc = SkyCoord(t['skycoord_centroid']).icrs
    fl = farr(t['flux_fit']); ok = np.isfinite(sc.ra.deg) & np.isfinite(fl) & (fl > 0)
    sc = sc[ok]
    ci, s, _ = sc.match_to_catalog_sky(virac); ri, _, _ = virac.match_to_catalog_sky(sc)
    k = (ri[ci] == np.arange(len(ci))) & (s <= 0.25 * u.arcsec)
    if k.sum() < 8:
        continue
    mc, mr = sc[np.where(k)[0]], virac[ci[k]]
    dra = (mc.ra.deg - mr.ra.deg) * np.cos(np.radians(mc.dec.deg)) * 3.6e6
    dd = (mc.dec.deg - mr.dec.deg) * 3.6e6
    m, md = np.nanmedian(dra), np.nanmedian(dd)
    cl = np.hypot(dra - m, dd - md) <= 120
    dra, dd = dra[cl], dd[cl]
    n = len(dra)
    mr_re = re.search(r'(nrc[ab]\d)_visit(\d+)_vgroup\d+_exp(\d+)', fn)
    det, vis, exp = mr_re.group(1), int(mr_re.group(2)), int(mr_re.group(3))
    # measured offset = JWST - VIRAC2 (mas). To CORRECT, subtract it.
    rows.append(dict(Filter='F115W', Module=det, Visit=f'{vis:03d}', Exposure=exp,
                     n_match=n,
                     measured_dRA_mas=float(np.nanmedian(dra)), measured_dDec_mas=float(np.nanmedian(dd)),
                     correction_dRA_mas=float(-np.nanmedian(dra)), correction_dDec_mas=float(-np.nanmedian(dd)),
                     sem_dRA_mas=float(stats.mad_std(dra, ignore_nan=True) / np.sqrt(n)),
                     sem_dDec_mas=float(stats.mad_std(dd, ignore_nan=True) / np.sqrt(n)),
                     scatter_dRA_mas=float(stats.mad_std(dra, ignore_nan=True)),
                     scatter_dDec_mas=float(stats.mad_std(dd, ignore_nan=True))))

tab = Table(rows)
tab.meta['reference'] = 'VIRAC2 (II/387), Gaia DR3 frame, propagated to epoch 2022.70'
tab.meta['note'] = 'measured = median(JWST_centroid - VIRAC2); correction = -measured. Apply to skycoord_centroid.'
tab.meta['epoch'] = EPOCH
tab.write(OUTFN, overwrite=True)
print(f"Wrote {OUTFN}: {len(tab)} frames")
print(f"correction range dRA [{tab['correction_dRA_mas'].min():.0f},{tab['correction_dRA_mas'].max():.0f}] "
      f"dDec [{tab['correction_dDec_mas'].min():.0f},{tab['correction_dDec_mas'].max():.0f}] mas")
print(f"median per-frame SEM = ({np.median(tab['sem_dRA_mas']):.1f},{np.median(tab['sem_dDec_mas']):.1f}) mas")
