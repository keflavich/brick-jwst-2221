#!/usr/bin/env python
"""Produce a Gaia-frame F115W merged catalog by per-frame re-anchoring to VIRAC2.

Controlled re-merge (does NOT touch the production pipeline / offsets-table, whose
RAOFFSET unit convention is ambiguous).  For each per-frame catalog: apply the
per-frame VIRAC2 shift to skycoord_centroid, then accumulate the corrected
detections onto the existing merged-catalog source grid (reusing its clustering)
and average.  Output: a new catalog with Gaia-frame positions + verification.
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
MERGED = f'{BASE}/catalogs/f115w_merged_indivexp_merged_dao_basic.fits'
SHIFTS = f'{BASE}/astrometry_diag/f115w_virac2_perframe_shifts.ecsv'
OUT = f'{BASE}/catalogs/f115w_merged_indivexp_merged_dao_basic_GAIAFRAME.fits'


def farr(x):
    return np.asarray(np.ma.filled(np.ma.masked_invalid(np.asarray(x, float)), np.nan), float)


def prop(ra, dec, pmra, pmde, dt):
    pmra = np.where(np.isfinite(pmra), pmra, 0.); pmde = np.where(np.isfinite(pmde), pmde, 0.)
    return ra + (pmra * dt / 3.6e6) / np.cos(np.radians(dec)), dec + (pmde * dt / 3.6e6)


def off(a, b, sep=0.3*u.arcsec, clip=150):
    i, s, _ = a.match_to_catalog_sky(b); ri, _, _ = b.match_to_catalog_sky(a)
    k = (ri[i] == np.arange(len(i))) & (s <= sep)
    if k.sum() < 5: return None
    mc, mr = a[np.where(k)[0]], b[i[k]]
    dra = (mc.ra.deg - mr.ra.deg) * np.cos(np.radians(mc.dec.deg)) * 3.6e6
    dd = (mc.dec.deg - mr.dec.deg) * 3.6e6
    m, md = np.nanmedian(dra), np.nanmedian(dd); cl = np.hypot(dra-m, dd-md) <= clip
    if cl.sum() >= 5: dra, dd = dra[cl], dd[cl]
    n = len(dra)
    return dict(N=n, dRA=round(float(np.nanmedian(dra)),2), dDec=round(float(np.nanmedian(dd)),2),
                madRA=round(float(stats.mad_std(dra,ignore_nan=True)),1), madDec=round(float(stats.mad_std(dd,ignore_nan=True)),1),
                semRA=round(float(stats.mad_std(dra,ignore_nan=True)/np.sqrt(n)),2),
                semDec=round(float(stats.mad_std(dd,ignore_nan=True)/np.sqrt(n)),2))


# shift table keyed by (module, visit, exposure)
sh = Table.read(SHIFTS)
shift = {(r['Module'], int(r['Visit']), int(r['Exposure'])): (r['correction_dRA_mas'], r['correction_dDec_mas']) for r in sh}

# base grid = existing merged catalog source positions
m = Table.read(MERGED)
base = SkyCoord(m['skycoord']).icrs
nbase = len(base)
sum_ra = np.zeros(nbase); sum_dec = np.zeros(nbase); cnt = np.zeros(nbase, int)
sumsq_ra = np.zeros(nbase); sumsq_dec = np.zeros(nbase)

frames = sorted(glob.glob(f'{BASE}/F115W/f115w_*_visit*_exp*_daophot_basic.fits'))
frames = [f for f in frames if not any(t in f for t in ('_m1','_m2','_m3','_m4','iter','resbgsub','_group'))]
nused = 0
for fn in frames:
    t = Table.read(fn)
    mr = re.search(r'(nrc[ab]\d)_visit(\d+)_vgroup\d+_exp(\d+)', fn)
    key = (mr.group(1), int(mr.group(2)), int(mr.group(3)))
    if key not in shift:
        continue
    cdra, cddec = shift[key]   # correction in mas (to ADD)
    sc = SkyCoord(t['skycoord_centroid']).icrs
    ok = np.isfinite(sc.ra.deg)
    sc = sc[ok]
    # corrected positions (mas -> deg) -- these are what we average
    ra = sc.ra.deg + (cdra / 3.6e6) / np.cos(np.radians(sc.dec.deg))
    dec = sc.dec.deg + (cddec / 3.6e6)
    # cluster assignment uses the UNCORRECTED positions (the merge was built from
    # those, so they sit near the base grid); we accumulate the CORRECTED positions.
    idx, sep, _ = sc.match_to_catalog_sky(base)
    good = sep <= 60 * u.mas
    bi = idx[good]
    np.add.at(sum_ra, bi, ra[good]); np.add.at(sum_dec, bi, dec[good])
    np.add.at(sumsq_ra, bi, ra[good]**2); np.add.at(sumsq_dec, bi, dec[good]**2)
    np.add.at(cnt, bi, 1)
    nused += 1

print(f"Re-merged {nused} frames onto {nbase} base sources")
has = cnt >= 1
ra_new = sum_ra / np.where(cnt > 0, cnt, 1)
dec_new = sum_dec / np.where(cnt > 0, cnt, 1)
# internal scatter per source (std across frames), mas
var_ra = np.maximum(sumsq_ra / np.where(cnt > 0, cnt, 1) - ra_new**2, 0)
var_dec = np.maximum(sumsq_dec / np.where(cnt > 0, cnt, 1) - dec_new**2, 0)
std_ra_mas = np.sqrt(var_ra) * np.cos(np.radians(dec_new)) * 3.6e6
std_dec_mas = np.sqrt(var_dec) * 3.6e6

out = m.copy()
out['skycoord'] = SkyCoord(ra_new * u.deg, dec_new * u.deg)
out['nmatch_gaiaframe'] = cnt
out['std_ra_gaiaframe_mas'] = std_ra_mas
out['std_dec_gaiaframe_mas'] = std_dec_mas
out = out[has]
out.meta['FRAME'] = 'Gaia DR3 (via per-frame VIRAC2 alignment), epoch 2022.70'
out.write(OUT, overwrite=True)
print(f"Wrote {OUT}: {len(out)} sources")

multi = cnt[has] >= 3
print(f"\nInternal frame-to-frame repeatability (>=3 frames): "
      f"median std=({np.nanmedian(std_ra_mas[has][multi]):.1f},{np.nanmedian(std_dec_mas[has][multi]):.1f}) mas")

# verify vs references
g = Table.read(f'{BASE}/astrometry_analysis/reference_cache/basic_merged_photometry_tables_merged_gaia.fits')
gaia = SkyCoord(*prop(farr(g['RA_ICRS']), farr(g['DE_ICRS']), farr(g['pmRA']), farr(g['pmDE']), EPOCH-2016.0), unit='deg')
v = Table.read(f'{BASE}/astrometry_diag/refcache/virac2.fits')
virac = SkyCoord(*prop(farr(v['RAJ2000']), farr(v['DEJ2000']), farr(v['pmRA']), farr(v['pmDE']), EPOCH-2016.0), unit='deg')
gsc = Table.read(f'{BASE}/astrometry_diag/refcache/gsc32.fits')
gscsc = SkyCoord(*prop(farr(gsc['ra']), farr(gsc['dec']), farr(gsc['rapm']), farr(gsc['decpm']), EPOCH-farr(gsc['epoch'])), unit='deg')
# Gaia-backed GSC subset (bright, G finite) for a clean absolute check
gbright = np.isfinite(farr(gsc['gaiaGmag']))
gscsc_gaia = gscsc[gbright]

new_sc = out['skycoord']
old_sc = SkyCoord(m['skycoord']).icrs
print("\n=== ABSOLUTE OFFSETS: original vs Gaia-frame re-merge ===")
for lab, ref in [('Gaia DR3', gaia), ('GSC3.2(Gaia subset)', gscsc_gaia), ('VIRAC2', virac)]:
    print(f"  {lab:20} original: {off(old_sc, ref)}")
    print(f"  {lab:20} re-merged:{off(new_sc, ref)}")
