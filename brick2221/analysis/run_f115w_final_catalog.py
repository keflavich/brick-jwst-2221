#!/usr/bin/env python
"""Final F115W catalog on the Gaia/VIRAC2 frame, from the NEW (seed-reduced) crf.

Runs after recatalog (catalog_long --each-exposure --daophot on the
new crf). Steps:
  1. Measure per-frame offset (new crf catalogs) -> VIRAC2, build offsets table.
  2. merge_individual_frames(dao/basic) with that offsets table.
  3. Build trimmed canonical from _allcols; apply final rigid null to VIRAC2.
  4. Verify vs VIRAC2 / Gaia / GSC3.2(Gaia).
"""
import sys, glob, re, warnings
sys.path.insert(0, '/orange/adamginsburg/repos/jwst-gc-pipeline')
import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import stats
from jwst_gc_pipeline.photometry.merge_catalogs import merge_individual_frames

warnings.simplefilter('ignore')
BASE = '/orange/adamginsburg/jwst/brick/'
EPOCH = 2022.70
FRAMEDIR = f'{BASE}/F115W'
OFFSETS_OUT = f'{BASE}/offsets/Offsets_JWST_Brick1182_F115W_VIRAC2frame_newcrf.csv'
CANON = f'{BASE}/catalogs/f115w_merged_indivexp_merged_dao_basic.fits'
ALLCOLS = f'{BASE}/catalogs/f115w_merged_indivexp_merged_dao_basic_allcols.fits'


def farr(x):
    return np.asarray(np.ma.filled(np.ma.masked_invalid(np.asarray(x, float)), np.nan), float)


def prop(ra, dec, pmra, pmde, dt):
    pmra = np.where(np.isfinite(pmra), pmra, 0.); pmde = np.where(np.isfinite(pmde), pmde, 0.)
    return ra + (pmra * dt / 3.6e6) / np.cos(np.radians(dec)), dec + (pmde * dt / 3.6e6)


def load_seed_components():
    v = Table.read(f'{BASE}/astrometry_diag/refcache/virac2.fits')
    virac = SkyCoord(*prop(farr(v['RAJ2000']), farr(v['DEJ2000']), farr(v['pmRA']), farr(v['pmDE']), EPOCH - 2016.0), unit='deg')
    g = Table.read(f'{BASE}/astrometry_analysis/reference_cache/basic_merged_photometry_tables_merged_gaia.fits')
    gaia = SkyCoord(*prop(farr(g['RA_ICRS']), farr(g['DE_ICRS']), farr(g['pmRA']), farr(g['pmDE']), EPOCH - 2016.0), unit='deg')
    gsc = Table.read(f'{BASE}/astrometry_diag/refcache/gsc32.fits'); gb = np.isfinite(farr(gsc['gaiaGmag']))
    gscg = SkyCoord(*prop(farr(gsc['ra'])[gb], farr(gsc['dec'])[gb], farr(gsc['rapm'])[gb], farr(gsc['decpm'])[gb], EPOCH - farr(gsc['epoch'])[gb]), unit='deg')
    return virac, gaia, gscg


def off(a, b, sep=0.3 * u.arcsec, clip=120):
    i, s, _ = a.match_to_catalog_sky(b); ri, _, _ = b.match_to_catalog_sky(a)
    k = (ri[i] == np.arange(len(i))) & (s <= sep)
    if k.sum() < 5: return None
    mc, mr = a[np.where(k)[0]], b[i[k]]
    dra = (mc.ra.deg - mr.ra.deg) * np.cos(np.radians(mc.dec.deg)) * 3.6e6
    dd = (mc.dec.deg - mr.dec.deg) * 3.6e6
    m, md = np.nanmedian(dra), np.nanmedian(dd); cl = np.hypot(dra - m, dd - md) <= clip
    if cl.sum() >= 5: dra, dd = dra[cl], dd[cl]
    n = len(dra)
    return dict(N=n, dRA=round(float(np.nanmedian(dra)), 2), dDec=round(float(np.nanmedian(dd)), 2),
                madRA=round(float(stats.mad_std(dra)), 1), madDec=round(float(stats.mad_std(dd)), 1))


virac, gaia, gscg = load_seed_components()

# ---- 1. per-frame offsets (new crf catalogs) -> VIRAC2 ----
frames = sorted(glob.glob(f'{FRAMEDIR}/f115w_*_visit*_exp*_daophot_basic.fits'))
frames = [f for f in frames if not any(t in f for t in ('_m1', '_m2', '_m3', '_m4', 'iter', 'resbgsub', '_group'))]
rows = []
for fn in frames:
    t = Table.read(fn)
    sc = SkyCoord(t['skycoord_centroid']).icrs
    ok = np.isfinite(sc.ra.deg)
    sc = sc[ok]
    ci, s, _ = sc.match_to_catalog_sky(virac); ri, _, _ = virac.match_to_catalog_sky(sc)
    k = (ri[ci] == np.arange(len(ci))) & (s <= 0.25 * u.arcsec)
    if k.sum() < 8:
        continue
    mc, mr = sc[np.where(k)[0]], virac[ci[k]]
    dra = (mc.ra.deg - mr.ra.deg) * np.cos(np.radians(mc.dec.deg)) * 3.6e6
    dd = (mc.dec.deg - mr.dec.deg) * 3.6e6
    m, md = np.nanmedian(dra), np.nanmedian(dd); cl = np.hypot(dra - m, dd - md) <= 120
    cdra, cddec = -float(np.nanmedian(dra[cl])), -float(np.nanmedian(dd[cl]))   # correction (on-sky)
    RA0, DE0 = float(t.meta['RAOFFSET']), float(t.meta['DEOFFSET'])
    dec0 = float(np.nanmedian(mc.dec.deg))
    mrf = re.search(r'(nrc[ab]\d)_visit(\d+)_vgroup\d+_exp(\d+)', fn)
    mod, vis, exp = mrf.group(1), int(mrf.group(2)), int(mrf.group(3))
    rows.append(dict(Filter='F115W', Module=mod, Visit=f'jw01182004{vis:03d}', Exposure=exp,
                     dra=RA0 + (cdra / 1000.0) / np.cos(np.radians(dec0)),
                     ddec=DE0 + (cddec / 1000.0)))
ot = Table(rows)
ot.write(OFFSETS_OUT, overwrite=True)
print(f"[1] offsets table: {len(ot)} frames -> {OFFSETS_OUT}")

# ---- 2. merge ----
merge_individual_frames(module='merged', desat=False, filtername='f115w', progid='1182',
                        bgsub=False, epsf=False, fitpsf=False, blur=False, suffix='_basic',
                        method='dao', target='brick', exposure_numbers=np.arange(1, 25),
                        offsets_table=ot, iteration_label=None, resbgsub=False, basepath=BASE)
print("[2] merge_individual_frames done")

# ---- 3. trimmed canonical from _allcols (skip replace_saturated) + final VIRAC2 null ----
ac = Table.read(ALLCOLS)
column_names = ('flux_fit', 'flux_err', 'skycoord', 'qfit', 'cfit', 'flux_init', 'flags', 'local_bkg', 'iter_detected', 'group_size')
minimal = {c: ac[f'{c}_avg'] for c in column_names if f'{c}_avg' in ac.colnames}
for key in ('dra_avg', 'ddec_avg', 'std_ra', 'std_dec', 'nmatch', 'nmatch_good', 'flux_err_prop'):
    if key in ac.colnames: minimal[key.split('_avg')[0]] = ac[key]
mt = Table(minimal); mt.meta = dict(ac.meta)
sc = SkyCoord(mt['skycoord']); mt = mt[~(np.isnan(sc.ra.deg) | np.isnan(sc.dec.deg))]
# final rigid null to VIRAC2
sc = SkyCoord(mt['skycoord']).icrs
o = off(sc, virac, sep=0.3 * u.arcsec)
ra = sc.ra.deg - (o['dRA'] / 3.6e6) / np.cos(np.radians(sc.dec.deg))
dec = sc.dec.deg - o['dDec'] / 3.6e6
mt['skycoord'] = SkyCoord(ra * u.deg, dec * u.deg)
mt.meta['FRAME'] = 'VIRAC2 (Gaia-tied) epoch 2022.70; new-crf per-frame align + final VIRAC2 null'
mt.write(CANON, overwrite=True)
print(f"[3] final canonical: {CANON} ({len(mt)} rows); pre-null bulk vs VIRAC2 was ({o['dRA']},{o['dDec']})")

# ---- 4. verify ----
scf = SkyCoord(mt['skycoord']).icrs
print("[4] FINAL F115W catalog vs references:")
for lab, ref in [('VIRAC2', virac), ('Gaia DR3', gaia), ('GSC3.2(Gaia)', gscg)]:
    print(f"     vs {lab:13}: {off(scf, ref, sep=0.25*u.arcsec, clip=60)}")
multi = farr(mt['nmatch']) >= 3
print(f"     internal std (nmatch>=3): ({np.nanmedian(farr(mt['std_ra'])[multi])*3.6e6:.1f},{np.nanmedian(farr(mt['std_dec'])[multi])*3.6e6:.1f}) mas")
print("DONE final F115W catalog")
