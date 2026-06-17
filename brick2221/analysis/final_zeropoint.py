#!/usr/bin/env python
"""Final global zero-point correction of the Gaia-frame F115W catalog to Gaia/GSC3.2.

The per-frame VIRAC2 alignment leaves a small bulk residual (estimator/sampling +
VIRAC2's own ~5 mas tie to Gaia). Anchor the absolute zero-point directly to the
Gaia DR3 / GSC3.2 frame (the NIRSpec frame) with one rigid shift, then verify.
"""
import warnings
import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import stats

warnings.simplefilter('ignore')
BASE = '/orange/adamginsburg/jwst/brick'
EPOCH = 2022.70
IN = f'{BASE}/catalogs/f115w_merged_indivexp_merged_dao_basic_GAIAFRAME.fits'
OUT = f'{BASE}/catalogs/f115w_merged_indivexp_merged_dao_basic_GAIAFRAME.fits'


def farr(x):
    return np.asarray(np.ma.filled(np.ma.masked_invalid(np.asarray(x, float)), np.nan), float)


def prop(ra, dec, pmra, pmde, dt):
    pmra = np.where(np.isfinite(pmra), pmra, 0.); pmde = np.where(np.isfinite(pmde), pmde, 0.)
    return ra + (pmra * dt / 3.6e6) / np.cos(np.radians(dec)), dec + (pmde * dt / 3.6e6)


def off(a, b, sep=0.3*u.arcsec, clip=60):
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


# references
g = Table.read(f'{BASE}/astrometry_analysis/reference_cache/basic_merged_photometry_tables_merged_gaia.fits')
gaia = SkyCoord(*prop(farr(g['RA_ICRS']), farr(g['DE_ICRS']), farr(g['pmRA']), farr(g['pmDE']), EPOCH-2016.0), unit='deg')
gsc = Table.read(f'{BASE}/astrometry_diag/refcache/gsc32.fits')
gb = np.isfinite(farr(gsc['gaiaGmag']))
gscg = SkyCoord(*prop(farr(gsc['ra'])[gb], farr(gsc['dec'])[gb], farr(gsc['rapm'])[gb], farr(gsc['decpm'])[gb], EPOCH-farr(gsc['epoch'])[gb]), unit='deg')
v = Table.read(f'{BASE}/astrometry_diag/refcache/virac2.fits')
virac = SkyCoord(*prop(farr(v['RAJ2000']), farr(v['DEJ2000']), farr(v['pmRA']), farr(v['pmDE']), EPOCH-2016.0), unit='deg')

t = Table.read(IN)
sc = SkyCoord(t['skycoord']).icrs

# anchor zero-point to Gaia DR3 directly (tight 0.2" match, bright real counterparts)
o = off(sc, gaia, sep=0.25*u.arcsec, clip=40)
print("before final shift, vs Gaia:", o)
dra_fix, ddec_fix = o['dRA'], o['dDec']
ra = sc.ra.deg - (dra_fix/3.6e6)/np.cos(np.radians(sc.dec.deg))
dec = sc.dec.deg - ddec_fix/3.6e6
sc2 = SkyCoord(ra*u.deg, dec*u.deg)
print(f"applied final rigid shift (-{dra_fix}, -{ddec_fix}) mas to anchor on Gaia DR3\n")

for lab, ref in [('Gaia DR3', gaia), ('GSC3.2(Gaia)', gscg), ('VIRAC2', virac)]:
    print(f"  vs {lab:14}: {off(sc2, ref, sep=0.25*u.arcsec, clip=40)}")

t['skycoord'] = sc2
t.meta['ZEROPOINT'] = f'final rigid shift ({-dra_fix},{-ddec_fix}) mas to Gaia DR3 @2022.70'
t.write(OUT, overwrite=True)
print(f"\nUpdated {OUT}")
