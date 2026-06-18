#!/usr/bin/env python
"""Re-anchor the per-frame-aligned F115W catalog to the VIRAC2 (DR2, II/387) frame.

Decision: adopt VIRAC2 as the reference frame -- it covers the full pointing
footprint contiguously (unlike GNS, which is a bounded HAWK-I mosaic), giving one
uniform NIR frame across all pointings. VIRAC2 is tied to Gaia DR3 (~5 mas), so
this is the Gaia frame realized densely; it is NOT the 2MASS-tied guide-star
position frame. Use the SAME frame for all filters.
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
OUT = f'{BASE}/catalogs/f115w_merged_indivexp_merged_dao_basic_VIRAC2FRAME.fits'


def farr(x):
    return np.asarray(np.ma.filled(np.ma.masked_invalid(np.asarray(x, float)), np.nan), float)


def prop(ra, dec, pmra, pmde, dt):
    pmra = np.where(np.isfinite(pmra), pmra, 0.); pmde = np.where(np.isfinite(pmde), pmde, 0.)
    return ra + (pmra * dt / 3.6e6) / np.cos(np.radians(dec)), dec + (pmde * dt / 3.6e6)


def off(a, b, sep=0.3*u.arcsec, clip=120):
    i, s, _ = a.match_to_catalog_sky(b); ri, _, _ = b.match_to_catalog_sky(a)
    k = (ri[i] == np.arange(len(i))) & (s <= sep)
    mc, mr = a[np.where(k)[0]], b[i[k]]
    dra = (mc.ra.deg - mr.ra.deg) * np.cos(np.radians(mc.dec.deg)) * 3.6e6
    dd = (mc.dec.deg - mr.dec.deg) * 3.6e6
    m, md = np.nanmedian(dra), np.nanmedian(dd); cl = np.hypot(dra-m, dd-md) <= clip
    dra, dd = dra[cl], dd[cl]; n = len(dra)
    return dict(N=n, dRA=round(float(np.nanmedian(dra)),2), dDec=round(float(np.nanmedian(dd)),2),
                madRA=round(float(stats.mad_std(dra)),1), madDec=round(float(stats.mad_std(dd)),1),
                semRA=round(float(stats.mad_std(dra)/np.sqrt(n)),2), semDec=round(float(stats.mad_std(dd)/np.sqrt(n)),2))


v = Table.read(f'{BASE}/astrometry_diag/refcache/virac2.fits')
virac = SkyCoord(*prop(farr(v['RAJ2000']), farr(v['DEJ2000']), farr(v['pmRA']), farr(v['pmDE']), EPOCH-2014.0), unit='deg')  # VIRAC2 ref epoch 2014.0 (II/387), NOT 2016.0
g = Table.read(f'{BASE}/astrometry_analysis/reference_cache/basic_merged_photometry_tables_merged_gaia.fits')
gaia = SkyCoord(*prop(farr(g['RA_ICRS']), farr(g['DE_ICRS']), farr(g['pmRA']), farr(g['pmDE']), EPOCH-2016.0), unit='deg')

t = Table.read(IN)
sc = SkyCoord(t['skycoord']).icrs
o = off(sc, virac, sep=0.3*u.arcsec)
print("current catalog vs VIRAC2:", o)
ra = sc.ra.deg - (o['dRA']/3.6e6)/np.cos(np.radians(sc.dec.deg)); dec = sc.dec.deg - o['dDec']/3.6e6
sc2 = SkyCoord(ra*u.deg, dec*u.deg)
print(f"applied rigid shift (-{o['dRA']}, -{o['dDec']}) mas to anchor on VIRAC2\n")
print("  vs VIRAC2 :", off(sc2, virac, sep=0.3*u.arcsec))
print("  vs Gaia   :", off(sc2, gaia, sep=0.25*u.arcsec, clip=40), "(~VIRAC2-Gaia frame diff, expected)")

t['skycoord'] = sc2
t.meta['FRAME'] = 'VIRAC2 (VVV IR Astrometric Cat v2, II/387), epoch 2022.70; tied to Gaia DR3 ~5 mas'
t.meta['ZEROPOINT'] = f'anchored to VIRAC2 (rigid shift {-o["dRA"]},{-o["dDec"]} mas from prior GAIAFRAME)'
t.write(OUT, overwrite=True)
print(f"\nWrote {OUT}: {len(t)} sources")
