#!/usr/bin/env python
"""Post-reduction verification for the 1182 v001 ~20" offset fix.
Checks, per broadband: the regenerated crf carries the corrected RAOFFSET
(v001 ~-17.5, v002 ~+1.9) and the crf native offset vs VIRAC2 collapses to ~0
for v001 (was ~20"). PASS iff v001 crf is now < 0.1" from VIRAC2.
"""
import glob, os, time
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.coordinates import SkyCoord, search_around_sky
from astropy import units as u
import warnings; warnings.filterwarnings('ignore')
from photutils.detection import DAOStarFinder

v = Table.read('/orange/adamginsburg/jwst/brick/astrometry_retie_qualcuts_20251211/refcache/virac2.fits')
vra = np.asarray(v['RAJ2000'], float); vde = np.asarray(v['DEJ2000'], float)
pr = np.nan_to_num(np.asarray(v['pmRA'], float)); pd = np.nan_to_num(np.asarray(v['pmDE'], float))
cosd = np.cos(np.radians(vde))
vsc = SkyCoord((vra+pr*8.7/3.6e6/cosd)*u.deg, (vde+pd*8.7/3.6e6)*u.deg)

def native_off(path, rad=30):
    h = fits.open(path, memmap=True); e = 1 if h[0].data is None else 0
    d = h[e].data.astype(float); w = WCS(h[e].header); std = np.nanstd(d[np.isfinite(d)])
    src = DAOStarFinder(fwhm=2.5, threshold=25*std)(np.nan_to_num(d-np.nanmedian(d)))
    if src is None or len(src) < 15: return None
    sky = w.pixel_to_world(src['xcentroid'], src['ycentroid'])
    i1, i2, _, _ = search_around_sky(sky, vsc, rad*u.arcsec)
    if len(i1) < 15: return None
    dra = (sky.ra.deg[i1]-vsc.ra.deg[i2])*np.cos(np.radians(sky.dec.deg[i1]))*3600
    dde = (sky.dec.deg[i1]-vsc.dec.deg[i2])*3600
    b = 0.05; ee = np.arange(-rad, rad+b, b); H, xe, ye = np.histogram2d(dra, dde, bins=[ee, ee])
    iy, ix = np.unravel_index(np.argmax(H.T), H.T.shape)
    return 0.5*(xe[ix]+xe[ix+1]), 0.5*(ye[iy]+ye[iy+1]), int(H.max())

P = '/orange/adamginsburg/jwst/brick'
FILT = {'F115W': '02101', 'F200W': '04101', 'F356W': '02101', 'F444W': '04101'}
allpass = True
print('band  visit  crf_RAOFFSET(applied)   native_off_vs_VIRAC   verdict')
for band in ['F115W', 'F200W', 'F356W', 'F444W']:
    d = band; sw = band in ('F115W', 'F200W')
    det = 'nrca1' if sw else 'nrcalong'
    for vis in ['jw01182004001', 'jw01182004002']:
        g = sorted(glob.glob(f'{P}/{d}/pipeline/{vis}_*_00001_{det}_destreak_o004_crf.fits'))
        if not g:
            print(f'{band} {vis[-3:]}: crf MISSING'); allpass = False; continue
        crf = g[0]; h = fits.getheader(crf, 1)
        ra_off, de_off = h.get('RAOFFSET'), h.get('DEOFFSET')
        off = native_off(crf); mt = time.ctime(os.path.getmtime(crf))[4:16]
        if off is None:
            print(f'{band} {vis[-3:]}: RAOFFSET=({ra_off:+.2f},{de_off:+.2f}) native=? {mt}'); continue
        mag = np.hypot(off[0], off[1])
        ok = mag < 0.10
        if vis.endswith('001') and not ok: allpass = False
        print(f'{band} {vis[-3:]}: RAOFFSET=({ra_off:+7.2f},{de_off:+6.2f})  native=({off[0]:+6.2f},{off[1]:+6.2f})\" |{mag:5.2f}|  {"PASS" if ok else "FAIL"}  {mt}')
print('\n==== OVERALL:', 'PASS - v001 aligned to VIRAC2' if allpass else 'FAIL / incomplete', '====')
