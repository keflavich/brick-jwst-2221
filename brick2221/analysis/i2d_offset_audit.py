#!/usr/bin/env python
"""Measure each 1182 i2d mosaic's WCS offset vs the (correct) catalog, by
projecting bright catalog stars into the mosaic and centroiding them.  A correct
i2d puts each star at its predicted pixel (offset ~0); a misaligned i2d shows a
systematic centroid offset = the mosaic WCS error.
"""
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.nddata import Cutout2D
import warnings, os; warnings.filterwarnings('ignore')

snap = Table.read('/blue/adamginsburg/adamginsburg/jwst/brick/astrometry_diag/m8_dedup_1182_snapshot_20260708.fits')

def bright_stars(band, nmax=800):
    sc = snap[f'skycoord_{band}']; f = np.asarray(snap[f'flux_jy_{band}'], float)
    ef = np.asarray(snap[f'eflux_jy_{band}'], float)
    q = np.asarray(snap[f'qfit_{band}'], float) if f'qfit_{band}' in snap.colnames else np.zeros(len(snap))
    ok = np.isfinite(sc.ra.deg) & np.isfinite(f) & (f > 0) & (ef > 0) & (f/ef > 30) & (q < 0.1)
    idx = np.argsort(f[ok])[::-1][:nmax*3]
    return SkyCoord(sc.ra.deg[ok][idx]*u.deg, sc.dec.deg[ok][idx]*u.deg), f[ok][idx]

def measure(path, sc, box=9):
    h = fits.open(path, memmap=True)
    ext = 1 if h[0].data is None else 0
    if len(h) > 1 and h[1].name == 'SCI': ext = 1
    data = h[ext].data; w = WCS(h[ext].header)
    scale = np.sqrt(np.abs(np.linalg.det(w.pixel_scale_matrix)))*3600*1000  # mas/pix
    px, py = w.world_to_pixel(sc)
    offx, offy = [], []
    ny, nx = data.shape
    for x0, y0 in zip(px, py):
        if not (box < x0 < nx-box and box < y0 < ny-box): continue
        xi, yi = int(round(x0)), int(round(y0))
        cut = data[yi-box:yi+box+1, xi-box:xi+box+1].astype(float)
        if not np.isfinite(cut).all() or cut.max() <= 0: continue
        cut = cut - np.median(cut); cut[cut < 0] = 0
        if cut.sum() <= 0: continue
        yy, xx = np.mgrid[0:2*box+1, 0:2*box+1]
        cx = (cut*xx).sum()/cut.sum(); cy = (cut*yy).sum()/cut.sum()
        # centroid vs the sub-pixel predicted position
        offx.append((xi-box+cx) - x0); offy.append((yi-box+cy) - y0)
        if len(offx) >= 600: break
    offx = np.array(offx); offy = np.array(offy)
    # robust: median, in mas (pixel offset * scale); note pixel x = -RA roughly
    return len(offx), np.median(offx)*scale, np.median(offy)*scale, scale

BASE = '/orange/adamginsburg/jwst/brick'
for band in ['f200w', 'f356w', 'f115w', 'f444w']:
    sc, _ = bright_stars(band)
    print(f'\n=== {band.upper()} (N bright cat stars={len(sc)}) ===')
    B = band; P = f'{BASE}/{B.upper()}/pipeline/jw01182-o004_t001_nircam_clear-{B}'
    for tag in ['-merged_i2d', '-merged_data_i2d', '-merged-reproject_i2d', '-nrcb_i2d', '-nrca_i2d']:
        path = P + tag + '.fits'
        if not os.path.exists(path): continue
        n, ox, oy, sc_mas = measure(path, sc)
        mag = np.hypot(ox, oy)
        flag = '  <-- ARCSEC' if mag > 500 else ('  <- off' if mag > 60 else '')
        mt = __import__('time').ctime(os.path.getmtime(path))[4:16]
        print(f'  {tag:24s} n={n:4d} scale={sc_mas:.1f}mas  centroid-offset=({ox:+7.1f},{oy:+7.1f}) mas |{mag:6.1f}| {mt}{flag}')
