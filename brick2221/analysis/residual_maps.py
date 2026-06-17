#!/usr/bin/env python
"""Spatial residual maps of merged catalogs vs VIRAC2 (Gaia frame, dense).

Distinguishes a rigid frame offset (uniform arrows) from a distortion field
(spatially-coherent structure) -- determines whether a per-filter rigid shift,
a per-frame shift, or a full distortion model is needed to reach the Gaia frame.
"""
import os, warnings
import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.simplefilter('ignore')
BASE = '/orange/adamginsburg/jwst/brick'
OUT = f'{BASE}/astrometry_diag'
EPOCH = 2022.70
CATS = {'F115W': (f'{BASE}/catalogs/f115w_merged_indivexp_merged_dao_basic.fits', 'Jmag'),
        'F200W': (f'{BASE}/catalogs/f200w_merged_indivexp_merged_dao_basic.fits', 'Ksmag'),
        'F182M': (f'{BASE}/catalogs/f182m_merged_indivexp_merged_dao_basic.fits', 'Hmag')}


def farr(x):
    return np.asarray(np.ma.filled(np.ma.masked_invalid(np.asarray(x, float)), np.nan), float)


def load_virac():
    v = Table.read(f'{BASE}/astrometry_diag/refcache/virac2.fits')
    pmra = np.where(np.isfinite(farr(v['pmRA'])), farr(v['pmRA']), 0.0)
    pmde = np.where(np.isfinite(farr(v['pmDE'])), farr(v['pmDE']), 0.0)
    dt = EPOCH - 2016.0
    ra = farr(v['RAJ2000']) + (pmra * dt / 3.6e6) / np.cos(np.radians(farr(v['DEJ2000'])))
    dec = farr(v['DEJ2000']) + (pmde * dt / 3.6e6)
    return SkyCoord(ra * u.deg, dec * u.deg), v


def main():
    vsc, v = load_virac()
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    for ax, (filt, (fn, band)) in zip(axes, CATS.items()):
        t = Table.read(fn)
        sc = SkyCoord(t['skycoord']).icrs
        fl = farr(t['flux']); ok = np.isfinite(sc.ra.deg) & np.isfinite(fl) & (fl > 0)
        sc, fl = sc[ok], fl[ok]; mi = -2.5 * np.log10(fl)
        ci, sep, _ = sc.match_to_catalog_sky(vsc)
        ri, _, _ = vsc.match_to_catalog_sky(sc)
        keep = (ri[ci] == np.arange(len(ci))) & (sep <= 0.3 * u.arcsec)
        idx, rid = np.where(keep)[0], ci[keep]
        mc, mr = sc[idx], vsc[rid]
        dra = (mc.ra.deg - mr.ra.deg) * np.cos(np.radians(mc.dec.deg)) * 3.6e6
        ddec = (mc.dec.deg - mr.dec.deg) * 3.6e6
        # flux match to nearest VIRAC2 band
        rm = farr(v[band])[rid]; fin = np.isfinite(rm) & np.isfinite(mi[idx])
        if fin.sum() > 20:
            dm = mi[idx][fin] - rm[fin]; md = np.nanmedian(dm); s = max(stats.mad_std(dm, ignore_nan=True), 0.3)
            f2 = fin.copy(); f2[fin] = np.abs(dm - md) <= 3 * s
        else:
            f2 = np.ones(len(idx), bool)
        ra_d, dec_d = mc.ra.deg[f2], mc.dec.deg[f2]
        dra, ddec = dra[f2], ddec[f2]
        # sep clip around median
        mdra, mddec = np.nanmedian(dra), np.nanmedian(ddec)
        cl = np.hypot(dra - mdra, ddec - mddec) <= 150
        ra_d, dec_d, dra, ddec = ra_d[cl], dec_d[cl], dra[cl], ddec[cl]

        # binned median residual map
        nb = 12
        ra0, dec0 = np.median(ra_d), np.median(dec_d)
        x = (ra_d - ra0) * np.cos(np.radians(dec0)) * 3600
        y = (dec_d - dec0) * 3600
        xb = np.linspace(x.min(), x.max(), nb + 1); yb = np.linspace(y.min(), y.max(), nb + 1)
        gx, gy, gu, gv = [], [], [], []
        for i in range(nb):
            for j in range(nb):
                m = (x >= xb[i]) & (x < xb[i+1]) & (y >= yb[j]) & (y < yb[j+1])
                if m.sum() >= 8:
                    gx.append((xb[i]+xb[i+1])/2); gy.append((yb[j]+yb[j+1])/2)
                    gu.append(np.median(dra[m])); gv.append(np.median(ddec[m]))
        gx, gy, gu, gv = map(np.array, (gx, gy, gu, gv))
        q = ax.quiver(gx, gy, gu, gv, np.hypot(gu, gv), cmap='viridis', angles='xy',
                      scale_units='xy', scale=8.0, width=0.005)
        ax.quiverkey(q, 0.85, 1.03, 50, '50 mas', labelpos='E')
        plt.colorbar(q, ax=ax, label='|residual| mas', pad=0.02)
        ax.set_aspect('equal'); ax.set_xlabel('rel RA [arcsec]'); ax.set_ylabel('rel Dec [arcsec]')
        gvec = np.hypot(np.median(dra), np.median(ddec))
        spatial = stats.mad_std(np.concatenate([gu, gv]), ignore_nan=True)
        ax.set_title(f'{filt} - VIRAC2  N={len(dra)}\nglobal med=({np.median(dra):.0f},{np.median(ddec):.0f}) '
                     f'|{gvec:.0f}| mas; binned-scatter={spatial:.0f} mas', fontsize=10)
    fig.suptitle('Merged catalog residuals vs VIRAC2 (Gaia frame): rigid offset vs distortion field', fontsize=12)
    fig.tight_layout()
    fig.savefig(f'{OUT}/residual_maps_vs_virac2.png', dpi=150, bbox_inches='tight')
    print(f"Wrote {OUT}/residual_maps_vs_virac2.png")


if __name__ == '__main__':
    main()
