#!/usr/bin/env python
"""Verify the 1182 F200W crowdsource model & residual mosaics are right:
data | model | residual at the PM-discrepancy pair positions.  A correct model
reproduces both resolved stars and leaves a flat residual (no star-shaped
positive/negative residuals) -> catalog positions & fluxes are trustworthy."""
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy import units as u
from astropy.table import Table

P = '/orange/adamginsburg/jwst/brick/F200W/pipeline'
DATA = f'{P}/jw01182-o004_t001_nircam_clear-f200w-merged_i2d.fits'
MODEL = f'{P}/jw01182-o004_t001_nircam_clear-f200w-merged_resbgsub_m7_daophot_basic_mergedcat_model_i2d.fits'
RESID = f'{P}/jw01182-o004_t001_nircam_clear-f200w-merged_resbgsub_m7_daophot_basic_mergedcat_residual_i2d.fits'
TOP = '/orange/adamginsburg/jwst/brick/astrometry_diag/pm_binaries/pm_discrepancy_top.fits'
OUT = '/orange/adamginsburg/jwst/brick/astrometry_diag/pm_binaries/verify_1182_model_residual.png'

d = fits.open(DATA, memmap=True); dd, dw = d[1].data, WCS(d[1].header)
m = fits.open(MODEL, memmap=True); md, mw = m[1].data, WCS(m[1].header)
rf = fits.open(RESID, memmap=True); rd, rw = rf[1].data, WCS(rf[1].header)
top = Table.read(TOP)

pick = [0, 4, 9, 10]              # bright, clean pairs
CUT = 2.5*u.arcsec
fig, axes = plt.subplots(len(pick), 3, figsize=(9, 3.0*len(pick)), squeeze=False)
for r, ti in enumerate(pick):
    c = SkyCoord(top['ra'][ti]*u.deg, top['dec'][ti]*u.deg)
    cuts = []
    for (data, wcs) in [(dd, dw), (md, mw), (rd, rw)]:
        cuts.append(Cutout2D(data, c, CUT, wcs=wcs))
    # shared stretch from the data cutout
    di = cuts[0].data.astype(float); fin = np.isfinite(di) & (di != 0)
    lo, hi = np.nanpercentile(di[fin], [3, 99.7])
    for col, (cut, name) in enumerate(zip(cuts, ['data', 'model', 'residual'])):
        ax = axes[r][col]; img = cut.data.astype(float); ny, nx = img.shape
        ax.imshow(img, origin='lower', cmap='gray', vmin=lo, vmax=hi)
        ax.set_xlim(-0.5, nx-0.5); ax.set_ylim(-0.5, ny-0.5)
        cx, cy = cut.wcs.world_to_pixel(c); ax.plot(cx, cy, '+', color='cyan', ms=11, mew=1.5)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0: ax.set_title(name, fontsize=10)
        if col == 0:
            ax.set_ylabel(f'#{ti} sep={top["compsep"][ti]:.2f}"\nKs={top["Ks"][ti]:.1f}', fontsize=8)
        if col == 2:  # residual RMS relative to data peak, as a subtraction-quality number
            rr = img[np.isfinite(img)]
            frac = np.std(rr) / (hi - lo)
            ax.text(0.03, 0.03, f'resid std/data-range={frac:.2f}', color='yellow',
                    fontsize=7, transform=ax.transAxes)
fig.suptitle('1182 F200W m7: data | model | residual  (cyan+ = VIRAC pos).  '
             'Flat residual at pair = correct model.', fontsize=10)
fig.tight_layout()
fig.savefig(OUT, dpi=130, bbox_inches='tight')
print('wrote', OUT)
