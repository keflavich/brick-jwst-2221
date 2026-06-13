#!/usr/bin/env python
"""
SED + cutout plots for the F2550W-detected sources in the Brick's
F770W-overlap region (sickle obs-003 = SICKLE-MIR-BACKGROUND footprint).

Adapted from sgrb2/NB/sgrb2_jwst/aperture_photometry_F2550W.py (Budaiev).
Differences:
  * source positions = brick F2550W detections inside the o003 footprint
    (catalogs/f2550w_sources_o003_validation.fits, already on the NIRCam
    refcat frame);
  * brick band set (NIRCam 2221+1182 + sickle MIRI o003 + brick F2550W);
  * all images carry the MIRIDRA WCS correction / are NIRCam-native, so
    aperture photometry by sky coordinate is consistent across bands;
  * background annulus 2.5-4.5x F2550W FWHM lies OUTSIDE the ~1" first-Airy
    null ("negative ring"), so the moat does not bias the sky estimate;
  * adds a per-source SED panel (lambda F_lambda vs lambda) beside the
    cutout grid.

Single-process; run on a node.  ~15 sources x 13 bands.
"""
import os
import warnings
import traceback
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.nddata import Cutout2D
from astropy.visualization import simple_norm
from astropy.stats import SigmaClip
from photutils.aperture import (SkyCircularAperture, SkyCircularAnnulus,
                                ApertureStats, aperture_photometry)
warnings.filterwarnings('ignore')

NRC = '/orange/adamginsburg/jwst/brick/{B}/pipeline/jw02221-o001_t001_nircam_clear-{b}-merged_i2d.fits'
NRC1182 = '/orange/adamginsburg/jwst/brick/{B}/pipeline/jw01182-o004_t001_nircam_clear-{b}-merged_i2d.fits'
SICKLE = '/orange/adamginsburg/jwst/sickle/{B}/pipeline/jw03958-o003_t001_miri_{b}_i2d.fits'

image_filenames = {
    'f115w':  NRC1182.format(B='F115W', b='f115w'),
    'f182m':  NRC.format(B='F182M', b='f182m'),
    'f187n':  NRC.format(B='F187N', b='f187n'),
    'f200w':  NRC1182.format(B='F200W', b='f200w'),
    'f212n':  NRC.format(B='F212N', b='f212n'),
    'f356w':  NRC1182.format(B='F356W', b='f356w'),
    'f405n':  NRC.format(B='F405N', b='f405n'),
    'f410m':  NRC.format(B='F410M', b='f410m'),
    'f444w':  NRC1182.format(B='F444W', b='f444w'),
    'f466n':  NRC.format(B='F466N', b='f466n'),
    'f770w':  SICKLE.format(B='F770W', b='f770w'),
    'f1130w': SICKLE.format(B='F1130W', b='f1130w'),
    'f1500w': SICKLE.format(B='F1500W', b='f1500w'),
    'f2550w': '/orange/adamginsburg/jwst/brick/F2550W/pipeline/jw02221-o002_t001_miri_f2550w_i2d.fits',
}
wavelengths = {'f115w':1.15,'f182m':1.82,'f187n':1.87,'f200w':2.00,'f212n':2.12,
               'f356w':3.56,'f405n':4.05,'f410m':4.10,'f444w':4.44,'f466n':4.66,
               'f770w':7.70,'f1130w':11.30,'f1500w':15.00,'f2550w':25.50}

F2550W_FWHM = 0.85
APER = 1.0 * F2550W_FWHM * u.arcsec
BKG_IN = 2.5 * F2550W_FWHM * u.arcsec      # outside the ~1" first-null ring
BKG_OUT = 4.5 * F2550W_FWHM * u.arcsec
CUTOUT_SIZE = 6.0 * u.arcsec
F2550W_APCORR = 100.0 / 68.0   # 0.85" aperture, per JWST MIRI docs

OUTDIR = '/orange/adamginsburg/jwst/brick/F2550W/sed_cutouts'
os.makedirs(OUTDIR, exist_ok=True)


def open_image(path):
    with fits.open(path) as hdul:
        hdu = hdul['SCI'] if 'SCI' in hdul else hdul[0]
        data = hdu.data.astype(float)
        ww = WCS(hdu.header, naxis=2)
    ps = ww.proj_plane_pixel_scales()
    pixarea_sr = abs((ps[0].to(u.rad) * ps[1].to(u.rad)).value)
    return data, ww, pixarea_sr


def phot_one(band, path, sky, t_out):
    data, ww, pixarea = open_image(path)
    src = SkyCircularAperture(sky, r=APER).to_pixel(ww)
    ann = SkyCircularAnnulus(sky, r_in=BKG_IN, r_out=BKG_OUT).to_pixel(ww)
    sc = SigmaClip(sigma=3.0, maxiters=10)
    bs = ApertureStats(data, ann, sigma_clip=sc)
    npix = np.array([m.data.sum() for m in src.to_mask(method='exact')])
    raw = np.array(aperture_photometry(data, src)['aperture_sum'])
    net = raw - bs.median * npix                 # MJy/sr * pix
    flux_jy = net * pixarea * 1e6
    if band == 'f2550w':
        flux_jy = flux_jy * F2550W_APCORR
    nann = np.array([a.area for a in ann])
    eflux = bs.std * np.sqrt(npix + npix**2 / nann) * pixarea * 1e6
    xy = np.array(ww.world_to_pixel(sky)).T
    off = (xy[:,0] < 0) | (xy[:,0] >= data.shape[1]) | (xy[:,1] < 0) | (xy[:,1] >= data.shape[0])
    flux_jy = np.where(off, np.nan, flux_jy)
    t_out[f'{band}_flux_jy'] = flux_jy
    t_out[f'{band}_eflux_jy'] = np.where(off, np.nan, eflux)


def cutout_and_sed(sid, coord, imgs, row):
    bands = sorted(imgs, key=lambda b: wavelengths[b])
    ncols = 5
    nrows = int(np.ceil(len(bands)/ncols)) + 1   # +1 row for the SED
    fig = plt.figure(figsize=(3.2*ncols, 3.2*nrows))
    for i, band in enumerate(bands):
        ax = fig.add_subplot(nrows, ncols, i+1)
        data, ww = imgs[band]
        try:
            x, y = ww.world_to_pixel(coord)
            pix = np.mean(np.abs(ww.proj_plane_pixel_scales()[0]).to(u.arcsec).value)
            npx = int(CUTOUT_SIZE.to(u.arcsec).value / pix)
            co = Cutout2D(data, (x,y), (npx,npx), wcs=ww, mode='partial', fill_value=np.nan)
            if not np.isfinite(co.data).any():
                raise ValueError('all-NaN')
            ax.imshow(co.data, origin='lower', cmap='inferno',
                      norm=simple_norm(co.data[np.isfinite(co.data)], stretch='log', percent=99.5))
            x0, y0 = co.data.shape[1]/2, co.data.shape[0]/2
            for rr, c, ls in [(APER,'red','-'), (BKG_IN,'cyan','--'), (BKG_OUT,'cyan','--')]:
                ax.add_patch(plt.Circle((x0,y0), rr.value/pix, ec=c, fc='none', lw=1.2, ls=ls))
        except Exception as e:
            ax.text(0.5,0.5,f'{type(e).__name__}', ha='center', va='center', transform=ax.transAxes, fontsize=8)
        ax.set_title(f'{band.upper()} {wavelengths[band]:.2f}um', fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    # SED panel (bottom row, spanning)
    axs = fig.add_subplot(nrows, 1, nrows)
    wl = np.array([wavelengths[b] for b in bands])
    fx = np.array([row[f'{b}_flux_jy'] for b in bands])
    ex = np.array([row[f'{b}_eflux_jy'] for b in bands])
    lflf = wl * 0 + fx / wl       # nu F_nu proportional to F_jy/lambda (lambda F_lambda)
    ok = np.isfinite(fx) & (fx > 0)
    axs.errorbar(wl[ok], fx[ok]*1e3, yerr=ex[ok]*1e3, fmt='o-', color='k')
    det = ok & (fx > 3*ex)
    axs.plot(wl[~ok | (fx <= 3*ex)], np.abs(fx[~ok | (fx<=3*ex)])*1e3, 'v', color='0.6', ms=5)
    axs.set_xscale('log'); axs.set_yscale('log')
    axs.set_xlabel('wavelength (um)'); axs.set_ylabel('flux (mJy)')
    axs.set_xticks([1,2,3,5,7,10,15,25]); axs.set_xticklabels(['1','2','3','5','7','10','15','25'])
    axs.grid(alpha=0.3)
    fig.suptitle(f"Brick F2550W source {sid}  ({coord.to_string('hmsdms', precision=1)})", fontsize=13)
    plt.tight_layout()
    out = f'{OUTDIR}/source_{sid:03d}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
    return out


def main():
    cat = Table.read('/orange/adamginsburg/jwst/brick/catalogs/f2550w_sources_o003_validation.fits')
    sky = SkyCoord(cat['ra'], cat['dec'], unit='deg')
    print(f'{len(sky)} F2550W sources in F770W-overlap region')

    out = Table()
    out['source_id'] = np.arange(len(sky))
    out['ra'] = sky.ra.deg; out['dec'] = sky.dec.deg
    imgs = {}
    for band, path in image_filenames.items():
        if not os.path.exists(path):
            print(f'  MISSING {band}: {path}'); continue
        phot_one(band, path, sky, out)
        d, w, _ = open_image(path)
        imgs[band] = (d, w)
        print(f'  {band}: done')
    out.write(f'{OUTDIR}/brick_f2550w_overlap_sed_photometry.ecsv', overwrite=True)
    print(f'wrote {OUTDIR}/brick_f2550w_overlap_sed_photometry.ecsv')
    for sid in range(len(sky)):
        f = cutout_and_sed(sid, sky[sid], imgs, out[sid])
        print(f'  source {sid}: {f}')
    print('DONE')


if __name__ == '__main__':
    main()
