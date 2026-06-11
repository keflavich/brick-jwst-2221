#!/usr/bin/env python
"""
Validate brick F2550W point sources against the sickle program's
SICKLE-MIR-BACKGROUND (jw03958 obs 003) F770W/F1130W/F1500W images, which
lie entirely within the brick F2550W mosaic, and classify the sources'
7.7-25.5 um SEDs (YSO candidates vs evolved stars).

Method:
- DAOFind on the (median-filter high-passed) F2550W mosaic, restricted to
  the o003 footprint.
- Forced aperture photometry (r = 1.5 FWHM, sky annulus 2.5-4 FWHM) at each
  position in all four bands; MJy/sr -> Jy via pixel area.
- alpha = dlog(lambda F_lambda)/dlog(lambda) over available 7.7-25.5 bands.
  alpha > 0.3 Class I-like (rising); -0.3..0.3 flat; < -0.3 falling
  (disk/photosphere/evolved).  Caveat: dusty AGB (OH/IR) can also rise.
- NIR counterpart check vs the 2023-05-19 crowdsource long-wave refcat.
"""
import numpy as np
import glob
import os
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.stats import mad_std
import astropy.units as u
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry, ApertureStats
from scipy.ndimage import median_filter
import warnings
warnings.filterwarnings('ignore')

# Per-band astrometric offsets (band frame minus f182m-refcat frame, arcsec),
# measured 2026-06-11 by offset-histogram stacking: o003 bands from 466-1574
# stars each (mad ~0.1"); F2550W from 6 F2550W<->F1500W common sources that
# agree to 10 mas after correcting F1500W (the refcat itself cannot register
# F2550W: dusty 25um sources rarely have NIR counterparts, the histogram is
# chance-dominated).  true_sky = measured - offset.
BANDS = {
    'F770W': dict(wl=7.7, fwhm_as=0.269, dra=-3.435, ddec=+1.381,
                  fn='/orange/adamginsburg/jwst/sickle/F770W/pipeline/jw03958-o003_t001_miri_f770w_i2d.fits'),
    'F1130W': dict(wl=11.3, fwhm_as=0.375, dra=-3.428, ddec=+1.404,
                   fn='/orange/adamginsburg/jwst/sickle/F1130W/pipeline/jw03958-o003_t001_miri_f1130w_i2d.fits'),
    'F1500W': dict(wl=15.0, fwhm_as=0.488, dra=-3.432, ddec=+1.405,
                   fn='/orange/adamginsburg/jwst/sickle/F1500W/pipeline/jw03958-o003_t001_miri_f1500w_i2d.fits'),
    'F2550W': dict(wl=25.5, fwhm_as=0.803, dra=-4.697, ddec=-2.469,
                   fn='/orange/adamginsburg/jwst/brick/F2550W/pipeline/jw02221-o002_t001_miri_f2550w_i2d.fits'),
}


def to_true(coords, band):
    """Measured position in `band`'s frame -> refcat ('true') frame."""
    info = BANDS[band]
    return SkyCoord(ra=coords.ra - (info['dra'] * u.arcsec) / np.cos(coords.dec),
                    dec=coords.dec - info['ddec'] * u.arcsec)


def to_band(coords, band):
    """Refcat-frame position -> `band`'s (uncorrected) image frame."""
    info = BANDS[band]
    return SkyCoord(ra=coords.ra + (info['dra'] * u.arcsec) / np.cos(coords.dec),
                    dec=coords.dec + info['ddec'] * u.arcsec)

# --- detect on F2550W ---
fh = fits.open(BANDS['F2550W']['fn'])
sci = fh['SCI']
ww25 = WCS(sci.header)
data25 = sci.data
mf = median_filter(np.nan_to_num(data25, nan=np.nanmedian(data25)), size=31)
hp = np.nan_to_num(data25) - mf
err = mad_std(hp)
srcs = DAOStarFinder(threshold=5 * err, fwhm=7.3)(hp)
sc25 = to_true(ww25.pixel_to_world(srcs['xcentroid'], srcs['ycentroid']), 'F2550W')
print(f'F2550W: {len(srcs)} sources at 5 sigma over whole mosaic (positions corrected to refcat frame)')

# restrict to o003 footprint
fh7 = fits.open(BANDS['F770W']['fn'])
ww7 = WCS(fh7['SCI'].header)
x7, y7 = ww7.world_to_pixel(to_band(sc25, 'F770W'))
ny7, nx7 = fh7['SCI'].data.shape
in_o003 = (x7 > 10) & (x7 < nx7 - 10) & (y7 > 10) & (y7 < ny7 - 10)
srcs = srcs[in_o003]
sc25 = sc25[in_o003]
print(f'{len(srcs)} F2550W sources inside the o003 (Brick background) footprint')


def forced_phot(fn, coords, fwhm_as, band):
    fhb = fits.open(fn)
    h = fhb['SCI']
    wwb = WCS(h.header)
    db = h.data
    pixscale = np.sqrt(np.abs(np.linalg.det(wwb.pixel_scale_matrix))) * 3600  # arcsec/pix
    pixar_sr = (pixscale / 206265.)**2
    x, y = wwb.world_to_pixel(to_band(coords, band))
    r = 1.5 * fwhm_as / pixscale
    ap = CircularAperture(np.transpose([x, y]), r=r)
    ann = CircularAnnulus(np.transpose([x, y]), r_in=2.5 * fwhm_as / pixscale,
                          r_out=4.0 * fwhm_as / pixscale)
    bkg = ApertureStats(db, ann).median
    tab = aperture_photometry(db, ap)
    flux_mjysr = tab['aperture_sum'] - bkg * ap.area
    flux_jy = flux_mjysr * 1e6 * pixar_sr
    # detection significance against annulus scatter
    noise = ApertureStats(db, ann).std * np.sqrt(ap.area)
    snr = (tab['aperture_sum'] - bkg * ap.area) / noise
    off = (x < 0) | (x > db.shape[1] - 1) | (y < 0) | (y > db.shape[0] - 1)
    flux_jy[off] = np.nan
    snr[off] = np.nan
    return np.asarray(flux_jy), np.asarray(snr)


out = Table()
out['ra'] = sc25.ra.deg
out['dec'] = sc25.dec.deg
for band, info in BANDS.items():
    fj, snr = forced_phot(info['fn'], sc25, info['fwhm_as'], band)
    out[f'flux_{band}'] = fj
    out[f'snr_{band}'] = snr

# spectral index over detected bands (SNR>3), lambda F_lambda power law
wls = np.array([BANDS[b]['wl'] for b in BANDS])
alphas = []
ndet = []
for row in out:
    lf, lw = [], []
    for b in BANDS:
        f = row[f'flux_{b}']
        if np.isfinite(f) and f > 0 and row[f'snr_{b}'] > 3:
            # lambda F_lambda ~ nu F_nu ~ F_jy / lambda
            lf.append(np.log10(f / BANDS[b]['wl']))
            lw.append(np.log10(BANDS[b]['wl']))
    ndet.append(len(lf))
    if len(lf) >= 2:
        alphas.append(np.polyfit(lw, lf, 1)[0])
    else:
        alphas.append(np.nan)
out['n_det'] = ndet
out['alpha_ir'] = alphas

# NIR counterparts
ref = Table.read('/orange/adamginsburg/jwst/brick/catalogs/2023-05-19_crowdsource_based_nircam-long_reference_astrometric_catalog.fits')
refsc = SkyCoord(ref['skycoord']) if 'skycoord' in ref.colnames else SkyCoord(ref['RA'], ref['DEC'], unit='deg')
idx, sep, _ = sc25.match_to_catalog_sky(refsc)
out['nir_sep_arcsec'] = sep.to(u.arcsec).value
out['has_nir'] = sep < 0.5 * u.arcsec

confirmed = (out['n_det'] >= 2)
print(f'\nconfirmed in >=2 MIRI bands: {confirmed.sum()}/{len(out)}')
print(f'with NIRCam counterpart (<0.5"): {out["has_nir"].sum()}/{len(out)}')

rising = confirmed & (out['alpha_ir'] > 0.3)
flat = confirmed & (out['alpha_ir'] > -0.3) & (out['alpha_ir'] <= 0.3)
falling = confirmed & (out['alpha_ir'] <= -0.3)
print(f'alpha>0.3 (rising, YSO-candidate-like): {rising.sum()}')
print(f'flat (-0.3..0.3): {flat.sum()}')
print(f'falling (<-0.3, photosphere/evolved-like): {falling.sum()}')

print('\nPer-source table (confirmed sources):')
out['alpha_ir'].format = '.2f'
for b in BANDS:
    out[f'flux_{b}'].format = '.4g'
    out[f'snr_{b}'].format = '.1f'
out['ra'].format = '.5f'
out['dec'].format = '.5f'
out['nir_sep_arcsec'].format = '.2f'
out[confirmed | (out['snr_F2550W'] > 10)].pprint(max_lines=200, max_width=250)

outfn = '/orange/adamginsburg/jwst/brick/catalogs/f2550w_sources_o003_validation.fits'
out.write(outfn, overwrite=True)
print(f'\nwrote {outfn}')
