#!/usr/bin/env python
"""Per-frame astrometric registration of sickle prop-3958 o002 F770W crf frames
directly to the NIRCam F480M reference catalog (2026-06-13).

WHY: MIRI tweakreg (abs_searchrad=0.4") cannot bridge the ~2-6" per-dither
pointing error of these frames, so it silently fails and every crf keeps its raw
commanded pointing.  A prior UNIFORM correction (MIRIDRA=3.33, MIRIDDE=-1.37)
was applied to all 5 frames, but the true error is PER-FRAME (offset-histogram vs
F480M: |off| 2.3-6.2", direction varies), so per-frame residuals remain and the
o002 catalog sits ~3-4" off truth (0/36 hand-selected captured).

This bypasses tweakreg: for each crf, detect bright sources, offset-histogram the
(image - F480M) offsets (dense-field-safe; nearest-neighbor median is meaningless
here), and subtract the peak from the FITS-header WCS (CRVAL).  FITS-only, matching
apply_measured_miri_wcs_offsets.py: the manual cataloging pipeline reads the FITS
WCS (cataloging.py: ``wcs.WCS(im1[1].header)``).  MIRIDRA/MIRIDDE are updated
CUMULATIVELY (old applied + new residual) so the keyword always records the total
correction vs the original commanded WCS.

Run, then re-run o002 cataloging and re-score hand-selected capture.
"""
import os
import glob
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.stats import mad_std
import astropy.units as u
from photutils.detection import DAOStarFinder
from scipy.ndimage import median_filter
import warnings
warnings.filterwarnings('ignore')

FRAMEGLOB = '/orange/adamginsburg/jwst/sickle/F770W/pipeline/jw03958002001_*_mirimage_o002_crf.fits'
REFCAT = '/orange/adamginsburg/jwst/sickle/catalogs/f480m_nrcb_indivexp_merged_resbgsub_m6_dao_basic.fits'
FWHM_PIX = 2.445   # F770W at 0.11"/px
MFSIZE = 31
THR = 5
SEARCH = 6.0       # arcsec


def load_ref():
    t = Table.read(REFCAT)
    if 'skycoord' in t.colnames:
        return SkyCoord(t['skycoord'])
    return SkyCoord(t['skycoord_ra'], t['skycoord_dec'], unit='deg')


def measure_offset(fn, refsc):
    """Offset-histogram (image - ref) in arcsec from the current FITS WCS."""
    fh = fits.open(fn)
    d = fh['SCI'].data
    ww = WCS(fh['SCI'].header)
    mfd = median_filter(np.nan_to_num(d, nan=np.nanmedian(d)), size=MFSIZE)
    hp = np.nan_to_num(d) - mfd
    s = DAOStarFinder(threshold=THR * mad_std(hp), fwhm=FWHM_PIX)(hp)
    if s is None or len(s) == 0:
        return None, None, 0
    xcol = 'xcentroid' if 'xcentroid' in s.colnames else 'x_centroid'
    ycol = 'ycentroid' if 'ycentroid' in s.colnames else 'y_centroid'
    sc = ww.pixel_to_world(s[xcol], s[ycol])
    i_b, i_a, sep, _ = search_around_sky(refsc, sc, SEARCH * u.arcsec)
    if len(i_a) < 5:
        return None, None, len(i_a)
    dra = ((sc.ra[i_a] - refsc.ra[i_b]) * np.cos(sc.dec[i_a].rad)).to(u.arcsec).value
    ddec = (sc.dec[i_a] - refsc.dec[i_b]).to(u.arcsec).value
    bins = np.arange(-SEARCH, SEARCH + 0.1, 0.1)
    H, xe, ye = np.histogram2d(dra, ddec, bins=[bins, bins])
    pk = np.unravel_index(np.argmax(H), H.shape)
    cx, cy = 0.5 * (xe[pk[0]] + xe[pk[0] + 1]), 0.5 * (ye[pk[1]] + ye[pk[1] + 1])
    m = (np.abs(dra - cx) < 0.5) & (np.abs(ddec - cy) < 0.5)
    return float(np.median(dra[m])), float(np.median(ddec[m])), int(m.sum())


def apply(fn, dra_as, ddec_as):
    """Subtract (image - ref) residual from FITS CRVAL; update MIRIDRA cumulatively."""
    fh = fits.open(fn)
    h = fh[1].header
    cosd = np.cos(np.deg2rad(h['CRVAL2']))
    prev_ra = float(h.get('MIRIDRA', 0.0))
    prev_de = float(h.get('MIRIDDE', 0.0))
    if 'OLCRVAL1' not in h:
        h['OLCRVAL1'] = h['CRVAL1']
        h['OLCRVAL2'] = h['CRVAL2']
    h['CRVAL1'] = h['CRVAL1'] - dra_as / 3600. / cosd
    h['CRVAL2'] = h['CRVAL2'] - ddec_as / 3600.
    # MIRIDRA records TOTAL applied correction (= -(image-ref)) vs commanded WCS
    h['MIRIDRA'] = (prev_ra + (-dra_as), 'total applied RA corr [arcsec], per-frame 2026-06-13')
    h['MIRIDDE'] = (prev_de + (-ddec_as), 'total applied Dec corr [arcsec], per-frame 2026-06-13')
    h['MIRIWCSN'] = ('FITS WCS corrected per-frame to F480M; ASDF gwcs NOT corrected',
                     'offset-histogram registration')
    fh.writeto(fn, overwrite=True)


def main():
    refsc = load_ref()
    print(f'reference {len(refsc)} F480M sources', flush=True)
    for fn in sorted(glob.glob(FRAMEGLOB)):
        tag = os.path.basename(fn).split('_mirimage')[0]
        dra, ddec, n = measure_offset(fn, refsc)
        if dra is None:
            print(f'{tag}: insufficient matches (n={n}); SKIP', flush=True)
            continue
        apply(fn, dra, ddec)
        print(f'{tag}: residual (image-ref)=({dra:+.2f}",{ddec:+.2f}") n={n} '
              f'-> applied ({-dra:+.2f}",{-ddec:+.2f}")', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
