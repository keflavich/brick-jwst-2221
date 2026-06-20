"""NON-circular test: compare the SAME physical stars' sky coords between the MAST
o001 i2d (ground truth, correctly aligned per user) and our pipeline o001 i2d.
If our WCS is rotated/shifted, the brightest stars land at different sky coords."""
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy import units as u
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats

mast = '/orange/adamginsburg/jwst/sickle/mastDownload/JWST/jw03958-o001_t001_miri_f770w-brightsky/jw03958-o001_t001_miri_f770w-brightsky_i2d.fits'
ours = '/orange/adamginsburg/jwst/sickle/F770W/pipeline/jw03958-o001_t001_miri_clear-f770w-mirimage_data_i2d.fits'

def load(fn):
    h = fits.open(fn)
    for hdu in h:
        if hdu.data is not None and getattr(hdu.data,'ndim',0)==2 and 'CRVAL1' in hdu.header:
            return hdu.data, WCS(hdu.header), hdu.header
    raise RuntimeError(fn)

def pa(w):
    cd = w.pixel_scale_matrix
    return np.degrees(np.arctan2(cd[0,1], cd[1,1]))

def detect(data, w, n=80):
    mean, med, std = sigma_clipped_stats(data, sigma=3.0)
    src = DAOStarFinder(fwhm=3.0, threshold=20*std)(data - med)
    src.sort('flux'); src.reverse(); src = src[:n]
    sky = w.pixel_to_world(src['xcentroid'], src['ycentroid'])
    return sky, np.array(src['flux'])

dm, wm, hm = load(mast)
do, wo, ho = load(ours)

print(f"MAST  o001 i2d: shape={dm.shape} PA={pa(wm):8.3f}  CRVAL=({hm['CRVAL1']:.5f},{hm['CRVAL2']:.5f})  CRPIX=({hm.get('CRPIX1')},{hm.get('CRPIX2')})")
print(f"OURS  o001 i2d: shape={do.shape} PA={pa(wo):8.3f}  CRVAL=({ho['CRVAL1']:.5f},{ho['CRVAL2']:.5f})  CRPIX=({ho.get('CRPIX1')},{ho.get('CRPIX2')})")

# field centers
cm = wm.pixel_to_world(dm.shape[1]/2, dm.shape[0]/2)
co = wo.pixel_to_world(do.shape[1]/2, do.shape[0]/2)
print(f"\nMAST field center sky: {cm.ra.deg:.5f} {cm.dec.deg:.5f}")
print(f"OURS field center sky: {co.ra.deg:.5f} {co.dec.deg:.5f}")
print(f"center-to-center separation: {cm.separation(co).arcsec:.2f}\"")

# brightest star in each
sky_m, fm = detect(dm, wm)
sky_o, fo = detect(do, wo)
print(f"\nMAST brightest star sky: {sky_m[0].ra.deg:.5f} {sky_m[0].dec.deg:.5f}  flux={fm[0]:.0f}")
print(f"OURS brightest star sky: {sky_o[0].ra.deg:.5f} {sky_o[0].dec.deg:.5f}  flux={fo[0]:.0f}")
print(f"brightest-to-brightest separation: {sky_m[0].separation(sky_o[0]).arcsec:.2f}\"")

# cross-match our stars to MAST stars in SKY (should be ~0 if WCS correct)
idx, sep, _ = match_coordinates_sky(sky_o, sky_m)
print(f"\nnearest-neighbor sky match (our stars -> MAST stars):")
print(f"  median sep={np.median(sep.arcsec):.2f}\"  min={sep.arcsec.min():.2f}\"  max={sep.arcsec.max():.2f}\"")
print(f"  matches <1\": {np.sum(sep.arcsec<1)}/{len(sep)}")
