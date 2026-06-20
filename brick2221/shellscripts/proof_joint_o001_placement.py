"""Visual proof: render the JOINT o001-002 F770W mosaic and overplot stars whose
sky coords were measured INDEPENDENTLY in the individual o001 i2d. If the joint
o001 half were rotated/shifted, the markers would not land on stars."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import simple_norm
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats

P = '/orange/adamginsburg/jwst/sickle/F770W/pipeline/'
indiv = P + 'jw03958-o001_t001_miri_clear-f770w-mirimage_data_i2d.fits'
jdat  = P + 'jw03958-o001-002_t001_miri_clear-f770w-mirimage_data_i2d.fits'
jres  = P + 'jw03958-o001-002_t001_miri_clear-f770w-mirimage_resbgsub_group_m6_daophot_basic_mergedcat_residual_i2d.fits'

def load(fn):
    h = fits.open(fn)
    for hdu in h:
        if hdu.data is not None and getattr(hdu.data,'ndim',0)==2:
            return hdu.data, WCS(hdu.header)
    raise RuntimeError(fn)

di, wi = load(indiv)
dd, wd = load(jdat)
dr, wr = load(jres)

# stars detected independently in individual o001 i2d
mean, med, std = sigma_clipped_stats(di, sigma=3.0)
src = DAOStarFinder(fwhm=3.0, threshold=30*std)(di - med)
src.sort('flux'); src.reverse(); src = src[:60]
sky = wi.pixel_to_world(src['xcentroid'], src['ycentroid'])

fig, axes = plt.subplots(1, 2, figsize=(20, 9))
for ax, (data, w, title) in zip(axes, [
        (dd, wd, 'JOINT data i2d  (o001 stars overplotted by sky coord)'),
        (dr, wr, 'JOINT m6 residual i2d  (stars subtracted)')]):
    norm = simple_norm(data, 'asinh', min_percent=20, max_percent=99.5)
    ax.imshow(data, origin='lower', cmap='gray', norm=norm)
    xs, ys = w.world_to_pixel(sky)
    ax.scatter(xs, ys, s=120, facecolors='none', edgecolors='lime', lw=1.2)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    # label the o001 tile corner region
    ax.text(0.02, 0.98, 'green circles = stars found in INDIVIDUAL o001 i2d,\n'
            'placed here by their sky coordinate', transform=ax.transAxes,
            va='top', color='yellow', fontsize=11)

plt.tight_layout()
out = P + 'PROOF_joint_o001_placement.png'
plt.savefig(out, dpi=80, bbox_inches='tight')
print("saved", out)

# also report: how many markers landed on a local peak in the JOINT data
def onpeak(data, w, sc, box=6):
    x, y = w.world_to_pixel(sc); x, y = float(x), float(y)
    if not (box<x<data.shape[1]-box and box<y<data.shape[0]-box): return None
    sub = data[int(y-box):int(y+box+1), int(x-box):int(x+box+1)]
    if not np.isfinite(sub).any(): return None
    return np.nanmax(sub) > med + 5*std
hits = [onpeak(dd, wd, sc) for sc in sky]
hits = [h for h in hits if h is not None]
print(f"o001 stars landing on a real peak in JOINT data: {sum(hits)}/{len(hits)}")
