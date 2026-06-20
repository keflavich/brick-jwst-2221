"""Final proof: new JOINT o001-002 F770W mosaic (corrected crf) with MAST o001
bright stars overplotted by PREDICTED position (MAST true sky -> joint pixel via
joint WCS). Markers land on the stars => o001 correctly placed, rotation gone."""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import simple_norm
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats

P='/orange/adamginsburg/jwst/sickle/'
mast=P+'mastDownload/JWST/jw03958-o001_t001_miri_f770w-brightsky/jw03958-o001_t001_miri_f770w-brightsky_i2d.fits'
jdat=P+'F770W/pipeline/jw03958-o001-002_t001_miri_clear-f770w-mirimage_data_i2d.fits'
jres=P+'F770W/pipeline/jw03958-o001-002_t001_miri_clear-f770w-mirimage_resbgsub_group_m6_daophot_basic_mergedcat_residual_i2d.fits'
def load(fn):
    h=fits.open(fn)
    for hdu in h:
        if hdu.data is not None and getattr(hdu.data,'ndim',0)==2 and 'CRVAL1' in hdu.header:
            return hdu.data,WCS(hdu.header)
dm,wm=load(mast); dd,wd=load(jdat); dr,wr=load(jres)
mm,medm,sm_=sigma_clipped_stats(dm,sigma=3.0)
src=DAOStarFinder(fwhm=2.8,threshold=15*sm_)(dm-medm);src.sort('flux');src.reverse();src=src[:25]
sky=wm.pixel_to_world(src['xcentroid'],src['ycentroid'])
fig,axes=plt.subplots(1,2,figsize=(20,9))
for ax,(data,w,t) in zip(axes,[(dd,wd,'JOINT data i2d (corrected) -- MAST o001 stars overplotted by true sky'),
                                (dr,wr,'JOINT m6 residual (stars subtracted)')]):
    ax.imshow(data,origin='lower',cmap='gray',norm=simple_norm(data,'asinh',min_percent=20,max_percent=99.5))
    xs,ys=w.world_to_pixel(sky)
    ax.scatter(xs,ys,s=140,facecolors='none',edgecolors='lime',lw=1.3)
    ax.set_title(t,fontsize=12)
    ax.text(0.02,0.98,'green = bright stars from MAST o001 (correct Gaia frame),\nplaced by their TRUE sky coord. Landing on stars = o001 correct.',
            transform=ax.transAxes,va='top',color='yellow',fontsize=10)
plt.tight_layout()
out=P+'F770W/pipeline/PROOF_joint_o001_FIXED.png'
plt.savefig(out,dpi=80,bbox_inches='tight'); print("saved",out)
EOF=1
