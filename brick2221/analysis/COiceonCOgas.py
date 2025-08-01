"""
Compare CO ice to CO3-2 gas emission

largely taken from COemission_analysis.ipynb
"""
import numpy as np
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.visualization import simple_norm
from spectral_cube import SpectralCube
from astropy.table import Table
import pylab as pl
from dust_extinction.averages import CT06_MWGC, G21_MWAvg
import scipy.ndimage

from brick2221.analysis.analysis_setup import basepath
from brick2221.analysis.catalog_on_RGB import load_table, overlay_stars

basetable = load_table()

brick_center = SkyCoord(0.257, 0.018, unit=(u.deg, u.deg), frame='galactic')

#cube = SpectralCube.read('/orange/adamginsburg/cmz/CHIMPS/12CO_GC_359-000_mosaic.fits')
co_image = fits.open('/orange/adamginsburg/cmz/CHIMPS/12CO_CMZ_INTEG_LB.fits')
# co_image_cutout = Cutout2D(co_image[0].data, SkyCoord('17:46:12.752 -28:43:52.4', unit=(u.h, u.deg), frame='fk5'), size=10*u.arcsec, wcs=WCS(co_image[0].header)) ## this is that interesting cluster zone
co_image_cutout = Cutout2D(co_image[0].data, brick_center, size=8*u.arcmin, wcs=WCS(co_image[0].header))
coheader = co_image[0].header
cowcs = WCS(coheader)

ppmap_cdens_full = fits.open('/orange/adamginsburg/cmz/ppmap/PPMAP_Results/l000_results/l000_cdens.fits')
ppmap_tem_full = fits.open('/orange/adamginsburg/cmz/ppmap/PPMAP_Results/l000_results/l000_temp.fits')

ppmap_col = Cutout2D(ppmap_cdens_full[0].data,
                     brick_center,
                     size=8*u.arcmin, wcs=WCS(ppmap_cdens_full[0].header))
ppmap_tem = Cutout2D(ppmap_tem_full[0].data,
                     brick_center,
                     size=8*u.arcmin, wcs=WCS(ppmap_tem_full[0].header))


# FIGURE 1
fig = pl.figure(dpi=200)
ax = fig.add_subplot(projection=co_image_cutout.wcs)
pl.imshow(co_image_cutout.data, norm=simple_norm(co_image_cutout.data, stretch='linear', ),
          cmap='gray')
ax.contour(ppmap_col.data, transform=ax.get_transform(ppmap_col.wcs),
           cmap='viridis', levels=6, linewidths=[0.5]*20)
ax.set_xlabel('Galactic Longitude')
ax.set_ylabel('Galactic Latitude')
ax.xaxis.set_units(u.deg)
ax.yaxis.set_units(u.deg)
lon = ax.coords['glon']
lat = ax.coords['glat']
lon.set_major_formatter('d.dd')
lat.set_major_formatter('d.dd')

pl.savefig(f"{basepath}/figures/PPMAP_on_CO32.pdf", dpi=200, bbox_inches='tight')
pl.savefig(f"{basepath}/figures/PPMAP_on_CO32.png", dpi=200, bbox_inches='tight')


unextincted_color, blue_ice = overlay_stars(basetable=basetable,
              ax=ax,
              color_filter1='F405N',
              color_filter2='F466N',
              threshold=-0.4,
              av_threshold=17,
              ref_filter='f410m',
              avfilts=['F182M', 'F212N'],
              ext=CT06_MWGC(),
              cmap='Reds',
              s=1,
              alpha=0.5,
              )

pl.savefig(f"{basepath}/figures/COice_on_PPMAP_on_CO32.png", dpi=200, bbox_inches='tight')


coords = co_image_cutout.wcs.world_to_pixel(basetable['skycoord_f410m'])
co32integ_values = scipy.ndimage.map_coordinates(co_image_cutout.data, coords, order=1)
ppmap_values = scipy.ndimage.map_coordinates(ppmap_col.data, coords, order=1)

pl.figure()
pl.hexbin(co32integ_values, unextincted_color,  cmap='viridis', gridsize=100, mincnt=1)
pl.ylim(-2.5, 0.2)
pl.xlabel('CO3-2 integrated intensity (K km/s)')
pl.ylabel('F405N - F466N (dereddened)')
pl.colorbar(label='PPMAP column density (cm$^{-2}$)')
pl.savefig(f"{basepath}/figures/F466Nblueness_vs_CO32_scatter.png", dpi=200, bbox_inches='tight')
