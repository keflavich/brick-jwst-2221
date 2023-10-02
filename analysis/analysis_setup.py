import numpy as np
import datetime
import os
import sys, imp
import regions
import warnings
import glob
from astropy.io import fits
from astropy.table import Table
from astropy import stats
from astropy.wcs import WCS
from astropy import coordinates
from astropy.coordinates import SkyCoord
from astropy.visualization import simple_norm
from astropy import wcs
from astropy import table
from astropy import units as u

from photutils import CircularAperture, EPSFBuilder, find_peaks, CircularAnnulus
from photutils.detection import DAOStarFinder, IRAFStarFinder
from photutils.psf import DAOGroup, IntegratedGaussianPRF, extract_stars, IterativelySubtractedPSFPhotometry, BasicPSFPhotometry
from photutils.background import MMMBackground, MADStdBackgroundRMS
from astropy.modeling.fitting import LevMarLSQFitter

import dust_extinction
from dust_extinction.parameter_averages import CCM89
from dust_extinction.averages import RRP89_MWGC, CT06_MWGC, F11_MWGC

import PIL
import pyavm

import pylab as pl
pl.rcParams['figure.facecolor'] = 'w'
pl.rcParams['image.origin'] = 'lower'
pl.rcParams['figure.figsize'] = (10,8)
pl.rcParams['figure.dpi'] = 100
pl.rcParams['font.size'] = 16


basepath = '/orange/adamginsburg/jwst/brick/'

sys.path.append(f'{basepath}/analysis/')
sys.path.append(f'{basepath}/reduction/')
import plot_tools
imp.reload(plot_tools)
from plot_tools import regzoomplot, starzoom, ccd, ccds, cmds, plot_extvec_ccd


distance_modulus = dm = 5*np.log10(8.3*u.kpc / (10*u.pc))

filternames = ['f410m', 'f212n', 'f466n', 'f405n', 'f187n', 'f182m']

reg = regions.Regions.read(f'{basepath}/regions/leftside_brick_zoom.reg')[0]
regzoom = regions.Regions.read(f'{basepath}/regions/leftside_brick_rezoom.reg')[0]


fh_nrca = fits.open(f'{basepath}/F410M/pipeline/jw02221-o001_t001_nircam_clear-f410m-nrca_i2d.fits')
ww410_nrca = wcs.WCS(fh_nrca[1].header)
ww_nrca = ww410_nrca

fh_nrcb = fits.open(f'{basepath}/F410M/pipeline/jw02221-o001_t001_nircam_clear-f410m-nrcb_i2d.fits')
ww410_nrcb = wcs.WCS(fh_nrcb[1].header)
ww_nrcb = ww410_nrcb

fh_merged = fits.open(f'{basepath}/F410M/pipeline/jw02221-o001_t001_nircam_clear-f410m-merged_i2d.fits')
ww410_merged = wcs.WCS(fh_merged[1].header)
ww_merged = ww410_merged

fh_merged_reproject = fits.open(f'{basepath}/F410M/pipeline/jw02221-o001_t001_nircam_clear-f410m-merged-reproject_i2d.fits')
ww410_merged_reproject = wcs.WCS(fh_merged_reproject[1].header)
ww_merged_reproject = ww410_merged_reproject

avm_nostars_nrca = pyavm.AVM.from_image(f'{basepath}/images/BrickJWST_longwave_RGB_unrotated.png')
img_nostars_nrca = np.array(PIL.Image.open(f'{basepath}/images/BrickJWST_longwave_RGB_unrotated.png'))[::-1,:,:]
wwi_nostars_nrca = wcs.WCS(fits.Header.fromstring(avm_nostars_nrca.Spatial.FITSheader))

avm_nrca = pyavm.AVM.from_image(f'{basepath}/images/BrickJWST_nrca_longwave_stars_rotated.png')
img_nrca = np.array(PIL.Image.open(f'{basepath}/images/BrickJWST_nrca_longwave_stars_rotated.png'))[::-1,:,:]
wwi_nrca = wcs.WCS(fits.Header.fromstring(avm_nrca.Spatial.FITSheader))


# placeholder until we replace this with merge
avm = avm_nrca
img = img_nrca
wwi = wwi_nrca

avm_nostars_merged = pyavm.AVM.from_image(f'{basepath}/images/BrickJWST_merged_longwave_narrowband_rotated.png')
img_nostars_merged = np.array(PIL.Image.open(f'{basepath}/images/BrickJWST_merged_longwave_narrowband_rotated.png'))[::-1,:,:]
wwi_nostars_merged = wcs.WCS(fits.Header.fromstring(avm_nostars_nrca.Spatial.FITSheader))

# the merged version is the one I *want* to work with, but nrca is the one I've tested most
# and can really vouch for
basetable_merged = basetable = Table.read(f'{basepath}/catalogs/crowdsource_nsky0_merged_photometry_tables_merged.fits')

# use locked-in 2023/07/02 version to make sure analysis stays stable
# (this version is confirmed to have good long-wave astrometry, at least)
basetable_merged_reproject = Table.read(f'{basepath}/catalogs/crowdsource_nsky0_merged-reproject_photometry_tables_merged_20230702.fits')
# updated version: has metadata about which filter was used as the reference
basetable_merged_reproject = Table.read(f'{basepath}/catalogs/crowdsource_nsky0_merged-reproject_photometry_tables_merged_20230827.fits')
# updated version: new magnitude calcs
basetable_merged_reproject = Table.read(f'{basepath}/catalogs/crowdsource_nsky0_merged-reproject_photometry_tables_merged_20231002.fits')

basetable_nrca = Table.read(f'{basepath}/catalogs/crowdsource_nsky0_nrca_photometry_tables_merged.fits')
basetable_nrcb = Table.read(f'{basepath}/catalogs/crowdsource_nsky0_nrcb_photometry_tables_merged.fits')

basetable_merged_reproject_dao_iter = Table.read(f'{basepath}/catalogs/iterative_merged-reproject_photometry_tables_merged.fits')
basetable_merged_reproject_dao_iter_epsf = Table.read(f'{basepath}/catalogs/iterative_merged-reproject_photometry_tables_merged_epsf.fits')
basetable_merged_reproject_dao_iter_bg_epsf = Table.read(f'{basepath}/catalogs/iterative_merged-reproject_photometry_tables_merged_bgsub_epsf.fits')

def getmtime(x):
    return datetime.datetime.fromtimestamp(os.path.getmtime(x)).strftime('%Y-%m-%d %H:%M:%S')

for module in ('nrca', 'nrcb', 'merged', 'merged-reproject'):
    fn = f'{basepath}/catalogs/crowdsource_nsky0_{module}_photometry_tables_merged.fits'
    print(f"For module {module} catalog {os.path.basename(fn)}, mod date is {getmtime(fn)}")
