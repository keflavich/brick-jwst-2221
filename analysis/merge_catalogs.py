import numpy as np
import os
import warnings
from astropy.io import fits
import glob
from photutils.background import MMMBackground, MADStdBackgroundRMS
from photutils import CircularAperture, EPSFBuilder, find_peaks, CircularAnnulus
from photutils.detection import DAOStarFinder, IRAFStarFinder
from photutils.psf import DAOGroup, IntegratedGaussianPRF, extract_stars, IterativelySubtractedPSFPhotometry, BasicPSFPhotometry
from astropy.modeling.fitting import LevMarLSQFitter
from astropy import stats
from astropy.table import Table
from astropy.wcs import WCS
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import coordinates
from astropy.visualization import simple_norm
from astropy import wcs
from astropy import table
from astropy import units as u
import pylab as pl
pl.rcParams['figure.facecolor'] = 'w'
pl.rcParams['image.origin'] = 'lower'
pl.rcParams['figure.figsize'] = (10,8)
pl.rcParams['figure.dpi'] = 100

basepath = '/orange/adamginsburg/jwst/brick/'
filternames = ['f410m', 'f212n', 'f466n', 'f405n', 'f187n', 'f182m']

def merge_catalogs(tbls, catalog_type='crowdsource', module='nrca'):
    basetable = master_tbl = tbls[0].copy()
    basecrds = basetable['skycoords']

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for colname in basetable.colnames:
            basetable.rename_column(colname, colname+"_"+basetable.meta['filter'])

        for tbl in tbls[1:]:
            wl = tbl.meta['filter']
            print(wl)
            crds = tbl['skycoords']
            matches, sep, _ = basecrds.match_to_catalog_sky(crds, nthneighbor=1)
            basetable.add_column(name=f"sep_{wl}", col=sep)
            basetable.add_column(name=f"id_{wl}", col=matches)
            matchtb = tbl[matches]
            for cn in matchtb.colnames:
                matchtb.rename_column(cn, f"{cn}_{wl}")
            basetable = table.hstack([basetable, matchtb], join_type='exact')
            basetable.meta[f'{wl}_pixelscale_deg2'] = tbl.meta['pixelscale_deg2']
            basetable.meta[f'{wl}_pixelscale_arcsec'] = tbl.meta['pixelscale_arcsec']

        # Line-subtract the F410 continuum band
        basetable.add_column(basetable['flux_jy_f410m'] - basetable['flux_jy_f405n'] * 0.16, name='flux_jy_410m405')
        basetable.add_column(basetable['flux_jy_410m405'].to(u.ABmag), name='mag_ab_410m405')
        # Then subtract that remainder back from the F405 band to get the continuum-subtracted F405
        basetable.add_column(basetable['flux_jy_f405n'] - basetable['flux_jy_410m405'], name='flux_jy_405m410')
        basetable.add_column(basetable['flux_jy_405m410'].to(u.ABmag), name='mag_ab_405m410')

        basetable.write(f"{basepath}/catalogs/{catalog_type}_{module}_photometry_tables_merged.ecsv", overwrite=True)
        basetable.write(f"{basepath}/catalogs/{catalog_type}_{module}_photometry_tables_merged.fits", overwrite=True)

def merge_crowdsource(module='nrca'):
    imgfns = [x
          for filn in filternames
          for x in glob.glob(f"{basepath}/{filn.upper()}/pipeline/jw02221-o001_t001_nircam*{filn.lower()}*{module}_i2d.fits")
          if f'{module}_' in x or f'{module}1_' in x
         ]

    catfns = [x
              for filn in filternames
              for x in glob.glob(f"{basepath}/{filn.upper()}/{filn.lower()}*{module}_crowdsource.fits")
             ]
    assert len(catfns) == 6
    tbls = [Table.read(catfn) for catfn in catfns]

    for catfn, tbl in zip(catfns, tbls):
        tbl.meta['filename'] = catfn
        tbl.meta['filter'] = os.path.basename(catfn).split("_")[0]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        #wcses = [wcs.WCS(fits.getheader(fn.replace("_crowdsource", "_crowdsource_skymodel"))) for fn in catfns]
        imgs = [fits.getdata(fn, ext=('SCI', 1)) for fn in imgfns]
        wcses = [wcs.WCS(fits.getheader(fn, ext=('SCI', 1))) for fn in imgfns]

    for tbl, ww in zip(tbls, wcses):
        tbl['y'],tbl['x'] = tbl['x'],tbl['y']
        crds = ww.pixel_to_world(tbl['x'], tbl['y'])
        tbl.add_column(crds, name='skycoords')
        tbl.meta['pixelscale_deg2'] = ww.proj_plane_pixel_area()
        tbl.meta['pixelscale_arcsec'] = (ww.proj_plane_pixel_area()**0.5).to(u.arcsec)
        flux_jy = (tbl['flux'] * u.MJy/u.sr * (2*np.pi / (8*np.log(2))) * tbl['fwhm']**2 * tbl.meta['pixelscale_deg2']).to(u.Jy)
        abmag = flux_jy.to(u.ABmag)
        tbl.add_column(flux_jy, name='flux_jy')
        tbl.add_column(abmag, name='mag_ab')

    merge_catalogs(tbls, catalog_type='crowdsource', module=module)


def merge_daophot(module='nrca', detector='', daophot_type='basic'):
    catfns = daocatfns = [
        f"{basepath}/{filtername.upper()}/{filtername.lower()}_{module}{detector}_daophot_{daophot_type}.fits"
        for filtername in filternames
    ]
    imgfns = [x
          for filn in filternames
          for x in glob.glob(f"{basepath}/{filn.upper()}/pipeline/jw02221-o001_t001_nircam*{filn.lower()}*{module}_i2d.fits")
          if f'{module}_' in x or f'{module}1_' in x
         ]

    tbls = [Table.read(catfn) for catfn in daocatfns]

    for catfn, tbl, filtername in zip(catfns, tbls, filternames):
        tbl.meta['filename'] = catfn
        tbl.meta['filter'] = filtername

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        #wcses = [wcs.WCS(fits.getheader(fn.replace("_crowdsource", "_crowdsource_skymodel"))) for fn in catfns]
        imgs = [fits.getdata(fn, ext=('SCI', 1)) for fn in imgfns]
        wcses = [wcs.WCS(fits.getheader(fn, ext=('SCI', 1))) for fn in imgfns]

    for tbl, ww in zip(tbls, wcses):
        crds = ww.pixel_to_world(tbl['x'], tbl['y'])
        tbl.add_column(crds, name='skycoords')
        tbl.meta['pixelscale_deg2'] = ww.proj_plane_pixel_area()
        tbl.meta['pixelscale_arcsec'] = (ww.proj_plane_pixel_area()**0.5).to(u.arcsec)
        flux_jy = (tbl['flux'] * u.MJy/u.sr * (2*np.pi / (8*np.log(2))) * tbl['fwhm']**2 * tbl.meta['pixelscale_deg2']).to(u.Jy)
        abmag = flux_jy.to(u.ABmag)
        tbl.add_column(flux_jy, name='flux_jy')
        tbl.add_column(abmag, name='mag_ab')

    merge_catalogs(tbls, catalog_type=daophot_type, module=module)


def main():
    for module in ('nrca', 'nrcb', 'merged',):
        print(module)
        merge_crowdsource(module=module)
        merge_daophot(daophot_type='basic', module=module)
        merge_daophot(daophot_type='iterative', module=module)
