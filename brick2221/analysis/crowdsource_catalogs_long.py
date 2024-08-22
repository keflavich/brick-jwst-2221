print("Starting crowdsource_catalogs_long", flush=True)
import glob
import time
import numpy
import crowdsource
import regions
import numpy as np
from functools import cache
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.visualization import simple_norm
from astropy.modeling.fitting import LevMarLSQFitter
from astropy import wcs
from astropy import table
from astropy import stats
from astropy import units as u
from astropy.nddata import NDData
from astropy.io import fits
from scipy import ndimage
import requests
import requests.exceptions
import urllib3
import urllib3.exceptions
from photutils.detection import DAOStarFinder, IRAFStarFinder
from photutils.psf import IntegratedGaussianPRF, extract_stars, EPSFStars, EPSFModel
try:
    # version >=1.7.0, doesn't work: the PSF is broken (https://github.com/astropy/photutils/issues/1580?)
    from photutils.psf import PSFPhotometry, IterativePSFPhotometry, SourceGrouper
except:
    # version 1.6.0, which works
    from photutils.psf import BasicPSFPhotometry as PSFPhotometry, IterativelySubtractedPSFPhotometry as IterativePSFPhotometry, DAOGroup as SourceGrouper
try:
    from photutils.background import MMMBackground, MADStdBackgroundRMS, MedianBackground, Background2D, LocalBackground
except:
    from photutils.background import MMMBackground, MADStdBackgroundRMS, MedianBackground, Background2D
    from photutils.background import MMMBackground as LocalBackground

from photutils.psf import EPSFBuilder
from photutils.psf import extract_stars

import warnings
from astropy.utils.exceptions import AstropyWarning, AstropyDeprecationWarning
warnings.simplefilter('ignore', category=AstropyWarning)
warnings.simplefilter('ignore', category=AstropyDeprecationWarning)

from crowdsource import crowdsource_base
from crowdsource.crowdsource_base import fit_im, psfmod

from astroquery.svo_fps import SvoFps

import pylab as pl
pl.rcParams['figure.facecolor'] = 'w'
pl.rcParams['image.origin'] = 'lower'

import os
print("Importing webbpsf", flush=True)
import webbpsf
print(f"Webbpsf version: {webbpsf.__version__}")
from webbpsf.utils import to_griddedpsfmodel
import datetime
print("Done with imports", flush=True)


def print(*args, **kwargs):
    now = datetime.datetime.now().isoformat()
    from builtins import print as printfunc
    return printfunc(f"{now}:", *args, **kwargs)


class WrappedPSFModel(crowdsource.psf.SimplePSF):
    """
    wrapper for photutils GriddedPSFModel
    """
    def __init__(self, psfgridmodel, stampsz=19):
        self.psfgridmodel = psfgridmodel
        self.default_stampsz = stampsz

    def __call__(self, col, row, stampsz=None, deriv=False):

        if stampsz is None:
            stampsz = self.default_stampsz

        parshape = numpy.broadcast(col, row).shape
        tparshape = parshape if len(parshape) > 0 else (1,)

        # numpy uses row, column notation
        rows, cols = np.indices((stampsz, stampsz)) - (np.array([stampsz, stampsz])-1)[:, None, None] / 2.

        # explicitly broadcast
        col = np.atleast_1d(col)
        row = np.atleast_1d(row)
        #rows = rows[:, :, None] + row[None, None, :]
        #cols = cols[:, :, None] + col[None, None, :]

        # photutils seems to use column, row notation
        # only works with photutils <= 1.6.0 - but is wrong there
        #stamps = self.psfgridmodel.evaluate(cols, rows, 1, col, row)
        # it returns something in (nstamps, row, col) shape
        # pretty sure that ought to be (col, row, nstamps) for crowdsource

        # andrew saydjari's version here:
        # it returns something in (nstamps, row, col) shape
        stamps = []
        for i in range(len(col)):
            # the +0.5 is required to actually center the PSF (empirically)
            #stamps.append(self.psfgridmodel.evaluate(cols+col[i]+0.5, rows+row[i]+0.5, 1, col[i], row[i]))
            # the above may have been true when we were using (incorrectly) offset PSFs
            stamps.append(self.psfgridmodel.evaluate(cols+col[i], rows+row[i], 1, col[i], row[i]))

        stamps = np.array(stamps)

        # for oversampled stamps, they may not be normalized
        stamps /= stamps.sum(axis=(1,2))[:,None,None]
        # this is evidently an incorrect transpose
        #stamps = np.transpose(stamps, axes=(0,2,1))

        if deriv:
            dpsfdrow, dpsfdcol = np.gradient(stamps, axis=(1, 2))

        ret = stamps
        if parshape != tparshape:
            ret = ret.reshape(stampsz, stampsz)
            if deriv:
                dpsfdrow = dpsfdrow.reshape(stampsz, stampsz)
                dpsfdcol = dpsfdcol.reshape(stampsz, stampsz)
        if deriv:
            ret = (ret, dpsfdcol, dpsfdrow)

        return ret

    def render_model(self, col, row, stampsz=None):
        """
        this function likely does nothing?
        """
        if stampsz is not None:
            self.stampsz = stampsz

        rows, cols = np.indices(self.stampsz, dtype=float) - (np.array(self.stampsz)-1)[:, None, None] / 2.

        return self.psfgridmodel.evaluate(cols, rows, 1, col, row).T.squeeze()


def save_epsf(epsf, filename, overwrite=True):
    header = {}
    header['OVERSAMP'] = list(epsf.oversampling)
    hdu = fits.PrimaryHDU(data=epsf.data, header=header)
    hdu.writeto(filename, overwrite=overwrite)


def read_epsf(filename):
    fh = fits.open(filename)
    hdu = fh[0]
    return EPSFModel(data=hdu.data, oversampling=hdu.header['OVERSAMP'])


def catalog_zoom_diagnostic(data, modsky, zoomcut, stars):

    # make sure stars is a table
    try:
        'qf' in stars.colnames
    except AttributeError:
        stars = Table(stars)

    pl.figure(figsize=(12,12))
    im = pl.subplot(2,2,1).imshow(data[zoomcut],
                                  norm=simple_norm(data[zoomcut],
                                                   stretch='log',
                                                   max_percent=99.95,
                                                   min_cut=0), cmap='gray')
    pl.xticks([]); pl.yticks([]); pl.title("Data")
    pl.colorbar(mappable=im)
    im = pl.subplot(2,2,2).imshow(modsky[zoomcut],
                                  norm=simple_norm(modsky[zoomcut],
                                                   stretch='log',
                                                   max_percent=99.95,
                                                   min_cut=0), cmap='gray')
    pl.xticks([]); pl.yticks([]); pl.title("fit_im model+sky")
    pl.colorbar(mappable=im)

    resid = (data[zoomcut] - modsky[zoomcut])
    rms = stats.mad_std(resid, ignore_nan=True)
    if np.isnan(rms):
        raise ValueError("RMS is nan, this shouldn't happen")

    norm = (simple_norm(resid, stretch='asinh', max_percent=99.95, min_percent=0.5)
            if np.nanmin(resid) > 0 else
            simple_norm(resid, stretch='log', max_cut=np.nanpercentile(resid, 99.95), min_cut=-2*rms))

    im = pl.subplot(2,2,3).imshow(resid,
                                  norm=norm,
                                  cmap='gray')
    pl.xticks([]); pl.yticks([]); pl.title(f"data-modsky (rms={rms:10.3g})")
    pl.colorbar(mappable=im)
    im = pl.subplot(2,2,4).imshow(data[zoomcut],
                                  norm=simple_norm(data[zoomcut],
                                                   stretch='log',
                                                   max_percent=99.95,
                                                   min_cut=0), cmap='gray')

    if 'qf' in stars.colnames:
        # used in analysis
        qgood = ((stars['qf'] > 0.9) &
                 (stars['spread_model'] < 0.25) &
                 (stars['fracflux'] > 0.75)
                )
        neg = stars['flux'] < 0
    elif 'qfit' in stars.colnames:
        # guesses, no tests don
        qgood = ((stars['qfit'] < 0.4) &
                 (stars['cfit'] < 0.1) &
                 (stars['flags'] == 0))
        neg = stars['flux_fit'] < 0
    else:
        qgood = np.ones(len(stars), dtype='bool')
        neg = np.zeros(len(stars), dtype='bool')

    axlims = pl.axis()
    if zoomcut[0].start:
        # pl.axis([0,zoomcut[0].stop-zoomcut[0].start, 0, zoomcut[1].stop-zoomcut[1].start])
        ok = ((stars['x'] >= zoomcut[1].start) &
              (stars['x'] <= zoomcut[1].stop) &
              (stars['y'] >= zoomcut[0].start) &
              (stars['y'] <= zoomcut[0].stop))
        pl.subplot(2,2,4).scatter(stars['x'][ok & ~qgood]-zoomcut[1].start,
                                  stars['y'][ok & ~qgood]-zoomcut[0].start,
                                  marker='+', color='y', s=8, linewidth=0.5)
        pl.subplot(2,2,4).scatter(stars['x'][ok & qgood]-zoomcut[1].start,
                                  stars['y'][ok & qgood]-zoomcut[0].start,
                                  marker='x', color='r', s=8, linewidth=0.5)
        pl.subplot(2,2,4).scatter(stars['x'][neg]-zoomcut[1].start,
                                  stars['y'][neg]-zoomcut[0].start,
                                  marker='1', color='b', s=8, linewidth=0.5)
    else:
        pl.subplot(2,2,4).scatter(stars['x'][~qgood],
                                  stars['y'][~qgood], marker='+', color='lime', s=5, linewidth=0.5)
        pl.subplot(2,2,4).scatter(stars['x'][qgood],
                                  stars['y'][qgood], marker='x', color='r', s=5, linewidth=0.5)
        pl.subplot(2,2,4).scatter(stars['x'][neg],
                                  stars['y'][neg], marker='1', color='b', s=5, linewidth=0.5)
    pl.axis(axlims)
    pl.xticks([]); pl.yticks([]); pl.title("Data with stars");
    pl.colorbar(mappable=im)
    pl.tight_layout()


def save_photutils_results(result, ww, filename,
                           im1, detector,
                           basepath, filtername, module, desat, bgsub, exposure_, visitid_, vgroupid_,
                           psf=None,
                           blur=False,
                           basic_or_iterative='basic',
                           options=None,
                           epsf_="",
                           group="",
                           fpsf=""):
    print("Saving photutils results.")
    blur_ = "_blur" if blur else ""

    pixscale = (ww.proj_plane_pixel_area()**0.5).to(u.arcsec)
    if 'x_fit' in result.colnames:
        coords = ww.pixel_to_world(result['x_fit'], result['y_fit'])
        result['skycoord_centroid'] = coords
    elif 'xcentroid' in result.colnames:
        coords = ww.pixel_to_world(result['xcentroid'], result['ycentroid'])
        result['skycoord_centroid'] = coords
    elif 'x_init' in result.colnames:
        coords = ww.pixel_to_world(result['x_init'], result['y_init'])
        result['skycoord_init'] = coords
    else:
        raise KeyError(f"No x value found in {result.colnames}")
    print(f'len(result) = {len(result)}, len(coords) = {len(coords)}, type(result)={type(result)}', flush=True)
    detector = "" # no detector #'s for long
    if options.each_exposure:
        result.meta['exposure'] = exposure_
    if visitid_ is not None:
        result.meta['visit'] = int(visitid_[-3:])
    if vgroupid_ is not None:
        result.meta['vgroup'] = int(vgroupid_)
    result.meta['filename'] = filename
    result.meta['filter'] = filtername
    result.meta['module'] = module
    result.meta['detector'] = detector
    result.meta['pixscale'] = pixscale.to(u.deg).value
    result.meta['pixscale_as'] = pixscale.to(u.arcsec).value
    result.meta['proposal_id'] = options.proposal_id

    if 'RAOFFSET' in im1[0].header:
        result.meta['RAOFFSET'] = im1[0].header['RAOFFSET']
        result.meta['DEOFFSET'] = im1[0].header['DEOFFSET']
    elif 'RAOFFSET' in im1[1].header:
        result.meta['RAOFFSET'] = im1[1].header['RAOFFSET']
        result.meta['DEOFFSET'] = im1[1].header['DEOFFSET']

    if 'x_err' in result.colnames:
        result['dra'] = result['x_err'] * pixscale
        result['ddec'] = result['y_err'] * pixscale

    tblfilename = f"{basepath}/{filtername}/{filtername.lower()}_{module}{detector}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}_daophot_{basic_or_iterative}.fits"

    result.write(tblfilename, overwrite=True)

    print("tblfilename={tblfilename}, filename={filename}, suffix={suffix}, filtername={filtername}, module={module}, desat={desat}, bgsub={bgsub}, fpsf={fpsf} blur={blur}")

    result.write(tblfilename, overwrite=True)
    print(f"Completed {basic_or_iterative} photometry, and wrote out file {tblfilename}")

    return result


def save_crowdsource_results(results, ww, filename, suffix,
                             im1, detector,
                             basepath, filtername, module, desat, bgsub, exposure_, visitid_, vgroupid_,
                             psf=None,
                             blur=False,
                             options=None,
                             fpsf=""):
    print("Saving crowdsource results.")
    blur_ = "_blur" if blur else ""

    stars, modsky, skymsky, psf_ = results
    stars = Table(stars)
    # crowdsource explicitly inverts x & y from the numpy convention:
    # https://github.com/schlafly/crowdsource/issues/11
    coords = ww.pixel_to_world(stars['y'], stars['x'])
    stars['skycoord'] = coords
    stars['x'], stars['y'] = stars['y'], stars['x']
    stars['dx'], stars['dy'] = stars['dy'], stars['dx']

    pixscale = (ww.proj_plane_pixel_area()**0.5).to(u.arcsec)
    stars['dra'] = stars['dx'] * pixscale
    stars['ddec'] = stars['dy'] * pixscale

    stars.meta['filename'] = filename
    stars.meta['filter'] = filtername
    stars.meta['module'] = module
    stars.meta['detector'] = detector
    stars.meta['pixscale'] = pixscale.to(u.deg).value
    stars.meta['pixscale_as'] = pixscale.to(u.arcsec).value
    stars.meta['proposal_id'] = options.proposal_id
    if exposure_:
        stars.meta['exposure'] = exposure_
    if visitid_:
        stars.meta['visit'] = int(visitid_[-3:])
    if vgroupid_:
        stars.meta['vgroup'] = int(vgroupid_)

    if 'RAOFFSET' in im1[0].header:
        stars.meta['RAOFFSET'] = im1[0].header['RAOFFSET']
        stars.meta['DEOFFSET'] = im1[0].header['DEOFFSET']
    elif 'RAOFFSET' in im1[1].header:
        stars.meta['RAOFFSET'] = im1[1].header['RAOFFSET']
        stars.meta['DEOFFSET'] = im1[1].header['DEOFFSET']

    tblfilename = (f"{basepath}/{filtername}/"
                   f"{filtername.lower()}_{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{fpsf}{blur_}"
                   f"_crowdsource_{suffix}.fits")

    print("tblfilename={tblfilename}, filename={filename}, suffix={suffix}, filtername={filtername}, module={module}, desat={desat}, bgsub={bgsub}, fpsf={fpsf} blur={blur}")

    stars.write(tblfilename, overwrite=True)
    # add WCS-containing header
    with fits.open(tblfilename, mode='update', output_verify='fix') as fh:
        fh[0].header.update(im1[1].header)
    skymskyhdu = fits.PrimaryHDU(data=skymsky, header=im1[1].header)
    modskyhdu = fits.ImageHDU(data=modsky, header=im1[1].header)
    # PSF doesn't need saving / can't be saved, it's a function
    # psfhdu = fits.ImageHDU(data=psf)
    hdul = fits.HDUList([skymskyhdu, modskyhdu])
    hdul.writeto(f"{basepath}/{filtername}/{filtername.lower()}_{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{fpsf}{blur_}_crowdsource_skymodel_{suffix}.fits", overwrite=True)

    if psf is not None:
        if hasattr(psf, 'stamp'):
            psfhdu = fits.PrimaryHDU(data=psf.stamp)
            psf_fn = (f"{basepath}/{filtername}/"
                      f"{filtername.lower()}_{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{fpsf}{blur_}"
                      f"_crowdsource_{suffix}_psf.fits")
            psfhdu.writeto(psf_fn, overwrite=True)
        else:
            raise ValueError(f"PSF did not have a stamp attribute.  It was: {psf}, type={type(psf)}")

    return stars


def load_data(filename):
    fh = fits.open(filename)
    im1 = fh
    data = im1[1].data
    try:
        wht = im1['WHT'].data
    except KeyError:
        wht = None
    err = im1['ERR'].data
    instrument = im1[0].header['INSTRUME']
    telescope = im1[0].header['TELESCOP']
    obsdate = im1[0].header['DATE-OBS']
    return fh, im1, data, wht, err, instrument, telescope, obsdate


def get_psf_model(filtername, proposal_id, field,
                  module,
                  use_webbpsf=False,
                  obsdate=None,
                  use_grid=False,
                  blur=False,
                  target='brick',
                  stampsz=19,
                  oversample=1,
                  basepath='/blue/adamginsburg/adamginsburg/jwst/'):
    """
    Return two types of PSF model, the first for DAOPhot and the second for Crowdsource
    """

    basepath = f'{basepath}/{target}'

    blur_ = "_blur" if blur else ""

    # psf_fn = f'{basepath}/{instrument.lower()}_{filtername}_samp{oversample}_nspsf{npsf}_npix{fov_pixels}.fits'
    # if os.path.exists(str(psf_fn)):
    #     # As a file
    #     print(f"Loading grid from psf_fn={psf_fn}", flush=True)
    #     grid = to_griddedpsfmodel(psf_fn)  # file created 2 cells above
    #     if isinstance(big_grid, list):
    #         print(f"PSF IS A LIST OF GRIDS!!! this is incompatible with the return from nrc.psf_grid")
    #         grid = grid[0]

    if use_webbpsf:
        with open(os.path.expanduser('~/.mast_api_token'), 'r') as fh:
            api_token = fh.read().strip()
        from astroquery.mast import Mast

        for ii in range(10):
            try:
                Mast.login(api_token.strip())
                break
            except (requests.exceptions.ReadTimeout, urllib3.exceptions.ReadTimeoutError, TimeoutError) as ex:
                print(f"Attempt {ii} to log in to MAST: {ex}")
                time.sleep(5)
        os.environ['MAST_API_TOKEN'] = api_token.strip()

        has_downloaded = False
        ntries = 0
        while not has_downloaded:
            ntries += 1
            try:
                print("Attempting to download WebbPSF data", flush=True)
                nrc = webbpsf.NIRCam()
                nrc.load_wss_opd_by_date(f'{obsdate}T00:00:00')
                nrc.filter = filtername
                if module in ('nrca', 'nrcb'):
                    if 'F4' in filtername.upper() or 'F3' in filtername.upper():
                        nrc.detector = f'{module.upper()}5' # I think NRCA5 must be the "long" detector?
                    else:
                        nrc.detector = f'{module.upper()}1' #TODO: figure out a way to use all 4?
                    grid = nrc.psf_grid(num_psfs=16, all_detectors=False, verbose=True, save=True)
                elif 'nrc' in module:
                    # Allow nrca1, nrca2, ...
                    nrc.detector = module.upper()
                    grid = nrc.psf_grid(num_psfs=16, all_detectors=False, verbose=True, save=True)
                else:
                    grid = nrc.psf_grid(num_psfs=16, all_detectors=True, verbose=True, save=True)
                has_downloaded = True
            except (urllib3.exceptions.ReadTimeoutError, requests.exceptions.ReadTimeout, requests.HTTPError) as ex:
                print(f"Failed to build PSF: {ex}", flush=True)
            except Exception as ex:
                print(ex, flush=True)
                if ntries > 10:
                    # avoid infinite loops
                    raise ValueError("Failed to download PSF, probably because of an error listed above")
                else:
                    continue

        if use_grid:
            return grid, WrappedPSFModel(grid, stampsz=stampsz)
        else:
            # there's no way to use a grid across all detectors.
            # the right way would be to use this as a grid of grids, but that apparently isn't supported.
            if isinstance(grid, list):
                grid = grid[0]

            #yy, xx = np.indices([31,31], dtype=float)
            #grid.x_0 = grid.y_0 = 15.5
            #psf_model = crowdsource.psf.SimplePSF(stamp=grid(xx,yy))

            # bigger PSF probably needed
            yy, xx = np.indices([61, 61], dtype=float)
            grid.x_0 = grid.y_0 = 30
            psf_model = crowdsource.psf.SimplePSF(stamp=grid(xx, yy))

            return grid, psf_model
    else:

        grid = psfgrid = to_griddedpsfmodel(f'{basepath}/psfs/{filtername.upper()}_{proposal_id}_{field}_merged_PSFgrid_oversample{oversample}{blur_}.fits')

        # if isinstance(grid, list):
        #     print(f"Grid is a list: {grid}")
        #     psf_model = WrappedPSFModel(grid[0])
        #     dao_psf_model = grid[0]
        # else:

        psf_model = WrappedPSFModel(grid, stampsz=stampsz)
        dao_psf_model = grid

        return grid, psf_model


def get_uncertainty(err, data, dq=None, wht=None):

    if dq is None:
        dq = np.zeros(data.shape, dtype='int')

    # crowdsource uses inverse-sigma, not inverse-variance
    weight = err**-1
    #maxweight = np.percentile(weight[np.isfinite(weight)], 95)
    #minweight = np.percentile(weight[np.isfinite(weight)], 5)
    #badweight =  np.percentile(weight[np.isfinite(weight)], 1)
    #weight[err < 1e-5] = 0
    #weight[(err == 0) | (wht == 0)] = np.nanmedian(weight)
    #weight[np.isnan(weight)] = 0
    bad = np.isnan(weight) | (data == 0) | np.isnan(data) | (weight == 0) | (err == 0)
    #if dq is not None:
    #    # only 0 is OK
    #    bad |= (dq != 0)
    if wht is not None:
        bad |= (wht == 0)

    #weight[weight > maxweight] = maxweight
    #weight[weight < minweight] = minweight
    # it seems that crowdsource doesn't like zero weights
    # may have caused broked f466n? weight[bad] = badweight
    #weight[bad] = minweight
    # crowdsource explicitly handles weight=0, so this _should_ work.
    weight[bad] = 0

    # Expand bad pixel zones for dq
    #bad_for_dq = ndimage.binary_dilation(bad, iterations=2)
    #dq[bad_for_dq] = 2 | 2**30 | 2**31
    #print(f"Total bad pixels = {bad.sum()}, total bad for dq={bad_for_dq.sum()}")

    return dq, weight, bad


def main(smoothing_scales={'f182m': 0.25, 'f187n':0.25, 'f212n':0.55,
                           'f410m': 0.55, 'f405n':0.55, 'f466n':0.55},
        bg_boxsizes={'f182m': 19, 'f187n':11, 'f212n':11,
                     'f410m': 11, 'f405n':11, 'f466n':11,
                     'f444w': 11, 'f356w':11,
                     'f200w':19, 'f115w':19,
                    },
        crowdsource_default_kwargs={'maxstars': 500000, },
        ):
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-f", "--filternames", dest="filternames",
                    default='F466N,F405N,F410M',
                    help="filter name list", metavar="filternames")
    parser.add_option("-m", "--modules", dest="modules",
                    default='nrca,nrcb,merged,merged-reproject',
                    help="module list", metavar="modules")
    parser.add_option("-d", "--desaturated", dest="desaturated",
                    default=False,
                    action='store_true',
                    help="use image with saturated stars removed?", metavar="desaturated")
    parser.add_option("--daophot", dest="daophot",
                    default=False,
                    action='store_true',
                    help="run daophot?", metavar="daophot")
    parser.add_option("--skip-crowdsource", dest="nocrowdsource",
                    default=False,
                    action='store_true',
                    help="skip crowdsource?", metavar="nocrowdsource")
    parser.add_option("--bgsub", dest="bgsub",
                    default=False,
                    action='store_true',
                    help="perform background-subtraction first?", metavar="bgsub")
    parser.add_option("--epsf", dest="epsf",
                    default=False,
                    action='store_true',
                    help="try to make & use an ePSF?", metavar="epsf")
    parser.add_option("--blur", dest="blur",
                    default=False,
                    action='store_true',
                    help="blur the PSF?", metavar="blur")
    parser.add_option("--proposal_id", dest="proposal_id",
                    default='2221',
                    help="proposal_id", metavar="proposal_id")
    parser.add_option("--target", dest="target",
                    default='brick',
                    help="target", metavar="target")
    parser.add_option("--group", dest="group",
                      default=False,
                      action='store_true')
    parser.add_option('--each-exposure', dest='each_exposure',
                      default=False, action='store_true',
                      help='Photometer _each_ exposure?', metavar='each_exposure')
    parser.add_option('--each-suffix', dest='each_suffix',
                      default='destreak_o001_crf',
                      help='Suffix for the level-2 products', metavar='each_suffix')
    (options, args) = parser.parse_args()

    filternames = options.filternames.split(",")
    modules = options.modules.split(",")
    proposal_id = options.proposal_id
    target = options.target

    nvisits = {'2221': {'brick': 1, 'cloudc': 2},
               '1182': {'brick': 2}
               }
    field_to_reg_mapping = {'2221': {'001': 'brick', '002': 'cloudc'},
                            '1182': {'004': 'brick'}}[proposal_id]
    reg_to_field_mapping = {v:k for k,v in field_to_reg_mapping.items()}
    field = reg_to_field_mapping[target]

    basepath = f'/blue/adamginsburg/adamginsburg/jwst/{field_to_reg_mapping[field]}/'

    pl.close('all')

    print(f"options: {options}")

    # need to have incrementing _before_ test
    index = -1

    for module in modules:
        detector = module # no sub-detectors for long-NIRCAM
        for filtername in filternames:
            if options.each_exposure:
                for visitid in range(1, nvisits[proposal_id][target] + 1):
                    visitid = f'{visitid:03d}'
                    filenames = get_filenames(basepath, filtername, proposal_id,
                                              field, visitid=visitid,
                                              each_suffix=options.each_suffix,
                                              module=module, pupil='clear')
                    if len(filenames) > 0:
                        print(f"Looping over filenames {filenames} for filter={filtername} proposal={proposal_id} field={field} visitid={visitid}")
                        # jw02221001001_07101_00024_nrcblong_destreak_o001_crf.fits
                        for filename in filenames:

                            index += 1
                            # enable array jobs
                            if os.getenv('SLURM_ARRAY_TASK_ID') is not None and int(os.getenv('SLURM_ARRAY_TASK_ID')) != index:
                                print(f'Task={os.getenv("SLURM_ARRAY_TASK_ID")} does not match index {index}')
                                continue

                            exposure_id = filename.split("_")[2]
                            visit_id = filename.split("_")[0][-3:]
                            vgroup_id = filename.split("_")[1]
                            do_photometry_step(options, filtername, module, detector,
                                               field, basepath, filename, proposal_id,
                                               crowdsource_default_kwargs, exposurenumber=int(exposure_id),
                                               visit_id=visit_id, vgroup_id=vgroup_id,
                                               use_webbpsf=True,
                                               bg_boxsizes=bg_boxsizes)
            else:
                filename = get_filename(basepath, filtername, proposal_id, field, module, options=options, pupil='clear')
                do_photometry_step(options, filtername, module, detector, field,
                                   basepath, filename, proposal_id, crowdsource_default_kwargs,
                                   bg_boxsizes=bg_boxsizes
                                   )


def get_filenames(basepath, filtername, proposal_id, field, each_suffix, module, pupil='clear', visitid='001'):

    # jw01182004002_02101_00012_nrcalong_destreak_o004_crf.fits
    # jw02221001001_07101_00012_nrcalong_destreak_o001_crf.fits
    # jw02221001001_05101_00022_nrcb3_destreak_o001_crf.fits
    glstr = f'{basepath}/{filtername}/pipeline/jw0{proposal_id}{field}{visitid}*{module}*{each_suffix}.fits'
    fglob = glob.glob(glstr)
    if len(fglob) == 0:
        raise ValueError(f"No matches found to {glstr}")
    else:
        return fglob


def get_filename(basepath, filtername, proposal_id, field, module, options, pupil='clear'):
    desat = '_unsatstar' if options.desaturated else ''
    bgsub = '_bgsub' if options.bgsub else ''
    #epsf_ = "_epsf" if options.epsf else ""
    #blur_ = "_blur" if options.blur else ""

    filename = f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filtername.lower()}-{module}_i2d.fits'
    if not os.path.exists(filename):
        filename = f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filtername.lower()}-{module}_realigned-to-refcat.fits'
    if not os.path.exists(filename):
        # merged-reproject_i2d.fits lives here
        # 12/22/2023: that is generally the best-behaved; it's the only one with no clear misalignments.  tweakreg-based merge just doesn't lock in coords
        filename = f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filtername.lower()}-{module}_i2d{desat}.fits'
    if not os.path.exists(filename):
        pupil = 'F444W'
        filename = f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filtername.lower()}-{module}_nodestreak_realigned-to-refcat.fits'
    if not os.path.exists(filename):
        filename = f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filtername.lower()}-{module}_realigned-to-refcat.fits'
    if not os.path.exists(filename):
        # merged-reproject_i2d.fits lives here
        # 12/22/2023: that is generally the best-behaved; it's the only one with no clear misalignments.  tweakreg-based merge just doesn't lock in coords
        filename = f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filtername.lower()}-{module}_i2d{desat}.fits'
    if not os.path.exists(filename):
        glstr = f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_*-{module}_realigned-to-refcat.fits'
        fglob = glob.glob(glstr)
        if len(fglob) == 1:
            filename = fglob[0]
        else:
            raise ValueError(f"File {filename} does not exist, and nothing matching {glstr} exists either.  pupil={pupil}")

    return filename


def do_photometry_step(options, filtername, module, detector, field, basepath,
                       filename, proposal_id, crowdsource_default_kwargs, exposurenumber=None,
                       visit_id=None, vgroup_id=None,
                       bg_boxsizes=None,
                       use_webbpsf=False,
                       pupil='clear'):
    print(f"Starting {field} filter {filtername} module {module} detector {detector} {exposurenumber}", flush=True)
    fwhm_tbl = Table.read(f'{basepath}/reduction/fwhm_table.ecsv')
    row = fwhm_tbl[fwhm_tbl['Filter'] == filtername]
    fwhm = fwhm_arcsec = float(row['PSF FWHM (arcsec)'][0])
    fwhm_pix = float(row['PSF FWHM (pixel)'][0])

    # redundant, saves me renaming variables....
    filt = filtername

    # file naming suffixes
    desat = '_unsatstar' if options.desaturated else ''
    bgsub = '_bgsub' if options.bgsub else ''
    epsf_ = "_epsf" if options.epsf else ""
    exposure_ = f'_exp{exposurenumber:05d}' if exposurenumber is not None else ''
    visitid_ = f'_visit{int(visit_id):03d}' if visit_id is not None else ''
    vgroupid_ = f'_vgroup{int(vgroup_id)}' if vgroup_id is not None else ''
    blur_ = "_blur" if options.blur else ""
    group = "_group" if options.group else ""

    print(f"Starting cataloging on {filename}", flush=True)
    fh, im1, data, wht, err, instrument, telescope, obsdate = load_data(filename)

    # set up coordinate system
    ww = wcs.WCS(im1[1].header)
    pixscale = ww.proj_plane_pixel_area()**0.5
    cen = ww.pixel_to_world(im1[1].shape[1]/2, im1[1].shape[0]/2)

    if options.bgsub:
        # background subtraction
        # see BackgroundEstimationExperiments.ipynb
        bkg = Background2D(data, box_size=bg_boxsizes[filt.lower()], bkg_estimator=MedianBackground())
        fits.PrimaryHDU(data=bkg.background,
                        header=im1['SCI'].header).writeto(filename.replace(".fits",
                                                                           "_background.fits"),
                                                          overwrite=True)

        # subtract background, but then re-zero the edges
        zeros = data == 0
        data = data - bkg.background
        data[zeros] = 0

        fits.PrimaryHDU(data=data, header=im1['SCI'].header).writeto(filename.replace(".fits", "_bgsub.fits"), overwrite=True)

    # try to limit memory use before we start photometry
    data = data.astype('float32')

    # Load PSF model
    grid, psf_model = get_psf_model(filtername, proposal_id, field,
                                    module=module,
                                    use_webbpsf=use_webbpsf,
                                    # if we're doing each exposure, we want the full grid
                                    use_grid=options.each_exposure,
                                    blur=options.blur,
                                    target=options.target,
                                    obsdate=obsdate,
                                    basepath='/blue/adamginsburg/adamginsburg/jwst/')
    dao_psf_model = grid

    # bound the flux to be >= 0 (no negative peak fitting)
    dao_psf_model.flux.min = 0

    dq, weight, bad = get_uncertainty(err, data, wht=wht, dq=im1['DQ'].data if 'DQ' in im1 else None)

    filter_table = SvoFps.get_filter_list(facility=telescope, instrument=instrument)
    filter_table.add_index('filterID')
    instrument = 'NIRCam'
    eff_wavelength = filter_table.loc[f'{telescope}/{instrument}.{filt}']['WavelengthEff'] * u.AA

    # DAO Photometry setup
    grouper = SourceGrouper(2 * fwhm_pix)
    mmm_bkg = MMMBackground()

    filtered_errest = stats.sigma_clipped_stats(data, stdfunc='mad_std')
    print(f'Error estimate for DAO from stats.: {filtered_errest}', flush=True)
    filtered_errest = np.nanmedian(err)
    print(f'Error estimate for DAO from median(err): {filtered_errest}', flush=True)

    daofind_tuned = DAOStarFinder(threshold=5 * filtered_errest,
                                  fwhm=fwhm_pix, roundhi=1.0, roundlo=-1.0,
                                  sharplo=0.30, sharphi=1.40)
    print("Finding stars with daofind_tuned", flush=True)
    finstars = daofind_tuned(np.nan_to_num(data))

    print(f"Found {len(finstars)} with daofind_tuned", flush=True)
    # for diagnostic plotting convenience
    finstars['x'] = finstars['xcentroid']
    finstars['y'] = finstars['ycentroid']
    finstars['skycoord'] = ww.pixel_to_world(finstars['x'], finstars['y'])

    result = save_photutils_results(finstars, ww, filename,
                                    im1=im1, detector=detector,
                                    basepath=basepath,
                                    filtername=filtername, module=module,
                                    desat=desat, bgsub=bgsub,
                                    blur="",
                                    exposure_=exposure_,
                                    visitid_=visitid_, vgroupid_=vgroupid_,
                                    basic_or_iterative='daofind',
                                    options=options,
                                    epsf_="",
                                    fpsf="",
                                    group=group,
                                    psf=None)

    stars = finstars # because I'm copy-pasting code...

    # Set up visualization
    reg = regions.RectangleSkyRegion(center=cen, width=1.5*u.arcmin, height=1.5*u.arcmin)
    preg = reg.to_pixel(ww)
    #mask = preg.to_mask()
    #cutout = mask.cutout(im1[1].data)
    #err = mask.cutout(im1[2].data)
    region_list = [y for x in glob.glob('regions_/*zoom*.reg') for y in
                   regions.Regions.read(x)]
    zoomcut_list = {reg.meta['text']: reg.to_pixel(ww).to_mask().get_overlap_slices(data.shape)[0]
                    for reg in region_list}
    zoomcut_list = {nm:slc for nm,slc in zoomcut_list.items()
                    if slc is not None and
                    slc[0].start > 0 and slc[1].start > 0
                    and slc[0].stop < data.shape[0] and slc[1].stop < data.shape[1]}

    zoomcut = slice(128, 256), slice(128, 256)
    modsky = data*0 # no model for daofind
    nullslice = (slice(None), slice(None))

    try:
        catalog_zoom_diagnostic(data, modsky, nullslice, stars)
        pl.suptitle(f"daofind Catalog Diagnostics zoomed {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}")
        pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}_catalog_diagnostics_daofind.png',
                bbox_inches='tight')

        catalog_zoom_diagnostic(data, modsky, zoomcut, stars)
        pl.suptitle(f"daofind Catalog Diagnostics {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}")
        pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}_catalog_diagnostics_zoom_daofind.png',
                bbox_inches='tight')

        for name, zoomcut in zoomcut_list.items():
            catalog_zoom_diagnostic(data, modsky, zoomcut, stars)
            pl.suptitle(f"daofind Catalog Diagnostics {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub} zoom {name}")
            pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}_catalog_diagnostics_zoom{name.replace(" ","_")}_daofind.png',
                    bbox_inches='tight')
    except Exception as ex:
        print(f'FAILURE to produce catalog zoom diagnostics for module {module} and filter {filtername} for basic daofinder: {ex}')

    if not options.nocrowdsource:

        t0 = time.time()

        if False: # why do the unweighted version?
            print()
            print("starting crowdsource unweighted", flush=True)
            results_unweighted = fit_im(np.nan_to_num(data), psf_model, weight=np.ones_like(data)*np.nanmedian(weight),
                                        # psfderiv=np.gradient(-psf_initial[0].data),
                                        dq=dq,
                                        nskyx=0, nskyy=0, refit_psf=False, verbose=True,
                                        **crowdsource_default_kwargs,
                                        )
            print(f"Done with unweighted crowdsource. dt={time.time() - t0}")
            stars, modsky, skymsky, psf = results_unweighted
            stars = save_crowdsource_results(results_unweighted, ww, filename,
                                             im1=im1, detector=detector,
                                             basepath=basepath,
                                             filtername=filtername, module=module,
                                             desat=desat, bgsub=bgsub,
                                             blur=options.blur,
                                             exposure_=exposure_,
                                             visitid_=visitid_,
                                             vgroupid_=vgroupid_,
                                             options=options,
                                             suffix="unweighted", psf=None)

            zoomcut = slice(128, 256), slice(128, 256)

            try:
                catalog_zoom_diagnostic(data, modsky, nullslice, stars)
                pl.suptitle(f"Crowdsource nsky=0 unweighted Catalog Diagnostics zoomed {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{blur_}")
                pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{blur_}_catalog_diagnostics_unweighted.png',
                        bbox_inches='tight')

                catalog_zoom_diagnostic(data, modsky, zoomcut, stars)
                pl.suptitle(f"Crowdsource nsky=0 unweighted Catalog Diagnostics {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{blur_}")
                pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{blur_}_catalog_diagnostics_zoom_unweighted.png',
                        bbox_inches='tight')
                for name, zoomcut in zoomcut_list.items():
                    catalog_zoom_diagnostic(data, modsky, zoomcut, stars)
                    pl.suptitle(f"Crowdsource nsky=0 Catalog Diagnostics {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{blur_} zoom {name}")
                    pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{blur_}_catalog_diagnostics_zoom{name.replace(" ","_")}_unweighted.png',
                            bbox_inches='tight')
            except Exception as ex:
                print(f'FAILURE to produce catalog zoom diagnostics for module {module} and filter {filtername} for unweighted crowdsource: {ex}')
                exc_tb = sys.exc_info()[2]
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(f"Exception {ex} was in {fname} line {exc_tb.tb_lineno}")

            fig = pl.figure(0, figsize=(10,10))
            fig.clf()
            ax = fig.gca()
            im = ax.imshow(weight, norm=simple_norm(weight, stretch='log'))
            pl.colorbar(mappable=im)
            pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}_weights.png',
                    bbox_inches='tight')

        #for refit_psf, fpsf in zip((False, True), ('', '_fitpsf',)):
        for refit_psf, fpsf in zip((False, ), ('', )):
            for nsky in (0, ): #1, ):
                t0 = time.time()
                print()
                print(f"Running crowdsource fit_im with weights & nskyx=nskyy={nsky} & fpsf={fpsf} & blur={blur_}")
                print(f"data.shape={data.shape} weight_shape={weight.shape}", flush=True)
                results = fit_im(np.nan_to_num(data), psf_model, weight=weight,
                                    nskyx=nsky, nskyy=nsky, refit_psf=refit_psf, verbose=True,
                                    dq=dq,
                                    **crowdsource_default_kwargs
                                    )
                print(f"Done with weighted, refit={fpsf}, nsky={nsky} crowdsource. dt={time.time() - t0}")
                stars, modsky, skymsky, psf = results
                stars = save_crowdsource_results(results, ww, filename,
                                                 im1=im1, detector=detector,
                                                 basepath=basepath,
                                                 filtername=filtername,
                                                 module=module, desat=desat,
                                                 bgsub=bgsub, fpsf=fpsf,
                                                 blur=options.blur,
                                                 exposure_=exposure_,
                                                 visitid_=visitid_,
                                                 vgroupid_=vgroupid_,
                                                 psf=psf if refit_psf else None,
                                                 options=options,
                                                 suffix=f"nsky{nsky}")

                zoomcut = slice(128, 256), slice(128, 256)

                try:
                    catalog_zoom_diagnostic(data, modsky, nullslice, stars)
                    pl.suptitle(f"Catalog Diagnostics {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{fpsf}{blur_} nsky={nsky} weighted")
                    pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{fpsf}{blur_}_nsky{nsky}_weighted_catalog_diagnostics.png',
                            bbox_inches='tight')

                    catalog_zoom_diagnostic(data, modsky, zoomcut, stars)
                    pl.suptitle(f"Catalog Diagnostics zoomed {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{fpsf}{blur_} nsky={nsky} weighted")
                    pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{fpsf}{blur_}_nsky{nsky}_weighted_catalog_diagnostics_zoom.png',
                            bbox_inches='tight')

                    for name, zoomcut in zoomcut_list.items():
                        catalog_zoom_diagnostic(data, modsky, zoomcut, stars)
                        pl.suptitle(f"Crowdsource nsky={nsky} weighted Catalog Diagnostics {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{fpsf}{blur_} zoom {name}")
                        pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{fpsf}{blur_}_nsky{nsky}_weighted_catalog_diagnostics_zoom{name.replace(" ","_")}.png',
                                bbox_inches='tight')
                except Exception as ex:
                    print(f'FAILURE to produce catalog zoom diagnostics for module {module} and filter {filtername} for crowdsource nsky={nsky} refitpsf={refit_psf} blur={options.blur}: {ex}')
                    exc_tb = sys.exc_info()[2]
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(f"Exception {ex} was in {fname} line {exc_tb.tb_lineno}")

    if options.daophot:
        t0 = time.time()
        print("Starting basic PSF photometry", flush=True)

        if options.epsf:
            print("Building EPSF")
            epsf_builder = EPSFBuilder(oversampling=3, maxiters=10,
                                       smoothing_kernel='quadratic',
                                       progress_bar=True)

            epsfsel = ((finstars['peak'] > 200) &
                       (finstars['roundness1'] > -0.25) &
                       (finstars['roundness1'] < 0.25) &
                       (finstars['roundness2'] > -0.25) &
                       (finstars['roundness2'] < 0.25) &
                       (finstars['sharpness'] > 0.4) &
                       (finstars['sharpness'] < 0.8))

            print(f"Extracting {epsfsel.sum()} stars")
            stars = extract_stars(NDData(data=np.nan_to_num(data)), finstars[epsfsel], size=25)

            # reject stars with negative pixels
            #stars = EPSFStars([x for x in stars if x.data.min() >= 0])
            # apparently this failed - too restrictive?

            for star in stars:
                # background subtraction
                star.data[:] -= np.nanpercentile(star.data, 5)

            epsf, fitted_stars = epsf_builder(stars)

            # trim edges
            epsf._data = epsf.data[2:-2, 2:-2]

            norm = simple_norm(epsf.data, 'log', percent=99.0)
            pl.figure(1).clf()
            pl.imshow(epsf.data, norm=norm, origin='lower', cmap='viridis')
            pl.colorbar()
            pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}_daophot_epsf.png',
                       bbox_inches='tight')
            dao_psf_model = epsf

            save_epsf(epsf,
                      f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}_daophot_epsf.fits')


        phot_basic = PSFPhotometry(finder=daofind_tuned,#finder_maker(),
                             #grouper=grouper, # the grouper is needed to jointly fit stars and avoid faint stars jumping onto bright ones
                             grouper=grouper if options.group else None,
                             # localbkg_estimator=None, # must be none or it un-saturates pixels
                             localbkg_estimator=LocalBackground(5, 15),
                             psf_model=dao_psf_model,
                             fitter=LevMarLSQFitter(),
                             fit_shape=(5, 5),
                             aperture_radius=2*fwhm_pix,
                             progress_bar=True,
                            )

        print("About to do BASIC photometry....")
        result = phot_basic(np.nan_to_num(data))
        print(f"Done with BASIC photometry.  len(result)={len(result)} dt={time.time() - t0}")

        # remove negative-peak and zero-peak sources (they affect the residuals badly)
        # we don't want to remove them now; we need to flag out objects that are too close to negative stars
        #bad = result['flux_fit'] <= 0
        #result = result[~bad]

        result = save_photutils_results(result, ww, filename,
                                        im1=im1, detector=detector,
                                        basepath=basepath,
                                        filtername=filtername, module=module,
                                        desat=desat, bgsub=bgsub,
                                        blur=options.blur,
                                        exposure_=exposure_,
                                        visitid_=visitid_,
                                        vgroupid_=vgroupid_,
                                        basic_or_iterative='basic',
                                        options=options,
                                        epsf_=epsf_,
                                        group=group,
                                        psf=None)

        stars = result
        stars['x'] = stars['x_fit']
        stars['y'] = stars['y_fit']
        print("Creating BASIC residual image, using 21x21 patches")
        modsky = phot_basic.make_model_image(data.shape, (21, 21), include_localbkg=False)
        residual = data - modsky
        print("Done creating BASIC residual image, using 21x21 patches")
        fits.PrimaryHDU(data=residual, header=im1[1].header).writeto(
            f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}_daophot_basic_residual.fits',
            overwrite=True)
        fits.PrimaryHDU(data=modsky, header=im1[1].header).writeto(
            f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}_daophot_basic_model.fits',
            overwrite=True)
        print("Saved BASIC residual image, now making diagnostics.")
        try:
            catalog_zoom_diagnostic(data, modsky, nullslice, stars)
            pl.suptitle(f"daophot basic Catalog Diagnostics zoomed {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}")
            pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}_catalog_diagnostics_daophot_basic.png',
                    bbox_inches='tight')

            catalog_zoom_diagnostic(data, modsky, zoomcut, stars)
            pl.suptitle(f"daophot basic Catalog Diagnostics {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}")
            pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}_catalog_diagnostics_zoom_daophot_basic.png',
                    bbox_inches='tight')

            for name, zoomcut in zoomcut_list.items():
                catalog_zoom_diagnostic(data, modsky, zoomcut, stars)
                pl.suptitle(f"daophot basic Catalog Diagnostics {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group} zoom {name}")
                pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}__catalog_diagnostics_zoom_daophot_basic{name.replace(" ","_")}.png',
                        bbox_inches='tight')
        except Exception as ex:
            print(f'FAILURE to produce catalog zoom diagnostics for module {module} and filter {filtername} BASIC photometry: {ex}')
            exc_tb = sys.exc_info()[2]
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(f"Exception {ex} was in {fname} line {exc_tb.tb_lineno}")

        print(f"Done with diagnostics for BASIC photometry.  dt={time.time() - t0}")
        pl.close('all')

    if options.daophot:
        t0 = time.time()

        print("Iterative PSF photometry")
        if options.epsf:
            print("Building EPSF")
            epsf_builder = EPSFBuilder(oversampling=3, maxiters=10,
                                       smoothing_kernel='quadratic',
                                       progress_bar=True)

            epsfsel = ((finstars['peak'] > 200) &
                       (finstars['roundness1'] > -0.25) &
                       (finstars['roundness1'] < 0.25) &
                       (finstars['roundness2'] > -0.25) &
                       (finstars['roundness2'] < 0.25) &
                       (finstars['sharpness'] > 0.4) &
                       (finstars['sharpness'] < 0.8))

            print(f"Extracting {epsfsel.sum()} stars")
            stars = extract_stars(NDData(data=np.nan_to_num(data)), finstars[epsfsel], size=35)

            # reject stars with negative pixels
            #stars = EPSFStars([x for x in stars if x.data.min() >= 0])
            # apparently this failed - too restrictive?

            for star in stars:
                # background subtraction
                star.data[:] -= np.nanpercentile(star.data, 5)


            epsf, fitted_stars = epsf_builder(stars)

            # trim edges
            epsf._data = epsf.data[2:-2, 2:-2]

            norm = simple_norm(epsf.data, 'log', percent=99.0)
            pl.figure(1).clf()
            pl.imshow(epsf.data, norm=norm, origin='lower', cmap='viridis')
            pl.colorbar()
            pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}_daophot_epsf.png',
                       bbox_inches='tight')
            dao_psf_model = epsf

        # iterative takes for-ev-er
        phot_iter = IterativePSFPhotometry(finder=daofind_tuned,
                                           localbkg_estimator=LocalBackground(5, 15),
                                           grouper=grouper if options.group else None,
                                           psf_model=dao_psf_model,
                                           fitter=LevMarLSQFitter(),
                                           maxiters=5,
                                           fit_shape=(5, 5),
                                           sub_shape=(15, 15),
                                           aperture_radius=2*fwhm_pix,
                                           progress_bar=True,
                                          )

        print("About to do ITERATIVE photometry....")
        result2 = phot_iter(data)
        print(f"Done with ITERATIVE photometry. len(result2)={len(result2)}  dt={time.time() - t0}")

        # need to flag stars near negative stars, so we don't want to exclude them _yet_
        #bad = result2['flux_fit'] <= 0
        #result2 = result2[~bad]

        result2 = save_photutils_results(result2, ww, filename,
                                         im1=im1, detector=detector,
                                         basepath=basepath,
                                         filtername=filtername, module=module,
                                         desat=desat, bgsub=bgsub,
                                         blur=options.blur,
                                         exposure_=exposure_,
                                         visitid_=visitid_,
                                         vgroupid_=vgroupid_,
                                         basic_or_iterative='iterative',
                                         options=options,
                                         epsf_=epsf_,
                                         group=group,
                                         psf=None)

        stars = result2
        stars['x'] = stars['x_fit']
        stars['y'] = stars['y_fit']

        print("Creating iterative residual")
        modsky = phot_iter.make_model_image(data.shape, (21, 21), include_localbkg=False)
        residual = data - modsky
        print("finished iterative residual")
        fits.PrimaryHDU(data=residual, header=im1[1].header).writeto(
            f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}_daophot_iterative_residual.fits',
            overwrite=True)
        fits.PrimaryHDU(data=modsky, header=im1[1].header).writeto(
            f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}_daophot_iterative_model.fits',
            overwrite=True)
        print("Saved iterative residual")
        try:
            catalog_zoom_diagnostic(data, modsky, nullslice, stars)
            pl.suptitle(f"daophot iterative Catalog Diagnostics zoomed {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}")
            pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}_catalog_diagnostics_daophot_iterative.png',
                    bbox_inches='tight')

            catalog_zoom_diagnostic(data, modsky, zoomcut, stars)
            pl.suptitle(f"daophot iterative Catalog Diagnostics {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}")
            pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}_catalog_diagnostics_zoom_daophot_iterative.png',
                    bbox_inches='tight')

            for name, zoomcut in zoomcut_list.items():
                catalog_zoom_diagnostic(data, modsky, zoomcut, stars)
                pl.suptitle(f"daophot iterative Catalog Diagnostics {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group} zoom {name}")
                pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}__catalog_diagnostics_zoom_daophot_iterative{name.replace(" ","_")}.png',
                        bbox_inches='tight')
        except Exception as ex:
            print(f'FAILURE to produce catalog zoom diagnostics for module {module} and filter {filtername} for ITERATIVE daophot: {ex}')
            exc_tb = sys.exc_info()[2]
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(f"Exception {ex} was in {fname} line {exc_tb.tb_lineno}")

if __name__ == "__main__":
    main()
