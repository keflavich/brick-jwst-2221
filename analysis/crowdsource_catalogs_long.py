print("Starting cataloging", flush=True)
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
from astropy.io import fits
import requests
import requests.exceptions
import urllib3
import urllib3.exceptions
from photutils.detection import DAOStarFinder, IRAFStarFinder
from photutils.psf import DAOGroup, IntegratedGaussianPRF, extract_stars
from photutils.psf import PSFPhotometry, IterativePSFPhotometry, SourceGrouper
from photutils.background import MMMBackground, MADStdBackgroundRMS, MedianBackground, Background2D

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
from webbpsf.utils import to_griddedpsfmodel
import datetime

def print(*args, **kwargs):
    now = datetime.datetime.now().isoformat()
    from builtins import print as printfunc
    return printfunc(f"{now}:", *args, **kwargs)

print("Done with imports", flush=True)

basepath = '/blue/adamginsburg/adamginsburg/jwst/brick/'

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
        rows = rows[:, :, None] + row[None, None, :]
        cols = cols[:, :, None] + col[None, None, :]

        # photutils seems to use column, row notation
        stamps = self.psfgridmodel.evaluate(cols, rows, 1, col, row)
        # it returns something in (nstamps, row, col) shape
        # pretty sure that ought to be (col, row, nstamps) for crowdsource

        if deriv:
            dpsfdrow, dpsfdcol = np.gradient(stamps, axis=(0, 1))
            dpsfdrow = dpsfdrow.T
            dpsfdcol = dpsfdcol.T

        ret = stamps.T
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


def catalog_zoom_diagnostic(data, modsky, zoomcut, stars):
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
    im = pl.subplot(2,2,3).imshow((data-modsky)[zoomcut],
                                  norm=simple_norm((data-modsky)[zoomcut],
                                                   stretch='asinh',
                                                   max_percent=99.5,
                                                   min_percent=0.5),
                                  cmap='gray')
    pl.xticks([]); pl.yticks([]); pl.title("data-modsky")
    pl.colorbar(mappable=im)
    im = pl.subplot(2,2,4).imshow(data[zoomcut],
                                  norm=simple_norm(data[zoomcut],
                                                   stretch='log',
                                                   max_percent=99.95,
                                                   min_cut=0), cmap='gray')
    axlims = pl.axis()
    if zoomcut[0].start:
        pl.axis([0,zoomcut[0].stop-zoomcut[0].start, 0, zoomcut[1].stop-zoomcut[1].start])
        pl.subplot(2,2,4).scatter(stars['x']-zoomcut[1].start, stars['y']-zoomcut[0].start, marker='x', color='r', s=8, linewidth=0.5)
    else:
        pl.subplot(2,2,4).scatter(stars['x'], stars['y'], marker='x', color='r', s=5, linewidth=0.5)
    pl.axis(axlims)
    pl.xticks([]); pl.yticks([]); pl.title("Data with stars");
    pl.colorbar(mappable=im)
    pl.tight_layout()

def main(smoothing_scales={'f182m': 0.25, 'f187n':0.25, 'f212n':0.55,
                           'f410m': 0.55, 'f405n':0.55, 'f466n':0.55},
        bg_boxsizes={'f182m': 19, 'f187n':11, 'f212n':11,
                     'f410m': 11, 'f405n':11, 'f466n':11}
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
    (options, args) = parser.parse_args()

    filternames = options.filternames.split(",")
    modules = options.modules.split(",")
    use_desaturated = options.desaturated

    nullslice = (slice(None), slice(None))

    print(f"options: {options}")

    for module in modules:
        detector = module # no sub-detectors for long-NIRCAM
        for filtername in filternames:
            print(f"Starting filter {filtername}", flush=True)
            fwhm_tbl = Table.read(f'{basepath}/reduction/fwhm_table.ecsv')
            row = fwhm_tbl[fwhm_tbl['Filter'] == filtername]
            fwhm = fwhm_arcsec = float(row['PSF FWHM (arcsec)'][0])
            fwhm_pix = float(row['PSF FWHM (pixel)'][0])

            desat = '_unsatstar' if use_desaturated else ''
            bgsub = '_bgsub' if options.bgsub else ''

            try:
                pupil = 'clear'
                filename = f'{basepath}/{filtername}/pipeline/jw02221-o001_t001_nircam_{pupil}-{filtername.lower()}-{module}_i2d{desat}.fits'
                fh = fits.open(filename)
            except Exception:
                pupil = 'F444W'
                filename = f'{basepath}/{filtername}/pipeline/jw02221-o001_t001_nircam_{pupil}-{filtername.lower()}-{module}_i2d{desat}.fits'
                fh = fits.open(filename)
            print(f"Starting on {filename}", flush=True)

            im1 = fh
            data = im1[1].data
            wht = im1['WHT'].data
            err = im1['ERR'].data
            instrument = im1[0].header['INSTRUME']
            telescope = im1[0].header['TELESCOP']
            #filt = im1[0].header['FILTER']
            filt = filtername

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

            wavelength_table = SvoFps.get_transmission_data(f'{telescope}/{instrument}.{filt}')
            obsdate = im1[0].header['DATE-OBS']

            # psf_fn = f'{basepath}/{instrument.lower()}_{filtername}_samp{oversample}_nspsf{npsf}_npix{fov_pixels}.fits'
            # if os.path.exists(str(psf_fn)):
            #     # As a file
            #     print(f"Loading grid from psf_fn={psf_fn}", flush=True)
            #     grid = to_griddedpsfmodel(psf_fn)  # file created 2 cells above
            #     if isinstance(big_grid, list):
            #         print(f"PSF IS A LIST OF GRIDS!!! this is incompatible with the return from nrc.psf_grid")
            #         grid = grid[0]

            with open(os.path.expanduser('/home/adamginsburg/.mast_api_token'), 'r') as fh:
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
                    nrc.filter = filt
                    print(f"Running {module}{desat}{bgsub}")
                    if module in ('nrca', 'nrcb'):
                        if 'F4' in filt.upper():
                            nrc.detector = f'{module.upper()}5' # I think NRCA5 must be the "long" detector?
                        else:
                            nrc.detector = f'{module.upper()}1' #TODO: figure out a way to use all 4?
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

            # # there's no way to use a grid across all detectors.
            # # the right way would be to use this as a grid of grids, but that apparently isn't supported.
            # if isinstance(grid, list):
            #     grid = grid[0]

            # print("Done with WebbPSF downloading; now building model", flush=True)
            # # yy, xx = np.indices([31,31], dtype=float)
            # # grid.x_0 = grid.y_0 = 15.5
            # # psf_model = crowdsource.psf.SimplePSF(stamp=grid(xx,yy))

            # # bigger PSF probably needed
            # yy, xx = np.indices([61,61], dtype=float)
            # grid.x_0 = grid.y_0 = 30
            # psf_model = crowdsource.psf.SimplePSF(stamp=grid(xx,yy))
            if isinstance(grid, list):
                print(f"Grid is a list: {grid}")
                psf_model = WrappedPSFModel(grid[0])
                dao_psf_model = grid[0]
            else:
                psf_model = WrappedPSFModel(grid)
                dao_psf_model = grid
            psf_model_blur = psf_model

            ww = wcs.WCS(im1[1].header)
            cen = ww.pixel_to_world(im1[1].shape[1]/2, im1[1].shape[0]/2)
            reg = regions.RectangleSkyRegion(center=cen, width=1.5*u.arcmin, height=1.5*u.arcmin)
            preg = reg.to_pixel(ww)
            #mask = preg.to_mask()
            #cutout = mask.cutout(im1[1].data)
            #err = mask.cutout(im1[2].data)
            region_list = [y for x in glob.glob('regions/*zoom*.reg') for y in
                           regions.Regions.read(x)]
            zoomcut_list = {reg.meta['text']: reg.to_pixel(ww).to_mask().get_overlap_slices(data.shape)[0]
                            for reg in region_list}
            zoomcut_list = {nm:slc for nm,slc in zoomcut_list.items()
                            if slc is not None and
                            slc[0].start > 0 and slc[1].start > 0
                            and slc[0].stop < data.shape[0] and slc[1].stop < data.shape[1]}

            # crowdsource uses inverse-sigma, not inverse-variance
            weight = err**-1
            maxweight = np.percentile(weight[np.isfinite(weight)], 95)
            minweight = np.percentile(weight[np.isfinite(weight)], 5)
            badweight =  np.percentile(weight[np.isfinite(weight)], 1)
            weight[err < 1e-5] = 0
            #weight[(err == 0) | (wht == 0)] = np.nanmedian(weight)
            weight[np.isnan(weight)] = 0
            bad = np.isnan(weight) | (data == 0) | np.isnan(data) | (weight == 0) | (err == 0) | (wht == 0) | (data < 1e-5)

            weight[weight > maxweight] = maxweight
            weight[weight < minweight] = minweight
            # it seems that crowdsource doesn't like zero weights
            # may have caused broked f466n? weight[bad] = badweight
            weight[bad] = minweight
            # crowdsource explicitly handles weight=0, so this _should_ work.
            weight[bad] = 0


            filter_table = SvoFps.get_filter_list(facility=telescope, instrument=instrument)
            filter_table.add_index('filterID')
            instrument = 'NIRCam'
            eff_wavelength = filter_table.loc[f'{telescope}/{instrument}.{filt}']['WavelengthEff'] * u.AA

            #fwhm = (1.22 * eff_wavelength / (6.5*u.m)).to(u.arcsec, u.dimensionless_angles())

            pixscale = ww.proj_plane_pixel_area()**0.5
            #fwhm_pix = (fwhm / pixscale).decompose().value

            #daogroup = DAOGroup(2 * fwhm_pix)
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
            stars = finstars # because I'm copy-pasting code...

            zoomcut = slice(128, 256), slice(128, 256)
            modsky = data*0 # no model for daofind
            try:
                catalog_zoom_diagnostic(data, modsky, nullslice, stars)
                pl.suptitle(f"daofind Catalog Diagnostics zoomed {filtername} {module}{desat}{bgsub}")
                pl.savefig(f'{basepath}/{filtername}/pipeline/jw02221-o001_t001_nircam_{pupil}-{filtername.lower()}-{module}{desat}{bgsub}_catalog_diagnostics_daofind.png',
                        bbox_inches='tight')

                catalog_zoom_diagnostic(data, modsky, zoomcut, stars)
                pl.suptitle(f"daofind Catalog Diagnostics {filtername} {module}{desat}{bgsub}")
                pl.savefig(f'{basepath}/{filtername}/pipeline/jw02221-o001_t001_nircam_{pupil}-{filtername.lower()}-{module}{desat}{bgsub}_catalog_diagnostics_zoom_daofind.png',
                        bbox_inches='tight')

                for name, zoomcut in zoomcut_list.items():
                    catalog_zoom_diagnostic(data, modsky, zoomcut, stars)
                    pl.suptitle(f"daofind Catalog Diagnostics {filtername} {module}{desat}{bgsub} zoom {name}")
                    pl.savefig(f'{basepath}/{filtername}/pipeline/jw02221-o001_t001_nircam_{pupil}-{filtername.lower()}-{module}{desat}{bgsub}_catalog_diagnostics_zoom{name.replace(" ","_")}_daofind.png',
                            bbox_inches='tight')
            except Exception as ex:
                print(f'FAILURE: {ex}')
            #grid.x_0 = 0
            #grid.y_0 = 0
            # not needed? def evaluate(x, y, flux, x_0, y_0):
            # not needed?     """
            # not needed?     Evaluate the `GriddedPSFModel` for the input parameters.
            # not needed?     """
            # not needed?     # Get the local PSF at the (x_0,y_0)
            # not needed?     psfmodel = grid._compute_local_model(x_0+slcs[1].start, y_0+slcs[0].start)

            # not needed?     # now evaluate the PSF at the (x_0, y_0) subpixel position on
            # not needed?     # the input (x, y) values
            # not needed?     return psfmodel.evaluate(x, y, flux, x_0, y_0)
            # not needed? grid.evaluate = evaluate

            if not options.nocrowdsource:

                t0 = time.time()

                print("starting crowdsource unweighted", flush=True)
                results_unweighted  = fit_im(np.nan_to_num(data), psf_model, weight=np.ones_like(data)*np.nanmedian(weight),
                                                #psfderiv=np.gradient(-psf_initial[0].data),
                                                nskyx=1, nskyy=1, refit_psf=False, verbose=True)
                print(f"Done with unweighted crowdsource. dt={time.time() - t0}")
                stars, modsky, skymsky, psf = results_unweighted
                stars = Table(stars)
                # crowdsource explicitly inverts x & y from the numpy convention:
                # https://github.com/schlafly/crowdsource/issues/11
                coords = ww.pixel_to_world(stars['y'], stars['x'])
                stars['skycoord'] = coords
                stars['x'], stars['y'] = stars['y'], stars['x']

                stars.meta['filename'] = filename
                stars.meta['filter'] = filtername
                stars.meta['module'] = module
                stars.meta['detector'] = detector

                tblfilename = (f"{basepath}/{filtername}/"
                               f"{filtername.lower()}_{module}{desat}{bgsub}"
                               "_crowdsource_unweighted.fits")
                stars.write(tblfilename, overwrite=True)
                # add WCS-containing header
                with fits.open(tblfilename, mode='update', output_verify='fix') as fh:
                    fh[0].header.update(im1[1].header)
                fits.PrimaryHDU(data=skymsky, header=im1[1].header).writeto(f"{basepath}/{filtername}/{filtername.lower()}_{module}{desat}{bgsub}_crowdsource_skymodel_unweighted.fits", overwrite=True)

                zoomcut = slice(128, 256), slice(128, 256)

                try:
                    catalog_zoom_diagnostic(data, modsky, nullslice, stars)
                    pl.suptitle(f"Crowdsource nsky=1 unweighted Catalog Diagnostics zoomed {filtername} {module}{desat}{bgsub}")
                    pl.savefig(f'{basepath}/{filtername}/pipeline/jw02221-o001_t001_nircam_{pupil}-{filtername.lower()}-{module}{desat}{bgsub}_catalog_diagnostics_unweighted.png',
                            bbox_inches='tight')

                    catalog_zoom_diagnostic(data, modsky, zoomcut, stars)
                    pl.suptitle(f"Crowdsource nsky=1 unweighted Catalog Diagnostics {filtername} {module}{desat}{bgsub}")
                    pl.savefig(f'{basepath}/{filtername}/pipeline/jw02221-o001_t001_nircam_{pupil}-{filtername.lower()}-{module}{desat}{bgsub}_catalog_diagnostics_zoom_unweighted.png',
                            bbox_inches='tight')
                    for name, zoomcut in zoomcut_list.items():
                        catalog_zoom_diagnostic(data, modsky, zoomcut, stars)
                        pl.suptitle(f"Crowdsource nsky=1 Catalog Diagnostics {filtername} {module}{desat}{bgsub} zoom {name}")
                        pl.savefig(f'{basepath}/{filtername}/pipeline/jw02221-o001_t001_nircam_{pupil}-{filtername.lower()}-{module}{desat}{bgsub}_catalog_diagnostics_zoom{name.replace(" ","_")}_unweighted.png',
                                bbox_inches='tight')
                except Exception as ex:
                    print(f'FAILURE: {ex}')

                # pl.figure(figsize=(10,5))
                # pl.subplot(1,2,1).imshow(psf_model(30,30), norm=simple_norm(psf_model(30,30), stretch='log'), cmap='cividis')
                # pl.title("Input model")
                # pl.subplot(1,2,2).imshow(psf(30,30), norm=simple_norm(psf(30,30), stretch='log'), cmap='cividis')
                # pl.title("Fitted model")



                # yy, xx = np.indices([61, 61], dtype=float)
                # grid.x_0 = preg.center.x+30
                # grid.y_0 = preg.center.y+30
                # gpsf2 = grid(xx+preg.center.x, yy+preg.center.y)
                # psf_model = crowdsource.psf.SimplePSF(stamp=gpsf2)

                # smoothing_scale = smoothing_scales[filt.lower()]
                # gpsf3 = convolve(gpsf2, Gaussian2DKernel(smoothing_scale))
                # psf_model_blur = crowdsource.psf.SimplePSF(stamp=gpsf3)

                fig = pl.figure(0, figsize=(10,10))
                fig.clf()
                ax = fig.gca()
                im = ax.imshow(weight, norm=simple_norm(weight, stretch='log')); pl.colorbar(mappable=im);
                pl.savefig(f'{basepath}/{filtername}/pipeline/jw02221-o001_t001_nircam_{pupil}-{filtername.lower()}-{module}{desat}{bgsub}_weights.png',
                        bbox_inches='tight')

                # t0 = time.time()
                # print("Running crowdsource fit_im with weights")
                # results_blur  = fit_im(np.nan_to_num(data), psf_model_blur, weight=weight,
                #                     nskyx=1, nskyy=1, refit_psf=False, verbose=True)
                # print(f"Done with weighted, nsky=1 crowdsource. dt={time.time() - t0}")
                # stars, modsky, skymsky, psf = results_blur
                # stars = Table(stars)

                # # crowdsource explicitly inverts x & y from the numpy convention:
                # # https://github.com/schlafly/crowdsource/issues/11
                # coords = ww.pixel_to_world(stars['y'], stars['x'])
                # stars['skycoord'] = coords
                # stars['x'], stars['y'] = stars['y'], stars['x']

                # tblfilename = f"{basepath}/{filtername}/{filtername.lower()}_{module}{desat}{bgsub}_crowdsource.fits"
                # stars.write(tblfilename, overwrite=True)
                # # add WCS-containing header
                # with fits.open(tblfilename, mode='update', output_verify='fix') as fh:
                #     fh[0].header.update(im1[1].header)

                # fits.PrimaryHDU(data=skymsky, header=im1[1].header).writeto(f"{basepath}/{filtername}/{filtername.lower()}_{module}{desat}{bgsub}_crowdsource_skymodel.fits", overwrite=True)
                # fits.PrimaryHDU(data=data-modsky,
                #                 header=im1[1].header).writeto(f"{basepath}/{filtername}/{filtername.lower()}_{module}{desat}{bgsub}_crowdsource_data-modsky.fits", overwrite=True)

                # zoomcut = slice(128, 256), slice(128, 256)

                # catalog_zoom_diagnostic(data, modsky, nullslice, stars)
                # pl.suptitle(f"Crowdsource nsky=1 weighted Catalog Diagnostics zoomed {filtername} {module}{desat}{bgsub}")
                # pl.savefig(f'{basepath}/{filtername}/pipeline/jw02221-o001_t001_nircam_{pupil}-{filtername.lower()}-{module}{desat}{bgsub}_catalog_diagnostics_weighted_nsky1.png',
                #         bbox_inches='tight')

                # catalog_zoom_diagnostic(data, modsky, zoomcut, stars)
                # pl.suptitle(f"Crowdsource nsky=1 weighted Catalog Diagnostics {filtername} {module}{desat}{bgsub}")
                # pl.savefig(f'{basepath}/{filtername}/pipeline/jw02221-o001_t001_nircam_{pupil}-{filtername.lower()}-{module}{desat}{bgsub}_catalog_diagnostics_zoom_weighted_nsky1.png',
                #         bbox_inches='tight')



                for refit_psf, fpsf in zip((True, False), ('_fitpsf', '')):
                    for nsky in (0, 1, ):
                        t0 = time.time()
                        print(f"Running crowdsource fit_im with weights & nskyx=nskyy={nsky}")
                        print(f"data.shape={data.shape} weight_shape={weight.shape}", flush=True)
                        results_blur  = fit_im(np.nan_to_num(data), psf_model_blur, weight=weight,
                                            nskyx=nsky, nskyy=nsky, refit_psf=refit_psf, verbose=True)
                        print(f"Done with weighted, refit={fpsf}, nsky={nsky} crowdsource. dt={time.time() - t0}")
                        stars, modsky, skymsky, psf = results_blur
                        stars = Table(stars)

                        # crowdsource explicitly inverts x & y from the numpy convention:
                        # https://github.com/schlafly/crowdsource/issues/11
                        coords = ww.pixel_to_world(stars['y'], stars['x'])
                        stars['skycoord'] = coords
                        stars['x'], stars['y'] = stars['y'], stars['x']

                        tblfilename = f"{basepath}/{filtername}/{filtername.lower()}_{module}{desat}{bgsub}_crowdsource_nsky{nsky}.fits"
                        stars.write(tblfilename, overwrite=True)
                        # add WCS-containing header
                        with fits.open(tblfilename, mode='update', output_verify='fix') as fh:
                            fh[0].header.update(im1[1].header)

                        fits.PrimaryHDU(data=skymsky, header=im1[1].header).writeto(f"{basepath}/{filtername}/{filtername.lower()}_{module}{desat}{bgsub}{fpsf}_crowdsource_nsky{nsky}_skymodel.fits", overwrite=True)
                        fits.PrimaryHDU(data=data-modsky,
                                        header=im1[1].header).writeto(f"{basepath}/{filtername}/{filtername.lower()}_{module}{desat}{bgsub}{fpsf}_crowdsource_nsky{nsky}_data-modsky.fits", overwrite=True)

                        zoomcut = slice(128, 256), slice(128, 256)


                        catalog_zoom_diagnostic(data, modsky, nullslice, stars)
                        pl.suptitle(f"Catalog Diagnostics {filtername} {module}{desat}{bgsub}{fpsf} nsky={nsky} weighted")
                        pl.savefig(f'{basepath}/{filtername}/pipeline/jw02221-o001_t001_nircam_{pupil}-{filtername.lower()}-{module}{desat}{bgsub}{fpsf}_nsky{nsky}_weighted_catalog_diagnostics.png',
                                bbox_inches='tight')

                        catalog_zoom_diagnostic(data, modsky, zoomcut, stars)
                        pl.suptitle(f"Catalog Diagnostics zoomed {filtername} {module}{desat}{bgsub}{fpsf} nsky={nsky} weighted")
                        pl.savefig(f'{basepath}/{filtername}/pipeline/jw02221-o001_t001_nircam_{pupil}-{filtername.lower()}-{module}{desat}{bgsub}{fpsf}_nsky{nsky}_weighted_catalog_diagnostics_zoom.png',
                                bbox_inches='tight')

                        for name, zoomcut in zoomcut_list.items():
                            catalog_zoom_diagnostic(data, modsky, zoomcut, stars)
                            pl.suptitle(f"Crowdsource nsky={nsky} weighted Catalog Diagnostics {filtername} {module}{desat}{bgsub}{fpsf} zoom {name}")
                            pl.savefig(f'{basepath}/{filtername}/pipeline/jw02221-o001_t001_nircam_{pupil}-{filtername.lower()}-{module}{desat}{bgsub}{fpsf}_nsky{nsky}_weighted_catalog_diagnostics_zoom{name.replace(" ","_")}.png',
                                    bbox_inches='tight')



            if options.daophot:
                t0 = time.time()
                print("Starting basic PSF photometry", flush=True)

                phot = PSFPhotometry(finder=daofind_tuned,#finder_maker(),
                                     #grouper=grouper,
                                     localbkg_estimator=None, # must be none or it un-saturates pixels
                                     psf_model=dao_psf_model,
                                     fitter=LevMarLSQFitter(),
                                     fit_shape=(5, 5),
                                     aperture_radius=2*fwhm_pix,
                                     progress_bar=True,
                                    )

                print("About to do BASIC photometry....")
                result = phot(np.nan_to_num(data))
                print(f"Done with BASIC photometry.  len(result)={len(result)} dt={time.time() - t0}")
                coords = ww.pixel_to_world(result['x_fit'], result['y_fit'])
                print(f'len(result) = {len(result)}, len(coords) = {len(coords)}, type(result)={type(result)}', flush=True)
                result['skycoord_centroid'] = coords
                detector = "" # no detector #'s for long
                result.write(f"{basepath}/{filtername}/{filtername.lower()}_{module}{detector}{desat}{bgsub}_daophot_basic.fits", overwrite=True)

                stars = result
                stars['x'] = stars['x_fit']
                stars['y'] = stars['y_fit']
                modsky = phot.make_residual_image()
                fits.PrimaryHDU(data=modsky, header=im1[1].header).writeto(
                    filename.replace(".fits", "_daophot_residual.fits"),
                    overwrite=True)
                try:
                    catalog_zoom_diagnostic(data, modsky, nullslice, stars)
                    pl.suptitle(f"daophot basic Catalog Diagnostics zoomed {filtername} {module}{desat}{bgsub}")
                    pl.savefig(f'{basepath}/{filtername}/pipeline/jw02221-o001_t001_nircam_{pupil}-{filtername.lower()}-{module}{desat}{bgsub}_catalog_diagnostics_daophot_basic.png',
                            bbox_inches='tight')

                    catalog_zoom_diagnostic(data, modsky, zoomcut, stars)
                    pl.suptitle(f"daophot basic Catalog Diagnostics {filtername} {module}{desat}{bgsub}")
                    pl.savefig(f'{basepath}/{filtername}/pipeline/jw02221-o001_t001_nircam_{pupil}-{filtername.lower()}-{module}{desat}{bgsub}_catalog_diagnostics_zoom_daophot_basic.png',
                            bbox_inches='tight')

                    for name, zoomcut in zoomcut_list.items():
                        catalog_zoom_diagnostic(data, modsky, zoomcut, stars)
                        pl.suptitle(f"daophot basic Catalog Diagnostics {filtername} {module}{desat}{bgsub} zoom {name}")
                        pl.savefig(f'{basepath}/{filtername}/pipeline/jw02221-o001_t001_nircam_{pupil}-{filtername.lower()}-{module}{desat}{bgsub}__catalog_diagnostics_zoom_daophot_basic{name.replace(" ","_")}.png',
                                bbox_inches='tight')
                except Exception as ex:
                    print(f'FAILURE: {ex}')
                print(f"Done with diagnostics for BASIC photometry.  dt={time.time() - t0}")

            if options.daophot:
                t0 = time.time()
                # iterative takes for-ev-er
                phot_ = IterativePSFPhotometry(finder=daofind_tuned,
                                               #grouper=grouper,
                                               localbkg_estimator=mmm_bkg,
                                               psf_model=dao_psf_model,
                                               fitter=LevMarLSQFitter(),
                                               maxiters=2,
                                               fit_shape=(5, 5),
                                               aperture_radius=2*fwhm_pix,
                                               progress_bar=True
                                              )

                print("About to do ITERATIVE photometry....")
                result2 = phot_(data)
                print(f"Done with ITERATIVE photometry. len(result2)={len(result2)}  dt={time.time() - t0}")
                coords2 = ww.pixel_to_world(result2['x_fit'], result2['y_fit'])
                result2['skycoord_centroid'] = coords2
                print(f'len(result2) = {len(result2)}, len(coords) = {len(coords)}', flush=True)
                result2.write(f"{basepath}/{filtername}/{filtername.lower()}"
                              f"_{module}{detector}{desat}{bgsub}"
                              f"_daophot_iterative.fits", overwrite=True)
                stars = result2
                stars['x'] = stars['x_fit']
                stars['y'] = stars['y_fit']

                modsky = phot_.make_residual_image()
                try:
                    catalog_zoom_diagnostic(data, modsky, nullslice, stars)
                    pl.suptitle(f"daophot iterative Catalog Diagnostics zoomed {filtername} {module}{desat}{bgsub}")
                    pl.savefig(f'{basepath}/{filtername}/pipeline/jw02221-o001_t001_nircam_{pupil}-{filtername.lower()}-{module}{desat}{bgsub}_catalog_diagnostics_daophot_iterative.png',
                            bbox_inches='tight')

                    catalog_zoom_diagnostic(data, modsky, zoomcut, stars)
                    pl.suptitle(f"daophot iterative Catalog Diagnostics {filtername} {module}{desat}{bgsub}")
                    pl.savefig(f'{basepath}/{filtername}/pipeline/jw02221-o001_t001_nircam_{pupil}-{filtername.lower()}-{module}{desat}{bgsub}_catalog_diagnostics_zoom_daophot_iterative.png',
                            bbox_inches='tight')

                    for name, zoomcut in zoomcut_list.items():
                        catalog_zoom_diagnostic(data, modsky, zoomcut, stars)
                        pl.suptitle(f"daophot iterative Catalog Diagnostics {filtername} {module}{desat}{bgsub} zoom {name}")
                        pl.savefig(f'{basepath}/{filtername}/pipeline/jw02221-o001_t001_nircam_{pupil}-{filtername.lower()}-{module}{desat}{bgsub}__catalog_diagnostics_zoom_daophot_iterative{name.replace(" ","_")}.png',
                                bbox_inches='tight')
                except Exception as ex:
                    print(f'FAILURE: {ex}')

if __name__ == "__main__":
    main()
