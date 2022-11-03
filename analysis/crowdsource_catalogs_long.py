print("Starting long-wavelength cataloging", flush=True)
import numpy as np
import crowdsource
import regions
import numpy as np
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
import urllib3
import urllib3.exceptions
from photutils.detection import DAOStarFinder, IRAFStarFinder
from photutils.psf import DAOGroup, IntegratedGaussianPRF, extract_stars, IterativelySubtractedPSFPhotometry, BasicPSFPhotometry
from photutils.background import MMMBackground, MADStdBackgroundRMS

from crowdsource import crowdsource_base
from crowdsource.crowdsource_base import fit_im, psfmod

from astroquery.svo_fps import SvoFps

import pylab as pl
pl.rcParams['figure.facecolor'] = 'w'
pl.rcParams['image.origin'] = 'lower'

import os
print("Importing webbpsf", flush=True)
os.environ['WEBBPSF_PATH'] = '/blue/adamginsburg/adamginsburg/jwst/webbpsf-data/'
with open(os.path.expanduser('~/.mast_api_token'), 'r') as fh:
    os.environ['MAST_API_TOKEN'] = fh.read().strip()
import webbpsf
from webbpsf.utils import to_griddedpsfmodel

print("Done with imports", flush=True)

basepath = '/blue/adamginsburg/adamginsburg/jwst/brick/'

from optparse import OptionParser
parser = OptionParser()
parser.add_option("-f", "--filternames", dest="filternames",
                  default='F466N,F405N,F410M',
                  help="filter name list", metavar="filternames")
(options, args) = parser.parse_args()

filternames = options.filternames.split(",")

for module in ('merged', 'nrca', 'nrcb', 'merged-reproject', ):
    detector = module # no sub-detectors for long-NIRCAM
    for filtername in filternames:
        print(f"Starting filter {filtername}", flush=True)
        fwhm_tbl = Table.read(f'{basepath}/reduction/fwhm_table.ecsv')
        row = fwhm_tbl[fwhm_tbl['Filter'] == filtername]
        fwhm = fwhm_arcsec = float(row['PSF FWHM (arcsec)'][0])
        fwhm_pix = float(row['PSF FWHM (pixel)'][0])

        try:
            pupil = 'clear'
            filename = f'{basepath}/{filtername}/pipeline/jw02221-o001_t001_nircam_{pupil}-{filtername.lower()}-{module}_i2d.fits'
            fh = fits.open(filename)
        except Exception:
            pupil = 'F444W'
            filename = f'{basepath}/{filtername}/pipeline/jw02221-o001_t001_nircam_{pupil}-{filtername.lower()}-{module}_i2d.fits'
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

        has_downloaded = False
        while not has_downloaded:
            try:
                print("Attempting to download WebbPSF data", flush=True)
                nrc = webbpsf.NIRCam()
                nrc.load_wss_opd_by_date(f'{obsdate}T00:00:00')
                nrc.filter = filt
                print(f"Running {module}")
                if module in ('nrca', 'nrcb'):
                    nrc.detector = f'{module.upper()}5' # I think NRCA5 must be the "long" detector?
                    grid = nrc.psf_grid(num_psfs=16, all_detectors=False, verbose=True, save=True)
                else:
                    grid = nrc.psf_grid(num_psfs=16, all_detectors=True, verbose=True, save=True)
                has_downloaded = True
            except (urllib3.exceptions.ReadTimeoutError, requests.exceptions.ReadTimeout, requests.HTTPError) as ex:
                print(f"Failed to build PSF: {ex}", flush=True)
            except Exception as ex:
                print(ex, flush=True)
                continue

        # there's no way to use a grid across all detectors.
        # the right way would be to use this as a grid of grids, but that apparently isn't supported.
        if isinstance(grid, list):
            grid = grid[0]

        print("Done with WebbPSF downloading; now building model", flush=True)
        yy, xx = np.indices([31,31], dtype=float)
        grid.x_0 = grid.y_0 = 15.5
        psf_model = crowdsource.psf.SimplePSF(stamp=grid(xx,yy))

        ww = wcs.WCS(im1[1].header)
        cen = ww.pixel_to_world(im1[1].shape[1]/2, im1[1].shape[0]/2) 
        reg = regions.RectangleSkyRegion(center=cen, width=1.5*u.arcmin, height=1.5*u.arcmin)
        preg = reg.to_pixel(ww)
        #mask = preg.to_mask()
        #cutout = mask.cutout(im1[1].data)
        #err = mask.cutout(im1[2].data)

        # crowdsource uses inverse-sigma, not inverse-variance
        weight = err**-1
        maxweight = np.percentile(weight[np.isfinite(weight)], 95)
        minweight = np.percentile(weight[np.isfinite(weight)], 5)
        badweight =  np.percentile(weight[np.isfinite(weight)], 1)
        weight[err < 1e-5] = 0
        weight[(err == 0) | (wht == 0)] = np.nanmedian(weight)
        weight[np.isnan(weight)] = 0
        bad = np.isnan(weight) | (data == 0) | np.isnan(data) | (weight == 0)

        weight[weight > maxweight] = maxweight
        weight[weight < minweight] = minweight
        # it seems that crowdsource doesn't like zero weights
        weight[bad] = badweight
        weight[bad] = minweight




        filter_table = SvoFps.get_filter_list(facility=telescope, instrument=instrument)
        filter_table.add_index('filterID')
        instrument = 'NIRCam'
        eff_wavelength = filter_table.loc[f'{telescope}/{instrument}.{filt}']['WavelengthEff'] * u.AA

        #fwhm = (1.22 * eff_wavelength / (6.5*u.m)).to(u.arcsec, u.dimensionless_angles())

        pixscale = ww.proj_plane_pixel_area()**0.5
        #fwhm_pix = (fwhm / pixscale).decompose().value

        daogroup = DAOGroup(2 * fwhm_pix)
        mmm_bkg = MMMBackground()

        filtered_errest = stats.sigma_clipped_stats(data, stdfunc='mad_std')
        print(f'Error estimate for DAO from stats.: {filtered_errest}', flush=True)
        filtered_errest = np.nanmedian(err)
        print(f'Error estimate for DAO from median(err): {filtered_errest}', flush=True)

        daofind_tuned = DAOStarFinder(threshold=5 * filtered_errest, fwhm=fwhm_pix, roundhi=1.0, roundlo=-1.0,
                                    sharplo=0.30, sharphi=1.40)
        print("Finding stars with daofind_tuned", flush=True)
        finstars = daofind_tuned(np.nan_to_num(data))

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



        print("starting crowdsource unweighted", flush=True)
        results_unweighted  = fit_im(np.nan_to_num(data), psf_model, weight=np.ones_like(data)*np.nanmedian(weight),
                                        #psfderiv=np.gradient(-psf_initial[0].data),
                                        nskyx=1, nskyy=1, refit_psf=False, verbose=True)
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

        tblfilename = f"{basepath}/{filtername}/{filtername.lower()}_{module}_crowdsource_unweighted.fits"
        stars.write(tblfilename, overwrite=True)
        # add WCS-containing header
        with fits.open(tblfilename, mode='update', output_verify='fix') as fh:
            fh[0].header.update(im1[1].header)
        fits.PrimaryHDU(data=skymsky, header=im1[1].header).writeto(f"{basepath}/{filtername}/{filtername.lower()}_{module}_crowdsource_skymodel_unweighted.fits", overwrite=True)

        zoomcut = slice(128, 256), slice(128, 256)

        try:
            pl.figure(figsize=(12,12))
            pl.subplot(2,2,1).imshow(data, norm=simple_norm(data, stretch='log', max_percent=99.95), cmap='gray')
            pl.xticks([]); pl.yticks([]); pl.title("Data")
            pl.subplot(2,2,2).imshow(modsky, norm=simple_norm(modsky, stretch='log', max_percent=99.95), cmap='gray')
            pl.xticks([]); pl.yticks([]); pl.title("fit_im model+sky")
            pl.subplot(2,2,3).imshow(skymsky, norm=simple_norm(skymsky, stretch='asinh'), cmap='gray')
            pl.xticks([]); pl.yticks([]); pl.title("fit_im sky+skym")
            pl.subplot(2,2,4).imshow(data, norm=simple_norm(data, stretch='log', max_percent=99.95), cmap='gray')
            pl.subplot(2,2,4).scatter(stars['x']+zoomcut[1].start, stars['y']+zoomcut[0].start, marker='x', color='r', s=5, linewidth=0.5)
            pl.xticks([]); pl.yticks([]); pl.title("Data with stars");
            pl.savefig(f'{basepath}/{filtername}/pipeline/jw02221-o001_t001_nircam_{pupil}-{filtername.lower()}-{module}_catalog_diagnostics_unweighted.png',
                    bbox_inches='tight')


            pl.figure(figsize=(12,12))
            pl.subplot(2,2,1).imshow(data[zoomcut], norm=simple_norm(data[zoomcut], stretch='log', max_percent=99.95), cmap='gray')
            pl.xticks([]); pl.yticks([]); pl.title("Data")
            pl.subplot(2,2,2).imshow(modsky[zoomcut], norm=simple_norm(modsky[zoomcut], stretch='log', max_percent=99.95), cmap='gray')
            pl.xticks([]); pl.yticks([]); pl.title("fit_im model+sky")
            pl.subplot(2,2,3).imshow(skymsky[zoomcut], norm=simple_norm(skymsky[zoomcut], stretch='asinh'), cmap='gray')
            pl.xticks([]); pl.yticks([]); pl.title("fit_im sky+skym")
            pl.subplot(2,2,4).imshow(data[zoomcut], norm=simple_norm(data[zoomcut], stretch='log', max_percent=99.95), cmap='gray')
            pl.subplot(2,2,4).scatter(stars['x']+zoomcut[1].start, stars['y']+zoomcut[0].start, marker='x', color='r', s=8, linewidth=0.5)
            pl.axis([0,128,0,128])
            pl.xticks([]); pl.yticks([]); pl.title("Data with stars");
            pl.savefig(f'{basepath}/{filtername}/pipeline/jw02221-o001_t001_nircam_{pupil}-{filtername.lower()}-{module}_catalog_diagnostics_zoom_unweighted.png',
                    bbox_inches='tight')
        except Exception as ex:
            print(f'FAILURE: {ex}')

        # pl.figure(figsize=(10,5))
        # pl.subplot(1,2,1).imshow(psf_model(30,30), norm=simple_norm(psf_model(30,30), stretch='log'), cmap='cividis')
        # pl.title("Input model")
        # pl.subplot(1,2,2).imshow(psf(30,30), norm=simple_norm(psf(30,30), stretch='log'), cmap='cividis')
        # pl.title("Fitted model")



        yy, xx = np.indices([61, 61], dtype=float)
        grid.x_0 = preg.center.x+30
        grid.y_0 = preg.center.y+30
        gpsf2 = grid(xx+preg.center.x, yy+preg.center.y)
        psf_model = crowdsource.psf.SimplePSF(stamp=gpsf2)

        smoothing_scale = 0.55 # pixels
        gpsf3 = convolve(gpsf2, Gaussian2DKernel(smoothing_scale))
        psf_model_blur = crowdsource.psf.SimplePSF(stamp=gpsf3)

        fig = pl.figure(0, figsize=(10,10))
        fig.clf()
        ax = fig.gca()
        im = ax.imshow(weight, norm=simple_norm(weight, stretch='log')); pl.colorbar(mappable=im);
        pl.savefig(f'{basepath}/{filtername}/pipeline/jw02221-o001_t001_nircam_{pupil}-{filtername.lower()}-{module}_weights.png',
                   bbox_inches='tight')

        print("Running crowdsource fit_im with weights")
        print(f"psf_model_blur={psf_model_blur}")

        results_blur  = fit_im(np.nan_to_num(data), psf_model_blur, weight=weight,
                               nskyx=1, nskyy=1, refit_psf=False, verbose=True)
        stars, modsky, skymsky, psf = results_blur
        stars = Table(stars)

        # crowdsource explicitly inverts x & y from the numpy convention:
        # https://github.com/schlafly/crowdsource/issues/11
        coords = ww.pixel_to_world(stars['y'], stars['x'])
        stars['skycoord'] = coords
        stars['x'], stars['y'] = stars['y'], stars['x']

        tblfilename = f"{basepath}/{filtername}/{filtername.lower()}_{module}_crowdsource.fits"
        stars.write(tblfilename, overwrite=True)
        # add WCS-containing header
        with fits.open(tblfilename, mode='update', output_verify='fix') as fh:
            fh[0].header.update(im1[1].header)

        fits.PrimaryHDU(data=skymsky, header=im1[1].header).writeto(f"{basepath}/{filtername}/{filtername.lower()}_{module}_crowdsource_skymodel.fits", overwrite=True)
        fits.PrimaryHDU(data=data-modsky,
                        header=im1[1].header).writeto(f"{basepath}/{filtername}/{filtername.lower()}_{module}_crowdsource_data-modsky.fits", overwrite=True)

        zoomcut = slice(128, 256), slice(128, 256)

        pl.figure(figsize=(12,12))
        pl.subplot(2,2,1).imshow(data[zoomcut], norm=simple_norm(data[zoomcut], stretch='log', max_percent=99.95), cmap='gray')
        pl.xticks([]); pl.yticks([]); pl.title("Data")
        pl.subplot(2,2,2).imshow(modsky[zoomcut], norm=simple_norm(modsky[zoomcut], stretch='log', max_percent=99.95), cmap='gray')
        pl.xticks([]); pl.yticks([]); pl.title("fit_im model+sky")
        pl.subplot(2,2,3).imshow((data-modsky)[zoomcut], norm=simple_norm((data-modsky)[zoomcut], stretch='asinh', max_percent=99.5, min_percent=0.5), cmap='gray')
        pl.xticks([]); pl.yticks([]); pl.title("data-modsky")
        pl.subplot(2,2,4).imshow(data[zoomcut], norm=simple_norm(data[zoomcut], stretch='log', max_percent=99.95), cmap='gray')
        pl.subplot(2,2,4).scatter(stars['x']+zoomcut[1].start, stars['y']+zoomcut[0].start, marker='x', color='r', s=8, linewidth=0.5)
        pl.axis([0,128,0,128])
        pl.xticks([]); pl.yticks([]); pl.title("Data with stars");
        pl.suptitle("Using WebbPSF model blurred a little")
        pl.tight_layout()
        pl.savefig(f'{basepath}/{filtername}/pipeline/jw02221-o001_t001_nircam_{pupil}-{filtername.lower()}-{module}_catalog_diagnostics_zoom.png',
                   bbox_inches='tight')

        pl.figure(figsize=(12,12))
        pl.subplot(2,2,1).imshow(data, norm=simple_norm(data, stretch='log', max_percent=99.95), cmap='gray')
        pl.xticks([]); pl.yticks([]); pl.title("Data")
        pl.subplot(2,2,2).imshow(modsky, norm=simple_norm(modsky, stretch='log', max_percent=99.95), cmap='gray')
        pl.xticks([]); pl.yticks([]); pl.title("fit_im model+sky")
        pl.subplot(2,2,3).imshow(skymsky, norm=simple_norm(skymsky, stretch='asinh'), cmap='gray')
        pl.xticks([]); pl.yticks([]); pl.title("fit_im sky+skym")
        pl.subplot(2,2,4).imshow(data, norm=simple_norm(data, stretch='log', max_percent=99.95), cmap='gray')
        pl.subplot(2,2,4).scatter(stars['x']+zoomcut[1].start, stars['y']+zoomcut[0].start, marker='x', color='r', s=5, linewidth=0.5)
        pl.xticks([]); pl.yticks([]); pl.title("Data with stars");
        pl.savefig(f'{basepath}/{filtername}/pipeline/jw02221-o001_t001_nircam_{pupil}-{filtername.lower()}-{module}_catalog_diagnostics.png',
                   bbox_inches='tight')


        print("Starting basic PSF photometry", flush=True)
        phot = BasicPSFPhotometry(finder=daofind_tuned,#finder_maker(),
                                  group_maker=daogroup,
                                  bkg_estimator=None, # must be none or it un-saturates pixels
                                  psf_model=grid,
                                  fitter=LevMarLSQFitter(),
                                  fitshape=(11, 11),
                                  aperture_radius=5*fwhm_pix)

        print("About to do BASIC photometry....")
        result = phot(np.nan_to_num(data))
        print("Done with BASIC photometry")
        coords = ww.pixel_to_world(result['x_fit'], result['y_fit'])
        print(f'len(result) = {len(result)}, len(coords) = {len(coords)}, type(result)={type(result)}', flush=True)
        result['skycoord_centroid'] = coords
        detector = "" # no detector #'s for long
        result.write(f"{basepath}/{filtername}/{filtername.lower()}_{module}{detector}_daophot_basic.fits", overwrite=True)

        if True:
            # iterative takes for-ev-er
            phot_ = IterativelySubtractedPSFPhotometry(finder=daofind_tuned, group_maker=daogroup,
                                                        bkg_estimator=mmm_bkg,
                                                        psf_model=grid,
                                                        fitter=LevMarLSQFitter(),
                                                        niters=2, fitshape=(11, 11),
                                                        aperture_radius=2*fwhm_pix)

            result2 = phot_(data)
            coords2 = ww.pixel_to_world(result2['x_fit'], result2['y_fit'])
            result2['skycoord_centroid'] = coords2
            print(f'len(result2) = {len(result2)}, len(coords) = {len(coords)}', flush=True)
            result2.write(f"{basepath}/{filtername}/{filtername.lower()}_{module}{detector}_daophot_iterative.fits", overwrite=True)
