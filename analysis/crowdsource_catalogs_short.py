raise NotImplementedError("Try the long version, it has more options built in")
"""
Trying to declare this obsolete & use the LONG version for all wavelengths
"""
import crowdsource
import time
import regions
import numpy as np
import datetime
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
import urllib3.exceptions
import requests

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
os.environ['WEBBPSF_PATH'] = '/orange/adamginsburg/jwst/webbpsf-data/'
with open(os.path.expanduser('~/.mast_api_token'), 'r') as fh:
    os.environ['MAST_API_TOKEN'] = fh.read().strip()
import webbpsf

print("Completed imports", flush=True)

basepath = '/blue/adamginsburg/adamginsburg/jwst/brick/'

from optparse import OptionParser
parser = OptionParser()
parser.add_option("-f", "--filternames", dest="filternames",
                  default='F212N,F182M,F187N',
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
(options, args) = parser.parse_args()

filternames = options.filternames.split(",")
modules = options.modules.split(",")
use_desaturated = options.desaturated
print(f'options={options}', flush=True)
print(f'args={args}', flush=True)

desat = '_unsatstar' if use_desaturated else ''

for filtername in filternames:
    print(f"Starting filter {filtername}", flush=True)
    fwhm_tbl = Table.read(f'{basepath}/reduction/fwhm_table.ecsv')
    row = fwhm_tbl[fwhm_tbl['Filter'] == filtername]
    fwhm = fwhm_arcsec = float(row['PSF FWHM (arcsec)'][0])
    fwhm_pix = float(row['PSF FWHM (pixel)'][0])

    for module in modules:
        for detector in ("", ):  # or range(1,5)
            # detector="" is for the merged version, which should be OK
            pupil = 'clear'
            filename = f'{basepath}/{filtername}/pipeline/jw02221-o001_t001_nircam_{pupil}-{filtername.lower()}-{module}{detector}_i2d{desat}.fits'
            fh = fits.open(filename)

            print(f"Starting {filename}", flush=True)

            im1 = fh
            data = im1[1].data
            wht = im1['WHT'].data
            err = im1['ERR'].data
            instrument = im1[0].header['INSTRUME']
            telescope = im1[0].header['TELESCOP']
            filt = im1[0].header['FILTER']

            wavelength_table = SvoFps.get_transmission_data(f'{telescope}/{instrument}.{filt}')
            obsdate = im1[0].header['DATE-OBS']

            attempts = 0
            success = False
            while not success:
                now = datetime.datetime.now()
                attempts += 1
                try:
                    nrc = webbpsf.NIRCam()
                    nrc.load_wss_opd_by_date(f'{obsdate}T00:00:00')
                    nrc.filter = filt

                    # TODO: figure out whether a blank detector works here (i.e., if it's possible to specify only the module)
                    if detector:
                        nrc.detector = f'{module.upper()}{detector}'
                        print(f"Retrieving detector {nrc.detector} with filter {nrc.filter}", flush=True)
                        all_detectors = False
                    else:
                        # should be "A" or "B"
                        # https://github.com/spacetelescope/webbpsf/blob/ad994bf6619a483061c5e7f8f8c2e327eb4d6145/webbpsf/webbpsf_core.py#L1969
                        # nrc.module = module.upper()[-1]
                        # "NIRCam module is not directly settable; set detector instead."
                        # I tried using 'all detectors' but that changes the returned data structure
                        all_detectors = False
                        pass
                    grid = nrc.psf_grid(num_psfs=16, all_detectors=all_detectors)
                    success = True
                except (urllib3.exceptions.ReadTimeoutError, requests.exceptions.ReadTimeout, requests.HTTPError) as ex:
                    print(f"Failed to build PSF: {ex}\n{now}", flush=True)
                except Exception as ex:
                    print(f"EXCEPTION: {ex}\n{now}", flush=True)
                    continue

                if attempts > 10:
                    raise ValueError("Failed to download WebbPSF PSFs after 10 attempts")

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

            t0 = time.time()
            print(f"Calculating inverse error weights. t={t0}", flush=True)
            # crowdsource uses inverse-sigma, not inverse-variance
            weight = err**-1
            print(f'weight shape: {weight.shape} weight dtype: {weight.dtype}')
            print(f"Calculating finite weights.  t={time.time() - t0}", flush=True)
            finiteweights = weight[np.isfinite(weight)]
            print(f"Calculating percentiles.  t={time.time() - t0}", flush=True)
            # allow overwriting input b/c it's a copy
            badweight, minweight, midweight, maxweight = np.percentile(finiteweights, [1, 5, 50, 95], overwrite_input=True)
            print(f"Flagging low weights.  t={time.time() - t0}", flush=True)
            del finiteweights
            weight[err < 1e-5] = 0
            weight[(err == 0) | (wht == 0)] = midweight
            weight[np.isnan(weight)] = 0
            bad = np.isnan(weight) | (data == 0) | np.isnan(data) | (weight == 0) | (data < 1e-5)

            weight[weight > maxweight] = maxweight
            weight[weight < minweight] = minweight
            weight[bad] = badweight

            unweight = np.ones_like(data)*midweight
            assert np.all(np.isfinite(unweight))
            print("Done calculating weights", flush=True)

            """
            This error has repeatedly occurred.  My only guess is that it's
            coming from NaNs in the data, so I'm trying `nan_to_num`, even
            though I don't know if that's correct.

            Traceback (most recent call last):
              File "/orange/adamginsburg/jwst/brick/analysis/crowdsource_catalogs_short.py", line 109, in <module>
                results_unweighted  = fit_im(data, psf_model, weight=np.ones_like(data)*np.nanmedian(weight),
              File "/blue/adamginsburg/adamginsburg/repos/crowdsource/crowdsource/crowdsource_base.py", line 836, in fit_im
                raise ValueError("Model is all NaNs")
            ValueError: Model is all NaNs

            On further inspection, it seemed to come up when there were any zero weights
            """
            t0 = time.time()
            print("Running crowdsource unweighted", flush=True)
            results_unweighted = fit_im(np.nan_to_num(data), psf_model, weight=unweight,
                                        #psfderiv=np.gradient(-psf_initial[0].data),
                                        nskyx=1, nskyy=1, refit_psf=False, verbose=True)
            print(f"Done with unweighted crowdsource. dt={time.time() - t0}", flush=True)
            stars, modsky, skymsky, psf = results_unweighted
            # crowdsource explicitly inverts x & y from the numpy convention:
            # https://github.com/schlafly/crowdsource/issues/11
            stars = Table(stars)
            coords = ww.pixel_to_world(stars['y'], stars['x'])
            stars['skycoord'] = coords
            stars['x'], stars['y'] = stars['y'], stars['x']

            stars.meta['filename'] = filename
            stars.meta['filter'] = filtername
            stars.meta['module'] = module
            stars.meta['detector'] = detector

            tblfilename = f"{basepath}/{filtername}/{filtername.lower()}_{module}{detector}{desat}_crowdsource_unweighted.fits"
            stars.write(tblfilename, overwrite=True)
            # add WCS-containing header
            with fits.open(tblfilename, mode='update', output_verify='fix') as fh:
                fh[0].header.update(im1[1].header)

            fits.PrimaryHDU(data=skymsky, header=im1[1].header).writeto(f"{basepath}/{filtername}/{filtername.lower()}_{module}{detector}{desat}_crowdsource_skymodel_unweighted.fits", overwrite=True)


            pl.figure(figsize=(12,12))
            pl.subplot(2,2,1).imshow(data, norm=simple_norm(data, stretch='log', min_cut=0, max_percent=99.95), cmap='gray')
            pl.xticks([]); pl.yticks([]); pl.title("Data")
            pl.subplot(2,2,2).imshow(modsky, norm=simple_norm(modsky, stretch='log', min_cut=0, max_percent=99.95), cmap='gray')
            pl.xticks([]); pl.yticks([]); pl.title("fit_im model+sky")
            pl.subplot(2,2,3).imshow(skymsky, norm=simple_norm(skymsky, stretch='asinh'), cmap='gray')
            pl.xticks([]); pl.yticks([]); pl.title("fit_im sky+skym")
            pl.subplot(2,2,4).imshow(data, norm=simple_norm(data, stretch='log', min_cut=0, max_percent=99.95), cmap='gray')
            pl.subplot(2,2,4).scatter(stars['x'], stars['y'], marker='x', color='r', s=5, linewidth=0.5)
            pl.xticks([]); pl.yticks([]); pl.title("Data with stars");
            pl.savefig(f'{basepath}/{filtername}/pipeline/jw02221-o001_t001_nircam_clear-{filtername.lower()}-{module}{detector}{desat}_catalog_diagnostics_unweighted.png',
                    bbox_inches='tight')

            zoomcut = slice(512, 512+128), slice(512, 512+128)
            pl.figure(figsize=(12,12))
            pl.subplot(2,2,1).imshow(data[zoomcut], norm=simple_norm(data[zoomcut], stretch='log', min_cut=0, max_percent=99.95), cmap='gray')
            pl.xticks([]); pl.yticks([]); pl.title("Data")
            pl.subplot(2,2,2).imshow(modsky[zoomcut], norm=simple_norm(modsky[zoomcut], stretch='log', min_cut=0, max_percent=99.95), cmap='gray')
            pl.xticks([]); pl.yticks([]); pl.title("fit_im model+sky")
            pl.subplot(2,2,3).imshow(skymsky[zoomcut], norm=simple_norm(skymsky[zoomcut], stretch='asinh'), cmap='gray')
            pl.xticks([]); pl.yticks([]); pl.title("fit_im sky+skym")
            pl.subplot(2,2,4).imshow(data[zoomcut], norm=simple_norm(data[zoomcut], stretch='log', min_cut=0, max_percent=99.95), cmap='gray')
            pl.subplot(2,2,4).scatter(stars['x']+zoomcut[1].start, stars['y']+zoomcut[0].start, marker='x', color='r', s=8, linewidth=0.5)
            pl.axis([0,128,0,128])
            pl.xticks([]); pl.yticks([]); pl.title("Data with stars");
            pl.savefig(f'{basepath}/{filtername}/pipeline/jw02221-o001_t001_nircam_clear-{filtername.lower()}-{module}{detector}{desat}_catalog_diagnostics_zoom_unweighted.png',
                    bbox_inches='tight')

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

            # First try: use 0.05 pixels (minimal blur)
            smoothing_scale = 0.05 # pixels
            gpsf3 = convolve(gpsf2, Gaussian2DKernel(smoothing_scale))
            psf_model_blur = crowdsource.psf.SimplePSF(stamp=gpsf3)

            fig = pl.figure(0, figsize=(10,10))
            fig.clf()
            ax = fig.gca()
            im = ax.imshow(weight, norm=simple_norm(weight, stretch='log')); pl.colorbar(mappable=im);
            fig.savefig(f'{basepath}/{filtername}/pipeline/jw02221-o001_t001_nircam_{pupil}-{filtername.lower()}-{module}{detector}{desat}_weights.png',
                    bbox_inches='tight')


            t0 = time.time()
            print("Starting weighted fit_im crowdsource", flush=True)
            # see note above about NaN models
            results_blur  = fit_im(np.nan_to_num(data), psf_model_blur, weight=weight,
                                   nskyx=1, nskyy=1, refit_psf=False, verbose=True)
            print(f"Done with weighted fit_im crowdsource dt={time.time()-t0}", flush=True)
            stars, modsky, skymsky, psf = results_blur
            stars = Table(stars)

            stars.meta['filename'] = filename
            stars.meta['filter'] = filtername
            stars.meta['module'] = module
            stars.meta['detector'] = detector

            # crowdsource explicitly inverts x & y from the numpy convention:
            # https://github.com/schlafly/crowdsource/issues/11
            coords = ww.pixel_to_world(stars['y'], stars['x'])
            stars['skycoord'] = coords
            stars['x'], stars['y'] = stars['y'], stars['x']

            tblfilename = f"{basepath}/{filtername}/{filtername.lower()}_{module}{detector}{desat}_crowdsource.fits"
            stars.write(tblfilename, overwrite=True)
            # add WCS-containing header
            with fits.open(tblfilename, mode='update', output_verify='fix') as fh:
                fh[0].header.update(im1[1].header)
            fits.PrimaryHDU(data=skymsky, header=im1[1].header).writeto(f"{basepath}/{filtername}/{filtername.lower()}_{module}{detector}{desat}_crowdsource_skymodel.fits", overwrite=True)



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
            pl.savefig(f'{basepath}/{filtername}/pipeline/jw02221-o001_t001_nircam_{pupil}-{filtername.lower()}-{module}{detector}{desat}_catalog_diagnostics_zoom.png',
                    bbox_inches='tight')

            pl.figure(figsize=(12,12))
            pl.subplot(2,2,1).imshow(data, norm=simple_norm(data, stretch='log', max_percent=99.95), cmap='gray')
            pl.xticks([]); pl.yticks([]); pl.title("Data")
            pl.subplot(2,2,2).imshow(modsky, norm=simple_norm(modsky, stretch='log', max_percent=99.95), cmap='gray')
            pl.xticks([]); pl.yticks([]); pl.title("fit_im model+sky")
            pl.subplot(2,2,3).imshow(skymsky, norm=simple_norm(skymsky, stretch='asinh'), cmap='gray')
            pl.xticks([]); pl.yticks([]); pl.title("fit_im sky+skym")
            pl.subplot(2,2,4).imshow(data, norm=simple_norm(data, stretch='log', max_percent=99.95), cmap='gray')
            pl.subplot(2,2,4).scatter(stars['x'], stars['y'], marker='x', color='r', s=5, linewidth=0.5)
            pl.xticks([]); pl.yticks([]); pl.title("Data with stars");
            pl.savefig(f'{basepath}/{filtername}/pipeline/jw02221-o001_t001_nircam_{pupil}-{filtername.lower()}-{module}{detector}{desat}_catalog_diagnostics.png',
                    bbox_inches='tight')



            # nsky=15 led to OOM errors
            for nsky in (0, 1, 5):
                t0 = time.time()
                print(f"Starting weighted fit_im crowdsource w/nsky={nsky}", flush=True)
                # see note above about NaN models
                results_blur  = fit_im(np.nan_to_num(data), psf_model_blur, weight=weight,
                                    nskyx=nsky, nskyy=nsky, refit_psf=False, verbose=True)
                print(f"Done with weighted fit_im w/nsky={nsky} crowdsource dt={time.time()-t0}", flush=True)
                stars, modsky, skymsky, psf = results_blur
                stars = Table(stars)

                stars.meta['filename'] = filename
                stars.meta['filter'] = filtername
                stars.meta['module'] = module
                stars.meta['detector'] = detector

                # crowdsource explicitly inverts x & y from the numpy convention:
                # https://github.com/schlafly/crowdsource/issues/11
                coords = ww.pixel_to_world(stars['y'], stars['x'])
                stars['skycoord'] = coords
                stars['x'], stars['y'] = stars['y'], stars['x']

                tblfilename = f"{basepath}/{filtername}/{filtername.lower()}_{module}{detector}{desat}_crowdsource_nsky{nsky}.fits"
                stars.write(tblfilename, overwrite=True)
                # add WCS-containing header
                with fits.open(tblfilename, mode='update', output_verify='fix') as fh:
                    fh[0].header.update(im1[1].header)
                fits.PrimaryHDU(data=skymsky, header=im1[1].header).writeto(f"{basepath}/{filtername}/{filtername.lower()}_{module}{detector}{desat}_crowdsource_nsky{nsky}_skymodel.fits", overwrite=True)



                pl.figure(figsize=(12,12))
                pl.subplot(2,2,1).imshow(data[zoomcut], norm=simple_norm(data[zoomcut], stretch='log', max_percent=99.95, min_cut=0), cmap='gray')
                pl.xticks([]); pl.yticks([]); pl.title("Data")
                pl.subplot(2,2,2).imshow(modsky[zoomcut], norm=simple_norm(modsky[zoomcut], stretch='log', max_percent=99.95, min_cut=0), cmap='gray')
                pl.xticks([]); pl.yticks([]); pl.title("fit_im model+sky")
                pl.subplot(2,2,3).imshow((data-modsky)[zoomcut], norm=simple_norm((data-modsky)[zoomcut], stretch='asinh', max_percent=99.5, min_percent=0.5), cmap='gray')
                pl.xticks([]); pl.yticks([]); pl.title("data-modsky")
                pl.subplot(2,2,4).imshow(data[zoomcut], norm=simple_norm(data[zoomcut], stretch='log', max_percent=99.95, min_cut=0), cmap='gray')
                pl.subplot(2,2,4).scatter(stars['x']+zoomcut[1].start, stars['y']+zoomcut[0].start, marker='x', color='r', s=8, linewidth=0.5)
                pl.axis([0,128,0,128])
                pl.xticks([]); pl.yticks([]); pl.title("Data with stars");
                pl.suptitle("Using WebbPSF model blurred a little")
                pl.tight_layout()
                pl.savefig(f'{basepath}/{filtername}/pipeline/jw02221-o001_t001_nircam_{pupil}-{filtername.lower()}-{module}{detector}{desat}_catalog_nsky{nsky}_diagnostics_zoom.png',
                        bbox_inches='tight')

                pl.figure(figsize=(12,12))
                pl.subplot(2,2,1).imshow(data, norm=simple_norm(data, stretch='log', max_percent=99.95, min_cut=0), cmap='gray')
                pl.xticks([]); pl.yticks([]); pl.title("Data")
                pl.subplot(2,2,2).imshow(modsky, norm=simple_norm(modsky, stretch='log', max_percent=99.95, min_cut=0), cmap='gray')
                pl.xticks([]); pl.yticks([]); pl.title("fit_im model+sky")
                pl.subplot(2,2,3).imshow(skymsky, norm=simple_norm(skymsky, stretch='asinh'), cmap='gray')
                pl.xticks([]); pl.yticks([]); pl.title("fit_im sky+skym")
                pl.subplot(2,2,4).imshow(data, norm=simple_norm(data, stretch='log', max_percent=99.95, min_cut=0), cmap='gray')
                pl.subplot(2,2,4).scatter(stars['x'], stars['y'], marker='x', color='r', s=5, linewidth=0.5)
                pl.xticks([]); pl.yticks([]); pl.title("Data with stars");
                pl.savefig(f'{basepath}/{filtername}/pipeline/jw02221-o001_t001_nircam_{pupil}-{filtername.lower()}-{module}{detector}{desat}_catalog_nsky{nsky}_diagnostics.png',
                        bbox_inches='tight')






            filter_table = SvoFps.get_filter_list(facility=telescope, instrument=instrument)
            filter_table.add_index('filterID')
            instrument = 'NIRCam'
            eff_wavelength = filter_table.loc[f'{telescope}/{instrument}.{filt}']['WavelengthEff'] * u.AA

            #fwhm = (1.22 * eff_wavelength / (6.5*u.m)).to(u.arcsec, u.dimensionless_angles())

            pixscale = ww.proj_plane_pixel_area()**0.5
            #fwhm_pix = (fwhm / pixscale).decompose().value

            daogroup = DAOGroup(2 * fwhm_pix)
            mmm_bkg = MMMBackground()

            filtered_errest = stats.mad_std(data, ignore_nan=True)
            print(f'Error estimate for DAO: {filtered_errest}', flush=True)

            daofind_fin = DAOStarFinder(threshold=7 * filtered_errest, fwhm=fwhm_pix, roundhi=1.0, roundlo=-1.0,
                                        sharplo=0.30, sharphi=1.40)
            finstars = daofind_fin(data)

            grid.x_0 = 0
            grid.y_0 = 0
            # def evaluate(x, y, flux, x_0, y_0):
            #     """
            #     Evaluate the `GriddedPSFModel` for the input parameters.
            #     """
            #     # Get the local PSF at the (x_0,y_0)
            #     psfmodel = grid._compute_local_model(x_0+slcs[1].start, y_0+slcs[0].start)

            #     # now evaluate the PSF at the (x_0, y_0) subpixel position on
            #     # the input (x, y) values
            #     return psfmodel.evaluate(x, y, flux, x_0, y_0)
            # grid.evaluate = evaluate

            if options.daophot:
                phot = BasicPSFPhotometry(finder=daofind_fin,#finder_maker(),
                                        group_maker=daogroup,
                                        bkg_estimator=None, # must be none or it un-saturates pixels
                                        psf_model=grid,
                                        fitter=LevMarLSQFitter(),
                                        fitshape=(11, 11),
                                        aperture_radius=5*fwhm_pix)

                result = phot(data)
                result['skycoord_centroid'] = ww.pixel_to_world(result['x_fit'], result['y_fit'])
                result.write(f"{basepath}/{filtername}/{filtername.lower()}_{module}{detector}{desat}_daophot_basic.fits", overwrite=True)

            if options.daophot:
                # iterative takes too long
                phot_ = IterativelySubtractedPSFPhotometry(finder=daofind_fin, group_maker=daogroup,
                                                        bkg_estimator=mmm_bkg,
                                                        psf_model=grid,
                                                        fitter=LevMarLSQFitter(),
                                                        niters=2, fitshape=(11, 11),
                                                        aperture_radius=2*fwhm_pix)

                result2 = phot_(data)
                result2['skycoord_centroid'] = ww.pixel_to_world(result2['x_fit'], result2['y_fit'])
                result2.write(f"{basepath}/{filtername}/{filtername.lower()}_{module}{detector}{desat}_daophot_iterative.fits", overwrite=True)
