import crowdsource
import regions
import numpy as np
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.visualization import simple_norm
from astropy import wcs
from astropy import table
from astropy import units as u
from astropy.io import fits
import urllib3.exceptions
import requests

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


basepath = '/orange/adamginsburg/jwst/brick/'

for filtername in ('F212N', 'F182M', 'F187N')[::-1]:
    for module in ('nrca', 'nrcb'):
        for detector in ("", ):  # or range(1,5)
            # detector="" is for the merged version, which should be OK
            pupil = 'clear'
            fh = fits.open(f'{basepath}/{filtername}/pipeline/jw02221-o001_t001_nircam_{pupil}-{filtername.lower()}-{module}{detector}_i2d.fits')

            im1 = fh
            data = im1[1].data
            wht = im1['WHT'].data
            err = im1['ERR'].data
            instrument = im1[0].header['INSTRUME']
            telescope = im1[0].header['TELESCOP']
            filt = im1[0].header['FILTER']

            wavelength_table = SvoFps.get_transmission_data(f'{telescope}/{instrument}.{filt}')
            obsdate = im1[0].header['DATE-OBS']

            success = False
            while not success:
                try:
                    nrc = webbpsf.NIRCam()
                    nrc.load_wss_opd_by_date(f'{obsdate}T00:00:00')
                    nrc.filter = filt
                    # TODO: figure out whether a blank detector works here (i.e., if it's possible to specify only the module)
                    if detector:
                        nrc.detector = f'{module.upper()}{detector}'
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
                except (urllib3.exceptions.ReadTimeoutError, requests.exceptions.ReadTimeout) as ex:
                    print(f"Failed to build PSF: {ex}")

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
            weight[err < 1e-5] = 0
            weight[(err == 0) | (wht == 0)] = np.nanmedian(weight)
            weight[np.isnan(weight)] = 0

            weight[weight > maxweight] = maxweight
            weight[weight < minweight] = minweight


            results_unweighted  = fit_im(data, psf_model, weight=np.ones_like(data)*np.nanmedian(weight),
                                            #psfderiv=np.gradient(-psf_initial[0].data),
                                            nskyx=1, nskyy=1, refit_psf=False, verbose=True)
            stars, modsky, skymsky, psf = results_unweighted

            fits.BinTableHDU(data=stars).writeto(f"{basepath}/{filtername}/{filtername.lower()}_{module}{detector}_crowdsource_unweighted.fits", overwrite=True)
            fits.PrimaryHDU(data=skymsky, header=im1[1].header).writeto(f"{basepath}/{filtername}/{filtername.lower()}_{module}{detector}_crowdsource_skymodel_unweighted.fits", overwrite=True)


            pl.figure(figsize=(12,12))
            pl.subplot(2,2,1).imshow(data, norm=simple_norm(data, stretch='log', max_percent=99.95), cmap='gray')
            pl.xticks([]); pl.yticks([]); pl.title("Data")
            pl.subplot(2,2,2).imshow(modsky, norm=simple_norm(modsky, stretch='log', max_percent=99.95), cmap='gray')
            pl.xticks([]); pl.yticks([]); pl.title("fit_im model+sky")
            pl.subplot(2,2,3).imshow(skymsky, norm=simple_norm(skymsky, stretch='asinh'), cmap='gray')
            pl.xticks([]); pl.yticks([]); pl.title("fit_im sky+skym")
            pl.subplot(2,2,4).imshow(data, norm=simple_norm(data, stretch='log', max_percent=99.95), cmap='gray')
            pl.subplot(2,2,4).scatter(stars['y'], stars['x'], marker='x', color='r', s=5, linewidth=0.5)
            pl.xticks([]); pl.yticks([]); pl.title("Data with stars");
            pl.savefig(f'{basepath}/{filtername}/pipeline/jw02221-o001_t001_nircam_clear-{filtername.lower()}-{module}{detector}_catalog_diagnostics_unweighted.png',
                    bbox_inches='tight')

            pl.figure(figsize=(12,12))
            pl.subplot(2,2,1).imshow(data[:128,:128], norm=simple_norm(data[:128,:128], stretch='log', max_percent=99.95), cmap='gray')
            pl.xticks([]); pl.yticks([]); pl.title("Data")
            pl.subplot(2,2,2).imshow(modsky[:128,:128], norm=simple_norm(modsky[:128,:128], stretch='log', max_percent=99.95), cmap='gray')
            pl.xticks([]); pl.yticks([]); pl.title("fit_im model+sky")
            pl.subplot(2,2,3).imshow(skymsky[:128,:128], norm=simple_norm(skymsky[:128,:128], stretch='asinh'), cmap='gray')
            pl.xticks([]); pl.yticks([]); pl.title("fit_im sky+skym")
            pl.subplot(2,2,4).imshow(data[:128,:128], norm=simple_norm(data[:128,:128], stretch='log', max_percent=99.95), cmap='gray')
            pl.subplot(2,2,4).scatter(stars['y'], stars['x'], marker='x', color='r', s=8, linewidth=0.5)
            pl.axis([0,128,0,128])
            pl.xticks([]); pl.yticks([]); pl.title("Data with stars");
            pl.savefig(f'{basepath}/{filtername}/pipeline/jw02221-o001_t001_nircam_clear-{filtername.lower()}-{module}{detector}_catalog_diagnostics_zoom_unweighted.png',
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
            fig.savefig(f'{basepath}/{filtername}/pipeline/jw02221-o001_t001_nircam_{pupil}-{filtername.lower()}-{module}{detector}_weights.png',
                    bbox_inches='tight')


            results_blur  = fit_im(data, psf_model_blur, weight=weight,
                                nskyx=1, nskyy=1, refit_psf=False, verbose=True)
            stars, modsky, skymsky, psf = results_blur
            fits.BinTableHDU(data=stars).writeto(f"{basepath}/{filtername}/{filtername.lower()}_{module}{detector}_crowdsource.fits", overwrite=True)
            fits.PrimaryHDU(data=skymsky, header=im1[1].header).writeto(f"{basepath}/{filtername}/{filtername.lower()}_{module}{detector}_crowdsource_skymodel.fits", overwrite=True)



            stars, modsky, skymsky, psf = results_blur
            pl.figure(figsize=(12,12))
            pl.subplot(2,2,1).imshow(data[:128,:128], norm=simple_norm(data[:256,:256], stretch='log', max_percent=99.95), cmap='gray')
            pl.xticks([]); pl.yticks([]); pl.title("Data")
            pl.subplot(2,2,2).imshow(modsky[:128,:128], norm=simple_norm(modsky[:256,:256], stretch='log', max_percent=99.95), cmap='gray')
            pl.xticks([]); pl.yticks([]); pl.title("fit_im model+sky")
            pl.subplot(2,2,3).imshow((data-modsky)[:128,:128], norm=simple_norm((data-modsky)[:256,:256], stretch='asinh', max_percent=99.5, min_percent=0.5), cmap='gray')
            pl.xticks([]); pl.yticks([]); pl.title("data-modsky")
            pl.subplot(2,2,4).imshow(data[:128,:128], norm=simple_norm(data[:256,:256], stretch='log', max_percent=99.95), cmap='gray')
            pl.subplot(2,2,4).scatter(stars['y'], stars['x'], marker='x', color='r', s=8, linewidth=0.5)
            pl.axis([0,128,0,128])
            pl.xticks([]); pl.yticks([]); pl.title("Data with stars");
            pl.suptitle("Using WebbPSF model blurred a little")
            pl.tight_layout()
            pl.savefig(f'{basepath}/{filtername}/pipeline/jw02221-o001_t001_nircam_{pupil}-{filtername.lower()}-{module}{detector}_catalog_diagnostics_zoom.png',
                    bbox_inches='tight')

            pl.figure(figsize=(12,12))
            pl.subplot(2,2,1).imshow(data, norm=simple_norm(data, stretch='log', max_percent=99.95), cmap='gray')
            pl.xticks([]); pl.yticks([]); pl.title("Data")
            pl.subplot(2,2,2).imshow(modsky, norm=simple_norm(modsky, stretch='log', max_percent=99.95), cmap='gray')
            pl.xticks([]); pl.yticks([]); pl.title("fit_im model+sky")
            pl.subplot(2,2,3).imshow(skymsky, norm=simple_norm(skymsky, stretch='asinh'), cmap='gray')
            pl.xticks([]); pl.yticks([]); pl.title("fit_im sky+skym")
            pl.subplot(2,2,4).imshow(data, norm=simple_norm(data, stretch='log', max_percent=99.95), cmap='gray')
            pl.subplot(2,2,4).scatter(stars['y'], stars['x'], marker='x', color='r', s=5, linewidth=0.5)
            pl.xticks([]); pl.yticks([]); pl.title("Data with stars");
            pl.savefig(f'{basepath}/{filtername}/pipeline/jw02221-o001_t001_nircam_{pupil}-{filtername.lower()}-{module}{detector}_catalog_diagnostics.png',
                    bbox_inches='tight')