import photutils
import regions
from photutils import CircularAperture, EPSFBuilder, find_peaks, CircularAnnulus
from photutils.detection import DAOStarFinder, IRAFStarFinder
from photutils.psf import extract_stars, BasicPSFPhotometry

try:
    # version >=1.7.0, doesn't work: the PSF is broken (https://github.com/astropy/photutils/issues/1580?)
    from photutils.psf import PSFPhotometry, SourceGrouper
except:
    # version 1.6.0, which works
    from photutils.psf import (BasicPSFPhotometry as PSFPhotometry,
                               DAOGroup as SourceGrouper)

import numpy as np
import time
from astropy.stats import mad_std
from astropy import stats
from astropy.io import fits
from astropy import wcs
from astropy import log
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel
from astropy import units as u
from astropy.table import Table
from astropy.nddata import NDData
from photutils.background import MMMBackground, MADStdBackgroundRMS
from astropy.modeling.fitting import LevMarLSQFitter
from astropy import table
from tqdm.autonotebook import tqdm
from astroquery.svo_fps import SvoFps
from scipy.ndimage import median_filter
from scipy import ndimage
from astropy.stats import sigma_clipped_stats, SigmaClip
from photutils.segmentation import detect_threshold, detect_sources
from photutils.utils import circular_footprint
import pylab as pl
from astropy.visualization import simple_norm

import os
os.environ['WEBBPSF_PATH'] = '/orange/adamginsburg/jwst/webbpsf-data/'
import webbpsf
from webbpsf.utils import to_griddedpsfmodel

try:
    from paths import basepath
except ImportError:
    basepath = '/blue/adamginsburg/adamginsburg/jwst/brick/'

def get_fwhm(header, instrument_replacement='NIRCam'):
    """
    Paramters
    ---------
    header : fits.Header
        The header of the file of interest
    instrument_replacement : str
        Case-sensitive version of instrument name

    Returns
    -------
    fwhm : u.Quantity
        The FWHM in arcseconds
    fwhm_pix : float
        The FWHM in pixels
    """

    instrument = header['INSTRUME']
    telescope = header['TELESCOP']
    #filtername = header['FILTER']
    filtername = get_filtername(header)

    fwhm_tbl = Table.read(f'{basepath}/reduction/fwhm_table.ecsv')
    row = fwhm_tbl[fwhm_tbl['Filter'] == filtername]
    fwhm = fwhm_arcsec = float(row['PSF FWHM (arcsec)'][0])
    fwhm_pix = float(row['PSF FWHM (pixel)'][0])

    return fwhm, fwhm_pix

    if False:
        # old way
        wavelength_table = SvoFps.get_transmission_data(f'{telescope}/{instrument}.{filtername}')
        filter_table = SvoFps.get_filter_list(facility=telescope, instrument=instrument)
        filter_table.add_index('filterID')
        eff_wavelength = filter_table.loc[f'{telescope}/{instrument_replacement}.{filtername}']['WavelengthEff'] * u.AA

        fwhm = (1.22 * eff_wavelength / (6.5*u.m)).to(u.arcsec, u.dimensionless_angles())

        ww = wcs.WCS(header)
        pixscale = ww.proj_plane_pixel_area()**0.5
        fwhm_pix = (fwhm / pixscale).decompose().value

        return fwhm, fwhm_pix


def get_filtername(header):

    filtername = header['FILTER']
    if 'PUPIL' in header:
        # only for NIRCAM
        filtername2 = header['PUPIL']
        if filtername == 'CLEAR':
            filtername = filtername2
        elif filtername2 == 'CLEAR':
            # do nothing here
            pass
        elif filtername2 != 'CLEAR':
            # filtername is real, but so is filtername2
            filtername = filtername2

    assert filtername != 'CLEAR'

    return filtername

def estimate_background(data, header, medfilt_size=[15,15], do_segment_mask=False, save_products=True,
                        path_prefix='./',
                        psf_size=31, nsigma_threshold=7):
    """
    holy side effects batman
    """

    fwhm, fwhm_pix = get_fwhm(header)
    filtername = get_filtername(header)
    instrument = header['INSTRUME']

    obsdate = header['DATE-OBS']


    ### First iteration: simple median filter to produce ePSF estimate

    t0 = time.time()

    medfilt = median_filter(data, size=medfilt_size)
    log.info(f"Median filter done in {time.time()-t0:0.1f}s")


    medfilt_sub = data - medfilt

    if do_segment_mask:
        # this was an exploration to see if segmentation-based-masking was a good idea.
        # It's not.
        sigma_clip = SigmaClip(sigma=3.0, maxiters=10)


        threshold = detect_threshold(medfilt_sub, nsigma=5.0, sigma_clip=sigma_clip)
        segment_img = detect_sources(medfilt_sub, threshold, npixels=10)
        footprint = circular_footprint(radius=2)
        segment_mask = segment_img.make_source_mask(footprint=footprint)
        mean, median, std = sigma_clipped_stats(medfilt_sub, sigma=3.0, mask=segment_mask)


    # make webbpsf
    nc = webbpsf.NIRCam()
    nc.load_wss_opd_by_date(f'{obsdate}T00:00:00')

    nc.filter = filtername

    # mask out the pixels that are saturated
    medfilt_sub[data==0] = np.nan

    # calculate a PSF and use it as a kernel
    psf_kernel = nc.calc_psf(fov_pixels=psf_size, oversample=1)[0].data
    log.info(f"Calculating PSF kernel done at {time.time()-t0:0.1f}s")

    # replace the saturated pixels with interpolated ones
    datafilt_conv_psf = (convolve(medfilt_sub, psf_kernel, nan_treatment='interpolate'))
    log.info(f"Convolution done at {time.time()-t0:0.1f}s")

    filled_in_pixels = (data==0) & ~np.isnan(datafilt_conv_psf)

    datafilt_conv_psf = np.nan_to_num(datafilt_conv_psf)


    # create a PSF-based mask to mask out diffraction spikes
    # (the psf_size+1 here is a bit of a hack that might break if you try to put in different PSF sizes...)
    oversample = 1
    psf_fn = f'{path_prefix}/{instrument.lower()}_{filtername}_samp{oversample}_simple_psf.fits'
    if os.path.exists(psf_fn):
        # As a file
        pp = fits.getdata(psf_fn)  # file created 2 cells above
    else:
        psf_hdu = nc.calc_psf(oversample=1, fov_pixels=psf_size+1)
        psf_hdu.writeto(psf_fn)
        pp = psf_hdu[0].data
    log.info(f"Calculating PSF for mask done at {time.time()-t0:0.1f}s")

    # total guess heuristics...
    psfmask1000 = pp > 0.001*pp.max()
    psfmask100 = ndimage.binary_erosion(ndimage.binary_dilation(pp > 0.01*pp.max()))
    psfmask7pct = ndimage.binary_erosion(ndimage.binary_dilation(pp > 0.07*pp.max()))


    # find stars to mask out
    err_est_conv = stats.mad_std(datafilt_conv_psf)
    daofind_deep = DAOStarFinder(threshold=10 * err_est_conv,
                                 fwhm=fwhm_pix*2**0.5, roundhi=0.25,
                                 roundlo=-1.0, sharplo=0.30, sharphi=1.40)

    stars_deep_conv = daofind_deep(datafilt_conv_psf)

    # allow any shape at all, but strong cut on S/N
    daofind_shallow_conv = DAOStarFinder(threshold=250 * err_est_conv,
                                         fwhm=fwhm_pix, roundhi=4,
                                         roundlo=-1.0, sharplo=0.001,
                                         sharphi=4.40)

    stars_shallow_conv = daofind_shallow_conv(datafilt_conv_psf)
    log.info(f"Starfinding for mask done at {time.time()-t0:0.1f}s.  n_shallow={len(stars_shallow_conv)}, n_deep={len(stars_deep_conv)}")


    # Use the original data, unfiltered and unconvolved, because we're masking out the stars
    masked_data = data.copy()
    starfish_mask = np.zeros_like(data, dtype='bool')
    for row in stars_deep_conv:
        xc,yc = row['xcentroid'], row['ycentroid']
        #mreg = regions.CirclePixelRegion(regions.PixCoord(xc, yc), radius=fwhm_pix*1.5)
        mreg = regions.RectanglePixelRegion(regions.PixCoord(xc, yc), width=31, height=31)

        msk = mreg.to_mask()
        slcs, sslcs = msk.get_overlap_slices(masked_data.shape)
        try:
            #masked_data[slcs][msk.data.astype('bool')] = np.nan
            if row['flux'] > 10:
                masked_data[slcs][psfmask100.astype('bool')[sslcs]] = np.nan
            else:
                masked_data[slcs][psfmask7pct.astype('bool')[sslcs]] = np.nan

        except IndexError:
            # border case
            pass
    for row in stars_shallow_conv:
        xc,yc = row['xcentroid'], row['ycentroid']
        #mreg = regions.CirclePixelRegion(regions.PixCoord(xc, yc), radius=15)
        msksz = 31
        mreg = regions.RectanglePixelRegion(regions.PixCoord(xc, yc), width=msksz, height=msksz)
        msk = mreg.to_mask()
        slcs, sslcs = msk.get_overlap_slices(masked_data.shape)
        try:
            #masked_data[slcs][msk.data.astype('bool')] = np.nan
            masked_data[slcs][psfmask1000.astype('bool')[sslcs]] = np.nan

            # we want to allow stars at the center of the mask, but not in its wings
            center_false = psfmask1000.astype('bool').copy()
            center_false[msksz//2, msksz//2] = False
            starfish_mask[slcs][center_false[sslcs]] = True
        except IndexError:
            # border case
            pass


    # commented out experiments
    #medfilt_masked = median_filter(masked_data, size=[9,9])
    #conv = convolve(medfilt_masked, kernel=Gaussian2DKernel(fwhm_pix), nan_treatment='interpolate')
    #medfilt_masked = median_filter(medfilt_masked, size=[9,9])

    # now that we've masked out the stars, we fill back in the star positions by interpolating into them by smoothing the background
    xsize = np.ceil(10*fwhm_pix)
    if xsize % 2 == 0:
        xsize += 1
    kernel = Gaussian2DKernel(fwhm_pix, x_size=xsize)
    conv = convolve(masked_data, kernel=kernel, nan_treatment='interpolate')
    log.info(f"Masked convolution done at {time.time()-t0:0.1f}s")

    #medfilt_masked[np.isnan(medfilt_masked)] = conv[np.isnan(medfilt_masked)]

    # now we can replace the nans in the original data
    data_replacenans = data.copy()
    data_replacenans[filled_in_pixels] = datafilt_conv_psf[filled_in_pixels]

    # and then subtract off our star-masked convolved image
    # filtered_data = data_replacenans - medfilt_masked
    filtered_data = data_replacenans - conv


    if save_products:

        # fits.PrimaryHDU(data=medfilt_masked, header=im1[1].header).writeto('F444W_filter-based-background.fits', overwrite=True)

        fits.PrimaryHDU(data=conv, header=header).writeto(f'{path_prefix}/{filtername}_convolution-based-background.fits', overwrite=True)
        fits.PrimaryHDU(data=filtered_data, header=header).writeto(f'{path_prefix}/{filtername}_filter-based-background-subtraction.fits', overwrite=True)



    # create an empirical PSF
    data_cts = filtered_data / header['PHOTMJSR']

    bad_shallow = (np.abs(stars_shallow_conv['roundness2']) > 0.25) | (np.abs(stars_shallow_conv['roundness1']) > 0.25)

    stars_tbl = Table()
    stars_tbl['x'] = stars_shallow_conv['xcentroid'][~bad_shallow]
    stars_tbl['y'] = stars_shallow_conv['ycentroid'][~bad_shallow]

    # extract_stars does poorly with nans
    nddata = NDData(data=np.nan_to_num(filtered_data))
    sz = psf_size
    stars_ = extract_stars(nddata, stars_tbl, size=sz)

    # Remove off-center stars
    # and stars with saturated pixels
    stars = photutils.psf.epsf_stars.EPSFStars([star for star in stars_
                                                if np.unravel_index(np.argmax(star), star.shape) == (sz//2, sz//2)
                                                and data[int(star.center[1]), int(star.center[0])] > 0 # don't want the ones we replaced
                                               ])

    log.info(f"EPSF calculation beginning at {time.time()-t0:0.1f}s")
    epsf_builder = EPSFBuilder(oversampling=4, maxiters=10, smoothing_kernel='quadratic')

    epsf_quadratic_filtered, fitted_stars = epsf_builder(stars)
    if save_products:
        fits.PrimaryHDU(data=epsf_quadratic_filtered.data).writeto(f'{filtername}_ePSF_quadratic_filtered-background-subtracted.fits',
                overwrite=True)
    log.info(f"EPSF calculation done at {time.time()-t0:0.1f}s")

    # Compare  PSFs

    npsf = 16
    oversample = 2
    fov_pixels = 256
    psf_fn = f'{path_prefix}/{instrument.lower()}_{filtername}_samp{oversample}_nspsf{npsf}_npix{fov_pixels}.fits'
    if os.path.exists(psf_fn):
        # As a file
        grid = to_griddedpsfmodel(psf_fn)
    elif os.path.exists(psf_fn.replace(".fits", "_nrca5.fits")):
        # apparently even with outfile specified, nrca5 gets appended?
        grid = to_griddedpsfmodel(psf_fn.replace(".fits", "_nrca5.fits"))
    else:
        log.info(f"filtering: Calculating grid for psf_fn={psf_fn}")
        grid = nc.psf_grid(num_psfs=npsf, oversample=oversample,
                           all_detectors=False, save=True, outfile=psf_fn,
                           fov_pixels=fov_pixels)

    yy,xx = np.indices(epsf_quadratic_filtered.data.shape)
    xc,yc = np.unravel_index(epsf_quadratic_filtered.data.argmax(), epsf_quadratic_filtered.data.shape)
    grid.x_0 = xc/oversample
    grid.y_0 = yc/oversample
    grid.flux = 1
    fitter = LevMarLSQFitter()

    # there's some hoop-jumping here to get the PSFs to be comparable; I really
    # hope the PSF photometry toolkit understands how to use these PSFs...
    fitted_gridmod = fitter(model=grid, x=xx/oversample/2 + grid.x_0/oversample, y=yy/oversample/2 + grid.y_0/oversample, z=epsf_quadratic_filtered.data,)
    gridmodpsf = (fitted_gridmod(xx/oversample/2 + grid.x_0/oversample, yy/oversample/2 + grid.y_0/oversample))    

    pl.figure(1, figsize=(16,5))
    pl.clf()
    ax = pl.subplot(1,3,1)
    norm_epsf = simple_norm(epsf_quadratic_filtered.data, 'log', percent=99.)
    ax.set_title(f"{filtername} quadratic\nfrom median-filtered data")
    im = ax.imshow(epsf_quadratic_filtered.data, norm=norm_epsf, origin='lower')
    pl.colorbar(mappable=im)
    ax.set_xlabel('X [px]', fontsize=20)
    ax.set_ylabel('Y [px]', fontsize=20)
    ax2 = pl.subplot(1,3,2)
    ax2.set_title("WebbPSF model")
    dd = gridmodpsf
    norm = simple_norm(dd, 'log', percent=99.)
    im2 = ax2.imshow(dd, norm=norm, origin='lower')
    pl.colorbar(mappable=im2)
    ax3 = pl.subplot(1,3,3)
    ax3.set_title("Difference")
    dd = (epsf_quadratic_filtered.data) - gridmodpsf
    norm = simple_norm(dd, 'asinh', percent=99)
    im3 = ax3.imshow(dd, norm=norm, origin='lower')
    pl.colorbar(mappable=im3)
    pl.savefig(f"{filtername}_ePSF_quadratic_filtered_vs_webbpsf.png")


    # ## Do the PSF photometry
    #
    # DAOGroup decides which subset of stars needs to be simultaneously fitted together - i.e., it deals with blended sources.
    daogroup = SourceGrouper(5 * fwhm_pix)
    mmm_bkg = MMMBackground()

    filtered_errest = stats.mad_std(filtered_data, ignore_nan=True)

    daofind_fin = DAOStarFinder(threshold=nsigma_threshold * filtered_errest,
                                fwhm=fwhm_pix, roundhi=1.0, roundlo=-1.0,
                                sharplo=0.30, sharphi=1.40)
    finstars = daofind_fin(filtered_data, mask=starfish_mask)
    log.info(f"First-pass starfinding calculation done at {time.time()-t0:0.1f}s.  Found {len(finstars)} stars.")

    # criteria are based on examining some plots; they probably don't hold universally
    def filtered_finder(data, *args, **kwargs):
        """
        Wrap the star finder to reject bad stars
        """
        finstars = daofind_fin(data)
        bad = ((finstars['roundness1'] > finstars['mag']*0.4/8+0.65) | (finstars['roundness1'] < finstars['mag']*-0.4/8-0.5) |
               # bad! (finstars['sharpness'] < 0.48) | (finstars['sharpness'] > 0.6) |
               (finstars['roundness2'] > finstars['mag']*0.4/8+0.55) | (finstars['roundness2'] < finstars['mag']*-0.4/8-0.5))
        finstars = finstars[~bad]
        finstars['id'] = np.arange(1, len(finstars)+1)

        # this will print at each iteration
        log.info(f"Filtered {bad.sum()} bad stars out.  t={time.time()-t0:0.1f}")

        return finstars

    log.info(f"Finding sources.  t={time.time()-t0:0.1f}")
    star_list = finstars #filtered_finder(filtered_data)
    log.info(f"Found {len(star_list)} sources.  t={time.time()-t0:0.1f}")

    star_list['x_0'] = star_list['xcentroid']
    star_list['y_0'] = star_list['ycentroid']
    group_list = daogroup(star_list)
    log.info(f"Found {len(group_list)} sources.  t={time.time()-t0:0.1f}")

    # there may be fewer groups than stars
    #pb = tqdm(len(group_list))
    #lmfitter = LevMarLSQFitter()
    #def fitter(*args, **kwargs):
    #    pb.update()
    #    return lmfitter(*args, **kwargs)

    phot = BasicPSFPhotometry(finder=None, #filtered_finder,
                              group_maker=None,
                              bkg_estimator=None, #mmm_bkg,
                              #psf_model=psf_modelgrid[0],
                              psf_model=grid,
                              fitter=LevMarLSQFitter(),
                              fitshape=(11, 11),
                              aperture_radius=2*fwhm_pix)

    # operate on the full data
    log.info(f"Doing full photometry.  t={time.time()-t0:0.1f}")
    result_full = phot(np.nan_to_num(filtered_data), init_guesses=group_list)
    log.info(f"Done with full photometry.  t={time.time()-t0:0.1f}")
    resid = phot.get_residual_image()
    log.info(f"Done with final residual estimate.  t={time.time()-t0:0.1f}")
    if save_products:
        result_full.write(f'{path_prefix}/{filtername}_fullfield_WebbPSF_photometry.ecsv', overwrite=True)
        result_full.write(f'{path_prefix}/{filtername}_fullfield_WebbPSF_photometry.fits', overwrite=True)
        fits.PrimaryHDU(data=resid, header=header).writeto(f'{path_prefix}/{filtername}_psfphot_stars_removed.fits', overwrite=True)

    resid_orig = photutils.psf.utils.subtract_psf(data, grid, result_full)
    log.info(f"Done with final star subtraction from original data.  t={time.time()-t0:0.1f}")
    if save_products:
        fits.PrimaryHDU(data=resid_orig, header=header).writeto(f'{path_prefix}/{filtername}_originalimage_stars_removed.fits', overwrite=True)

    resid_orig_filled = photutils.psf.utils.subtract_psf(datafilt_conv_psf, grid, result_full)
    log.info(f"Done with final star subtraction from original filled in data.  t={time.time()-t0:0.1f}")
    if save_products:
        fits.PrimaryHDU(data=resid_orig_filled, header=header).writeto(f'{path_prefix}/{filtername}_psfphot_stars_filled_then_removed.fits', overwrite=True)

    starsubtracted_background = convolve(resid_orig, kernel=kernel, nan_treatment='interpolate')
    log.info(f"Done smoothing star-subtracted image for new background.  t={time.time()-t0:0.1f}")
    if save_products:
        fits.PrimaryHDU(data=resid_orig_filled,
                        header=header).writeto(f'{path_prefix}/{filtername}_background_convolved_from_starsubtracted.fits',
                                               overwrite=True)

    filtered_data_two = data_replacenans - starsubtracted_background
    if save_products:
        fits.PrimaryHDU(data=filtered_data,
                        header=header).writeto(f'{path_prefix}/{filtername}_starsubtraction-based-background-subtraction.fits',
                                               overwrite=True)

    log.info(f"Doing full photometry after star subtraction background.  t={time.time()-t0:0.1f}")
    finstars2 = daofind_fin(filtered_data_two)
    star_list2 = finstars2
    log.info(f"Found {len(star_list2)} sources.  t={time.time()-t0:0.1f}")

    star_list2['x_0'] = star_list2['xcentroid']
    star_list2['y_0'] = star_list2['ycentroid']
    group_list2 = daogroup(star_list2)
    result_full2 = phot(np.nan_to_num(filtered_data_two), init_guesses=group_list2)
    log.info(f"Done with full photometry.  t={time.time()-t0:0.1f}")
    resid2 = phot.get_residual_image()
    log.info(f"Done with final residual estimate.  t={time.time()-t0:0.1f}")
    if save_products:
        result_full2.write(f'{path_prefix}/{filtername}_fullfield_WebbPSF_photometry_iter2.ecsv', overwrite=True)
        result_full2.write(f'{path_prefix}/{filtername}_fullfield_WebbPSF_photometry_iter2.fits', overwrite=True)
        fits.PrimaryHDU(data=resid2, header=header).writeto(f'{path_prefix}/{filtername}_psfphot_stars_removed_iter2.fits', overwrite=True)

    resid_orig2 = photutils.psf.utils.subtract_psf(data, grid, result_full)
    log.info(f"Done with second final star subtraction from original data.  t={time.time()-t0:0.1f}")
    if save_products:
        fits.PrimaryHDU(data=resid_orig2, header=header).writeto(f'{path_prefix}/{filtername}_originalimage_stars_removed_iter2.fits', overwrite=True)

    return result_full2, resid_orig2


def make_noisemap(data, noisemap_filter_size=31, noisefunc=mad_std):
    """
    Create a noise map by applying a mad_std filter to the image.

    This is best done on an image that has only stuff considered noise.
    """
    yy,xx = np.indices([noisemap_filter_size, noisemap_filter_size])
    rr = ((xx-xx.max()/2)**2 + (yy-yy.max()/2)**2)**0.5
    footprint = rr<noisemap_filter_size/2
    noisemap = ndimage.generic_filter(data, noisefunc,
                                      footprint=footprint)

    return noisemap
