import glob
from astropy.io import fits
from scipy.ndimage import label, find_objects, center_of_mass, sum_labels
from astropy.modeling.fitting import LevMarLSQFitter

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


from tqdm.notebook import tqdm
from tqdm import tqdm
from astropy import wcs
from astropy.wcs import WCS
import numpy as np
from scipy import ndimage
from astropy.table import Table, QTable
from astropy import table
from astropy import log
from filtering import get_filtername, get_fwhm
import functools
import requests
import urllib3

def debug_wrap(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        print(function.__name__, flush=True)
        return function(*args, **kwargs)
    return wrapper

import os
try:
    import webbpsf
    from webbpsf.utils import to_griddedpsfmodel
    os.environ['WEBBPSF_PATH'] = '/orange/adamginsburg/jwst/webbpsf-data/'
except ImportError:
    import stpsf
    from stpsf.utils import to_griddedpsfmodel
    os.environ['STPSF_PATH'] = '/orange/adamginsburg/jwst/stpsf-data/'


def is_star(data, sources, srcid, slc, rindsize=3, min_flux=500, require_gradient=False):
    """
    Attempt to determine if a collection of blank pixels is actually a star by
    assuming the pixels closest to the center will be brighter than their
    surroundings
    """
    slc = tuple(slice(max(ss.start-rindsize, 0),
                      min(ss.stop+rindsize, shp)) for ss,shp in zip(slc, data.shape))

    labelmask = sources[slc] == srcid
    assert np.any(labelmask)

    rind1 = ndimage.binary_dilation(labelmask, iterations=2).astype('bool')
    rind2 = ndimage.binary_dilation(rind1, iterations=2).astype('bool')
    rind2sum = np.nansum(data[slc][rind2 & ~rind1])
    rind1sum = np.nansum(data[slc][rind1 & ~labelmask])

    rind3 = ndimage.binary_dilation(labelmask, iterations=rindsize)
    rind3sum = np.nansum(data[slc][rind3 & ~labelmask])

    return ((rind1sum > rind2sum) or not require_gradient) and rind3sum > min_flux

def finder_maker(max_size=100, min_size=0, min_sep_from_edge=5, min_flux=500,
                 edge_npix=10000,
                 progressbar=False,
                 rindsize=3, require_gradient=False, raise_for_nosources=True, *args, **kwargs):
    """
    Create a saturated star finder that can select on the number of saturated pixels and the
    distance from the edge of the image
    """
    # criteria are based on examining some plots; they probably don't hold universally
    def saturated_finder(data, *args, raise_for_nosources=raise_for_nosources, rind_threshold=1000, **kwargs):
        """
        Wrap the star finder to reject bad stars
        """
        saturated = np.logical_or((data==0), np.isnan(data))
        if not np.any(saturated):
            raise ValueError("No saturated (data==0) pixels found")
        sources_, nsources_ = label(saturated)
        if raise_for_nosources and nsources_ == 0:
            raise ValueError("No saturated sources found")

        sizes = sum_labels(saturated, sources_, np.arange(nsources_)+1)
        msfe = min_sep_from_edge

        # which sources are edge sources?  Anything w/ more than edge_npix contiguous "saturated" pixels
        # add +1 because 0 is the non-saturated zone that we've excluded
        edge_ids = np.where(sizes > edge_npix)[0] + 1
        edge_mask = np.isin(sources_, edge_ids)
        no_edge_saturated = saturated & (~ndimage.binary_dilation(edge_mask, iterations=msfe))

        # now re-calculate sources
        sources, nsources = label(no_edge_saturated)
        print(f"Reduced nsources from {nsources_} to {nsources} by excluding edge zone with size {msfe}", flush=True)
        if raise_for_nosources and nsources == 0:
            raise ValueError("No saturated sources found")

        slices = find_objects(sources)
        sizes = sum_labels(saturated, sources, np.arange(nsources)+1)

        coms = center_of_mass(saturated, sources, np.arange(nsources)+1)
        coms = np.array(coms)

        # Create additional mask for rind threshold check using scipy tools
        def check_rind_threshold(label_id, rindsz=3):
            # Get the slice for this source using find_objects
            src_slice = slices[label_id - 1]  # label_id is 1-indexed, slices is 0-indexed
            # Expand slice by 3 pixels on each side
            src_slice = (slice(max(0, src_slice[0].start - rindsz),
                               min(data.shape[0], src_slice[0].stop + rindsz)),
                        slice(max(0, src_slice[1].start - rindsz),
                              min(data.shape[1], src_slice[1].stop + rindsz)))
            src_mask = sources[src_slice] == label_id
            # Create 3-pixel rind around the source
            rind = ndimage.binary_dilation(src_mask, iterations=rindsz) & ~src_mask
            # Check if sum of values in rind exceeds threshold
            rind_sum = np.nansum(data[src_slice][rind])
            return rind_sum > rind_threshold

        # Apply rind threshold check to all sources
        rind_ok = np.array([check_rind_threshold(label_id) for label_id in range(1, nsources + 1)])

        # Update sources to only include those that pass rind threshold
        print(f"Reduced nsources from {len(slices)} to {rind_ok.sum()} by applying rind threshold {rind_threshold}", flush=True)


        # progressbar isn't super necessary as this is rarely the bottleneck
        # (but I included it because I didn't know that up front)
        if progressbar:
            pb = tqdm
        else:
            pb = lambda x: x

        sizes_ok = (sizes < max_size) & (sizes >= min_size)
        coms_finite = np.isfinite(coms).all(axis=1)
        coms_inbounds = (
            (coms[:,1] > msfe) & (coms[:,0] > msfe) &
            (coms[:,1] < data.shape[1]-msfe) &
            (coms[:,0] < data.shape[0]-msfe)
        )
        all_ok = sizes_ok & coms_finite & coms_inbounds & rind_ok
        is_star_ok = np.array([szok and is_star(data, sources, srcid+1, slcs, min_flux=min_flux, rindsize=rindsize)
                               for srcid, (szok, slcs) in enumerate(pb(zip(all_ok, slices)))])
        all_ok &= is_star_ok
        print(f"inside saturated_finder, with minmax nsaturated = {min_size,max_size} and min_flux={min_flux}, number of is_star={is_star_ok.sum()}, ", end="", flush=True)
        print(f"sizes={sizes_ok.sum()}, centerofmass_finite={coms_finite.sum()}, coms_inbounds={coms_inbounds.sum()}, total={all_ok.sum()} candidates", flush=True)


        print("Creating table", flush=True)
        tbl = Table()
        tbl['id'] = np.arange(1,all_ok.sum()+1)
        tbl['xcentroid'] = [cc[1] for cc, ok in zip(coms, all_ok) if ok]
        tbl['ycentroid'] = [cc[0] for cc, ok in zip(coms, all_ok) if ok]
        print("Table created; returning table", flush=True)

        return tbl
    return saturated_finder

def get_psf(header, path_prefix='.'):
    if header['INSTRUME'].lower() == 'nircam':
        psfgen = webbpsf.NIRCam()
        fwhm, fwhm_pix = get_fwhm(header, instrument_replacement='NIRCam')
    elif header['INSTRUME'].lower() == 'miri':
        psfgen = webbpsf.MIRI()
        fwhm, fwhm_pix = get_fwhm(header, instrument_replacement='MIRI')
    instrument = header['INSTRUME']
    filtername = get_filtername(header)
    module = header['MODULE']

    ww = wcs.WCS(header)
    try:
        assert ww.wcs.cdelt[1] != 1, "This is not a valid WCS!!! CDELT is wrong!! how did this HAPPEN!?!?"
    except AssertionError as ex:
        print(ex)
        print("ignoring WCS failure so check that stuff is right...")

    psfgen.filter = filtername
    obsdate = header['DATE-OBS']

    with open(os.path.expanduser('~/.mast_api_token'), 'r') as fh:
        api_token = fh.read().strip()

    npsf = 16
    oversample = 2
    fov_pixels = 512
    psf_fn = f'{path_prefix}/{instrument.lower()}_{filtername}_samp{oversample}_nspsf{npsf}_npix{fov_pixels}_{module}.fits'
    if module == 'merged':
        project_id = header['PROGRAM'][1:5]
        obs_id = header['OBSERVTN'].strip()
        merged_psf_fn = f'{basepath}/psfs/{filtername.upper()}_{project_id}_{obs_id}_merged_PSFgrid.fits'
        if os.path.exists(psf_fn):
            psf_fn = merged_psf_fn
        else:
            print("webbPSF is being used for merged data because merged PSF does not exist", flush=True)

    if os.path.exists(str(psf_fn)):
        # As a file
        log.info(f"Loading grid from psf_fn={psf_fn}")
        big_grid = to_griddedpsfmodel(psf_fn)  # file created 2 cells above
        if isinstance(big_grid, list):
            print(f"PSF IS A LIST OF GRIDS!!!", flush=True)
            big_grid = big_grid[0]
    else:
        log.info(f'PSF file {psf_fn} does not exist; downloading from MAST')
        from astroquery.mast import Mast

        print(f"Attempting to load PSF for {obsdate}")
        try:
            Mast.login(api_token.strip())
            os.environ['MAST_API_TOKEN'] = api_token.strip()

            psfgen.load_wss_opd_by_date(f'{obsdate}T00:00:00')
        except (urllib3.exceptions.ReadTimeoutError, requests.exceptions.ReadTimeout, requests.HTTPError) as ex:
            print(f"Failed to build PSF: {ex}")
        except Exception as ex:
            print("psfgen load_wss_opd_by_date failed")
            print(ex)

        log.info(f"starfinding: Calculating grid for psf_fn={psf_fn}")
        # https://github.com/spacetelescope/webbpsf/blob/cc16c909b55b2a26e80b074b9ab79ed9a312f14c/webbpsf/webbpsf_core.py#L640
        # https://github.com/spacetelescope/webbpsf/blob/cc16c909b55b2a26e80b074b9ab79ed9a312f14c/webbpsf/gridded_library.py#L424
        big_grid = psfgen.psf_grid(num_psfs=npsf, oversample=oversample,
                                   all_detectors=True, fov_pixels=fov_pixels,
                                   outdir=path_prefix,
                                   save=True, outfile=psf_fn, overwrite=True)
        # now the PSF should be written
        assert glob.glob(psf_fn.replace(".fits", "*"))
        if isinstance(big_grid, list):
            print(f"PSF FROM PSF_GEN IS A LIST OF GRIDS!!!", flush=True)
            big_grid = big_grid[0]
            # if we really want to get this right, we need to create a new grid of PSF models
            # that is some sort of average of the PSF model grid.
            # There's no way to do it _right_ right without going back to the original data,
            # which is untenable with this approach.  It's a huge project.

    return big_grid


def iteratively_remove_saturated_stars(data, header,
                                       fit_sizes=[351, 351, 201, 201, 101, 51],
                                       nsaturated=[(100, 500), (50, 100), (25, 50), (10, 25), (5, 10), (0, 5)],
                                       min_flux=[1000, 750, 500, 250, 200, 150],
                                       ap_rad=[15, 15, 15, 15, 10, 5],
                                       require_gradient=[False, False, False, False, False, True],
                                       dilations=[3, 3, 3, 2, 2, 1],
                                       rindsize=[6, 5, 5, 4, 4, 3],
                                       path_prefix='.',
                                       verbose=False
                                      ):
    """
    Iteratively remove the most saturated down to the least saturated stars until all such stars are fitted & subtracted.

    Parameters
    ----------
    fit_sizes : list of integers
        The size along each axis to cut out around the bright star to fit the
        PSF and subtract it.  Bigger ``fit_sizes`` are _much_ more expensive,
        but they are also needed for the brightest sources that affect large
        areas on the image
    nsaturated : list of tuples of integer pairs
        The minimum and maximum number of saturated pixels for stars included in
        each iteration.
        For example, the range 100-500 will include all saturated sources with
        100 < npixels < 500 contiguous saturated pixels.
    min_flux : list of floats
        The minimum flux value to consider an object a star.  This is to reject
        regions that are set to zero but are not saturated.  It includes a region
        that is by default from the saturated zone edge to a region 3 pixels away
        using binary dilation.
    ap_rad : list of integers
        The aperture radius in which to perform aperture photometry.  It's not
        clear that this parameter is at all important.
    require_gradient : list of booleans
        Check that the "star" pixels around the saturated pixels are brighter
        than those inside.  This can be used to reject "fake" saturated stars
        caused by other problems in the image.  It's not clear that this parameter
        should ever be set, but it should likely be limited to the smallest
        (~few saturated pixel) sources, since those are the only ones likely to
        be fake sources.  I think these usually arise from places where the dither
        failed to fill in bad pixels.
    dilations : list of integers
        Dilate the mask to be used when calculating photometry?
        This masks out pixels around the saturated pixels, which are themselves
        often unreliable.  This is probably most important for aperture
        photometry, but it likely also affects PSF fitting.
    """
    print("Beginning to iteratively remove saturated stars", flush=True)

    assert len(fit_sizes) == len(nsaturated) == len(min_flux) == len(ap_rad) == len(dilations) == len(require_gradient)

    big_grid = get_psf(header, path_prefix=path_prefix)
    ww = wcs.WCS(header)

    # We force the centroid to be fixed b/c the fitter doesn't do a great job with this...
    # ....this is not optimal...
    #big_grid.fixed['x_0'] = True
    #big_grid.fixed['y_0'] = True

    # daogroup should be set super high to avoid fitting lots of "stars"... if there are a lot of saturated pixels near each other, they're probably all junk
    daogroup = SourceGrouper(min_separation=25)

    resid = data

    results = []

    lmfitter = LevMarLSQFitter()
    # def levmarverbosewrapper(self, *args, **kwargs):
    #     print("Running lmfitter")
    #     log.info(f"Running lmfitter with args {args} and kwargs {kwargs}")
    #     return self(*args, **kwargs)
    # #lmfitter.__call__ = levmarverbosewrapper
    # lmfitter._run_fitter = levmarverbosewrapper

    if header['INSTRUME'].lower() == 'nircam':
        psfgen = webbpsf.NIRCam()
        fwhm, fwhm_pix = get_fwhm(header, instrument_replacement='NIRCam')
    elif header['INSTRUME'].lower() == 'miri':
        psfgen = webbpsf.MIRI()
        fwhm, fwhm_pix = get_fwhm(header, instrument_replacement='MIRI')

    satpix = data == 0

    for (minsz, maxsz), minflx, grad, fitsz, apsz, diliter, rsz in zip(nsaturated, min_flux, require_gradient, fit_sizes, ap_rad, dilations, rindsize):
        finder = finder_maker(min_size=minsz, max_size=maxsz, require_gradient=grad, min_flux=minflx)
        print(f"Created finder {finder}", flush=True)

        # do the search on data b/c PSF subtraction can change zeros to non-zeros
        if np.any(np.logical_or(resid == 0, np.isnan(resid))):
            sources = finder(resid,
                             mask=ndimage.binary_dilation(resid==0, iterations=1),
                             raise_for_nosources=False, rindsize=rsz)
            print(f"Found {len(sources)} sources running finder {finder}", flush=True)
        else:
            log.warning(f"Skipped iteration with fit size={fitsz}, range={minsz}-{maxsz} because there are no saturated pixels")
            continue

        if len(sources) == 0:
            log.warning(f"Skipped iteration with fit size={fitsz}, range={minsz}-{maxsz}")
            continue

        if verbose:
            print(f"Before BasicPSFPhotometry: {len(sources)} sources.  min,max sz: {minsz,maxsz}  minflx={minflx}, grad={grad}, fitsz={fitsz}, apsz={apsz}, diliter={diliter}", flush=True)

        phot = PSFPhotometry(finder=finder,
                             # grouping disabled b/c it apepars to be a major bottleneck grouper=daogroup,
                             localbkg_estimator=None, # must be none or it un-saturates pixels
                             #psf_model=epsf_model,
                             psf_model=big_grid,
                             fitter=lmfitter,
                             fit_shape=fitsz,
                             aperture_radius=apsz*fwhm_pix,
                             )

        # Mask out the inner portion of the PSF when fitting it
        if diliter > 0:
            mask = ndimage.binary_dilation(np.logical_or(resid==0, np.isnan(resid)), iterations=diliter)
        else:
            mask = np.logical_or(resid==0, np.isnan(resid))

        #log.info("Doing photometry")
        try:
            print(f'Before trying with progressbar: resid shape={resid.shape}, mask shape={mask.shape}', flush=True)
            result = phot(resid, mask=mask, progressbar=tqdm)
        except TypeError:
            print(f'Before trying without: resid shape={resid.shape}, mask shape={mask.shape}', flush=True)
            result = phot(resid, mask=mask)

        result['skycoord_fit'] = ww.pixel_to_world(result['x_fit'], result['y_fit'])
        results.append(result)
        #log.info(f"Done; len(result) = {len(result)})")
        print(result, flush=True)

        # manually subtract off PSFs because get_residual_image seems to (never?) work
        # (it might work but I just had other errors masking that it was working, but this is fine - it's just more manual steps)
        #resid = subtract_psf(resid, phot.psf_model, result['x_fit', 'y_fit', 'flux_fit'], subshape=phot.fitshape)
        print(f"Making residual image.", flush=True)
        resid = phot.make_residual_image(resid, (fitsz, fitsz), include_localbkg=False)

        # reset saturated pixels back to zero
        resid[satpix] = 0

        # an option here, to make this work at an earlier phase in the pipeline, is to *replace* the masked
        # pixels with the values from the fitted model.  This will be tricky.
        print(f"Finished iteration with fit size={fitsz}, range={minsz}-{maxsz} with {len(result)} sources", flush=True, end='\n\n')

    final_table = table.vstack(results)

    return final_table, resid


def remove_saturated_stars(filename, save_suffix='_unsatstar', **kwargs):
    print(f"Removing saturated stars from {filename}", flush=True)
    fh = fits.open(filename)
    data = fh['SCI'].data

    # there are examples, especially in F405, where the variance is NaN but the value
    # is negative
    print(f"Setting NaN variance to 0", flush=True)
    data[np.isnan(fh['VAR_POISSON'].data)] = 0

    header = fh[0].header
    if 'CRPIX1' not in header:
        header.update(wcs.WCS(fh['SCI'].header).to_header())
    print("Running iteratively_remove_saturated_stars", flush=True)
    satstar_table, satstar_resid = iteratively_remove_saturated_stars(data, header, **kwargs)
    satstar_table.meta.update(header)
    print("Finished iteratively_remove_saturated_stars", flush=True)

    satstar_table.write(filename.replace(".fits", '_satstar_catalog.fits'), overwrite=True)
    fh['SCI'].data = satstar_resid
    fh.writeto(filename.replace(".fits", save_suffix+".fits"), overwrite=True)


def main():

    #with open(os.path.expanduser('/home/adamginsburg/.mast_api_token'), 'r') as fh:
    #    api_token = fh.read().strip()
    #from astroquery.mast import Mast
    #Mast.login(api_token.strip())
    #os.environ['MAST_API_TOKEN'] = api_token.strip()

    fn = '/orange/adamginsburg/jwst/cloudc/F405N/pipeline/jw02221002001_02201_00001_nrcalong_destreak_o002_crf.fits'
    remove_saturated_stars(fn, verbose=True)


    # skipping 'merged' b/c we don't expect PSFs to fit well enough
    for module in ('nrca', 'nrcb', ):#'merged'):
        for fn in glob.glob(f"/orange/adamginsburg/jwst/brick/F*/pipeline/*{module}*crf.fits"):
            print()
            print(fn)
            remove_saturated_stars(fn, verbose=True)

    for module in ('nrca', 'nrcb', ):#'merged'):
        for fn in glob.glob(f"/orange/adamginsburg/jwst/cloudc/F*/pipeline/*{module}*crf.fits"):
            print()
            print(fn)
            remove_saturated_stars(fn, verbose=True)


if __name__ == "__main__":
    main()
