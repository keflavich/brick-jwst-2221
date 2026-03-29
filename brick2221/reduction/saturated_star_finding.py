# original file : https://github.com/keflavich/brick-jwst-2221/blob/main/brick2221/reduction/saturated_star_finding.py
import os
if not os.getenv('STPSF_PATH'):
    raise ValueError("STPSF_PATH must be specified")

import glob
from astropy.io import fits
from scipy.ndimage import label, find_objects, center_of_mass, sum_labels
from astropy.modeling.fitting import LevMarLSQFitter
from jwst.datamodels import dqflags
import matplotlib.pyplot as plt

#os.environ['stpsf_PATH'] = '/orange/adamginsburg/jwst/stpsf-data/'
import stpsf
from stpsf.utils import to_griddedpsfmodel


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
from astropy.coordinates import SkyCoord
from .filtering import get_filtername, get_fwhm
import functools
import requests
import urllib3
import builtins

def get_psf(header, path_prefix='.', use_merged_psf_for_merged=False):
    if header['INSTRUME'].lower() == 'nircam':
        psfgen = stpsf.NIRCam()
        fwhm, fwhm_pix = get_fwhm(header, instrument_replacement='NIRCam')
    elif header['INSTRUME'].lower() == 'miri':
        psfgen = stpsf.MIRI()
        fwhm, fwhm_pix = get_fwhm(header, instrument_replacement='MIRI')
    instrument = header['INSTRUME']
    filtername = get_filtername(header)
    try:
        module = header['MODULE']
    except KeyError:
        module = header['DETECTOR']
    detector = header['DETECTOR']

    ww = wcs.WCS(header)
    try:
        assert ww.wcs.cdelt[1] != 1, "This is not a valid WCS!!! CDELT is wrong!! how did this HAPPEN!?!?  (might happen if fitting a non-i2d file)"
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
    if detector == 'NRCALONG':
        detector = 'nrca5'
    elif detector == 'NRCBLONG':
        detector = 'nrcb5'
    if detector.lower() == 'mirimage':
        detector = 'mirim'

    psfgen.detector = detector.upper()

    psf_fn = f'{path_prefix}/{instrument.lower()}_{detector.lower()}_{filtername.lower()}_fovp{fov_pixels}_samp{oversample}_npsf{npsf}.fits'

    if module == 'merged':
        project_id = header['PROGRAM'][1:5]
        obs_id = header['OBSERVTN'].strip()
        merged_psf_fn = f'{basepath}/psfs/{filtername.upper()}_{project_id}_{obs_id}_merged_PSFgrid.fits'
        if use_merged_psf_for_merged and os.path.exists(merged_psf_fn):
            psf_fn = merged_psf_fn
            log.info(f"Using merged PSF grid {psf_fn}")
        else:
            print("Using detector-specific WebbPSF grid for this frame", flush=True)

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
                       all_detectors=False, fov_pixels=fov_pixels,
                                   outdir=path_prefix,
                                   save=True, outfile=None, overwrite=True)
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


def debug_wrap(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        print(function.__name__, flush=True)
        return function(*args, **kwargs)
    return wrapper


def find_saturated_stars(fitsdata, min_sep_from_edge=5, edge_npix=10000):
    """
    Identify candidate saturated stars from the DQ plane.

    This helper builds a boolean mask of saturated pixels from
    ``dqflags.pixel['SATURATED']``, explicitly removes pixels flagged as cosmic
    rays (``dqflags.pixel['JUMP_DET']``), labels connected components, and
    suppresses large edge-adjacent saturated regions.

    Parameters
    ----------
    fitsdata : astropy.io.fits.HDUList
        Open FITS HDU list containing a ``DQ`` extension.
    min_sep_from_edge : int, optional
        Dilation iterations applied when masking edge-associated saturated
        structures.
    edge_npix : int, optional
        Minimum connected saturated area (pixels) used to classify a component
        as an edge source.

    Returns
    -------
    saturated : numpy.ndarray
        Boolean mask of saturated, non-cosmic-ray pixels after edge masking.
    sources : numpy.ndarray
        Integer connected-component label image returned by
        ``scipy.ndimage.label``.
    coms : list[tuple[float, float]]
        Centers of mass (y, x) for labeled saturated components.
    """

    dq = fitsdata['DQ'].data
    saturated = (dq & dqflags.pixel['SATURATED']) > 0
    cosmic_rays = (dq & dqflags.pixel['JUMP_DET']) > 0
    saturated = saturated & (~cosmic_rays)

    sources, nsource = label(saturated)
    print('Saturated starfinding: nsources=', nsource, flush=True)
    sizes = sum_labels(saturated, sources, np.arange(nsource)+1)
    msfe = min_sep_from_edge

    # which sources are edge sources?  Anything w/ more than edge_npix contiguous "saturated" pixels
    edge_ids = np.where(sizes > edge_npix)[0]
    # id 0 is the non-saturated zone that we've excluded [but reading this code 3/28/2026, I'm skeptical this makes sense]
    edge_ids = edge_ids[1:]
    edge_mask = np.isin(sources, edge_ids)
    saturated = saturated & (~ndimage.binary_dilation(edge_mask, iterations=msfe))

    coms = center_of_mass(saturated, labels=sources, index=np.arange(nsource)+1)

    return saturated, sources, coms


def get_saturated_stars(fitsdata, path_prefix='/orange/adamginsburg/jwst/w51/psfs/', pad=81, size=None, min_sep_from_edge=5, edge_npix=10000, mask_buffer=1, plot=True, rindsz=3, use_merged_psf_for_merged=False):
    """
    Detect and PSF-fit saturated sources in a JWST image.

    This routine identifies connected saturated-pixel regions using the ``DQ``
    extension, excludes large edge-associated saturated structures, and then
    fits one source per remaining region with ``PSFPhotometry``.  Fits are
    performed on local cutouts with saturated pixels masked, and accepted
    results are stacked into a single output table.

    Parameters
    ----------
    fitsdata : astropy.io.fits.HDUList
        Open FITS HDU list containing at least ``SCI``, ``DQ``, and
        ``VAR_POISSON`` extensions.
    path_prefix : str, optional
        Directory used to load or cache PSF grid files.
    pad : int, optional
        Half-size (pixels) of the square cutout centered on each saturated
        source.
    size : int or tuple, optional
        Fit shape passed to ``PSFPhotometry``.
    min_sep_from_edge : int, optional
        Number of dilation iterations used to mask around large edge-saturated
        regions.
    edge_npix : int, optional
        Minimum saturated-pixel area used to classify a region as an edge
        source to be excluded.
    mask_buffer : int, optional
        Number of dilation iterations applied to saturated masks before fitting.
    plot : bool, optional
        If ``True``, display per-source diagnostic plots (cutout, model,
        residual, mask, thresholded model).
    rindsz : int, optional
        Reserved/legacy parameter; currently unused.
    use_merged_psf_for_merged : bool, optional
        If ``True`` and a merged PSF grid file exists, use it for merged
        mosaics.  Default is ``False`` to prefer detector-specific WebbPSF
        grids for individual frame fitting.

    Returns
    -------
    astropy.table.Table or None
        Stacked table of accepted saturated-source fits, or ``None`` if no
        valid sources are found.  Typical columns include fit parameters such
        as ``x_fit``, ``y_fit``, ``flux_fit``, uncertainties, and derived
        ``xcentroid``, ``ycentroid``, and ``skycoord_fit``.

    Notes
    -----
    - Requires ``STPSF_PATH`` to be defined before this module is imported.
    - Source acceptance currently requires finite flux uncertainty,
      ``snr > 1``, and positive fitted flux.
    - Large contiguous saturated edge structures are removed prior to fitting.
    """
    header = fitsdata[0].header
    data = fitsdata['SCI'].data
    assert data is not None

    # nan_to_num data to avoid fitting NaNs
    data[np.isnan(fitsdata['VAR_POISSON'].data)] = 0
    dq = fitsdata['DQ'].data

    saturated, sources, coms = find_saturated_stars(fitsdata, min_sep_from_edge=min_sep_from_edge, edge_npix=edge_npix)
    nsource = len(coms)

    big_grid = get_psf(header, path_prefix=path_prefix, use_merged_psf_for_merged=use_merged_psf_for_merged)
    ww = wcs.WCS(header)

    # We force the centroid to be fixed b/c the fitter doesn't do a great job with this...
    # ....this is not optimal...
    #big_grid.fixed['x_0'] = True
    #big_grid.fixed['y_0'] = True

    # daogroup should be set super high to avoid fitting lots of "stars"... if there are a lot of saturated pixels near each other, they're probably all junk
    #daogroup = SourceGrouper(min_separation=25)

    #resid = data

    #results = []

    lmfitter = LevMarLSQFitter()
    # def levmarverbosewrapper(self, *args, **kwargs):
    #     print("Running lmfitter")
    #     log.info(f"Running lmfitter with args {args} and kwargs {kwargs}")
    #     return self(*args, **kwargs)
    # #lmfitter.__call__ = levmarverbosewrapper
    # lmfitter._run_fitter = levmarverbosewrapper

    if header['INSTRUME'].lower() == 'nircam':
        psfgen = stpsf.NIRCam()
        fwhm, fwhm_pix = get_fwhm(header, instrument_replacement='NIRCam')
    elif header['INSTRUME'].lower() == 'miri':
        psfgen = stpsf.MIRI()
        fwhm, fwhm_pix = get_fwhm(header, instrument_replacement='MIRI')

    slices = find_objects(saturated)

    if size is None:
        size = pad

    index = 0
    print(f"Found {nsource} saturated sources to process", flush=True)
    for ii in range(nsource):
        # get the center of pixels with this label

        com = coms[ii] #center_of_mass(saturated, labels=sources, index=ii+1)
        # center_of_mass can return (nan, nan) for degenerate labels; guard against that
        if com is None:
            print(f"Source {ii+1}: center_of_mass returned None; skipping", flush=True)
            continue
        yf, xf = com
        if not (np.isfinite(yf) and np.isfinite(xf)):
            print(f"Source {ii+1}: center_of_mass returned NaN or infinite values ({yf}, {xf}); skipping", flush=True)
            continue
        ycen = int(round(yf))
        xcen = int(round(xf))
        print(f"Source {ii+1}: center at (x, y) = ({xcen}, {ycen})")
        y0 = int(max(0, ycen - pad))
        y1 = int(min(data.shape[0], ycen + pad))
        x0 = int(max(0, xcen - pad))
        x1 = int(min(data.shape[1], xcen + pad))
        size_saturated = int(np.sqrt(sum_labels(saturated, labels=sources, index=ii+1))/2)
        # area_saturated = sum_labels(saturated, labels=sources, index=ii+1)
        cutout = data[y0:y1, x0:x1]
        init_params = QTable()
        init_params['x'] = [xcen - x0]
        init_params['y'] = [ycen - y0]
        cutout[np.isnan(cutout)] = 0.0
        # if isinstance(grid, list):
        #     print(f"Grid is a list: {grid}")
        #     psf_model = WrappedPSFModel(grid[0])
        #     dao_psf_model = grid[0]
        # else:

        #psf_model = WrappedPSFModel(grid, stampsz=(size,size))

        psfphot = PSFPhotometry(
                                localbkg_estimator=LocalBackground(15, 30),
                                fitter=lmfitter,
                                psf_model=big_grid,
                                fit_shape=size,
                                aperture_radius=15*fwhm_pix)
        low_x  = xcen - x0 - size_saturated
        high_x = xcen - x0 + size_saturated
        low_y  = ycen - y0 - size_saturated
        high_y = ycen - y0 + size_saturated

        # get the underlying model and set bounds there
        model = getattr(psfphot, "psf_model", None)
        if model is None:
            raise RuntimeError("psfphot.psf_model is None — can't set parameter bounds")

        for pname, bounds in (("x_0", (low_x, high_x)), ("y_0", (low_y, high_y))):
            if hasattr(model, pname):
                param = getattr(model, pname)
                # try the supported API first
                try:
                    param.bounds = bounds
                    print(f"Set {pname}.bounds = {bounds}")
                except Exception:
                    # fallback (private attribute) if necessary
                    try:
                        param._bounds = bounds
                        print(f"Set {pname}._bounds = {bounds} (fallback)")
                    except Exception:
                        print(f"Could not set bounds for {pname}; parameter object: {param}")
            else:
                print(f"Model does not have parameter '{pname}'; param_names={getattr(model,'param_names',None)}")
        # let x_0 be bounds at saturated pixels
        #psfphot.x_0.bounds = (xcen - x0 - size_saturated , xcen - x0 + size_saturated)
        #psfphot.y_0.bounds = (ycen - y0 - size_saturated, ycen - y0 + size_saturated)
        saturated_mask = saturated[y0:y1, x0:x1]


        # expand a few pixels of saturated area to be masked
        saturated_mask_expanded = ndimage.binary_dilation(saturated_mask, iterations=mask_buffer)
        mask = np.logical_or(cutout==0, np.isnan(cutout), saturated_mask_expanded)
        try:
            result = psfphot(cutout, init_params=init_params, mask=mask)
        except Exception as ex:
            print(f"PSF photometry failed for source {ii+1} at (x,y)=({xcen},{ycen}): {ex}", flush=True)
            continue

        if len(result) == 0:
            print(f"PSF photometry returned no rows for source {ii+1}; skipping", flush=True)
            continue

        if hasattr(result['x_fit'], 'mask'):
            bad_fit_rows = result['x_fit'].mask | result['y_fit'].mask
        else:
            bad_fit_rows = (~np.isfinite(result['x_fit'])) | (~np.isfinite(result['y_fit']))

        if np.any(bad_fit_rows):
            print(f"Removing {bad_fit_rows.sum()} invalid fit rows for source {ii+1}", flush=True)
            result = result[~bad_fit_rows]

        if len(result) == 0:
            print(f"All fit rows were invalid for source {ii+1}; skipping", flush=True)
            continue

        result['xcentroid'] = result['x_fit'] + x0
        result['ycentroid'] = result['y_fit'] + y0
        x_centroid = np.asarray(result['xcentroid'], dtype=float)
        y_centroid = np.asarray(result['ycentroid'], dtype=float)
        world_fit = ww.pixel_to_world(x_centroid, y_centroid)
        if isinstance(world_fit, SkyCoord):
            result['skycoord_fit'] = world_fit
        else:
            log.warning("pixel_to_world did not return SkyCoord; setting skycoord_fit to None")
            result['skycoord_fit'] = [None] * len(result)


        result.pprint(max_width=-1)

        ny = cutout.shape[0]
        nx = cutout.shape[1]
        model_image = np.zeros_like(cutout)


        for x_fit, y_fit, flux in zip(result['x_fit'], result['y_fit'], result['flux_fit']):
            # Make a local grid around the source
            if np.isnan(flux):
                raise ValueError("Flux is NaN; cannot build PSF model image")
            y, x = np.mgrid[0:ny, 0:nx]
            #psf_eval = big_grid(x, y, flux=flux, x_0=x0, y_0=y0)  # works for analytic PSF
            psf_eval = big_grid(x-x_fit, y-y_fit) * flux  # works for GriddedPSFModel
            # cut psf_eval to the image size
            model_image += psf_eval[0:ny, 0:nx]

        threshold_image = np.zeros_like(cutout)

        #count the number of pixels above local background in the model_image
        threshold = np.nanpercentile(cutout, 99)
        threshold_image[model_image>threshold]=1
        num_pixels_above_threshold = np.nansum(threshold_image)
        print(np.nanmax(model_image), flush=True)
        print(f"Number of pixels above threshold ({threshold}): {num_pixels_above_threshold}", flush=True)

        if len(result) > 0:
            flux = result['flux_fit'][0]
            fluxerr = result['flux_err'][0]
            snr = result['flux_fit'][0] / result['flux_err'][0]
            qfit = result['qfit'][0]
            cfit = result['cfit'][0]

        if plot:
            fig = plt.figure(figsize=(12,12))
            ax1 = fig.add_subplot(2,3,1)
            ax1.imshow(cutout, origin='lower', cmap='viridis', vmin=0, vmax=np.nanpercentile(cutout, 99))
            ax1.set_title('Cutout')
            ax2 = fig.add_subplot(2,3,2)
            ax2.set_title('Model')
            ax2.imshow(model_image, origin='lower', cmap='viridis', vmin=0, vmax=np.nanpercentile(cutout, 99))
            ax3 = fig.add_subplot(2,3,3)
            resid_image = cutout - model_image
            ax3.imshow(resid_image, origin='lower', cmap='viridis', vmin=0, vmax=np.nanpercentile(cutout, 99))
            ax3.set_title('Residual')
            ax4 = fig.add_subplot(2,3,4)
            ax4.imshow(mask, origin='lower', cmap='gray')
            ax4.set_title('Mask')
            ax5 = fig.add_subplot(2,3,5)
            ax5.imshow(threshold_image, origin='lower', cmap='gray')
            ax5.set_title('Thresholded Model Pixels')
            # print flux, fluxerr, snr, cfit in the title
            if len(result) > 0:


                ax1.set_title(f'Cutout\nFlux={flux:.2f}, FluxErr={fluxerr:.2f}, SNR={snr:.2f}, qfit={qfit:.2f}, cfit={cfit:.2f}')

            plt.show()
            plt.close()

        # check whether the pixels with dqflag = saturated are also flagged as HOT, DEAD, and RC
        # this is a sanity check to make sure that the saturated pixels are not being used in the fit
        idx_saturated_in_cutout = (dq[y0:y1, x0:x1] & dqflags.pixel['SATURATED']) > 0
        saturated_dqflags = dq[y0:y1, x0:x1][idx_saturated_in_cutout]
        if np.any((saturated_dqflags & dqflags.pixel['HOT'])!=0):
            print(f"Warning: Some saturated pixels are flagged as HOT; skipping source", flush=True)
            continue
        if np.any((saturated_dqflags & dqflags.pixel['DEAD'])!=0):
            print(f"Warning: Some saturated pixels are flagged as DEAD; skipping source", flush=True)
            continue
        #if np.any((saturated_dqflags & dqflags.pixel['RC'])!=0):
        #    print(f"Warning: Some saturated pixels are flagged as RC; skipping source", flush=True)
        #    continue



        # compare FWHM of the model and the size of saturated pixels
        #if area_saturated > num_pixels_above_threshold:
        #    print(f"Warning: Saturated mask area ({area_saturated}) is larger than number of pixels above threshold ({num_pixels_above_threshold}); skipping source", flush=True)
        #    continue

        # process the result
        if result is not None and np.isfinite(fluxerr) and snr > 1 and flux > 0:
            print(f"Accepting source {ii+1} with flux={flux}, fluxerr={fluxerr}, snr={snr}", flush=True)
            if index == 0:
                base_tab = result
            else:
                base_tab = table.vstack([base_tab, result])

            index += 1
        else:
            print(f"Skipping source {ii+1} due to non-finite flux error or low SNR", flush=True)
            print(f"  fluxerr={fluxerr}, snr={snr}", flush=True)

    # if base_tab is not defined, return None
    # this happens if no saturated stars are found
    if index == 0:
        print('No saturated stars found after processing all sources', flush=True)
        return None
    else:
        if 'x_0' not in base_tab.colnames and 'xcentroid' in base_tab.colnames:
            base_tab['x_0'] = base_tab['xcentroid']
        if 'y_0' not in base_tab.colnames and 'ycentroid' in base_tab.colnames:
            base_tab['y_0'] = base_tab['ycentroid']
        builtins.satstar_table = base_tab
        builtins.satstar_resid = data.copy()
        return base_tab

def remove_saturated_stars(filename, save_suffix='_unsatstar', overwrite=True, **kwargs):
    print(f"Removing saturated stars from {filename}", flush=True)
    fh = fits.open(filename)
    data = fh['SCI'].data

    # there are examples, especially in F405, where the variance is NaN but the value
    # is negative
    print(f"Setting NaN variance to 0", flush=True)
    #data[np.isnan(fh['VAR_POISSON'].data)] = 0

    header = fh[0].header
    if 'CRPIX1' not in header:
        header.update(wcs.WCS(fh['SCI'].header).to_header())
    print("Running get_saturated_stars", flush=True)
    satstar_table = get_saturated_stars(fh,)
    if satstar_table is not None:
        satstar_table.meta.update(header)
        print("Finished get_saturated_stars", flush=True)

        satstar_table.write(filename.replace(".fits", '_satstar_catalog.fits'), overwrite=overwrite)
        print(f"Saved saturated star catalog to {filename.replace('.fits', '_satstar_catalog.fits')}", flush=True)
    else:
        print("No saturated stars found", flush=True)
        return



def main():
    if not os.get('STPSF_PATH'):
        raise ValueError("STPSF_PATH must be specified")

    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-f", "--filter", dest="filter",
                      default='F140M',
                      help="filter list", metavar="filter")
    parser.add_option("--target", dest="target",
                      default='w51',
                      help="target name", metavar="target")
    (options, args) = parser.parse_args()
    filt = options.filter
    if int(filt[1:4]) < 500:
        modules = ('nrca', 'nrcb')
    else:
        modules = ('mirim',)

    for module in modules:
        if int(filt[1:4]) < 500:
            globlist = glob.glob(f"/orange/adamginsburg/jwst/{options.target}/{filt}/pipeline/*{module}*align*crf.fits")
        else:
            globlist = glob.glob(f"/orange/adamginsburg/jwst/{options.target}/{filt}/pipeline/*mirimage_cal.fits")
        for i, fn in enumerate(globlist):
            print()
            print(fn)
            if True:
                remove_saturated_stars(fn)

    #for module in ('nrca', 'nrcb', 'merged'):
    #    for fn in glob.glob(f"/orange/adamginsburg/jwst/w51/F*/pipeline/*{module}*crf.fits"):
    #        print()
    #        print(fn)
    #        remove_saturated_stars(fn, verbose=True)




if __name__ == "__main__":
    main()
