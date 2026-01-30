# original file : https://github.com/keflavich/brick-jwst-2221/blob/main/brick2221/reduction/saturated_star_finding.py
import os
os.environ['STPSF_PATH'] = '/blue/adamginsburg/t.yoo/from_red/stpsf-data'

import glob
from astropy.io import fits
from scipy.ndimage import label, find_objects, center_of_mass, sum_labels
from astropy.modeling.fitting import LevMarLSQFitter
from jwst.datamodels import dqflags
import matplotlib.pyplot as plt
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
def get_psf(header, path_prefix='.'):
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
    if detector == 'NRCALONG':
        detector = 'nrca5'
    elif detector == 'NRCBLONG':
        detector = 'nrcb5'
    if detector.lower() == 'mirimage':
        detector = 'mirim'
    psf_fn = f'{path_prefix}/{instrument.lower()}_{detector.lower()}_{filtername.lower()}_fovp{fov_pixels}_samp{oversample}_npsf{npsf}.fits'

    if module == 'merged':
        project_id = header['PROGRAM'][1:5]
        obs_id = header['OBSERVTN'].strip()
        merged_psf_fn = f'{basepath}/psfs/{filtername.upper()}_{project_id}_{obs_id}_merged_PSFgrid.fits'
        if os.path.exists(psf_fn):
            psf_fn = merged_psf_fn
        else:
            print("stpsf is being used for merged data because merged PSF does not exist", flush=True)

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
        # https://github.com/spacetelescope/stpsf/blob/cc16c909b55b2a26e80b074b9ab79ed9a312f14c/stpsf/stpsf_core.py#L640
        # https://github.com/spacetelescope/stpsf/blob/cc16c909b55b2a26e80b074b9ab79ed9a312f14c/stpsf/gridded_library.py#L424
        big_grid = psfgen.psf_grid(num_psfs=npsf, oversample=oversample,
                                   all_detectors=True, fov_pixels=fov_pixels,
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

import os
#os.environ['stpsf_PATH'] = '/orange/adamginsburg/jwst/stpsf-data/'
import stpsf
from stpsf.utils import to_griddedpsfmodel



def get_saturated_stars(fitsdata,path_prefix='/orange/adamginsburg/jwst/w51/psfs/', pad=30, size=None, min_sep_from_edge=5, edge_npix=10000, mask_buffer=1, plot=True, rindsz=3):
    header = fitsdata[0].header
    data = fitsdata['SCI'].data
    data[np.isnan(fitsdata['VAR_POISSON'].data)] = 0
    dq = fitsdata['DQ'].data
    saturated = (dq & dqflags.pixel['SATURATED'])>0 
    sources, nsource = label(saturated)
    print('nsources=', nsource, flush=True)
    sizes = sum_labels(saturated, sources, np.arange(nsource)+1)
    msfe = min_sep_from_edge

    # which sources are edge sources?  Anything w/ more than edge_npix contiguous "saturated" pixels
    # add +1 because 0 is the non-saturated zone that we've excluded
    edge_ids = np.where(sizes > edge_npix)[0] + 1
    edge_mask = np.isin(sources, edge_ids)
    saturated = saturated & (~ndimage.binary_dilation(edge_mask, iterations=msfe))


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
        psfgen = stpsf.NIRCam()
        fwhm, fwhm_pix = get_fwhm(header, instrument_replacement='NIRCam')
    elif header['INSTRUME'].lower() == 'miri':
        psfgen = stpsf.MIRI()
        fwhm, fwhm_pix = get_fwhm(header, instrument_replacement='MIRI')

    
    slices = find_objects(saturated)
    
    # define pad/size/fwhm_pix (choose sensible defaults if not already set)
    size = pad = 81

   
    print(f"Found {nsource} saturated sources to process", flush=True)
    for ii in range(nsource):
        if True: # to keep indentation level same
            # get the center of pixels with this label
            
            com = center_of_mass(saturated, labels=sources, index=i+1)
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
            area_saturated = sum_labels(saturated, labels=sources, index=ii+1)
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
                    except (AttributeError, TypeError) as e:
                        # fallback (private attribute) if necessary
                        try:
                            param._bounds = bounds
                            print(f"Set {pname}._bounds = {bounds} (fallback)")
                        except (AttributeError, TypeError) as e:
                            print(f"Could not set bounds for {pname}; parameter object: {param}. Error: {e}")
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
                print(f"PSF photometry failed for source {i+1} at (x,y)=({xcen},{ycen}): {ex}", flush=True)
                continue
            
            result['xcentroid'] = result['x_fit'] + x0
            result['ycentroid'] = result['y_fit'] + y0
            result['skycoord_fit'] = ww.pixel_to_world(result['xcentroid'], result['ycentroid'])

            
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
                print(f"Accepting source {i+1} with flux={flux}, fluxerr={fluxerr}, snr={snr}", flush=True)
                if index == 0:
                    base_tab = result
                else:
                    base_tab = table.vstack([base_tab, result])
                
                index += 1
            else:
                print(f"Skipping source {i+1} due to non-finite flux error or low SNR", flush=True)
                print(f"  fluxerr={fluxerr}, snr={snr}", flush=True)
            
    # if base_tab is not defined, return None
    # this happens if no saturated stars are found
    if index == 0:
        print('No saturated stars found after processing all sources', flush=True)
        return None
    else:
        return base_tab

def remove_saturated_stars(filename, save_suffix='_unsatstar', **kwargs):
    print(f"Removing saturated stars from {filename}", flush=True)
    fh = fits.open(filename)
    data = fh['SCI'].data

    # there are examples, especially in Faaaaa405, where the variance is NaN but the value
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

        satstar_table.write(filename.replace(".fits", '_satstar_catalog_newnewnewnew.fits'), overwrite=True)
        print(f"Saved saturated star catalog to {filename.replace('.fits', '_satstar_catalog_newnewnewnew.fits')}", flush=True)
    else:
        print("No saturated stars found", flush=True)
        return
    
    

def main():
    import os
    os.environ['STPSF_PATH'] = '/blue/adamginsburg/t.yoo/from_red/stpsf-data'

    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-f", "--filter", dest="filter",
                      default='F140M',
                      help="filter list", metavar="filter")
    (options, args) = parser.parse_args()
    filt = options.filter
    if filt in ['F140M', 'F162M', 'F182M', 'F187N', 'F210M', 'F335M', 'F360M', 'F405N', 'F410M', 'F480M']:
        modules = ('nrca', 'nrcb')
    else:
        modules = ('mirim',)
    for module in modules:
        if filt in ['F140M', 'F162M', 'F182M', 'F187N', 'F210M', 'F335M', 'F360M', 'F405N', 'F410M', 'F480M']:
            globlist = glob.glob(f"/orange/adamginsburg/jwst/w51/{filt}/pipeline/*{module}*align*crf.fits")
        else:
            globlist = glob.glob(f"/orange/adamginsburg/jwst/w51/{filt}/pipeline/*mirimage_cal.fits")
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
