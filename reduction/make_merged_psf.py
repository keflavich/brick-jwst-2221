import numpy as np
import copy
import os
import stdatamodels.jwst.datamodels
from photutils.psf import GriddedPSFModel
from astropy.io import fits
from astropy.wcs import WCS
import glob
import webbpsf
from astropy.nddata import NDData
from tqdm.auto import tqdm

def footprint_contains(x, y, shape):
    return (x > 0) and (y > 0) and (y < shape[0]) and (x < shape[1])

def make_merged_psf(filtername, basepath, halfstampsize=25,
                    grid_step=200,
                    project_id='2221', obs_id='001', suffix='merged_i2d'):
    
    nrc = webbpsf.NIRCam()
    nrc.filter = filtername
    grids = {}
    for detector in ('NRCA5', 'NRCB5'):
        savefilename = f'nircam_{detector.lower()}_{filtername.lower()}_fovp101_samp4_npsf16.fits'
        if os.path.exists(savefilename):
            gridfh = fits.open(savefilename)
            ndd = NDData(gridfh[0].data, meta=dict(gridfh[0].header))
            ndd.meta['grid_xypos'] = [((float(ndd.meta[key].split(',')[1].split(')')[0])),
                                       (float(ndd.meta[key].split(',')[0].split('(')[1])))
                                      for key in ndd.meta.keys() if "DET_YX" in key]

            ndd.meta['oversampling'] = ndd.meta["OVERSAMP"]  # just pull the value
            ndd.meta = {key.lower(): ndd.meta[key] for key in ndd.meta}

            grid = GriddedPSFModel(ndd)

        else:
            nrc.detector = detector
            grid = nrc.psf_grid(num_psfs=16, all_detectors=False, verbose=True, save=True)

        grids[detector] = grid

    # it would make sense to replace this with a more careful approach for determining what files went into the mosaic
    files = {'nrca': f'{basepath}/{filtername}/pipeline/*nrca*_cal.fits',
             'nrcb': f'{basepath}/{filtername}/pipeline/*nrcb*_cal.fits',
            }
    parent_file = fits.open(f'{basepath}/{filtername}/pipeline/jw0{project_id}-o{obs_id}_t001_nircam_clear-{filtername.lower()}-{suffix}.fits')
    parent_wcs = WCS(parent_file[1].header)

    pshape = parent_file[1].data.shape
    psf_grid_y, psf_grid_x = np.mgrid[0:pshape[0]:grid_step, 0:pshape[1]:grid_step]
    psf_grid_coords = list(zip(psf_grid_x.flat, psf_grid_y.flat))

    psfmeta = {'grid_xypos': psf_grid_coords,
               'oversampling': 1
              }
    allpsfs = []

    for pgxc, pgyc in tqdm(psf_grid_coords):
        skyc1 = parent_wcs.pixel_to_world(pgxc, pgyc)

        psfs = []
        for module in ("nrca", "nrcb"):
            for fn in glob.glob(f'{basepath}/{filtername}/pipeline/*{module}*_cal.fits'):
                dmod = stdatamodels.jwst.datamodels.open(fn)
                xc, yc = dmod.meta.wcs.world_to_pixel(skyc1)
                if footprint_contains(xc, yc, dmod.data.shape):
                    # force xc, yc to integers so they stay centered
                    # (mgrid is forced to be integers, and allowing xc/yc not to be would result in arbitrary subpixel shifts)
                    yy, xx = np.mgrid[int(yc)-halfstampsize:int(yc)+halfstampsize, int(xc)-halfstampsize:int(xc)+halfstampsize]
                    psf = grids[f'{module.upper()}5'].evaluate(x=xx, y=yy, flux=1, x_0=int(xc), y_0=int(yc))
                    psfs.append(psf)

        if len(psfs) > 0:
            meanpsf = np.mean(psfs, axis=0)
        else:
            meanpsf = np.zeros((halfstampsize*2, halfstampsize*2))
        allpsfs.append(meanpsf)

    allpsfs = np.array(allpsfs)

    cdata = allpsfs / allpsfs.sum(axis=(1,2))[:,None,None]
    avg = np.nanmean(cdata, axis=0)
    cdata[np.any(np.isnan(cdata), axis=(1,2)), :, :] = avg

    return NDData(cdata, meta=psfmeta)

def save_psfgrid(psfg,  outfilename, overwrite=True):
    xypos = fits.ImageHDU(np.array(psfg.meta['grid_xypos']))
    meta = copy.copy(psfg.meta)
    del meta['grid_xypos']
    header = fits.Header(meta)
    psfhdu = fits.PrimaryHDU(data=psfg.data, header=header)
    fits.HDUList([psfhdu, xypos]).writeto(outfilename, overwrite=overwrite)

def load_psfgrid(filename):
    fh = fits.open(filename)
    data = fh[0].data
    grid_xypos = fh[1].data
    meta = dict(fh[0].header)
    meta['grid_xypos'] = grid_xypos
    ndd = NDData(data, meta=meta)
    return GriddedPSFModel(ndd)

if __name__ == "__main__":

    project_id = '2221'
    obs_id = '001'
    for filtername in ('F405N', 'F466N', 'F410M', 'F444W', 'F356W', 'F187N', 'F182M', 'F212N', 'F200W', 'F115W'):
        psfg = make_merged_psf(filtername,
                            basepath='/orange/adamginsburg/jwst/brick/',
                            halfstampsize=25, grid_step=200,
                            project_id=project_id, obs_id=obs_id, suffix='merged_i2d')
        save_psfgrid(psfg, outfilename=f'{basepath}/psfs/{filtername}_{project_id}_{obs_id}_merged_PSFgrid.fits', overwrite=True)