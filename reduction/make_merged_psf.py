import stdatamodels.jwst.datamodels
from astropy.io import fits
from astropy.wcs import WCS
import glob
import webbpsf
from astropy.nddata import NDData

def make_merged_psf(filtername, basepath, stampsize=25,
                    grid_step=200,
                    project_id='2221', obs_id='001', suffix='merged_i2d')
    

    nrc = webbpsf.NIRCam()
    nrc.filter = filtername
    grid = nrc.psf_grid(num_psfs=16, all_detectors=True, verbose=True, save=True)

    grids = {g.meta['detector'][0]: g for g in grid}

    files = {'nrca': f'{basepath}/{filtername}/pipeline/*nrca*_cal.fits',
             'nrcb': f'{basepath}/{filtername}/pipeline/*nrcb*_cal.fits',
            }
    parent_file = fits.open(f'{basepath}/{filtername}/pipeline/jw0{project_id}-o{obs_id}_t001_nircam_clear-{filtername.lower()}-{suffix}.fits')
    parent_wcs = WCS(parent_file[1].header)

    pshape = parent_file[1].data.shape
    psf_grid_y, psf_grid_x = np.mgrid[0:pshape[0]:grid_step, 0:pshape[1]:grid_step]
    psf_grid_coords = list(zip(psf_grid_x, psf_grid_y))

    psfmeta = {'grid_xypos': psf_grid_coords,
               'oversampling': 1
              }
    allpsfs = []

    for pgxc, pgyc in psf_grid_coords:
        skyc1 = parent_wcs.pixel_to_world(pgxc, pgyc)

        psfs = []
        for module in ("nrca", "nrcb"):
            for fn in tqdm(glob.glob(f'{basepath}/{filtername}/pipeline/*{module}*_cal.fits')):
                dmod = stdatamodels.jwst.datamodels.open(fn)
                xc, yc = dmod.meta.wcs.world_to_pixel(skyc1)
                if footprint_contains(xc, yc, dmod.data.shape):
                    yy, xx = np.mgrid[int(yc)-stampsize:int(yc)+stampsize, int(xc)-stampsize:int(xc)+stampsize]
                    psf = grids[f'{module.upper()}5'].evaluate(x=xx, y=yy, flux=1, x_0=xc, y_0=yc)
                    psfs.append(psf)

        meanpsf = np.mean(psfs, axis=0)
        allpsfs.append(meanpsf)

    return NDData(np.array(allpsfs), meta=psfmeta)