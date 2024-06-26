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
from webbpsf.utils import to_griddedpsfmodel
from astropy.convolution import convolve, Gaussian2DKernel

def footprint_contains(x, y, shape):
    return (x > 0) and (y > 0) and (y < shape[0]) and (x < shape[1])

def make_merged_psf(filtername, basepath, halfstampsize=25,
                    grid_step=200,
                    oversampling=1,
                    blur=False,
                    smoothing_scales={'f182m': 0.25, 'f187n':0.25, 'f212n':0.55, 'f200w': 0.55, 'f115w': 0.25,
                                      'f356w': 0.55, 'f444w': 0.55, 'f410m': 0.55, 'f405n':0.55, 'f466n':0.55},
                    project_id='2221', obs_id='001', suffix='merged_i2d'):
    """
    Halfstampsize allows us to force odd-shaped PSF
    """

    wavelength = int(filtername[1:-1])
    if wavelength < 230:
        detectors = [f'NRC{ab}{num}' for ab in 'AB' for num in (1,2,3,4)]
    else:
        detectors = [f'NRC{ab}5' for ab in 'AB']

    nrc = webbpsf.NIRCam()
    nrc.filter = filtername
    grids = {}
    for detector in detectors:
        savefilename = f'nircam_{detector.lower()}_{filtername.lower()}_fovp101_samp4_npsf16.fits'
        if os.path.exists(savefilename):
            # gridfh = fits.open(savefilename)
            # ndd = NDData(gridfh[0].data, meta=dict(gridfh[0].header))
            # ndd.meta['grid_xypos'] = [((float(ndd.meta[key].split(',')[1].split(')')[0])),
            #                            (float(ndd.meta[key].split(',')[0].split('(')[1])))
            #                           for key in ndd.meta.keys() if "DET_YX" in key]

            # ndd.meta['oversampling'] = ndd.meta["OVERSAMP"]  # just pull the value
            # if int(ndd.metadata['oversampling']) != oversampling:
            #     raise ValueError("Saved file mismatch oversampling")
            # ndd.meta = {key.lower(): ndd.meta[key] for key in ndd.meta}

            # grid = GriddedPSFModel(ndd)
            grid = to_griddedpsfmodel(savefilename)

        else:
            nrc.detector = detector
            grid = nrc.psf_grid(num_psfs=16, oversample=oversampling, all_detectors=False, verbose=True, save=True)

        grids[detector.upper()] = grid

    parent_file = fits.open(f'{basepath}/{filtername}/pipeline/jw0{project_id}-o{obs_id}_t001_nircam_clear-{filtername.lower()}-{suffix}.fits')
    parent_wcs = WCS(parent_file[1].header)

    pshape = parent_file[1].data.shape
    psf_grid_y, psf_grid_x = np.mgrid[0:pshape[0]:grid_step, 0:pshape[1]:grid_step]
    psf_grid_coords = list(zip(psf_grid_x.flat, psf_grid_y.flat))

    psfmeta = {'grid_xypos': psf_grid_coords,
               'oversampling': oversampling
              }
    allpsfs = []

    for ii,(pgxc, pgyc) in enumerate(tqdm(psf_grid_coords)):
        skyc1 = parent_wcs.pixel_to_world(pgxc, pgyc)

        psfs = []
        for detector in detectors:
            if detector.endswith('5'):
                # name scheme: short is nrca1 nrca2 ..., long is nrcalong
                detectorstr = detector.replace('5', 'long').lower()
            else:
                detectorstr = detector.lower()

            # loop over each frame dithered to make the final mosaic
            for fn in glob.glob(f'{basepath}/{filtername}/pipeline/*{detectorstr.lower()}*_cal.fits'):
                dmod = stdatamodels.jwst.datamodels.open(fn)
                xc, yc = dmod.meta.wcs.world_to_pixel(skyc1)
                if footprint_contains(xc, yc, dmod.data.shape):
                    # force xc, yc to integers so they stay centered
                    # (mgrid is forced to be integers, and allowing xc/yc not to be would result in arbitrary subpixel shifts)
                    # oversampling allows non-integers again though, and increases the grid size

                    # yy, xx was being used before, but it looks like that might've created a 90deg rotation?
                    # OVERSAMPLING SHIFTS BY HALF-PIXEL!
                    yy, xx = np.mgrid[int(yc)-halfstampsize:int(yc)+halfstampsize + 1/oversampling:1/oversampling,
                                      int(xc)-halfstampsize:int(xc)+halfstampsize + 1/oversampling:1/oversampling]
                    psf = grids[f'{detector.upper()}'].evaluate(x=xx, y=yy, flux=1, x_0=int(xc), y_0=int(yc))
                    psfs.append(psf)

        if len(psfs) > 0:
            meanpsf = np.mean(psfs, axis=0)
        else:
            meanpsf = np.zeros((halfstampsize*2*oversampling + 1, halfstampsize*2*oversampling + 1))

        if blur:
            kernwidth = smoothing_scales[filtername.lower()]
            kernel = Gaussian2DKernel(kernwidth * oversampling)
            meanpsf = convolve(meanpsf, kernel)
            psfmeta['BLUR'] = kernwidth

        allpsfs.append(meanpsf)
        psfmeta[f'DET_YX{ii}'] =  (str((float(pgyc), float(pgxc))),
                                   "The #{} PSF's (y,x) detector pixel position".format(ii))
    psfmeta['OVERSAMP'] = oversampling
    psfmeta['DET_SAMP'] = oversampling
    psfmeta['FILTER'] = filtername
    psfmeta['DETECTOR'] = detectorstr # only the last one in the loop...

    allpsfs = np.array(allpsfs)

    cdata = allpsfs / allpsfs.sum(axis=(1,2))[:,None,None]
    avg = np.nanmean(cdata, axis=0)
    cdata[np.any(np.isnan(cdata), axis=(1,2)), :, :] = avg

    return NDData(cdata, meta=psfmeta)

def save_psfgrid(psfg,  outfilename, overwrite=True):
    xypos = fits.ImageHDU(np.array(psfg.meta['grid_xypos']))
    meta = copy.copy(psfg.meta)
    del meta['grid_xypos']
    try:
        header = fits.Header(meta)
    except Exception as ex:
        header = fits.Header()
        for key in meta:
            header[key] = meta[key]
    if 'oversampling' in header:
        del header['oversampling']
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

def fix_psfs_with_bad_meta(filename):
    """
    cd /orange/adamginsburg/jwst/brick/reduction
    from make_merged_psf import fix_psfs_with_bad_meta
    cd /orange/adamginsburg/jwst/brick/psfs
    import glob
    for fn in glob.glob("*.fits"):
        fix_psfs_with_bad_meta(fn)
    """
    from webbpsf.utils import to_griddedpsfmodel

    fh = fits.open(filename, mode='update')
    if 'DET_YX0' in fh[0].header and 'OVERSAMP' in fh[0].header:
        if 'oversampling' in fh[0].header:
            oversampling = fh[0].header['oversampling']
            del fh[0].header['oversampling']
        # done!
        fh.close()
    else:
        for ii, row in enumerate(fh[1].data):
            fh[0].header[f'DET_YX{ii}'] = (str((float(row[0]), float(row[1]))), "The #{} PSF's (y,x) detector pixel position".format(ii))

        if 'oversampling' in fh[0].header:
            oversampling = fh[0].header['oversampling']
            del fh[0].header['oversampling']
        else:
            print("Assuming oversampling=1")
            oversampling = 1

        fh[0].header['OVERSAMP'] = (oversampling, "Oversampling factor for FFTs in computation")
        fh[0].header['DET_SAMP'] = (oversampling, "Oversampling factor for MFT to detector plane")

        print(to_griddedpsfmodel(fh))

        fh.close()

if __name__ == "__main__":

    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-f", "--filternames", dest="filternames",
                    default='F466N,F405N,F410M,F200W,F115W,F187N,F212N,F182M,F444W,F356W',
                    help="filter name list", metavar="filternames")
    parser.add_option("--proposal_id", dest="proposal_id",
                    default='2221',
                    help="proposal_id", metavar="proposal_id")
    parser.add_option("--target", dest="target",
                    default='brick',
                    help="target", metavar="target")
    parser.add_option("--blur", dest="blur",
                    default=False, action='store_true',
                    help="blur", metavar="blur")
    parser.add_option("--redo", dest="redo",
                    default=False, action='store_true',
                    help="redo", metavar="redo")
    (options, args) = parser.parse_args()
    print(options)

    selected_filters = options.filternames.upper().split(",")

    obs_filters = {'2221': ['F410M', 'F212N', 'F466N', 'F405N', 'F187N', 'F182M'],
                   '1182': ['F444W', 'F356W', 'F200W', 'F115W']
                  }
    obs_ids = {'2221': {'brick': '001', 'cloudc': '002'},
               '1182': {'brick': '004'}}

    target = options.target
    blur = '_blur' if options.blur else ''

    project_ids = options.proposal_id.split(",")

    basepath = f'/orange/adamginsburg/jwst/{target}/'

    for oversampling, halfstampsize in [(1, 50), (2, 50), (4, 50), ]:
        for project_id in project_ids:
            for filtername in set(obs_filters[project_id]) & set(selected_filters):

                wavelength = int(filtername[1:-1])

                obs_id = obs_ids[project_id][target]

                outfilename = f'{basepath}/psfs/{filtername.upper()}_{project_id}_{obs_id}_merged_PSFgrid_oversample{oversampling}{blur}.fits'
                if options.redo or not os.path.exists(outfilename):
                    print(f"Making PSF grid {outfilename}")
                    psfg = make_merged_psf(filtername.upper(),
                                           basepath=basepath,
                                           halfstampsize=halfstampsize,
                                           grid_step=200 if wavelength > 230 else 400,
                                           oversampling=oversampling,
                                           blur=options.blur,
                                           project_id=project_id, obs_id=obs_id, suffix='merged_i2d')
                    save_psfgrid(psfg, outfilename=outfilename, overwrite=True)
                else:
                    print(f"PSF grid {outfilename} exists!  Fixing metadata if needed")
                    fix_psfs_with_bad_meta(outfilename)
