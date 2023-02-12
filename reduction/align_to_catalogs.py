import numpy as np
import os
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
import regions
from astroquery.vizier import Vizier
from astropy.visualization import quantity_support
from astropy.table import Table
import warnings

from astropy.wcs import WCS
from astropy.io import fits

import datetime

def print(*args, **kwargs):
    now = datetime.datetime.now().isoformat()
    from builtins import print as printfunc
    return printfunc(f"{now}:", *args, **kwargs)

def main():
    for filtername in ('f182m', 'f187n', 'f212n', 'f405n', 'f410m', 'f466n'):
        for module in ('nrca', 'nrcb'):

            print(filtername, module)

            realign_to_vvv(filtername=filtername, module=module)

def realign_to_vvv(
    basepath = '/orange/adamginsburg/jwst/brick/',
    filtername = 'f212n',
    module = 'nrca',
    imfile = None,
    catfile = None,
    fov_regname='/orange/adamginsburg/jwst/brick/regions/nircam_fov.reg',
):
    fov = regions.Regions.read(fov_regname)

    coord = fov[0].center
    height = fov[0].height
    width = fov[0].width
    height, width = width, height # CARTA wrote it wrong

    vvvdr2filename = f'{basepath}/{filtername.upper()}/pipeline/jw02221-o001_t001_nircam_clear-{filtername}-{module}_vvvcat.ecsv'

    if os.path.exists(vvvdr2filename):
        vvvdr2 = Table.read(vvvdr2filename)
    else:
        Vizier.ROW_LIMIT = 5e4
        vvvdr2 = Vizier.query_region(coordinates=coord, width=width, height=height, catalog=['II/348/vvv2'])[0]
        vvvdr2.write(vvvdr2filename, overwrite=True)

    # FK5 because it says 'J2000' on the Vizier page (same as twomass)
    vvvdr2_crds = SkyCoord(vvvdr2['RAJ2000'], vvvdr2['DEJ2000'], frame='fk5')

    return realign_to_catalog(vvvdr2_crds, filtername=filtername,
                              module=module, basepath=basepath,
                              catfile=catfile, imfile=imfile)


def realign_to_catalog(reference_coordinates, filtername='f212n',
                       module='nrca',
                       basepath='/orange/adamginsburg/jwst/brick/',
                       catfile=None, imfile=None):
    if catfile is None:
        catfile = f'{basepath}/{filtername.upper()}/pipeline/jw02221-o001_t001_nircam_clear-{filtername}-{module}_cat.ecsv'
    if imfile is None:
        imfile = f'{basepath}/{filtername.upper()}/pipeline/jw02221-o001_t001_nircam_clear-{filtername}-{module}_i2d.fits'

    cat = Table.read(catfile)

    # HACKETY HACK HACK filtering by flux
    flux = (cat['aper30_abmag'].value * u.ABmag).to(u.Jy)
    # 7e-8 is the empirical MJy/sr in one pixel-to-ABmag-flux conversion
    # it seems to hold for all of the fields, kinda?
    sel = (flux > 7e-8*500*u.Jy) & (flux < 4000*7e-8*u.Jy)

    skycrds_cat_orig = cat['sky_centroid']
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ww =  WCS(fits.getheader(imfile, ext=('SCI', 1)))
    skycrds_cat = ww.pixel_to_world(cat['xcentroid'], cat['ycentroid'])

    idx, sidx, sep, sep3d = reference_coordinates.search_around_sky(skycrds_cat[sel], 0.4*u.arcsec)
    dra = (skycrds_cat[sel][idx].ra - reference_coordinates[sidx].ra).to(u.arcsec)
    ddec = (skycrds_cat[sel][idx].dec - reference_coordinates[sidx].dec).to(u.arcsec)

    print(f'Before realignment, offset is {np.median(dra)}, {np.median(ddec)}')

    ww.wcs.crval = ww.wcs.crval - [np.median(dra).to(u.deg).value, np.median(ddec).to(u.deg).value]

    with fits.open(imfile, mode='update') as hdulist:
        print("CRVAL before", hdulist[1].header['CRVAL1'], hdulist[1].header['CRVAL2'])
        hdulist[1].header['OLCRVAL1'] = (hdulist[1].header['CRVAL1'], "Original CRVAL before ralign")
        hdulist[1].header['OLCRVAL2'] = (hdulist[1].header['CRVAL2'], "Original CRVAL before ralign")
        hdulist[1].header.update(ww.to_header())
        print("CRVAL after", hdulist[1].header['CRVAL1'], hdulist[1].header['CRVAL2'])

    # re-load the WCS to make sure it worked
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ww =  WCS(hdulist['SCI'].header)
    skycrds_cat_new = ww.pixel_to_world(cat['xcentroid'], cat['ycentroid'])

    idx, sidx, sep, sep3d = reference_coordinates.search_around_sky(skycrds_cat_new[sel], 0.2*u.arcsec)
    dra = (skycrds_cat_new[sel][idx].ra - reference_coordinates[sidx].ra).to(u.arcsec)
    ddec = (skycrds_cat_new[sel][idx].dec - reference_coordinates[sidx].dec).to(u.arcsec)

    print(f'After realignment, offset is {np.median(dra)}, {np.median(ddec)}')

    ww.wcs.crval = ww.wcs.crval - [np.median(dra).to(u.deg).value, np.median(ddec).to(u.deg).value]

    with fits.open(imfile, mode='update') as hdulist:
        print("CRVAL before", hdulist[1].header['CRVAL1'], hdulist[1].header['CRVAL2'])
        hdulist[1].header['OMCRVAL1'] = (hdulist[1].header['CRVAL1'], "Old median CRVAL")
        hdulist[1].header['OMCRVAL2'] = (hdulist[1].header['CRVAL2'], "Old median CRVAL")
        hdulist[1].header.update(ww.to_header())
        print("CRVAL after", hdulist[1].header['CRVAL1'], hdulist[1].header['CRVAL2'])

    # re-load the WCS to make sure it worked
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ww =  WCS(hdulist['SCI'].header)
    skycrds_cat_new = ww.pixel_to_world(cat['xcentroid'], cat['ycentroid'])

    idx, sidx, sep, sep3d = reference_coordinates.search_around_sky(skycrds_cat_new[sel], 0.2*u.arcsec)
    dra = (skycrds_cat_new[sel][idx].ra - reference_coordinates[sidx].ra).to(u.arcsec)
    ddec = (skycrds_cat_new[sel][idx].dec - reference_coordinates[sidx].dec).to(u.arcsec)

    print(f'After re-realignment, offset is {np.median(dra)}, {np.median(ddec)}')

    return hdulist

def merge_a_plus_b(filtername, 
    basepath = '/orange/adamginsburg/jwst/brick/',
    parallel=True,
    ):
    import reproject
    from reproject.mosaicking import find_optimal_celestial_wcs, reproject_and_coadd
    filename_nrca = f'{basepath}/{filtername.upper()}/pipeline/jw02221-o001_t001_nircam_clear-{filtername.lower()}-nrca_i2d.fits'
    filename_nrcb = f'{basepath}/{filtername.upper()}/pipeline/jw02221-o001_t001_nircam_clear-{filtername.lower()}-nrcb_i2d.fits'
    files = [filename_nrca, filename_nrcb]

    hdus = [fits.open(fn)[('SCI', 1)] for fn in files]
    ehdus = [fits.open(fn)[('ERR', 1)] for fn in files]
    weights = [fits.open(fn)[('WHT', 1)] for fn in files]

    # headers are only attached to the SCI frame for some reason!?
    for ehdu, hdu in zip(ehdus, hdus):
        ehdu.header.update(WCS(hdu).to_header())

    target_wcs, target_shape = find_optimal_celestial_wcs(hdus)
    merged, weightmap = reproject_and_coadd(hdus,
                                            output_projection=target_wcs,
                                            input_weights=weights,
                                            shape_out=target_shape,
                                            parallel=parallel,
                                            reproject_function=reproject.reproject_exact)
    merged_err, weightmap_ = reproject_and_coadd(ehdus,
                                                 output_projection=target_wcs,
                                                 input_weights=weights,
                                                 shape_out=target_shape,
                                                 parallel=parallel,
                                                 reproject_function=reproject.reproject_exact)
    header = fits.getheader(filename_nrca)
    header.update(target_wcs.to_header())
    hdul = fits.HDUList([fits.PrimaryHDU(header=header),
                         fits.ImageHDU(data=merged, name='SCI', header=header),
                         fits.ImageHDU(data=merged_err, name='ERR', header=header),
                         fits.ImageHDU(data=weightmap, name='WHT', header=header),
                        ])
    hdul.writeto(f'{basepath}/{filtername.upper()}/pipeline/jw02221-o001_t001_nircam_clear-{filtername.lower()}-merged-reproject_i2d.fits', overwrite=True)

def mihais_versin():
    """
    You can use other kinds of clipping in the `tweakwcs` package but not in the pipeline.

    The 'tweakreg' pipeline step relies on the 'tweakwcs' package
    (https://tweakwcs.readthedocs.io) that was designed to provide a lot of
    flexibility in how to align images and it should not be too difficult to use
    (though it _will_ require a little bit of effort, at least initially). My
    apologies also for lack of notebooks dedicated to JWST data. I will be also
    working on updating documentation the tweakwcs package and adding notebooks
    illustrating typical workflows. I would suggest that you look at the second
    example in this notebook:
    https://github.com/spacetelescope/tweakwcs/tree/master/notebooks that
    illustrates how align images to an external catalog (except that instead of
    FITSWCS you should import JWSTgWCS corrector class - about this later). Also,
    keep in mind that all units for shifts, searchrad, tolerance, etc. for JWST are
    in arcsec.

    One important concept in the tweakreg/tweakwcs is grouping of images for the
    purpose of alignment. For example, if a camera has multiple chips (i.e.,
    NIRCAM), then these chips are not supposed to move relative to each other and
    their relative positions should be known from calibration work and stored in
    reference files. For this reason all these images observed during *the same
    exposure* are aligned *together* to other input images. They are assigned to be
    part of the same group. This means that there will be computed only one affine
    transformation that will be applied to all images in a group. Once all these
    groups are aligned, the entire "mosaic" (or "panorama" using photographic terms)
    is aligned to Gaia (in tweakreg step) as *one* large group (images in the group
    do not move relative to each other).

    Here is an example of how to create tangent-plane correction object (needed as
    input for tweakwcs), create "groups", and align images to an external catalog.
    This example assumes that images' catalogs are astropy tables with 'x' and 'y'
    columns and the reference catalog is also an astropy table with 'RA' and 'DEC'
    columns. This example loads 4 images, loads image and reference catalogs,
    creates tangent plane corrector for each image, assigns images to groups, aligns
    them to the reference catalog, and updates model's WCS (GWCS and optionally FITS
    WCS):


    NOTE: Based on info I could get from your log file, your images are one image
    per group. Hence you should be assigning a unique number to `'group_id':
    unique_value` member.

    Please give 'tweakwcs' a try and do not hesitate to ask questions should you
    encounter any issues with this package.

    Unfortunately I am leaving tomorrow morning for a 2 weeks vacation and I will be
    traveling a lot and have limited connectivity otherwise.
    """


    from tweakwcs import JWSTgWCS, align_wcs
    from astropy.table import Table
    from jwst.datamodels import ImageModel

    # to update FITS WCS of the data models next import
    # requires installing dev version of the pipeline
    from jwst.assign_wcs.util import update_fits_wcsinfo

    # load 4 images:
    dm1 = ImageModel('image_model1.fits')
    dm2 = ImageModel('image_model2.fits')
    dm3 = ImageModel('image_model3.fits')
    dm4 = ImageModel('image_model4.fits')

    # load 4 image catalogs:
    imcat1 = Table.read('image1_catalog1.ecsv')
    imcat2 = Table.read('image1_catalog2.ecsv')
    imcat3 = Table.read('image1_catalog3.ecsv')
    imcat4 = Table.read('image1_catalog4.ecsv')

    # load reference catalog:
    refcat = Table.read('reference_catalog.ecsv')

    # create corrector objects. at the same time, assign
    # group ids that effectively group images into
    # alignment groups. For example, let's align images
    # 1 and 3 as one group and images 2 and 4 as another group.
    # ID value is arbitrary.

    corr1 = JWSTgWCS(dm1.meta.wcs, dm1.meta.wcsinfo.instance,
    meta={'catalog': imcat1, 'group_id': 1111})
    corr2 = JWSTgWCS(dm2.meta.wcs, dm2.meta.wcsinfo.instance,
    meta={'catalog': imcat2, 'group_id': 2022})
    corr3 = JWSTgWCS(dm3.meta.wcs, dm3.meta.wcsinfo.instance,
    meta={'catalog': imcat3, 'group_id': 1111})
    corr4 = JWSTgWCS(dm4.meta.wcs, dm4.meta.wcsinfo.instance,
    meta={'catalog': imcat4, 'group_id': 2022})

    # align images to the reference catalog:
    align_wcs([corr1, corr2, corr3, corr4], refcat=refcat) # + add other arguments as needed - see docs

    # upon return from align_wcs, corr*.wcs will contain updates GWCS objects.
    # assign corrected WCS back to the data models:
    dm1.meta.wcs = corr1.wcs
    update_fits_wcsinfo(dm1, npoints=16) # <- update FITS WCS too (optionally). Repeat for each model below
    dm2.meta.wcs = corr2.wcs
    update_fits_wcsinfo(dm2)
    dm3.meta.wcs = corr3.wcs
    update_fits_wcsinfo(dm3)
    dm4.meta.wcs = corr4.wcs
    update_fits_wcsinfo(dm4)

    # save image models to different files but one can also overwrite existing models:
    dm1.write('image_model1_corrected.fits')
    dm2.write('image_model2_corrected.fits')
    dm3.write('image_model3_corrected.fits')
    dm4.write('image_model4_corrected.fits')
