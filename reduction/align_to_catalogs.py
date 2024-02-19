import numpy as np
import shutil
import os
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
import regions
from astroquery.vizier import Vizier
from astropy.visualization import quantity_support
from astropy import log
from astropy.table import Table
import warnings
from jwst.datamodels import ImageModel

from astropy.wcs import WCS
from astropy.io import fits

import datetime


def diagnostic_plots(fn, refcrds, meascrds, dra, ddec, savename=None):
    import pylab as pl
    from astropy.visualization import simple_norm
    fig = pl.figure(dpi=200)
    ax1 = pl.subplot(2, 2, 1)
    ra = meascrds.ra
    dec = meascrds.dec
    ax1.quiver(ra.to(u.deg).value, dec.to(u.deg).value, dra.to(u.arcsec).value, ddec.to(u.arcsec).value)

    img = ImageModel(fn)
    ww = img.meta.wcs
    ax2 = pl.subplot(2, 2, 2, projection=ww)
    ax2.imshow(img.data, cmap='gray_r', norm=simple_norm(img.data, min_percent=1, max_percent=99, stretch='asinh'))
    ax2.scatter_coord(refcrds, marker='x', color='r')

    ax3 = pl.subplot(2, 2, 3, projection=ww)
    ax3.imshow(img.data, cmap='gray_r', norm=simple_norm(img.data, min_percent=1, max_percent=99, stretch='asinh'))
    ax3.scatter_coord(refcrds, marker='x', color='r')
    ax3.scatter_coord(meascrds, marker='+', color='b')
    ax3.axis([1000,1200,1000,1200])

    ax4 = pl.subplot(2, 2, 4, projection=ww)
    ax4.imshow(img.data, cmap='gray_r', norm=simple_norm(img.data, min_percent=1, max_percent=99, stretch='asinh'))
    ax4.scatter_coord(refcrds, marker='x', color='r')
    ax4.scatter_coord(meascrds, marker='+', color='b')
    ax4.axis([200,400,200,400])

    if savename is not None:
        pl.tight_layout()
        pl.savefig(savename, bbox_inches='tight')

def print(*args, **kwargs):
    now = datetime.datetime.now().isoformat()
    from builtins import print as printfunc
    log.info(f"{now}: {' '.join(map(str, args))}")
    return printfunc(f"{now}:", *args, **kwargs)

def main(field='001',
         basepath = '/orange/adamginsburg/jwst/brick/',
         proposal_id='2221',
        ):
    for filtername in ( 'f405n', 'f410m', 'f466n', 'f182m', 'f187n', 'f212n',):
        print()
        print(f"Filter = {filtername}")
        for module in ('nrca', 'nrcb', 'merged', ): #'merged-reproject'):
            # merged-reproject shouldn't need realignment b/c it should be made from realigned images

            print(filtername, module)
            log.info(f"Realigning to vvv (module={module}")

            realigned_vvv_filename = f'{basepath}/{filtername.upper()}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_clear-{filtername.lower()}-{module}_realigned-to-vvv.fits'
            shutil.copy(f'{basepath}/{filtername.upper()}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_clear-{filtername.lower()}-{module}_i2d.fits',
                        realigned_vvv_filename)
            realigned = realign_to_vvv(filtername=filtername.lower(),
                                    basepath=basepath, module=module, fieldnumber=field,
                                    imfile=realigned_vvv_filename, ksmag_limit=15 if filtername=='f410m'
                                    else 11, mag_limit=17 if filtername == 'F115W' else 15, proposal_id=proposal_id)

            log.info(f"Realigning to refcat (module={module}")

            abs_refcat = f'{basepath}/catalogs/crowdsource_based_nircam-f405n_reference_astrometric_catalog.ecsv'
            reftbl = Table.read(abs_refcat)

            realigned_refcat_filename = f'{basepath}/{filtername.upper()}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_clear-{filtername.lower()}-{module}_realigned-to-refcat.fits'
            shutil.copy(f'{basepath}/{filtername.upper()}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_clear-{filtername.lower()}-{module}_i2d.fits',
                        realigned_refcat_filename)
            realigned = realign_to_catalog(reftbl['skycoord'],
                                           filtername=filtername.lower(),
                                           basepath=basepath, module=module,
                                           fieldnumber=field,
                                           mag_limit=20,
                                           proposal_id=proposal_id,
                                           imfile=realigned_refcat_filename)


def retrieve_vvv(
    basepath = '/orange/adamginsburg/jwst/brick/',
    filtername = 'f212n',
    proposal_id='2221',
    module = 'nrca',
    imfile = None,
    catfile = None,
    fov_regname='regions/nircam_brick_fov.reg',
    fieldnumber='001',
):
    fov = regions.Regions.read(os.path.join(basepath, fov_regname))

    coord = fov[0].center
    height = fov[0].height
    width = fov[0].width
    height, width = width, height # CARTA wrote it wrong

    vvvdr2filename = f'{basepath}/{filtername.upper()}/pipeline/jw0{proposal_id}-o{fieldnumber}_t001_nircam_clear-{filtername}-{module}_vvvcat.ecsv'

    if os.path.exists(vvvdr2filename):
        vvvdr2 = Table.read(vvvdr2filename)
        vvvdr2['RA'] = vvvdr2['RAJ2000']
        vvvdr2['DEC'] = vvvdr2['DEJ2000']

        # FK5 because it says 'J2000' on the Vizier page (same as twomass)
        vvvdr2_crds = SkyCoord(vvvdr2['RAJ2000'], vvvdr2['DEJ2000'], frame='fk5')
        if 'skycoord' not in vvvdr2.colnames:
            vvvdr2['skycoord'] = vvvdr2_crds
            vvvdr2.write(vvvdr2filename, overwrite=True)
            vvvdr2.write(vvvdr2filename.replace(".ecsv", ".fits"), overwrite=True)
    else:
        Vizier.ROW_LIMIT = 5e4
        vvvdr2 = Vizier.query_region(coordinates=coord, width=width, height=height, catalog=['II/348/vvv2'])[0]
        vvvdr2['RA'] = vvvdr2['RAJ2000']
        vvvdr2['DEC'] = vvvdr2['DEJ2000']

        # FK5 because it says 'J2000' on the Vizier page (same as twomass)
        vvvdr2_crds = SkyCoord(vvvdr2['RAJ2000'], vvvdr2['DEJ2000'], frame='fk5')
        vvvdr2['skycoord'] = vvvdr2_crds

        vvvdr2.write(vvvdr2filename, overwrite=True)
        vvvdr2.write(vvvdr2filename.replace(".ecsv", ".fits"), overwrite=True)

    assert 'skycoord' in vvvdr2.colnames
    return vvvdr2_crds, vvvdr2

def realign_to_vvv(
    basepath = '/orange/adamginsburg/jwst/brick/',
    filtername = 'f212n',
    module = 'nrca',
    imfile = None,
    catfile = None,
    fov_regname='regions/nircam_brick_fov.reg',
    fieldnumber='001',
    proposal_id='2221',
    ksmag_limit=15,
    mag_limit=15,
    max_offset=0.4*u.arcsec,
    raoffset=0*u.arcsec, decoffset=0*u.arcsec,
):
    """
    ksmag_limit is a *lower* limit (we want fainter sources from VVV), while mag_limit is an *upper limit* - we want brighter sources from JWST
    """

    vvvdr2_crds, vvvdr2 = retrieve_vvv(basepath=basepath, filtername=filtername, module=module, fov_regname=fov_regname, fieldnumber=fieldnumber)

    if ksmag_limit:
        ksmag_sel = vvvdr2['Ksmag3'] > ksmag_limit
        log.info(f"Kept {ksmag_sel.sum()} out of {len(vvvdr2)} VVV stars using ksmag_limit>{ksmag_limit}")
        vvvdr2_crds = vvvdr2_crds[ksmag_sel]

    return realign_to_catalog(vvvdr2_crds, filtername=filtername,
                              module=module, basepath=basepath,
                              fieldnumber=fieldnumber,
                              catfile=catfile, imfile=imfile,
                              mag_limit=mag_limit,
                              max_offset=max_offset,
                              raoffset=raoffset, decoffset=decoffset,
                              proposal_id=proposal_id,
                              )


def realign_to_catalog(reference_coordinates, filtername='f212n',
                       module='nrca',
                       basepath='/orange/adamginsburg/jwst/brick/',
                       fieldnumber='001',
                       proposal_id='2221',
                       max_offset=0.4*u.arcsec,
                       mag_limit=15,
                       catfile=None, imfile=None,
                       threshold=0.001*u.arcsec,
                       raoffset=0*u.arcsec, decoffset=0*u.arcsec):
    if catfile is None:
        catfile = f'{basepath}/{filtername.upper()}/pipeline/jw0{proposal_id}-o{fieldnumber}_t001_nircam_clear-{filtername}-{module}_cat.ecsv'
        print(f"Catalog file was None, so defaulting to {catfile}")
    if imfile is None:
        imfile = f'{basepath}/{filtername.upper()}/pipeline/jw0{proposal_id}-o{fieldnumber}_t001_nircam_clear-{filtername}-{module}_i2d.fits'
        print(f"imfile file was None, so defaulting to {imfile}")

    cat = Table.read(catfile)

    # HACKETY HACK HACK filtering by flux
    # 7e-8 is the empirical MJy/sr in one pixel-to-ABmag-flux conversion
    # it seems to hold for all of the fields, kinda?
    #sel = (flux > 7e-8*500*u.Jy) & (flux < 4000*7e-8*u.Jy)

    # Manual checking in CARTA: didn't look like any good matches at mag>15
    mag = cat['aper_total_vegamag']
    sel = mag < mag_limit
    log.info(f"For {filtername} {module} {fieldnumber} catalog {catfile}, found {sel.sum()} of {sel.size} sources meeting criteria mag<{mag_limit}")

    if sel.sum() == 0:
        print(f"min mag: {np.nanmin(mag)}, max mag: {np.nanmax(mag)}")
        raise ValueError("No sources passed basic selection criteria")

    # don't trust the sky coords, recompute them from the current WCS (otherwise we can double-update)
    # skycrds_cat_orig = cat['sky_centroid']
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ww = WCS(fits.getheader(imfile, ext=('SCI', 1)))
        skycrds_cat_orig = ww.pixel_to_world(cat['xcentroid'], cat['ycentroid'])
        ww.wcs.crval = ww.wcs.crval - [raoffset.to(u.deg).value, decoffset.to(u.deg).value] # visualize this adjustment separately from next, find out which step is wrong

    med_dra = 100*u.arcsec
    med_ddec = 100*u.arcsec
    iteration = 0
    while np.abs(med_dra) > threshold or np.abs(med_ddec) > threshold:
        skycrds_cat = ww.pixel_to_world(cat['xcentroid'], cat['ycentroid'])

        idx, sidx, sep, sep3d = reference_coordinates.search_around_sky(skycrds_cat[sel], max_offset)
        dra = (skycrds_cat[sel][idx].ra - reference_coordinates[sidx].ra).to(u.arcsec)
        ddec = (skycrds_cat[sel][idx].dec - reference_coordinates[sidx].dec).to(u.arcsec)

        med_dra = np.median(dra)
        med_ddec = np.median(ddec)

        print(f'At realignment iteration {iteration}, offset is {med_dra}, {med_ddec}.  Found {len(idx)} matches.')
        iteration += 1

        if np.isnan(med_dra):
            print(f'len(refcoords) = {len(reference_coordinates)}')
            print(f'len(cat) = {len(cat)}')
            print(f'len(idx) = {len(idx)}')
            print(f'len(sidx) = {len(sidx)}')
            print(cat, sel, idx, sidx, sep)
            raise ValueError(f"median(dra) = {med_dra}.  np.nanmedian(dra) = {np.nanmedian(dra)}")

        ww.wcs.crval = ww.wcs.crval - [med_dra.to(u.deg).value, med_ddec.to(u.deg).value]

    with fits.open(imfile, mode='update') as hdulist:
        print("CRVAL before", hdulist['SCI'].header['CRVAL1'], hdulist['SCI'].header['CRVAL2'])
        hdulist['SCI'].header['OLCRVAL1'] = (hdulist['SCI'].header['CRVAL1'], "Original CRVAL before ralign")
        hdulist['SCI'].header['OLCRVAL2'] = (hdulist['SCI'].header['CRVAL2'], "Original CRVAL before ralign")
        hdulist['SCI'].header.update(ww.to_header())
        print("CRVAL after", hdulist['SCI'].header['CRVAL1'], hdulist['SCI'].header['CRVAL2'])

    # re-load the WCS to make sure it worked
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ww =  WCS(hdulist['SCI'].header)
    skycrds_cat_new = ww.pixel_to_world(cat['xcentroid'], cat['ycentroid'])

    idx, sidx, sep, sep3d = reference_coordinates.search_around_sky(skycrds_cat_new[sel], max_offset)
    dra = (skycrds_cat_new[sel][idx].ra - reference_coordinates[sidx].ra).to(u.arcsec)
    ddec = (skycrds_cat_new[sel][idx].dec - reference_coordinates[sidx].dec).to(u.arcsec)

    pngname = f'{basepath}/{filtername.upper()}/pipeline/jw0{proposal_id}-o{fieldnumber}_t001_nircam_clear-{filtername}-{module}_xmatch_diagnostics.png'
    diagnostic_plots(imfile, reference_coordinates, skycrds_cat_new[sel][idx], dra, ddec, savename=pngname)

    print(f'After realignment, offset is {np.median(dra)}, {np.median(ddec)} with {len(idx)} matches')

    # redundant
    # ww.wcs.crval = ww.wcs.crval - [np.median(dra).to(u.deg).value, np.median(ddec).to(u.deg).value]
    # with fits.open(imfile, mode='update') as hdulist:
    #     print("CRVAL before", hdulist['SCI'].header['CRVAL1'], hdulist['SCI'].header['CRVAL2'])
    #     hdulist['SCI'].header['OMCRVAL1'] = (hdulist['SCI'].header['CRVAL1'], "Old median CRVAL")
    #     hdulist['SCI'].header['OMCRVAL2'] = (hdulist['SCI'].header['CRVAL2'], "Old median CRVAL")
    #     hdulist['SCI'].header.update(ww.to_header())
    #     print("CRVAL after", hdulist['SCI'].header['CRVAL1'], hdulist['SCI'].header['CRVAL2'])


    # re-reload the file by reading from disk, with non-update mode
    # this is a double-double-check that the solution was written to disk
    with fits.open(imfile) as hdulist:
        # re-load the WCS to make sure it worked
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ww =  WCS(hdulist['SCI'].header)
        skycrds_cat_new = ww.pixel_to_world(cat['xcentroid'], cat['ycentroid'])

        idx, sidx, sep, sep3d = reference_coordinates.search_around_sky(skycrds_cat_new[sel], max_offset)
        dra = (skycrds_cat_new[sel][idx].ra - reference_coordinates[sidx].ra).to(u.arcsec)
        ddec = (skycrds_cat_new[sel][idx].dec - reference_coordinates[sidx].dec).to(u.arcsec)

        print(f'After re-realignment, offset is {np.median(dra)}, {np.median(ddec)} using {len(idx)} matches')

    return hdulist

def merge_a_plus_b(filtername,
    basepath = '/orange/adamginsburg/jwst/brick/',
    parallel=True,
    fieldnumber='001',
    proposal_id='2221',
    suffix='realigned-to-vvv',
    outsuffix='merged-reproject'
    ):
    """suffix can be realigned-to-vvv, realigned-to-refcat, or i2d"""
    import reproject
    from reproject.mosaicking import find_optimal_celestial_wcs, reproject_and_coadd
    filename_nrca = f'{basepath}/{filtername.upper()}/pipeline/jw0{proposal_id}-o{fieldnumber}_t001_nircam_clear-{filtername.lower()}-nrca{suffix}.fits'
    filename_nrcb = f'{basepath}/{filtername.upper()}/pipeline/jw0{proposal_id}-o{fieldnumber}_t001_nircam_clear-{filtername.lower()}-nrcb{suffix}.fits'
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
    outfn = f'{basepath}/{filtername.upper()}/pipeline/jw0{proposal_id}-o{fieldnumber}_t001_nircam_clear-{filtername.lower()}-{outsuffix}_i2d.fits'
    hdul.writeto(outfn, overwrite=True)
    return outfn