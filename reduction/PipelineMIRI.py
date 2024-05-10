#!/usr/bin/env python
from glob import glob
from astroquery.mast import Mast, Observations
import copy
import os
import re
import shutil
import numpy as np
import json
import requests
import asdf # requires asdf < 3.0 (there is no replacement for this functionality w/o a major pattern change https://github.com/asdf-format/asdf/issues/1680)
import stdatamodels
try:
    from asdf.fits_embed import AsdfInFits
except ImportError:
    from stdatamodels import asdf_in_fits as AsdfInFits
from astropy import log
from astropy.coordinates import SkyCoord
from astropy.io import ascii, fits
from astropy.table import Table
from astropy.utils.data import download_file
from astropy.wcs import WCS
from astropy.visualization import ImageNormalize, ManualInterval, LogStretch, LinearStretch
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime

# do this before importing webb
os.environ["CRDS_PATH"] = "/orange/adamginsburg/jwst/brick/crds/"
os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"

from jwst.pipeline import calwebb_image3
from jwst.pipeline import Detector1Pipeline, Image2Pipeline

# Individual steps that make up calwebb_image3
from jwst.tweakreg import TweakRegStep
from jwst.skymatch import SkyMatchStep
from jwst.outlier_detection import OutlierDetectionStep
from jwst.resample import ResampleStep
from jwst.source_catalog import SourceCatalogStep
from jwst import datamodels
from jwst.associations import asn_from_list
from jwst.associations.lib.rules_level3_base import DMS_Level3_Base
from jwst.tweakreg.utils import adjust_wcs
from jwst.datamodels import ImageModel

from align_to_catalogs import realign_to_vvv, realign_to_catalog, merge_a_plus_b, retrieve_vvv
from saturated_star_finding import iteratively_remove_saturated_stars, remove_saturated_stars

import crds
import jwst

filter_regex = re.compile('f[0-9][0-9][0-9][nmw]')

import warnings
from astropy.utils.exceptions import AstropyWarning, AstropyDeprecationWarning
from astropy.wcs import FITSFixedWarning
warnings.simplefilter('ignore', category=AstropyWarning)
warnings.simplefilter('ignore', category=AstropyDeprecationWarning)
warnings.simplefilter('ignore', category=FITSFixedWarning)

def print(*args, **kwargs):
    now = datetime.datetime.now().isoformat()
    from builtins import print as printfunc
    # redundant log.info(f"{now}: {' '.join(map(str, args))}",)
    return printfunc(f"{now}:", *args, **kwargs)


print(jwst.__version__)

fov_regname = {'brick': 'regions_/nircam_brick_fov.reg',
               'cloudc': 'regions_/nircam_cloudc_fov.reg',
               }


def main(filtername, Observations=None, regionname='brick',
         field='001', proposal_id='2221', skip_step1and2=False, use_average=True):
    """
    skip_step1and2 will not re-fit the ramps to produce the _cal images.  This
    can save time if you just want to redo the tweakreg steps but already have
    the zero-frame stuff done.
    """
    print(f"Processing filter {filtername} with and skip_step1and2={skip_step1and2} for field {field} and proposal id {proposal_id} in region {regionname}")

    wavelength = int(filtername[1:4])

    basepath = f'/orange/adamginsburg/jwst/{regionname}/'
    fwhm_tbl = Table.read(f'{basepath}/reduction/fwhm_table.ecsv')
    row = fwhm_tbl[fwhm_tbl['Filter'] == filtername]
    fwhm = fwhm_arcsec = float(row['PSF FWHM (arcsec)'][0])
    fwhm_pix = float(row['PSF FWHM (pixel)'][0])

    # sanity check
    if regionname == 'brick':
        if proposal_id == '2221':
            # jw02221-o002_t001_miri_f2550w_i2d.fits
            assert field == '002'
    elif regionname == 'cloudc':
        # jw02221-o001_t001_miri_f2550w_i2d.fits
        assert field == '001'

    os.environ["CRDS_PATH"] = f"{basepath}/crds/"
    os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"
    mpl.rcParams['savefig.dpi'] = 80
    mpl.rcParams['figure.dpi'] = 80



    # Files created in this notebook will be saved
    # in a subdirectory of the base directory called `Stage3`
    output_dir = f'/orange/adamginsburg/jwst/{regionname}/{filtername}/pipeline/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)

    # the files are one directory up
    for fn in glob("../*cal.fits"):
        try:
            os.link(fn, './'+os.path.basename(fn))
        except Exception as ex:
            print(f'Failed to link {fn} to {os.path.basename(fn)} because of {ex}')

    Observations.cache_location = output_dir
    obs_table = Observations.query_criteria(
                                            proposal_id=proposal_id,
                                            #proposal_pi="Ginsburg*",
                                            #calib_level=3,
                                            )
    print("Obs table length:", len(obs_table))

    msk = ((np.char.find(obs_table['filters'], filtername.upper()) >= 0) |
           (np.char.find(obs_table['obs_id'], filtername.lower()) >= 0))
    data_products_by_obs = Observations.get_product_list(obs_table[msk])
    print("data prodcts by obs length: ", len(data_products_by_obs))

    products_asn = Observations.filter_products(data_products_by_obs, extension="json")
    print("products_asn length:", len(products_asn))
    #valid_obsids = products_asn['obs_id'][np.char.find(np.unique(products_asn['obs_id']), 'jw02221-o001', ) == 0]
    #match = [x for x in valid_obsids if filtername.lower() in x][0]

    asn_mast_data = products_asn#[products_asn['obs_id'] == match]
    print("asn_mast_data:", asn_mast_data)

    manifest = Observations.download_products(asn_mast_data, download_dir=output_dir)
    print("manifest:", manifest)

    # MAST creates deep directory structures we don't want
    for row in manifest:
        try:
            shutil.move(row['Local Path'], os.path.join(output_dir, os.path.basename(row['Local Path'])))
        except Exception as ex:
            print(f"Failed to move file with error {ex}")

    products_fits = Observations.filter_products(data_products_by_obs, extension="fits")
    print("products_fits length:", len(products_fits))
    uncal_mask = np.array([uri.endswith('_uncal.fits') and f'jw0{proposal_id}{field}' in uri for uri in products_fits['dataURI']])
    uncal_mask &= products_fits['productType'] == 'SCIENCE'
    print("uncal length:", (uncal_mask.sum()))

    already_downloaded = np.array([os.path.exists(os.path.basename(uri)) for uri in products_fits['dataURI']])
    uncal_mask &= ~already_downloaded
    print(f"uncal to download: {uncal_mask.sum()}; {already_downloaded.sum()} were already downloaded")

    if uncal_mask.any():
        manifest = Observations.download_products(products_fits[uncal_mask], download_dir=output_dir)
        print("manifest:", manifest)

        # MAST creates deep directory structures we don't want
        for row in manifest:
            try:
                shutil.move(row['Local Path'], os.path.join(output_dir, os.path.basename(row['Local Path'])))
            except Exception as ex:
                print(f"Failed to move file with error {ex}")


    if True: # just to preserve indendation
        print(f"Working on MIRI: running initial pipeline setup steps (skip_step1and2={skip_step1and2})")
        print(f"Searching for {os.path.join(output_dir, f'jw0{proposal_id}-o{field}*_image3_*0[0-9][0-9]_asn.json')}")
        asn_file_search = glob(os.path.join(output_dir, f'jw0{proposal_id}-o{field}*_image3_*0[0-9][0-9]_asn.json'))
        if len(asn_file_search) == 1:
            asn_file = asn_file_search[0]
        elif len(asn_file_search) > 1:
            asn_file = sorted(asn_file_search)[-1]
            print(f"Found multiple asn files: {asn_file_search}.  Using the more recent one, {asn_file}.")
        else:
            raise ValueError(f"Mismatch: Did not find any asn files for field {field} in {output_dir}")

        mapping = crds.rmap.load_mapping(f'/orange/adamginsburg/jwst/{regionname}/crds/mappings/jwst/jwst_miri_pars-tweakregstep_0003.rmap')
        print(f"Mapping: {mapping.todict()['selections']}")
        print(f"Filtername: {filtername}")
        filter_match = [x for x in mapping.todict()['selections'] if filtername.upper() in x]
        print(f"Filter_match: {filter_match} n={len(filter_match)}")
        tweakreg_asdf_filename = filter_match[0][3]
        tweakreg_asdf = asdf.open(f'https://jwst-crds.stsci.edu/unchecked_get/references/jwst/{tweakreg_asdf_filename}')
        tweakreg_parameters = tweakreg_asdf.tree['parameters']
        # may not be needed for MIRI
        #tweakreg_parameters.update({'fitgeometry': 'general',
        #                            # brightest = 5000 was causing problems- maybe the cross-alignment was getting caught on PSF artifacts?
        #                            'brightest': 5000,
        #                            'snr_threshold': 20, # was 5, but that produced too many stars
        #                            # define later 'abs_refcat': abs_refcat,
        #                            'save_catalogs': True,
        #                            'catalog_format': 'fits',
        #                            'kernel_fwhm': fwhm_pix,
        #                            'nclip': 5,
        #                            # expand_refcat: A boolean indicating whether or not to expand reference catalog with new sources from other input images that have been already aligned to the reference image. (Default=False)
        #                            'expand_refcat': True,
        #                            # based on DebugReproduceTweakregStep
        #                            'sharplo': 0.3,
        #                            'sharphi': 0.9,
        #                            'roundlo': -0.25,
        #                            'roundhi': 0.25,
        #                            'separation': 0.5, # minimum separation; default is 1
        #                            'tolerance': 0.1, # tolerance: Matching tolerance for xyxymatch in arcsec. (Default=0.7)
        #                            'save_results': True,
        #                            # 'clip_accum': True, # https://github.com/spacetelescope/tweakwcs/pull/169/files
        #                            })



        print(f'Filter {filtername} tweakreg parameters: {tweakreg_parameters}')


        with open(asn_file) as f_obj:
            asn_data = json.load(f_obj)

        print(f"In cwd={os.getcwd()}")
        if not skip_step1and2:
            # re-calibrate all uncal files -> cal files *without* suppressing first group
            for member in asn_data['products'][0]['members']:
                # example filename: jw02221002001_02201_00002_nrcalong_cal.fits
                assert f'jw0{proposal_id}{field}' in member['expname']
                print(f"DETECTOR PIPELINE on {member['expname']}")
                print("Detector1Pipeline step")
                # from Hosek: expand_large_events -> false; turn off "snowball" detection
                Detector1Pipeline.call(member['expname'].replace("_cal.fits",
                                                                 "_uncal.fits"),
                                       save_results=True, output_dir=output_dir,
                                       save_calibrated_ramp=True,
                                       steps={'ramp_fit': {'suppress_one_group':False},
                                              "refpix": {"use_side_ref_pixels": True}})

                print(f"IMAGE2 PIPELINE on {member['expname']}")
                Image2Pipeline.call(member['expname'].replace("_cal.fits",
                                                              "_rate.fits"),
                                    save_results=True, output_dir=output_dir,
                                   )
        else:
            print("Skipped step 1 and step2")


    if True:
        print(f"Filter {filtername}: doing tweakreg.  ")

        with open(asn_file) as f_obj:
            asn_data = json.load(f_obj)
        asn_data['products'][0]['name'] = f'jw0{proposal_id}-o{field}_t001_miri_{filtername.lower()}'
        asn_data['products'][0]['members'] = [row for row in asn_data['products'][0]['members']]

        for member in asn_data['products'][0]['members']:
            print(f"Running  maybe alignment on {member}")
            hdr = fits.getheader(member['expname'])
            fname = member['expname']
            assert fname.endswith('_cal.fits')
            member['expname'] = fname.replace("_cal.fits", "_align.fits")
            shutil.copy(fname, member['expname'])

            fix_alignment(member['expname'], proposal_id=proposal_id,
                          field=field, basepath=basepath,
                          filtername=filtername, use_average=use_average)

        asn_file_each = asn_file
        with open(asn_file_each, 'w') as fh:
            json.dump(asn_data, fh)

        if True:
            abs_refcat = f'{basepath}/catalogs/crowdsource_based_nircam-f405n_reference_astrometric_catalog.ecsv'
            reftbl = Table.read(abs_refcat)
            # For non-F410M, try aligning to F410M instead of VVV?
            reftblversion = reftbl.meta['VERSION']
            reftbl.meta['name'] = 'F405N Reference Astrometric Catalog'

            # truncate to top 10,000 sources
            reftbl[:10000].write(f'{basepath}/catalogs/crowdsource_based_nircam-f405n_reference_astrometric_catalog_truncated10000.ecsv', overwrite=True)
            abs_refcat = f'{basepath}/catalogs/crowdsource_based_nircam-f405n_reference_astrometric_catalog_truncated10000.ecsv'

            tweakreg_parameters['abs_searchrad'] = 0.4
            # try forcing searchrad to be tighter to avoid bad crossmatches
            # (the raw data are very well-aligned to begin with, though CARTA
            # can't display them b/c they are using SIP)
            tweakreg_parameters['searchrad'] = 0.05
            print(f"Reference catalog is {abs_refcat} with version {reftblversion}")

        tweakreg_parameters.update({'abs_refcat': abs_refcat,})

        print(f"Running tweakreg")
        calwebb_image3.Image3Pipeline.call(
            asn_file_each,
            steps={'tweakreg': tweakreg_parameters,
                   # Skip skymatch: looks like it causes problems (but maybe not doing this is worse?)
                   #'skymatch': {'save_results': True, 'skip': True,
                   #             'skymethod': 'match', 'match_down': False},
            },
            output_dir=output_dir,
            save_results=True)
        print(f"DONE running {asn_file_each}")

        print("After tweakreg step, checking WCS headers:")
        for member in asn_data['products'][0]['members']:
            check_wcs(member['expname'])
            check_wcs(member['expname'].replace('cal', 'i2d').replace('destreak', 'i2d'))
        check_wcs(asn_data['products'][0]['name'] + "_i2d.fits")

        print(f"Realigning to VVV (filter={filtername})")
        realigned_vvv_filename = f'{basepath}/{filtername.upper()}/pipeline/jw0{proposal_id}-o{field}_t001_miri_{filtername.lower()}_realigned-to-vvv.fits'
        shutil.copy(f'{basepath}/{filtername.upper()}/pipeline/jw0{proposal_id}-o{field}_t001_miri_{filtername.lower()}_i2d.fits',
                    realigned_vvv_filename)
        catfile = f'{basepath}/{filtername.upper()}/pipeline/jw02221-o002_t001_miri_f2550w_cat.ecsv'
        print(f"Realigned to VVV filename: {realigned_vvv_filename}")
        realigned = realign_to_vvv(filtername=filtername.lower(),
                                   fov_regname=fov_regname[regionname],
                                   module='miri',
                                   basepath=basepath,
                                   catfile=catfile,
                                   fieldnumber=field, proposal_id=proposal_id,
                                   imfile=realigned_vvv_filename,
                                   ksmag_limit=11,
                                   mag_limit=15,
                                   max_offset=0.4*u.arcsec,
                                   #raoffset=raoffset,
                                   #decoffset=decoffset
                                   )
        print(f"Done realigning to VVV (filtername={filtername})")

        print(f"Realigning to refcat (filtername={filtername})")
        realigned_refcat_filename = f'{basepath}/{filtername.upper()}/pipeline/jw0{proposal_id}-o{field}_t001_miri_{filtername.lower()}_realigned-to-refcat.fits'
        shutil.copy(f'{basepath}/{filtername.upper()}/pipeline/jw0{proposal_id}-o{field}_t001_miri_{filtername.lower()}_i2d.fits',
                    realigned_refcat_filename)
        print(f"Realigned refcat filename: {realigned_refcat_filename}")
        realigned = realign_to_catalog(reftbl['skycoord'],
                                       filtername=filtername.lower(),
                                       basepath=basepath,
                                       fieldnumber=field,
                                       catfile=catfile,
                                       mag_limit=20, proposal_id=proposal_id,
                                       max_offset=0.4*u.arcsec,
                                       imfile=realigned_refcat_filename,
                                       #raoffset=raoffset, decoffset=decoffset
                                       )
        print(f"Done realigning to refcat (filtername={filtername})")


    globals().update(locals())
    return locals()

def fix_alignment(fn, proposal_id=None, field=None, basepath=None, filtername=None,
                  use_average=True):
    if os.path.exists(fn):
        print(f"Running manual align for data ({proposal_id} + {field}): {fn}", flush=True)
    else:
        print(f"Skipping manual align for nonexistent file ({proposal_id} + {field}): {fn}", flush=True)
        return

    mod = ImageModel(fn)
    if proposal_id is None:
        proposal_id = os.path.basename(fn)[3:7]
    if filtername is None:
        try:
            filtername = filter_regex.search(fn).group()
        except AttributeError:
            filters = tuple(map(str.lower, (mod.meta.instrument.filter, mod.meta.instrument.pupil)))
            if 'clear' in filters:
                filtername = [x for x in filters if x != 'clear'][0]
            else:
                # any filter that is not the wideband filter
                filtername = [x for x in filters if 'W' not in x][0]
    if field is None:
        field = mod.meta.observation.observation_number
    if basepath is None:
        basepath = f'/orange/adamginsburg/jwst/{field_name}'

    if (field == '004' and proposal_id == '1182') or (field == '001' and proposal_id == '2221'):
        exposure = int(fn.split("_")[-3])
        thismodule = fn.split("_")[-2]
        visit = fn.split("_")[0]
        if use_average:
            tblfn = f'{basepath}/offsets/Offsets_JWST_Brick{proposal_id}_VVV_average.csv'
            print(f"Using average offset table {tblfn}")
            offsets_tbl = Table.read(tblfn)
            match = (
                    ((offsets_tbl['Module'] == thismodule) |
                     (offsets_tbl['Module'] == thismodule.strip('1234'))) &
                    (offsets_tbl['Filter'] == filtername)
                    )
            row = offsets_tbl[match]
            print(f'Running manual align for merged for {filtername} {row["Module"][0]}.')
        else:
            tblfn = f'{basepath}/offsets/Offsets_JWST_Brick{proposal_id}.csv'
            print(f"Using offset table {tblfn}")
            offsets_tbl = Table.read(tblfn)
            match = ((offsets_tbl['Visit'] == visit) &
                    (offsets_tbl['Exposure'] == exposure) &
                    ((offsets_tbl['Module'] == thismodule) | (offsets_tbl['Module'] == thismodule.strip('1234'))) &
                    (offsets_tbl['Filter'] == filtername)
                    )
            row = offsets_tbl[match]
            print(f'Running manual align for merged for {filtername} {row["Group"][0]} {row["Module"][0]} {row["Exposure"][0]}.')
        if match.sum() != 1:
            raise ValueError(f"too many or too few matches for {fn} (match.sum() = {match.sum()}).  exposure={exposure}, thismodule={thismodule}, filtername={filtername}")
        rashift = float(row['dra (arcsec)'][0])*u.arcsec
        decshift = float(row['ddec (arcsec)'][0])*u.arcsec
    elif (field == '002' and proposal_id == '2221'):
        visit = fn.split('_')[0][-3:]
        thismodule = fn.split("_")[-2].strip('1234')
        if visit == '001':
            decshift = 7.95*u.arcsec
            rashift = 0.6*u.arcsec
        elif visit == '002':
            decshift = 3.85*u.arcsec
            rashift = 1.57*u.arcsec
        else:
            decshift = 0*u.arcsec
            rashift = 0*u.arcsec
    else:
        rashift = 0*u.arsec
        decshift = 0*u.arsec
    print(f"Shift for {fn} is {rashift}, {decshift}")
    align_fits = fits.open(fn)
    if 'RAOFFSET' in align_fits[1].header:
        # don't shift twice if we re-run
        print(f"{fn} is already aligned ({align_fits[1].header['RAOFFSET']}, {align_fits[1].header['DEOFFSET']})")
    else:
        # ASDF header
        fa = ImageModel(fn)
        wcsobj = fa.meta.wcs
        print(f"Before shift, crval={wcsobj.to_fits()[0]['CRVAL1']}, {wcsobj.to_fits()[0]['CRVAL2']}, {wcsobj.forward_transform.param_sets[-1]}")
        fa.meta.oldwcs = copy.copy(wcsobj)
        ww = adjust_wcs(wcsobj, delta_ra=rashift, delta_dec=decshift)
        print(f"After shift, crval={ww.to_fits()[0]['CRVAL1']}, {ww.to_fits()[0]['CRVAL2']}, {wcsobj.forward_transform.param_sets[-1]}")
        fa.meta.wcs = ww
        fa.save(fn, overwrite=True)

        # FITS header
        align_fits = fits.open(fn)
        align_fits[1].header['OLCRVAL1'] = align_fits[1].header['CRVAL1']
        align_fits[1].header['OLCRVAL2'] = align_fits[1].header['CRVAL2']
        align_fits[1].header.update(ww.to_fits()[0])
        align_fits[1].header['RAOFFSET'] = rashift.value
        align_fits[1].header['DEOFFSET'] = decshift.value
        align_fits.writeto(fn, overwrite=True)
        assert 'RAOFFSET' in fits.getheader(fn, ext=1)
    check_wcs(fn)


def check_wcs(fn):
    if os.path.exists(fn):
        print(f"Checking WCS of {fn}")
        fa = ImageModel(fn)
        wcsobj = fa.meta.wcs
        print(f"fa['meta']['wcs'] crval={wcsobj.to_fits()[0]['CRVAL1']}, {wcsobj.to_fits()[0]['CRVAL2']}, {wcsobj.forward_transform.param_sets[-1]}")
        new_1024 = wcsobj.pixel_to_world(1024, 1024)
        print(f"new pixel_to_world(1024,1024) = {new_1024}")
        if 'oldwcs' in fa.meta:
            oldwcsobj = fa.meta.oldwcs
            print(f"fa['meta']['oldwcs'] crval={oldwcsobj.to_fits()[0]['CRVAL1']}, {oldwcsobj.to_fits()[0]['CRVAL2']}, {oldwcsobj.forward_transform.param_sets[-1]}")
            old_1024 = oldwcsobj.pixel_to_world(1024, 1024)
            print(f"old pixel_to_world(1024,1024) = {old_1024}, sep from new GWCS={old_1024.separation(new_1024).to(u.arcsec)}")
        fa.close()


        # FITS header
        fh = fits.open(fn)
        print(f"CRVAL1={fh[1].header['CRVAL1']}, CRVAL2={fh[1].header['CRVAL2']}")
        if 'OLCRVAL1' in fh[1].header:
            print(f"OLCRVAL1={fh[1].header['OLCRVAL1']}, OLCRVAL2={fh[1].header['OLCRVAL2']}")
        if 'RAOFFSET' in fh[1].header:
            print("RA, DE offset: ", fh[1].header['RAOFFSET'], fh[1].header['DEOFFSET'])
        ww = WCS(fh[1].header)
        fits_1024 = ww.pixel_to_world(1024, 1024)
        print(f"FITS pixel_to_world(1024,1024) = {fits_1024}, sep from new GWCS={fits_1024.separation(new_1024).to(u.arcsec)}")
        fh.close()
    else:
        print(f"COULD NOT CHECK WCS FOR {fn}: does not exist")

if __name__ == "__main__":
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-f", "--filternames", dest="filternames",
                      default='F2550W',
                      help="filter name list", metavar="filternames")
    parser.add_option("-d", "--field", dest="field",
                    default='002',
                    help="list of target fields", metavar="field")
    parser.add_option("-s", "--skip_step1and2", dest="skip_step1and2",
                      default=False,
                      action='store_true',
                      help="Skip the image-remaking step?", metavar="skip_Step1and2")
    parser.add_option("-p", "--proposal_id", dest="proposal_id",
                      default='2221',
                      help="proposal id (string)", metavar="proposal_id")
    (options, args) = parser.parse_args()

    filternames = options.filternames.split(",")
    fields = options.field.split(",")
    proposal_id = options.proposal_id
    skip_step1and2 = options.skip_step1and2
    print(options)

    with open(os.path.expanduser('~/.mast_api_token'), 'r') as fh:
        api_token = fh.read().strip()
        os.environ['MAST_API_TOKEN'] = api_token.strip()
    Mast.login(api_token.strip())
    Observations.login(api_token)


    field_to_reg_mapping = {'2221': {'002': 'brick', '001': 'cloudc'}, }[proposal_id]

    for field in fields:
        for filtername in filternames:
            print(f"Main Loop: {proposal_id} + {filtername} + {field}={field_to_reg_mapping[field]}")
            results = main(filtername=filtername, Observations=Observations, field=field,
                           regionname=field_to_reg_mapping[field],
                           proposal_id=proposal_id,
                           skip_step1and2=skip_step1and2,
                          )


