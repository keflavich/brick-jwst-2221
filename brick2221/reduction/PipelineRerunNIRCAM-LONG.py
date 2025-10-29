#!/usr/bin/env python
import os
# do this before importing webb
os.environ["CRDS_PATH"] = "/orange/adamginsburg/jwst/w51/crds/"
os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"

from glob import glob
from astroquery.mast import Mast, Observations
import copy
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

from destreak import destreak

#from align_to_catalogs import realign_to_vvv, realign_to_catalog, merge_a_plus_b, retrieve_vvv
#from saturated_star_finding import iteratively_remove_saturated_stars, remove_saturated_stars

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



def merge_a_plus_b(filtername,
    basepath = '/orange/adamginsburg/jwst/w51/',
    parallel=True,
    fieldnumber='001',
    proposal_id='6151',
    suffix='realigned-to-vvv',
    outsuffix='merged-reproject'
    ):
    """suffix can be realigned-to-vvv, realigned-to-refcat, or i2d"""
    import reproject
    from reproject.mosaicking import find_optimal_celestial_wcs, reproject_and_coadd
    filename_nrca = f'{basepath}{filtername.upper()}/pipeline/jw0{proposal_id}-o{fieldnumber}_t001_nircam_clear-{filtername.lower()}-nrca{suffix}.fits'
    filename_nrcb = f'{basepath}{filtername.upper()}/pipeline/jw0{proposal_id}-o{fieldnumber}_t001_nircam_clear-{filtername.lower()}-nrcb{suffix}.fits'
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
    outfn = f'{basepath}{filtername.upper()}/pipeline/jw0{proposal_id}-o{fieldnumber}_t001_nircam_clear-{filtername.lower()}-{outsuffix}_i2d.fits'
    hdul.writeto(outfn, overwrite=True)
    return outfn




print(jwst.__version__)

# see 'destreak410.ipynb' for tests of this
medfilt_size = {'F410M': 15, 'F405N': 256, 'F466N': 55,
                'F182M': 55, 'F187N': 512, 'F212N': 512}

fov_regname = {'brick': 'regions_/nircam_brick_fov.reg',
               'cloudc': 'regions_/nircam_cloudc_fov.reg',
               }

refnames = {'2221': 'F405ref',
            '1182': 'VVV',
            '6151': 'W51'
            }

# it's very difficult to modify the Webb pipeline in this way
# # replace Image2Pipeline's 'resample' with one that uses our hand-corrected coordinates
# def pre_resample(func):
#   def wrapper(self, input, *args, **kwargs):
#     print("Before resample is called, fixing coordinates")
#     for member in inputs:
#         print(f"Fixing alignment for {member.meta.filename}")
#         fix_alignment(member.meta.filename)
#     result = func(*args, **kwargs)
#     return result
#   return wrapper
#
# Image2Pipeline.step_defs['resample'] = pre_resample(Image2Pipeline.resample)


def main(filtername, module, Observations=None, regionname='w51', do_destreak=False,
         field='001', proposal_id='6151', skip_step1and2=False, use_average=True):
    """
    skip_step1and2 will not re-fit the ramps to produce the _cal images.  This
    can save time if you just want to redo the tweakreg steps but already have
    the zero-frame stuff done.
    """
    print(f"Processing filter {filtername} module {module} with do_destreak={do_destreak} and skip_step1and2={skip_step1and2} for field {field} and proposal id {proposal_id} in region {regionname}")

    wavelength = int(filtername[1:4])

    basepath = f'/orange/adamginsburg/jwst/{regionname}/'
    print(basepath)
    fwhm_tbl = Table.read(f'{basepath}/reduction/fwhm_table.ecsv')
    row = fwhm_tbl[fwhm_tbl['Filter'] == filtername]
    fwhm = fwhm_arcsec = float(row['PSF FWHM (arcsec)'][0])
    fwhm_pix = float(row['PSF FWHM (pixel)'][0])

    destreak_suffix = '' if do_destreak else '_nodestreak'

    # sanity check
    if regionname == 'sgrb2':
        if proposal_id == '5365':
            assert field == '001'
    if regionname == 'w51':
        if proposal_id == '6151':
            assert field == '001'

    os.environ["CRDS_PATH"] = f"{basepath}crds/"
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
                                           
                                            )
    
   
    if 'filters' in obs_table.colnames:
        filters = obs_table['filters'].filled('')
        obs_id = obs_table['obs_id'].filled('')  # Replace masked values with an empty string
        msk = ((np.char.find(filters, filtername.upper()) >= 0) |
           (np.char.find(obs_id, filtername.lower()) >= 0))
    else:
        print("Warning: 'filters' column not found in obs_table")
  # msk = np.char.find(obs_table['obs_id'], filtername.lower()) >= 0

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


    # all cases, except if you're just doing a merger?
    if module in ('nrca', 'nrcb', 'merged'):
        print(f"Working on module {module}: running initial pipeline setup steps (skip_step1and2={skip_step1and2})")
        print(f"Searching for {os.path.join(output_dir, f'jw0{proposal_id}-o{field}*_image3_*0[0-9][0-9]_asn.json')}")
        asn_file_search = glob(os.path.join(output_dir, f'jw0{proposal_id}-o{field}*_image3_*0[0-9][0-9]_asn.json'))
        if len(asn_file_search) == 1:
            asn_file = asn_file_search[0]
        elif len(asn_file_search) > 1:
            asn_file = sorted(asn_file_search)[-1]
            print(f"Found multiple asn files: {asn_file_search}.  Using the more recent one, {asn_file}.")
        else:
            raise ValueError(f"Mismatch: Did not find any asn files for module {module} for field {field} in {output_dir}")

        mapping = crds.rmap.load_mapping(f'/orange/adamginsburg/jwst/{regionname}/crds/mappings/jwst/jwst_nircam_pars-tweakregstep_0003.rmap')
        print(f"Mapping: {mapping.todict()['selections']}")
        print(f"Filtername: {filtername}")
        filter_match = [x for x in mapping.todict()['selections'] if filtername in x]
        print(f"Filter_match: {filter_match} n={len(filter_match)}")
        tweakreg_asdf_filename = filter_match[0][4]
        tweakreg_asdf = asdf.open(f'https://jwst-crds.stsci.edu/unchecked_get/references/jwst/{tweakreg_asdf_filename}')
        tweakreg_parameters = tweakreg_asdf.tree['parameters']
        tweakreg_parameters.update({'skip': True,
                                    'fitgeometry': 'general',
                                    # brightest = 5000 was causing problems- maybe the cross-alignment was getting caught on PSF artifacts?
                                    'brightest': 5000,
                                    'snr_threshold': 20, # was 5, but that produced too many stars
                                    # define later 'abs_refcat': abs_refcat,
                                    'save_catalogs': True,
                                    'catalog_format': 'fits',
                                    'kernel_fwhm': fwhm_pix,
                                    'nclip': 5,
                                    #'starfinder': 'dao',
                                    # expand_refcat: A boolean indicating whether or not to expand reference catalog with new sources from other input images that have been already aligned to the reference image. (Default=False)
                                    'expand_refcat': True,
                                    # based on DebugReproduceTweakregStep
                                    'sharplo': 0.3,
                                    'sharphi': 0.9,
                                    'roundlo': -0.25,
                                    'roundhi': 0.25,
                                    'separation': 0.5, # minimum separation; default is 1
                                    'tolerance': 0.1, # tolerance: Matching tolerance for xyxymatch in arcsec. (Default=0.7)
                                    'save_results': True,
                                    # 'clip_accum': True, # https://github.com/spacetelescope/tweakwcs/pull/169/files
                                    })

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
                print('diagnostics:')
                print(member['expname'])
                print(member['expname'].replace("_cal.fits", "_uncal.fits"))
                
                Detector1Pipeline.call(member['expname'].replace("_cal.fits",
                                                                 "_uncal.fits"),
                                       save_results=True, output_dir=output_dir,
                                       save_calibrated_ramp=True,
                                       steps={'ramp_fit': {'suppress_one_group':False, 'save_results':True},
                                              "refpix": {"use_side_ref_pixels": True},
                                              "jump":{"save_results":True}})

                # apparently "rate" files have no WCS, but this is where it's needed...
                # print("Aligning RATE images before doing IMAGE2 pipeline")
                # for member in asn_data['products'][0]['members']:
                #     align_image = member['expname'].replace("_cal.fits", "_rate.fits")
                #     fix_alignment(align_image, proposal_id=proposal_id, module=module, field=field, basepath=basepath, filtername=filtername)
                #else:
                #    print(f"Field {field} proposal {proposal_id} did not require re-alignment")
                print(f"IMAGE2 PIPELINE on {member['expname']}")
                Image2Pipeline.call(member['expname'].replace("_cal.fits",
                                                              "_rate.fits"),
                                    save_results=True, output_dir=output_dir,
                                   )
        else:
            print("Skipped step 1 and step2")

        # don't need to do this / it affects Savannah's fixing approach
        #print("Doing pre-alignment from offsets tables")
        #for member in asn_data['products'][0]['members']:
        #    if (field == '004' and proposal_id == '1182') or ((field == '001' or field  == '002') and proposal_id == '2221'):
        #        for suffix in ("_cal.fits", "_destreak.fits"):
        #            align_image = member['expname'].replace("_cal.fits", suffix)
        #            fix_alignment(align_image, proposal_id=proposal_id, module=module, field=field, basepath=basepath, filtername=filtername)
        #    else:
        #        print(f"Field {field} proposal {proposal_id} did not require re-alignment")


    else:
        raise ValueError(f"Module is {module} - not allowed!")

    if module in ('nrca', 'nrcb'):
        print(f"Filter {filtername} module {module}: doing tweakreg.  do_destreak={do_destreak}")

        with open(asn_file) as f_obj:
            asn_data = json.load(f_obj)
        asn_data['products'][0]['name'] = f'jw0{proposal_id}-o{field}_t001_nircam_clear-{filtername.lower()}-{module}'
        asn_data['products'][0]['members'] = [row for row in asn_data['products'][0]['members']
                                                if f'{module}' in row['expname']]

        for member in asn_data['products'][0]['members']:
            print(f"Running destreak={do_destreak} and maybe alignment on {member} for module={module}")
            hdr = fits.getheader(member['expname'])
            do_destreak=False
            if do_destreak:
                if filtername in (hdr['PUPIL'], hdr['FILTER']):
                    outname = destreak(member['expname'],
                                    use_background_map=True,
                                    median_filter_size=2048)  # median_filter_size=medfilt_size[filtername])
                    member['expname'] = outname

                    # fix_alignment(outname, proposal_id=proposal_id,
                    #               module=module, field=field,
                    #               basepath=basepath, filtername=filtername,
                    #               use_average=use_average)
            else: # make align files
                fname = member['expname']
                assert fname.endswith('_cal.fits')
                member['expname'] = fname.replace("_cal.fits", "_align.fits")
                shutil.copy(fname, member['expname'])
#                if not os.path.exists(member['expname']): #check align file
#                    print(member['expname'])
 #                   raise FileNotFoundError(f"Alignment file {member['expname']} was not created from {fname}")
                fix_alignment(member['expname'], proposal_id=proposal_id,
                               module=module, field=field, basepath=basepath,
                               filtername=filtername, use_average=use_average)

        asn_file_each = asn_file.replace("_asn.json", f"_{module}_asn.json")
        with open(asn_file_each, 'w') as fh:
            json.dump(asn_data, fh)

        # don't use VVV at all; the catalog does not play nicely with JWST pipe catalogs
        # if False: #filtername.lower() == 'f405n':
        #     # for the VVV cat, use the merged version: no need for independent versions
        #     abs_refcat = vvvdr2fn = (f'{basepath}/{filtername.upper()}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_clear-{filtername}-merged_vvvcat.ecsv')
        #     print(f"Loaded VVV catalog {vvvdr2fn}")
        #     retrieve_vvv(basepath=basepath, filtername=filtername, fov_regname=fov_regname[regionname], module='merged', fieldnumber=field)
        #     tweakreg_parameters['abs_refcat'] = vvvdr2fn
        #     tweakreg_parameters['abs_searchrad'] = 1
        #     reftbl = Table.read(abs_refcat)
        #     reftbl.meta['name'] = f'VVV Reference Catalog {filtername}'
        #     assert 'skycoord' in reftbl.colnames
        # else:
        #     abs_refcat = f'{basepath}NB/catalogs/crowdsource_based_nircam-f405n_reference_astrometric_catalog.ecsv'
        #     reftbl = Table.read(abs_refcat)
        #     # For non-F410M, try aligning to F410M instead of VVV?
        #     reftblversion = reftbl.meta['VERSION']
        #     reftbl.meta['name'] = 'F405N Reference Astrometric Catalog'

        #     # truncate to top 10,000 sources
        #     # more recent versions are already truncated to only very high quality matches
        #     # reftbl[:10000].write(f'{basepath}/catalogs/crowdsource_based_nircam-f405n_reference_astrometric_catalog_truncated10000.ecsv', overwrite=True)
        #     # abs_refcat = f'{basepath}/catalogs/crowdsource_based_nircam-f405n_reference_astrometric_catalog_truncated10000.ecsv'

        #     tweakreg_parameters['abs_searchrad'] = 0.4
        #     # try forcing searchrad to be tighter to avoid bad crossmatches
        #     # (the raw data are very well-aligned to begin with, though CARTA
        #     # can't display them b/c they are using SIP)
        #     tweakreg_parameters['searchrad'] = 0.05
        #     print(f"Reference catalog is {abs_refcat} with version {reftblversion}")

        #tweakreg_parameters.update({'abs_refcat': abs_refcat})
        #tweakreg_parameters.update({'skip': True})

        print(f"Running tweakreg ({module})")
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
            check_wcs(member['expname'].replace('destreak', 'i2d'))
        check_wcs(asn_data['products'][0]['name'] + "_i2d.fits")

        # print(f"Realigning to VVV (module={module}, filter={filtername})")
        realigned_vvv_filename = f'{basepath}{filtername.upper()}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_clear-{filtername.lower()}-{module}{destreak_suffix}_realigned-to-vvv.fits'
        shutil.copy(f'{basepath}{filtername.upper()}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_clear-{filtername.lower()}-{module}_i2d.fits',
                     realigned_vvv_filename)
        # print(f"Realigned to VVV filename: {realigned_vvv_filename}")
        # realigned = realign_to_vvv(filtername=filtername.lower(),
        #                            fov_regname=fov_regname[regionname],
        #                            basepath=basepath, module=module,
        #                            fieldnumber=field, proposal_id=proposal_id,
        #                            imfile=realigned_vvv_filename,
        #                            ksmag_limit=15 if filtername.lower() == 'f410m' else 11,
        #                            mag_limit=18 if filtername.lower() == 'f115w' else 15,
        #                            max_offset=(0.4 if wavelength > 250 else 0.2)*u.arcsec,
        #                            #raoffset=raoffset,
        #                            #decoffset=decoffset
        #                            )
        # print(f"Done realigning to VVV (module={module}, filtername={filtername})")

        # print(f"Realigning to refcat (module={module}, filtername={filtername})")
        realigned_refcat_filename = f'{basepath}{filtername.upper()}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_clear-{filtername.lower()}-{module}{destreak_suffix}_realigned-to-refcat.fits'
        shutil.copy(f'{basepath}{filtername.upper()}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_clear-{filtername.lower()}-{module}_i2d.fits',
                     realigned_refcat_filename)
        # print(f"Realigned refcat filename: {realigned_refcat_filename}")
        # realigned = realign_to_catalog(reftbl['skycoord'],
        #                                filtername=filtername.lower(),
        #                                basepath=basepath, module=module,
        #                                fieldnumber=field,
        #                                mag_limit=20, proposal_id=proposal_id,
        #                                max_offset=(0.4 if wavelength > 250 else 0.2)*u.arcsec,
        #                                imfile=realigned_refcat_filename,
        #                                #raoffset=raoffset, decoffset=decoffset
        #                                )
        # print(f"Done realigning to refcat (module={module}, filtername={filtername})")

        # print(f"Removing saturated stars.  cwd={os.getcwd()}")
        # try:
        #     remove_saturated_stars(f'jw0{proposal_id}-o{field}_t001_nircam_clear-{filtername.lower()}-{module}_i2d.fits')
        #     remove_saturated_stars(f'jw0{proposal_id}-o{field}_t001_nircam_clear-{filtername.lower()}-{module}{destreak_suffix}_realigned-to-vvv.fits')
        # except (TimeoutError, requests.exceptions.ReadTimeout) as ex:
        #     print("Failed to run remove_saturated_stars with failure {ex}")


    if module == 'nrcb':
        # assume nrca is run before nrcb
        print("Merging already-combined nrca + nrcb modules")
        merge_a_plus_b(filtername, basepath=basepath, fieldnumber=field, suffix=f'{destreak_suffix}_realigned-to-refcat',
                       proposal_id=proposal_id)
        print("DONE Merging already-combined nrca + nrcb modules")

        #try:
        #    # this is probably wrong / has wrong path names.
        #    remove_saturated_stars(f'jw0{proposal_id}-o{field}_t001_nircam_clear-{filtername.lower()}-{module}-reproject_i2d.fits')
        #    remove_saturated_stars(f'jw0{proposal_id}-o{field}_t001_nircam_clear-{filtername.lower()}-{module}_realigned-to-vvv.fits')
        #except (TimeoutError, requests.exceptions.ReadTimeout) as ex:
        #    print("Failed to run remove_saturated_stars with failure {ex}")

    if module == 'merged':
        # try merging all frames & modules
        print(f"Working on merged reduction (both modules):  asn_file={asn_file}")

        # Load asn_data for both modules
        with open(asn_file) as f_obj:
            asn_data = json.load(f_obj)

        for member in asn_data['products'][0]['members']:
            print(f"Running destreak={do_destreak} and maybe alignment on {member} for module={module}")
            hdr = fits.getheader(member['expname'])
            do_destreak=False
            if do_destreak:
                if filtername in (hdr['PUPIL'], hdr['FILTER']):
                    outname = destreak(member['expname'],
                                    use_background_map=True,
                                    median_filter_size=2048)  # median_filter_size=medfilt_size[filtername])
                    member['expname'] = outname

                    # re-do alignment if destreak file doesn't exist at the earlier step above
                    #fix_alignment(outname, proposal_id=proposal_id, module=module, field=field, basepath=basepath, filtername=filtername, use_average=use_average)
            else: # make align files
                fname = member['expname']
                assert fname.endswith('_cal.fits')
                member['expname'] = fname.replace("_cal.fits", "_align.fits")
                shutil.copy(fname, member['expname'])

                fix_alignment(member['expname'], proposal_id=proposal_id, module=module, field=field, basepath=basepath, filtername=filtername, use_average=use_average)

        asn_data['products'][0]['name'] = f'jw0{proposal_id}-o{field}_t001_nircam_clear-{filtername.lower()}-merged'
        asn_file_merged = asn_file.replace("_asn.json", f"_merged_asn.json")
        with open(asn_file_merged, 'w') as fh:
            json.dump(asn_data, fh)

        # don't re-fit to VVV - it's not accurate enough with the JWST-derived
        # catalogs.  We needed to use our own much more extensive cataloging to
        # beat down the noise enough to make this approach viable
        # if False: # filtername.lower() == 'f405n':
        #     vvvdr2fn = (f'{basepath}/{filtername.upper()}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_clear-{filtername}-{module}_vvvcat.ecsv')
        #     print(f"Loaded VVV catalog {vvvdr2fn}")
        #     retrieve_vvv(basepath=basepath, filtername=filtername, fov_regname=fov_regname[regionname], module=module, fieldnumber=field)
        #     tweakreg_parameters['abs_refcat'] = abs_refcat = vvvdr2fn
        #     tweakreg_parameters['abs_searchrad'] = 1
        #     reftbl = Table.read(abs_refcat)
        #     assert 'skycoord' in reftbl.colnames
        # else:
        #     abs_refcat = f'{basepath}/catalogs/crowdsource_based_nircam-f405n_reference_astrometric_catalog.ecsv'
        #     reftbl = Table.read(abs_refcat)
        #     assert 'skycoord' in reftbl.colnames
        #     reftblversion = reftbl.meta['VERSION']

        #     # truncate to top 10,000 sources
        #     reftbl[:10000].write(f'{basepath}/catalogs/crowdsource_based_nircam-f405n_reference_astrometric_catalog_truncated10000.ecsv', overwrite=True)
        #     abs_refcat = f'{basepath}/catalogs/crowdsource_based_nircam-f405n_reference_astrometric_catalog_truncated10000.ecsv'

        #     tweakreg_parameters['abs_searchrad'] = 0.4
        #     tweakreg_parameters['searchrad'] = 0.05
        #     print(f"Reference catalog is {abs_refcat} with version {reftblversion}")

        #tweakreg_parameters.update({'abs_refcat': abs_refcat,})

        print("Running Image3Pipeline with tweakreg (merged)")
        calwebb_image3.Image3Pipeline.call(
            asn_file_merged,
            steps={'tweakreg': tweakreg_parameters,},
            #steps={'tweakreg': False,}
            output_dir=output_dir,
            save_results=True)
        print(f"DONE running Image3Pipeline {asn_file_merged}.  This should have produced file {asn_data['products'][0]['name']}_i2d.fits")

        print("After tweakreg step, checking WCS headers:")
        for member in asn_data['products'][0]['members']:
            check_wcs(member['expname'])
        check_wcs(asn_data['products'][0]['name'] + "_i2d.fits")

        print(f"Realigning to VVV (module={module})")# with raoffset={raoffset}, decoffset={decoffset}")
        realigned_vvv_filename = f'{basepath}{filtername.upper()}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_clear-{filtername.lower()}-{module}{destreak_suffix}_realigned-to-vvv.fits'
        print(f"Realigned to VVV filename: {realigned_vvv_filename}")
        shutil.copy(f'{basepath}{filtername.upper()}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_clear-{filtername.lower()}-{module}_i2d.fits',
                    realigned_vvv_filename)
        # realigned = realign_to_vvv(filtername=filtername.lower(),
        #                            fov_regname=fov_regname[regionname], basepath=basepath, module=module,
        #                            fieldnumber=field, proposal_id=proposal_id,
        #                            imfile=realigned_vvv_filename,
        #                            max_offset=(0.4 if wavelength > 250 else 0.2)*u.arcsec,
        #                            ksmag_limit=15 if filtername.lower() == 'f410m' else 11,
        #                            mag_limit=18 if filtername.lower() == 'f115w' else 15,
        #                            #raoffset=raoffset, decoffset=decoffset
        #                            )

        print(f"Realigning to refcat (module={module})")# with raoffset={raoffset}, decoffset={decoffset}")
        realigned_refcat_filename = f'{basepath}{filtername.upper()}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_clear-{filtername.lower()}-{module}{destreak_suffix}_realigned-to-refcat.fits'
        print(f"Realigned refcat filename: {realigned_refcat_filename}")
        shutil.copy(f'{basepath}{filtername.upper()}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_clear-{filtername.lower()}-{module}_i2d.fits',
                    realigned_refcat_filename)
        # realigned = realign_to_catalog(reftbl['skycoord'],
        #                                filtername=filtername.lower(),
        #                                basepath=basepath, module=module,
        #                                fieldnumber=field,
        #                                max_offset=(0.4 if wavelength > 250 else 0.2)*u.arcsec,
        #                                mag_limit=20,
        #                                proposal_id=proposal_id,
        #                                imfile=realigned_refcat_filename,
        #                                #raoffset=raoffset, decoffset=decoffset
        #                                )

        # print(f"Removing saturated stars.  cwd={os.getcwd()}")
        # try:
        #     remove_saturated_stars(f'jw0{proposal_id}-o{field}_t001_nircam_clear-{filtername.lower()}-merged_i2d.fits')
        #     remove_saturated_stars(f'jw0{proposal_id}-o{field}_t001_nircam_clear-{filtername.lower()}-{module}{destreak_suffix}_realigned-to-vvv.fits')
        # except (TimeoutError, requests.exceptions.ReadTimeout) as ex:
        #     print("Failed to run remove_saturated_stars with failure {ex}")

    globals().update(locals())
    return locals()


def fix_alignment(fn, proposal_id=None, module=None, field=None, basepath=None, filtername=None,
                  use_average=True):
    if os.path.exists(fn):
        print(f"Running manual align for {module} data ({proposal_id} + {field}): {fn}", flush=True)
    else:
        print(f"Skipping manual align for nonexistent file {module} ({proposal_id} + {field}): {fn}", flush=True)
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
        basepath = f'/orange/adamginsburg/jwst/{field}'
    if module is None:
        module = 'nrc' + mod.meta.instrument.module.lower()

    if (field == '004' and proposal_id == '1182') or (field == '001' and proposal_id == '2221'):
        refname = refnames[proposal_id]
        exposure = int(fn.split("_")[-3])
        thismodule = fn.split("_")[-2]
        visit = fn.split("_")[0]
        if use_average:
            tblfn = f'{basepath}/offsets/Offsets_JWST_Brick{proposal_id}_{refname}_average.csv'
            print(f"Using average offset table {tblfn}")
            offsets_tbl = Table.read(tblfn)
            match = (((offsets_tbl['Module'] == thismodule) |
                      (offsets_tbl['Module'] == thismodule.strip('1234'))) &
                     (offsets_tbl['Filter'] == filtername)
                     )
            if 'Visit' in offsets_tbl.colnames:
                match &= (offsets_tbl['Visit'] == visit)
            row = offsets_tbl[match]
            print(f'Running manual align for merged for {filtername} {row["Module"][0]}.')
        else:
            tblfn = f'{basepath}/offsets/Offsets_JWST_Brick{proposal_id}_{refname}.csv'
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
        if filtername.upper() in ('F212N', 'F187N', 'F182M'):
            print('Short wavelength offset correction.')
            if 'nrca' in thismodule.lower():
                decshift += 0.1*u.arcsec
                rashift += -0.23*u.arcsec
    else:
        rashift = 0*u.arcsec
        decshift = 0*u.arcsec
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
                      default='F480M,F187N',#'F210M,',#F466N,F405N,F410M,F212N,F182M,F150W,F300M,F360M, ### need more memory than 256 for 
                      help="filter name list", metavar="filternames")
    parser.add_option("-m", "--modules", dest="modules",
                    default='nrca,nrcb,merged',
                    help="module list", metavar="modules")
    parser.add_option("-d", "--field", dest="field",
                    default='001',
                    help="list of target fields", metavar="field")
    parser.add_option("-s", "--skip_step1and2", dest="skip_step1and2",
                      default=False,
                      action='store',
                      help="Skip the image-remaking step?", metavar="skip_Step1and2")
    parser.add_option("--no_destreak", dest="no_destreak",
                      default=False,
                      action='store',
                      help="Skip the destreaking step?", metavar="skip_destreak")
    parser.add_option("-p", "--proposal_id", dest="proposal_id",
                      default='6151',
                      help="proposal id (string)", metavar="proposal_id")
    (options, args) = parser.parse_args()

    filternames = options.filternames.split(",")
    modules = options.modules.split(",")
    fields = options.field.split(",")
    proposal_id = options.proposal_id
    skip_step1and2 = options.skip_step1and2
    no_destreak = bool(options.no_destreak)
    print(options)

    with open(os.path.expanduser('~/.mast_api_token'), 'r') as fh:
       api_token = fh.read().strip()
       os.environ['MAST_API_TOKEN'] = api_token.strip()
    Mast.login(api_token.strip())
    Observations.login(api_token)


    field_to_reg_mapping = {#'2221': {'001': 'brick', '002': 'cloudc'},
                            #'1182': {'004': 'brick'},
                            '5365': {'001': 'sgrb2'},
                            '6151': {'001': 'w51'},
                            }[proposal_id]

    for field in fields:
        for filtername in filternames:
            for module in modules:
                print(f"Main Loop: {proposal_id} + {filtername} + {module} + {field}={field_to_reg_mapping[field]}")
                results = main(filtername=filtername, module=module, Observations=Observations, field=field,
                               regionname=field_to_reg_mapping[field],
                               proposal_id=proposal_id,
                               skip_step1and2=skip_step1and2,
                               #do_destreak=not no_destreak,
                               do_destreak=no_destreak,
                              )


    # if proposal_id == '2221':
    #     print("Running notebooks")
    #     from run_notebook import run_notebook
    #     basepath = '/orange/adamginsburg/jwst/brick/'
    #     if 'merge' in modules:
    #         run_notebook(f'{basepath}/notebooks/BrA_Separation_nrca.ipynb')
    #         run_notebook(f'{basepath}/notebooks/BrA_Separation_nrcb.ipynb')
    #         run_notebook(f'{basepath}/notebooks/F466_separation_nrca.ipynb')
    #         run_notebook(f'{basepath}/notebooks/F466_separation_nrcb.ipynb')
    #         run_notebook(f'{basepath}/notebooks/StarDestroyer_nrca.ipynb')
    #         run_notebook(f'{basepath}/notebooks/StarDestroyer_nrcb.ipynb')
    #         run_notebook(f'{basepath}/notebooks/Stitch_A_to_B.ipynb')
    #         run_notebook(f'{basepath}/notebooks/PaA_Separation_nrcb.ipynb')
    #         run_notebook(f'{basepath}/notebooks/StarDestroyer_PaA_nrcb.ipynb')


"""
await app.openFile("/jwst/brick/F410M/pipeline/jw02221-o001_t001_nircam_clear-f410m-merged_i2d.fits")
await app.appendFile("/jwst/brick/F410M/pipeline/jw02221-o001_t001_nircam_clear-f410m-nrca_i2d.fits")
await app.appendFile("/jwst/brick/F410M/pipeline/jw02221-o001_t001_nircam_clear-f410m-nrcb_i2d.fits")
await app.appendFile("/jwst/brick/F182M/pipeline/jw02221-o001_t001_nircam_clear-f182m-merged_i2d.fits")
await app.appendFile("/jwst/brick/F182M/pipeline/jw02221-o001_t001_nircam_clear-f182m-nrca_i2d.fits")
await app.appendFile("/jwst/brick/F182M/pipeline/jw02221-o001_t001_nircam_clear-f182m-nrcb_i2d.fits")
await app.appendFile("/jwst/brick/F212N/pipeline/jw02221-o001_t001_nircam_clear-f212n-merged_i2d.fits")
await app.appendFile("/jwst/brick/F212N/pipeline/jw02221-o001_t001_nircam_clear-f212n-nrca_i2d.fits")
await app.appendFile("/jwst/brick/F212N/pipeline/jw02221-o001_t001_nircam_clear-f212n-nrcb_i2d.fits")
await app.appendFile("/jwst/brick/F466N/pipeline/jw02221-o001_t001_nircam_clear-f466n-merged_i2d.fits")
await app.appendFile("/jwst/brick/F466N/pipeline/jw02221-o001_t001_nircam_clear-f466n-nrca_i2d.fits")
await app.appendFile("/jwst/brick/F466N/pipeline/jw02221-o001_t001_nircam_clear-f466n-nrcb_i2d.fits")


"""
