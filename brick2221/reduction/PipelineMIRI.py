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

from brick2221.reduction.align_to_catalogs import realign_to_vvv, realign_to_catalog, merge_a_plus_b, retrieve_vvv

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
               'sickle': 'regions_/nircam_sickle_fov.reg',
               'w51': 'nope',
               'sgrb2': 'nope',
               }

# Reference catalog configuration by proposal and field.
# Paths are relative to basepath.
REFERENCE_ASTROMETRIC_CATALOG_CANDIDATES_BY_FIELD = {
    # Sickle MIRI fields. Prefer the short-wave merged astrometric catalog,
    # then bootstrapped catalogs if needed.
    '3958': {
        '001': (
            'catalogs/pipeline_based_nircam-f210m_reference_astrometric_catalog.fits',
            'catalogs/nircam_bootstrapped_to_gns_refcat.fits',
            'catalogs/nircam_bootstrapped_to_vvv_refcat.fits',
        ),
        '002': (
            'catalogs/pipeline_based_nircam-f210m_reference_astrometric_catalog.fits',
            'catalogs/nircam_bootstrapped_to_gns_refcat.fits',
            'catalogs/nircam_bootstrapped_to_vvv_refcat.fits',
        ),
        '003': (
            'catalogs/pipeline_based_nircam-f210m_reference_astrometric_catalog.fits',
            'catalogs/nircam_bootstrapped_to_gns_refcat.fits',
            'catalogs/nircam_bootstrapped_to_vvv_refcat.fits',
        ),
    },
    '2221': {
        '001': (
            'catalogs/crowdsource_based_nircam-f405n_reference_astrometric_catalog.fits',
            'catalogs/crowdsource_based_nircam-f405n_reference_astrometric_catalog.ecsv',
            'catalogs/twomass.fits',
        ),
        '002': (
            'catalogs/crowdsource_based_nircam-f405n_reference_astrometric_catalog.fits',
            'catalogs/crowdsource_based_nircam-f405n_reference_astrometric_catalog.ecsv',
            'catalogs/twomass.fits',
        ),
    },
}


def get_reference_astrometric_catalog_path(basepath, proposal_id, field, explicit_refcat=None):
    if explicit_refcat is not None:
        return explicit_refcat
    if proposal_id in REFERENCE_ASTROMETRIC_CATALOG_CANDIDATES_BY_FIELD:
        if field in REFERENCE_ASTROMETRIC_CATALOG_CANDIDATES_BY_FIELD[proposal_id]:
            for relpath in REFERENCE_ASTROMETRIC_CATALOG_CANDIDATES_BY_FIELD[proposal_id][field]:
                candidate = f'{basepath}/{relpath}'
                if os.path.exists(candidate):
                    return candidate
    twomass = f'{basepath}/catalogs/twomass.fits'
    if os.path.exists(twomass):
        return twomass
    return None


def relocate_manifest_products(manifest, output_dir):
    """Flatten MAST download tree into output_dir with idempotent relocation."""
    for row in manifest:
        src = str(row['Local Path'])
        dst = os.path.join(output_dir, os.path.basename(src))

        if os.path.exists(dst):
            # Common when rerunning and MAST points to a file already moved earlier.
            print(f"Relocation skipped: destination already exists ({dst})")
            continue

        try:
            shutil.move(src, dst)
        except FileNotFoundError:
            if os.path.exists(dst):
                print(f"Relocation skipped: source missing but destination exists ({dst})")
            else:
                raise FileNotFoundError(
                    f"MAST manifest source missing and destination not present: src={src} dst={dst}"
                )
        except shutil.Error as ex:
            print(f"Failed to move file with error {ex}")


def main(filtername, Observations=None, regionname='brick',
         field='001', proposal_id='2221', skip_step1and2=False, use_average=True,
         reference_catalog=None, skip_download_for_existing=False,
         marshall_tuning=False):
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
    elif regionname == 'sickle':
        assert proposal_id == '3958'
        assert field in ('001', '002', '003')

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
        except FileExistsError as ex:
            print(f'Failed to link {fn} to {os.path.basename(fn)} because of {ex}')

    Observations.cache_location = output_dir
    obs_table = Observations.query_criteria(
                                            proposal_id=proposal_id,
                                            #proposal_pi="Ginsburg*",
                                            #calib_level=3,
                                            )
    print("Obs table length:", len(obs_table))

    if 'filters' in obs_table.colnames and 'obs_id' in obs_table.colnames:
        try:
            filters_col = np.array([str(val).upper() for val in obs_table['filters'].filled('')])
            obs_id_col = np.array([str(val).lower() for val in obs_table['obs_id'].filled('')])
        except AttributeError:
            filters_col = np.array([str(val).upper() for val in obs_table['filters']])
            obs_id_col = np.array([str(val).lower() for val in obs_table['obs_id']])
        msk = ((np.char.find(filters_col, filtername.upper()) >= 0) |
               (np.char.find(obs_id_col, filtername.lower()) >= 0))
    else:
        print("Warning: 'filters' or 'obs_id' column missing in obs_table; selecting all observations for this proposal")
        msk = np.ones(len(obs_table), dtype=bool)
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
    relocate_manifest_products(manifest, output_dir)

    products_fits = Observations.filter_products(data_products_by_obs, extension="fits")
    print("products_fits length:", len(products_fits))
    uncal_mask = np.array([uri.endswith('_uncal.fits') and f'jw0{proposal_id}{field}' in uri for uri in products_fits['dataURI']])
    uncal_mask &= products_fits['productType'] == 'SCIENCE'
    print("uncal length:", (uncal_mask.sum()))

    if skip_download_for_existing:
        already_downloaded = np.array([os.path.exists(os.path.basename(uri)) for uri in products_fits['dataURI']])
        uncal_mask &= ~already_downloaded
        print(f"uncal to download: {uncal_mask.sum()}; {already_downloaded.sum()} were already downloaded")

    if uncal_mask.any():
        manifest = Observations.download_products(products_fits[uncal_mask], download_dir=output_dir)
        print("manifest:", manifest)

        # MAST creates deep directory structures we don't want
        relocate_manifest_products(manifest, output_dir)

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
        """
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
        """

        print(f'Filter {filtername} tweakreg parameters: {tweakreg_parameters}')

        with open(asn_file) as f_obj:
            asn_data = json.load(f_obj)

        print(f"In cwd={os.getcwd()}")
        members = asn_data['products'][0]['members']
        if skip_step1and2:
            missing_cal = [member['expname'] for member in members if not os.path.exists(member['expname'])]
            if len(missing_cal) == 0:
                print("Skipped step 1 and step2")
            else:
                print(f"skip_step1and2 requested, but {len(missing_cal)} _cal files are missing; running detector/image2 for missing files")

        if (not skip_step1and2) or (skip_step1and2 and len([member['expname'] for member in members if not os.path.exists(member['expname'])]) > 0):
            # re-calibrate uncal files -> cal files *without* suppressing first group
            for member in members:
                assert f'jw0{proposal_id}{field}' in member['expname']
                cal_name = member['expname']
                if skip_step1and2 and os.path.exists(cal_name):
                    continue

                print(f"DETECTOR PIPELINE on {cal_name}")
                print("Detector1Pipeline step")
                # from Hosek: expand_large_events -> false; turn off "snowball" detection
                detector1_steps = {'ramp_fit': {'suppress_one_group': False},
                                   'refpix': {'use_side_ref_pixels': True}}
                if marshall_tuning:
                    detector1_steps.update({'saturation': {'skip': True, 'n_pix_grow_sat': 0},
                                            'firstframe': {'skip': True},
                                            'rscd': {'skip': True}})
                Detector1Pipeline.call(cal_name.replace("_cal.fits", "_uncal.fits"),
                                       save_results=True, output_dir=output_dir,
                                       save_calibrated_ramp=True,
                                       steps=detector1_steps)

                print(f"IMAGE2 PIPELINE on {cal_name}")
                Image2Pipeline.call(cal_name.replace("_cal.fits", "_rate.fits"),
                                    save_results=True, output_dir=output_dir,
                                    #steps={'background': {'run': False}},
                                   )

    if True:
        print(f"Filter {filtername}: doing tweakreg.  ")

        with open(asn_file) as f_obj:
            asn_data = json.load(f_obj)
        asn_data['products'][0]['name'] = f'jw0{proposal_id}-o{field}_t001_miri_{filtername.lower()}'
        asn_data['products'][0]['members'] = [row for row in asn_data['products'][0]['members']]

        for member in asn_data['products'][0]['members']:
            print(f"Running  maybe alignment on {member}")
            fname = member['expname']
            assert fname.endswith('_cal.fits')
            member['expname'] = fname.replace("_cal.fits", "_align.fits")
            shutil.copy(fname, member['expname'])

            fix_alignment(member['expname'], proposal_id=proposal_id,
                          field=field, basepath=basepath,
                          regionname=regionname,
                          filtername=filtername,
                          use_average=use_average,
                          visit=fname[10:13])

        asn_file_each = asn_file
        with open(asn_file_each, 'w') as fh:
            json.dump(asn_data, fh)

        abs_refcat = get_reference_astrometric_catalog_path(basepath, proposal_id, field, explicit_refcat=reference_catalog)
        if abs_refcat is not None:
            reftbl = Table.read(abs_refcat)
            reftbl.meta['name'] = 'Reference Astrometric Catalog'

            tweakreg_parameters['abs_searchrad'] = 0.4
            # try forcing searchrad to be tighter to avoid bad crossmatches
            # (the raw data are very well-aligned to begin with, though CARTA
            # can't display them b/c they are using SIP)
            tweakreg_parameters['searchrad'] = 0.05
            # MIRI BRIGHTSKY fields can have very few matched stars per frame.
            tweakreg_parameters['minobj'] = 2
            tweakreg_parameters['abs_minobj'] = 2
            print(f"Reference catalog is {abs_refcat}")

            tweakreg_parameters.update({'abs_refcat': abs_refcat,})
        else:
            print(f"No reference catalog found for proposal_id={proposal_id} field={field} in {basepath}; running without abs_refcat")

        skymatch_params = {'save_results': True,
                           'subtract': False,
                           'skymethod': 'match',
                           'match_down': False}
        outlier_params = {'good_bits': "SATURATED, JUMP_DET"}
        if marshall_tuning:
            skymatch_params = {'save_results': True,
                               'subtract': True,
                               'skymethod': 'global',
                               'match_down': True}
            outlier_params = {'snr': "30.0 5.0",
                              'good_bits': "SATURATED, JUMP_DET",
                              'save_intermediate_results': True}

        print("Running tweakreg")
        calwebb_image3.Image3Pipeline.call(
            asn_file_each,
            steps={'tweakreg': tweakreg_parameters,
                   'skymatch': skymatch_params,
                   'outlier_detection': outlier_params,
            },
            output_dir=output_dir,
            save_results=True)
        print(f"DONE running {asn_file_each}")

        print("After tweakreg step, checking WCS headers:")
        for member in asn_data['products'][0]['members']:
            check_wcs(member['expname'])
            check_wcs(member['expname'].replace('cal', 'i2d').replace('destreak', 'i2d'))
        check_wcs(asn_data['products'][0]['name'] + "_i2d.fits")

    globals().update(locals())
    return locals()


def fix_alignment(fn, proposal_id=None, regionname='brick', field=None, basepath=None, filtername=None,
                  use_average=True, visit='003'):
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
        basepath = f'/orange/adamginsburg/jwst/{regionname}'

    print("TODO: calculate MIRI offsets and implement them")
    # Default Brick/CloudC offset.
    rashift = -3.895 * u.arcsec
    decshift = 1.28 * u.arcsec
    # Marshall W51 tuning: use a small global RA correction for MIRI.
    if regionname == 'w51':
        rashift = 0.2 * u.arcsec
        decshift = 0 * u.arcsec
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
    parser.add_option("--reference_catalog", dest="reference_catalog",
                      default=None,
                      help="Path to explicit astrometric reference catalog for tweakreg (optional)", metavar="reference_catalog")
    parser.add_option("--skip_download_for_existing", dest="skip_download_for_existing",
                      default=False, action='store_true',
                      help="Skip downloading _uncal files already present in output directory", metavar="skip_download_for_existing")
    parser.add_option("--marshall_tuning", dest="marshall_tuning",
                      default=False, action='store_true',
                      help="Enable Marshall W51-inspired MIRI tuning (Detector1/skymatch/outlier settings)", metavar="marshall_tuning")
    (options, args) = parser.parse_args()

    filternames = options.filternames.split(",")
    fields = options.field.split(",")
    proposal_id = options.proposal_id
    skip_step1and2 = options.skip_step1and2
    reference_catalog = options.reference_catalog
    skip_download_for_existing = options.skip_download_for_existing
    marshall_tuning = options.marshall_tuning
    print(options)

    with open(os.path.expanduser('~/.mast_api_token'), 'r') as fh:
        api_token = fh.read().strip()
        os.environ['MAST_API_TOKEN'] = api_token.strip()
    Mast.login(api_token.strip())
    Observations.login(api_token)


    field_to_reg_mapping = {'2221': {'002': 'brick', '001': 'cloudc'},
                            '3958': {'001': 'sickle', '002': 'sickle', '003': 'sickle'},
                            '5365': {'001': 'sgrb2'},
                            '6151': {'001': 'w51_background', '002': 'w51'},
                            }[proposal_id]

    for field in fields:
        for filtername in filternames:
            print(f"Main Loop: {proposal_id} + {filtername} + {field}={field_to_reg_mapping[field]}")
            results = main(filtername=filtername, Observations=Observations, field=field,
                           regionname=field_to_reg_mapping[field],
                           proposal_id=proposal_id,
                           skip_step1and2=skip_step1and2,
                           reference_catalog=reference_catalog,
                           skip_download_for_existing=skip_download_for_existing,
                           marshall_tuning=marshall_tuning,
                          )


