#!/usr/bin/env python
from glob import glob
from astroquery.mast import Mast, Observations
import os
import shutil
import numpy as np
import json
# import requests
import asdf
from astropy import log
from astropy.io import ascii, fits
from astropy.table import Table
from astropy.utils.data import download_file
from astropy.visualization import ImageNormalize, ManualInterval, LogStretch, LinearStretch
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
from destreak import destreak
from saturated_star_finding import iteratively_remove_saturated_stars, remove_saturated_stars

from align_to_catalogs import realign_to_vvv, realign_to_catalog, merge_a_plus_b

import crds

import jwst

def print(*args, **kwargs):
    now = datetime.datetime.now().isoformat()
    from builtins import print as printfunc
    return printfunc(f"{now}:", *args, **kwargs)

print(jwst.__version__)

# see 'destreak187.ipynb' for tests of this
# really this is just eyeballed; probably the 1/f noise is present at the same level in all of these, but you can't see it as well on
# the medium-band filters.
medfilt_size = {'F182M': 55, 'F187N': 512, 'F212N': 512}


def main(filtername, module, Observations=None, regionname='brick', field='001'):
    log.info(f"Processing filter {filtername} module {module}")

    # sanity check
    if regionname == 'brick':
        assert field == '001'
    elif regionname == 'cloudc':
        assert field == '002'

    basepath = f'/orange/adamginsburg/jwst/{regionname}/'
    os.environ["CRDS_PATH"] = f"/orange/adamginsburg/jwst/{regionname}/crds/"
    os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"
    mpl.rcParams['savefig.dpi'] = 80
    mpl.rcParams['figure.dpi'] = 80

    # Files created in this notebook will be saved
    # in a subdirectory of the base directory called `Stage3`
    output_dir = f'/orange/adamginsburg/jwst/{regionname}/{filtername}/pipeline/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    os.chdir(output_dir)

    # the files are one directory up
    for fn in glob("../*cal.fits"):
        try:
            os.link(fn, './'+os.path.basename(fn))
        except Exception as ex:
            print(f'Failed to link {fn} to {os.path.basename(fn)} because of {ex}')

    Observations.cache_location = output_dir
    obs_table = Observations.query_criteria(
                                            proposal_id="2221",
                                            proposal_pi="Ginsburg*",
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
    uncal_mask = np.array([uri.endswith('_uncal.fits') and f'jw02221{field}' in uri for uri in products_fits['dataURI']])
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
        print(f"Searching for {os.path.join(output_dir, f'jw02221-o{field}*_image3_*0[0-9][0-9]_asn.json')}")
        asn_file_search = glob(os.path.join(output_dir, f'jw02221-o{field}*_image3_*0[0-9][0-9]_asn.json'))
        if len(asn_file_search) == 1:
            asn_file = asn_file_search[0]
        elif len(asn_file_search) > 1:
            asn_file = sorted(asn_file_search)[-1]
            print(f"Found multiple asn files: {asn_file_search}.  Using the more recent one, {asn_file}.")
        else:
            raise ValueError(f"Mismatch: Did not find any asn files for module {module} for field {field} in {output_dir}")

        mapping = crds.rmap.load_mapping(f'/orange/adamginsburg/jwst/{regionname}/crds/mappings/jwst/jwst_nircam_pars-tweakregstep_0003.rmap')
        tweakreg_asdf_filename = [x for x in mapping.todict()['selections'] if x[1] == filtername][0][4]
        tweakreg_asdf = asdf.open(f'https://jwst-crds.stsci.edu/unchecked_get/references/jwst/{tweakreg_asdf_filename}')
        tweakreg_parameters = tweakreg_asdf.tree['parameters']
        print(f'Filter {filtername}: {tweakreg_parameters}')

        with open(asn_file) as f_obj:
            asn_data = json.load(f_obj)

        print(f"In cwd={os.getcwd()}")
        # re-calibrate all uncal files -> cal files *without* suppressing first group
        for member in asn_data['products'][0]['members']:
            # example filename: jw02221002001_02201_00002_nrcalong_cal.fits
            assert f'jw02221{field}' in member['expname']
            print(f"DETECTOR PIPELINE on {member['expname']}")
            print("Detector1Pipeline step")
            # from Hosek: expand_large_events -> false; turn off "snowball" detection
            Detector1Pipeline.call(member['expname'].replace("_cal.fits",
                                                             "_uncal.fits"),
                                   save_results=True, output_dir=output_dir,
                                   save_calibrated_ramp=True,
                                   steps={'ramp_fit': {'suppress_one_group':
                                                       False}})
            print(f"IMAGE2 PIPELINE on {member['expname']}")
            Image2Pipeline.call(member['expname'].replace("_cal.fits",
                                                          "_rate.fits"),
                                save_results=True, output_dir=output_dir,
                               )

    if module in ('nrca', 'nrcb',):
        print(f"Filter {filtername} module {module} ")

        with open(asn_file) as f_obj:
            asn_data = json.load(f_obj)
        asn_data['products'][0]['name'] = f'jw02221-o{field}_t001_nircam_clear-{filtername.lower()}-{module}'
        asn_data['products'][0]['members'] = [row for row in asn_data['products'][0]['members']
                                            if f'{module}' in row['expname']]

        for member in asn_data['products'][0]['members']:
            hdr = fits.getheader(member['expname'])
            if filtername in (hdr['PUPIL'], hdr['FILTER']):
                # changed filter size to be maximal now that we're using the background
                outname = destreak(member['expname'], median_filter_size=2048,
                                    use_background_map=True)  # medfilt_size[filtername])
                member['expname'] = outname

        asn_file_each = asn_file.replace("_asn.json", f"_{module}_alldetectors_asn.json")
        with open(asn_file_each, 'w') as fh:
            json.dump(asn_data, fh)

        # reference to long-wavelength catalogs
        abs_refcat = f'{basepath}/catalogs/crowdsource_based_nircam-f405n_reference_astrometric_catalog.ecsv'
        reftbl = Table.read(abs_refcat)
        reftblversion = reftbl.meta['VERSION']
        print(f"Reference catalog is {abs_refcat} with version {reftblversion}")

        # image3.tweakreg.searchrad = 1 # 1 arcsec instead of 2
        # image3.tweakreg.separation = 0.5 # min separation 0.4 arcsec instead of 1 (Mihai suggesteed separation = 2x tolerance)
        # image3.tweakreg.tolerance = 0.3 # max tolerance 0.2 instead of 0.7

        tweakreg_parameters.update({'fitgeometry': 'general',
                                    'brightest': 500,
                                    'snr_threshold': 15,
                                    'peakmax': 1400,
                                    'nclip': 7,
                                    'searchrad': 1,
                                    'abs_searchrad': 0.5,
                                    'abs_refcat': abs_refcat,
                                    'separation': 0.5,
                                    'tolerance': 0.3,
                                    'sharplo': 0.3,
                                    'sharphi': 0.9,
                                    'roundlo': -0.25,
                                    'roundhi': 0.25,
                                             })

        calwebb_image3.Image3Pipeline.call(
            asn_file_each,
            steps={'tweakreg': tweakreg_parameters,},
            output_dir=output_dir,
            save_results=True)
        print(f"DONE running {asn_file_each}")

        # realignment shouldn't be necessary, but at least the diagnostics from this
        # are useful
        realigned = realign_to_catalog(reftbl['skycoord_f410m'],
                                       filtername=filtername.lower(),
                                       module=module,
                                       fieldnumber=field)
        realigned.writeto(f'{basepath}/{filtername.upper()}/pipeline/jw02221-o{field}_t001_nircam_clear-{filtername.lower()}-{module}_realigned-to-refcat.fits', overwrite=True)

        with fits.open(f'jw02221-o{field}_t001_nircam_clear-{filtername.lower()}-{module}_i2d.fits', mode='update') as fh:
            fh[0].header['V_REFCAT'] = reftblversion

        log.info("Removing saturated stars")
        remove_saturated_stars(f'jw02221-o{field}_t001_nircam_clear-{filtername.lower()}-{module}_i2d.fits')
        remove_saturated_stars(f'jw02221-o{field}_t001_nircam_clear-{filtername.lower()}-{module}_realigned-to-refcat.fits')


    if module in ('nrcb', ):
        # June 30, 2023: previously, this was also being done for 'merged', but it timed out at that step, which appears to take >3 days
        # assume nrca is run before nrcb
        print("Merging already-combined nrca + nrcb modules", flush=True)
        merge_a_plus_b(filtername)
        print("DONE Merging already-combined nrca + nrcb modules")

    if module == 'merged':
        # May 31, 2023: commented this out to see if the problem was a memory problem or something else
        # it's not clear this code hsa ever been used or even makes sense.  But maybe?
        # raise ValueError("Don't try merging on disk any more, instead do the merging the other way.")
        print(f"Filter {filtername} module = merged nrca + nrcb ")
        log.info("Running merged frames")
        # try merging all frames & modules


        with open(asn_file) as f_obj:
            asn_data = json.load(f_obj)
        asn_data['products'][0]['name'] = f'jw02221-o{field}_t001_nircam_clear-{filtername.lower()}-merged'

        for member in asn_data['products'][0]['members']:
            hdr = fits.getheader(member['expname'])
            if filtername in (hdr['PUPIL'], hdr['FILTER']):
                # changed filter size to be maximal now that we're using the background
                outname = destreak(member['expname'], median_filter_size=2048,
                                    use_background_map=True)  # medfilt_size[filtername])
                member['expname'] = outname

        asn_file_merged = asn_file.replace("_asn.json", f"_merged_asn.json")
        with open(asn_file_merged, 'w') as fh:
            json.dump(asn_data, fh)

        # # TODO: instead, use F410M as the astrometric reference, since that matches _better_ to VVV

        # vvvdr2fn = (f'{basepath}/{filtername.upper()}/pipeline/jw02221-o001_t001_nircam_clear-{filtername}-{module}_vvvcat.ecsv')
        # print(vvvdr2fn)
        # if os.path.exists(vvvdr2fn):
        #     image3.tweakreg.abs_refcat = vvvdr2fn
        #     image3.tweakreg.abs_searchrad = 1
        # else:
        #     print(f"Did not find VVV catalog {vvvdr2fn}")

        abs_refcat = f'{basepath}/catalogs/crowdsource_based_nircam-f405n_reference_astrometric_catalog.ecsv'

        #image3.tweakreg.searchrad = 1 # 1 arcsec instead of 2
        #image3.tweakreg.separation = 0.6 # min separation 0.4 arcsec instead of 1 (Mihai suggesteed separation = 2x tolerance)
        #image3.tweakreg.tolerance = 0.3 # max tolerance 0.2 instead of 0.7

        tweakreg_parameters.update({'fitgeometry': 'general',
                                    'brightest': 500,
                                    'snr_threshold': 15,
                                    'peakmax': 1400,
                                    'nclip': 7,
                                    'searchrad': 1,
                                    'abs_searchrad': 0.5,
                                    'abs_refcat': abs_refcat,
                                    'separation': 0.6,
                                    'tolerance': 0.3,
                                    'sharplo': 0.3,
                                    'sharphi': 0.9,
                                    'roundlo': -0.25,
                                    'roundhi': 0.25,
                                   })

        calwebb_image3.Image3Pipeline.call(
            asn_file_merged,
            steps={'tweakreg': tweakreg_parameters,},
            output_dir=output_dir,
            save_results=True)
        print(f"DONE running {asn_file_merged}")


        abs_refcat = f'{basepath}/catalogs/crowdsource_based_nircam-f405n_reference_astrometric_catalog.ecsv'
        reftbl = Table.read(abs_refcat)
        reftblversion = reftbl.meta['VERSION']
        print(f"Reference catalog is {abs_refcat} with version {reftblversion}")

        realigned = realign_to_catalog(reftbl['skycoord_f410m'],
                                       filtername=filtername.lower(),
                                       module=module,
                                       fieldnumber=field)
        realigned.writeto(f'{basepath}/{filtername.upper()}/pipeline/jw02221-o{field}_t001_nircam_clear-{filtername.lower()}-{module}_realigned-to-refcat.fits', overwrite=True)

        with fits.open(f'jw02221-o{field}_t001_nircam_clear-{filtername.lower()}-{module}_i2d.fits', mode='update') as fh:
            fh[0].header['V_REFCAT'] = reftblversion

        log.info("Removing saturated stars")
        remove_saturated_stars(f'jw02221-o{field}_t001_nircam_clear-{filtername.lower()}-merged_i2d.fits')
        remove_saturated_stars(f'jw02221-o{field}_t001_nircam_clear-{filtername.lower()}-{module}_realigned-to-refcat.fits')


    globals().update(locals())
    return locals()

if __name__ == "__main__":
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-f", "--filternames", dest="filternames",
                      default='F212N,F182M,F187N',
                      help="filter name list", metavar="filternames")
    # merged requires >512 GB of memory, apparently - it fails with OOM kill
    parser.add_option("-m", "--modules", dest="modules", default='nrca,nrcb',
                      help="module list", metavar="modules")
    parser.add_option("-d", "--field", dest="field",
                    default='001,002',
                    help="list of target fields", metavar="field")
    (options, args) = parser.parse_args()

    filternames = options.filternames.split(",")
    modules = options.modules.split(",")
    fields = options.field.split(",")

    with open(os.path.expanduser('~/.mast_api_token'), 'r') as fh:
        api_token = fh.read().strip()
    Mast.login(api_token.strip())
    Observations.login(api_token)

    field_to_reg_mapping = {'001': 'brick', '002': 'cloudc'}

    for field in fields:
        for filtername in filternames:
            for module in modules:
                print(f"Main Loop: {filtername} + {module} + {field}")
                results = main(filtername=filtername, module=module, Observations=Observations, field=field,
                               regionname=field_to_reg_mapping[field])


    from run_notebook import run_notebook
    basepath = '/orange/adamginsburg/jwst/brick/'
    #run_notebook(f'{basepath}/notebooks/PaA_Separation_nrca.ipynb')
    run_notebook(f'{basepath}/notebooks/PaA_Separation_nrcb.ipynb')
    #run_notebook(f'{basepath}/notebooks/F466_separation_nrca.ipynb')
    #run_notebook(f'{basepath}/notebooks/F466_separation_nrcb.ipynb')
    #run_notebook(f'{basepath}/notebooks/StarDestroyer_PaA_nrca.ipynb')
    run_notebook(f'{basepath}/notebooks/StarDestroyer_PaA_nrcb.ipynb')
    #run_notebook(f'{basepath}/notebooks/Stitch_A_to_B.ipynb')
