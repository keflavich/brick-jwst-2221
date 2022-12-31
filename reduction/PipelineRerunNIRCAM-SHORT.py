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
from astropy.utils.data import download_file
from astropy.visualization import ImageNormalize, ManualInterval, LogStretch, LinearStretch
import matplotlib.pyplot as plt
import matplotlib as mpl
from jwst.pipeline import calwebb_image3

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

from align_to_catalogs import realign_to_vvv, merge_a_plus_b

import crds

import pprint

import jwst
print(jwst.__version__)

# see 'destreak187.ipynb' for tests of this
# really this is just eyeballed; probably the 1/f noise is present at the same level in all of these, but you can't see it as well on
# the medium-band filters.
medfilt_size = {'F182M': 55, 'F187N': 512, 'F212N': 512}


def main(filtername, module, Observations=None):
    log.info(f"Processing filter {filtername} module {module}")

    basepath = '/orange/adamginsburg/jwst/brick/'
    os.environ["CRDS_PATH"] = "/orange/adamginsburg/jwst/brick/crds/"
    os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"
    mpl.rcParams['savefig.dpi'] = 80
    mpl.rcParams['figure.dpi'] = 80

    # Files created in this notebook will be saved
    # in a subdirectory of the base directory called `Stage3`
    output_dir = f'/orange/adamginsburg/jwst/brick/{filtername}/pipeline/'
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
                                            #obs_id='jw02221-o001_t001_nircam_clear-f182m',
                                            filters=filtername.lower(),
                                            proposal_id="2221",
                                            proposal_pi="Ginsburg*",
                                            calib_level=3,
                                            )
    print(f"Observation table length: ", len(obs_table))

    data_products_by_obs = Observations.get_product_list(obs_table[obs_table['calib_level'] == 3])
    print(len(data_products_by_obs))

    products_asn = Observations.filter_products(data_products_by_obs, extension="json")
    print(len(products_asn))
    valid_obsids = products_asn['obs_id'][np.char.find(np.unique(products_asn['obs_id']), 'jw02221-o001', ) == 0]
    match = [x for x in valid_obsids if filtername.lower() in x][0]

    asn_mast_data = products_asn[products_asn['obs_id'] == match]
    print(asn_mast_data)

    manifest = Observations.download_products(asn_mast_data, download_dir=output_dir)
    print(manifest)

    # MAST creates deep directory structures we don't want
    for row in manifest:
        try:
            shutil.move(row['Local Path'], os.path.join(output_dir, os.path.basename(row['Local Path'])))
        except Exception as ex:
            print(f"Failed to move file with error {ex}")


    if module in ('nrca', 'nrcb'):
        print(f"Filter {filtername} module {module} ")
        print(f"Searching for {os.path.join(output_dir, f'jw02221-*_image3_0[0-9][0-9]_asn.json')}")
        asn_file_search = glob(os.path.join(output_dir, f'jw02221-*_image3_0[0-9][0-9]_asn.json'))
        if len(asn_file_search) == 1:
            asn_file = asn_file_search[0]
        elif len(asn_file_search) > 1:
            asn_file = sorted(asn_file_search)[-1]
            print(f"Found multiple asn files: {asn_file_search}.  Using the more recent one, {asn_file}.")
        else:
            raise ValueError("Mismatch")

        mapping = crds.rmap.load_mapping('/orange/adamginsburg/jwst/brick/crds/mappings/jwst/jwst_nircam_pars-tweakregstep_0003.rmap')
        tweakreg_asdf_filename = [x for x in mapping.todict()['selections'] if x[1] == filtername][0][4]
        tweakreg_asdf = asdf.open(f'https://jwst-crds.stsci.edu/unchecked_get/references/jwst/{tweakreg_asdf_filename}')
        tweakreg_parameters = tweakreg_asdf.tree['parameters']
        print(f'Filter {filtername}: {tweakreg_parameters}')

        with open(asn_file) as f_obj:
            asn_data = json.load(f_obj)
        asn_data['products'][0]['name'] = f'jw02221-o001_t001_nircam_clear-{filtername.lower()}-{module}'
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

        image3 = calwebb_image3.Image3Pipeline()

        image3.output_dir = output_dir
        image3.save_results = True
        for par in tweakreg_parameters:
            setattr(image3.tweakreg, par, tweakreg_parameters[par])

        image3.tweakreg.fit_geometry = 'general'
        # image3.tweakreg.brightest = 10000
        # image3.tweakreg.snr_threshold = 5
        # image3.tweakreg.nclip = 1

        # reference to long-wavelength catalogs
        image3.tweakreg.abs_refcat = f'{basepath}/catalogs/crowdsource_based_nircam-long_reference_astrometric_catalog.ecsv'
        image3.tweakreg.abs_searchrad = 0.5

        # try .... something else?
        image3.tweakreg.brightest = 2000
        image3.tweakreg.snr_threshold = 15
        image3.tweakreg.nclip = 7
        image3.tweakreg.peakmax = 1400

        image3.tweakreg.searchrad = 1 # 1 arcsec instead of 2
        image3.tweakreg.separation = 0.5 # min separation 0.4 arcsec instead of 1 (Mihai suggesteed separation = 2x tolerance)
        image3.tweakreg.tolerance = 0.3 # max tolerance 0.2 instead of 0.7


        image3.run(asn_file_each)
        print(f"DONE running {asn_file_each}")
        # don't realign now
        #realigned = realign_to_vvv(filtername=filtername.lower(), module=module)

        log.info("Removing saturated stars")
        remove_saturated_stars(f'jw02221-o001_t001_nircam_clear-{filtername.lower()}-{module}_i2d.fits')

    if module == 'nrcb':
        # assume nrca is run before nrcb
        print("Merging already-combined nrca + nrcb modules")
        merge_a_plus_b(filtername)
        print("DONE Merging already-combined nrca + nrcb modules")

    if module == 'merged':
        log.info("Running merged frames")
        # try merging all frames & modules

        asn_file_search = glob(os.path.join(output_dir, f'jw02221-*_image3_0[0-9][0-9]_asn.json'))
        if len(asn_file_search) == 1:
            asn_file = asn_file_search[0]
        elif len(asn_file_search) > 1:
            asn_file = sorted(asn_file_search)[-1]
            print(f"Found multiple asn files: {asn_file_search}.  Using the more recent one, {asn_file}.")
        else:
            raise ValueError("Mismatch")

        mapping = crds.rmap.load_mapping('/orange/adamginsburg/jwst/brick/crds/mappings/jwst/jwst_nircam_pars-tweakregstep_0003.rmap')
        tweakreg_asdf_filename = [x for x in mapping.todict()['selections'] if x[1] == filtername][0][4]
        tweakreg_asdf = asdf.open(f'https://jwst-crds.stsci.edu/unchecked_get/references/jwst/{tweakreg_asdf_filename}')
        tweakreg_parameters = tweakreg_asdf.tree['parameters']
        print(f'Filter {filtername}: {tweakreg_parameters}')

        with open(asn_file) as f_obj:
            asn_data = json.load(f_obj)
        asn_data['products'][0]['name'] = f'jw02221-o001_t001_nircam_clear-{filtername.lower()}-merged'

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

        image3 = calwebb_image3.Image3Pipeline()

        image3.output_dir = output_dir
        image3.save_results = True
        for par in tweakreg_parameters:
            setattr(image3.tweakreg, par, tweakreg_parameters[par])

        # # TODO: instead, use F410M as the astrometric reference, since that matches _better_ to VVV

        # vvvdr2fn = (f'{basepath}/{filtername.upper()}/pipeline/jw02221-o001_t001_nircam_clear-{filtername}-{module}_vvvcat.ecsv')
        # print(vvvdr2fn)
        # if os.path.exists(vvvdr2fn):
        #     image3.tweakreg.abs_refcat = vvvdr2fn
        #     image3.tweakreg.abs_searchrad = 1
        # else:
        #     print(f"Did not find VVV catalog {vvvdr2fn}")

        image3.tweakreg.abs_refcat = f'{basepath}/catalogs/crowdsource_based_nircam-long_reference_astrometric_catalog.ecsv'
        image3.tweakreg.abs_searchrad = 0.5

        # try .... something else?
        image3.tweakreg.brightest = 2000
        image3.tweakreg.snr_threshold = 15
        image3.tweakreg.nclip = 7
        image3.tweakreg.peakmax = 1400

        image3.tweakreg.searchrad = 1 # 1 arcsec instead of 2
        image3.tweakreg.separation = 0.6 # min separation 0.4 arcsec instead of 1 (Mihai suggesteed separation = 2x tolerance)
        image3.tweakreg.tolerance = 0.3 # max tolerance 0.2 instead of 0.7


        image3.run(asn_file_merged)
        print(f"DONE running {asn_file_merged}")

        # realignment doesn't work
        # realign_to_vvv(filtername=filtername.lower(), module='merged')

        log.info("Removing saturated stars")
        remove_saturated_stars(f'jw02221-o001_t001_nircam_clear-{filtername.lower()}-merged_i2d.fits')


    globals().update(locals())
    return locals()

if __name__ == "__main__":
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-f", "--filternames", dest="filternames",
                      default='F212N,F182M,F187N',
                      help="filter name list", metavar="filternames")
    # merged requires >512 GB of memory, apparently - it fails with OOM kill
    parser.add_option("-m", "--modules", dest="modules",
                    default='nrca,nrcb,merged',
                    help="module list", metavar="modules")
    (options, args) = parser.parse_args()

    filternames = options.filternames.split(",")
    modules = options.modules.split(",")

    with open(os.path.expanduser('/home/adamginsburg/.mast_api_token'), 'r') as fh:
        api_token = fh.read().strip()
    Mast.login(api_token.strip())
    Observations.login(api_token)


    for filtername in filternames:
        for module in modules:
            results = main(filtername=filtername, module=module, Observations=Observations)


    from run_notebook import run_notebook
    basepath = '/orange/adamginsburg/jwst/brick/'
    #run_notebook(f'{basepath}/notebooks/PaA_Separation_nrca.ipynb')
    run_notebook(f'{basepath}/notebooks/PaA_Separation_nrcb.ipynb')
    #run_notebook(f'{basepath}/notebooks/F466_separation_nrca.ipynb')
    #run_notebook(f'{basepath}/notebooks/F466_separation_nrcb.ipynb')
    #run_notebook(f'{basepath}/notebooks/StarDestroyer_PaA_nrca.ipynb')
    run_notebook(f'{basepath}/notebooks/StarDestroyer_PaA_nrcb.ipynb')
    #run_notebook(f'{basepath}/notebooks/Stitch_A_to_B.ipynb')
