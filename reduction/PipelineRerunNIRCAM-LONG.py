#!/usr/bin/env python
from glob import glob
from astroquery.mast import Mast, Observations
import os
import shutil
import numpy as np
import json
# import requests
import asdf
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

from align_to_catalogs import realign_to_vvv, merge_a_plus_b

from destreak import destreak

import crds

import pprint

import jwst
print(jwst.__version__)

# see 'destreak410.ipynb' for tests of this
medfilt_size = {'F410M': 15, 'F405N': 256, 'F466N': 55}

basepath = '/orange/adamginsburg/jwst/brick/'

def main():


    basepath = '/orange/adamginsburg/jwst/brick/'
    os.environ["CRDS_PATH"] = f"{basepath}/crds/"
    os.environ["CRDS_SERVER_URL"] = "https://jwst-crds-pub.stsci.edu"
    mpl.rcParams['savefig.dpi'] = 80
    mpl.rcParams['figure.dpi'] = 80

    with open(os.path.expanduser('/home/adamginsburg/.mast_api_token'), 'r') as fh:
        api_token = fh.read().strip()
    Mast.login(api_token.strip())
    Observations.login(api_token)


    for filtername in ('F405N', 'F466N', 'F410M'):
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
                                                proposal_id="2221",
                                                proposal_pi="Ginsburg*",
                                                calib_level=3,
                                               )
        print(len(obs_table))

        data_products_by_obs = Observations.get_product_list(obs_table[np.char.find(obs_table['obs_id'], filtername.lower()) >= 0])
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


        if True:
            for module in ('nrca', 'nrcb'):
                print(f"Filter {filtername} module {module}")
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
                tweakreg_asdf_filename = [x for x in mapping.todict()['selections'] if filtername in (x[1:3])][0][4]
                tweakreg_asdf = asdf.open(f'https://jwst-crds.stsci.edu/unchecked_get/references/jwst/{tweakreg_asdf_filename}')
                tweakreg_parameters = tweakreg_asdf.tree['parameters']
                print(f'Filter {filtername}: {tweakreg_parameters}')

                with open(asn_file) as f_obj:
                    asn_data = json.load(f_obj)
                asn_data['products'][0]['name'] = f'jw02221-o001_t001_nircam_clear-{filtername.lower()}-{module}'
                asn_data['products'][0]['members'] = [row for row in asn_data['products'][0]['members']
                                                        if f'{module}' in row['expname']]

                for member in asn_data['products'][0]['members']:
                    outname = destreak(member['expname'],
                                       use_background_map=True,
                                       median_filter_size=2048)  # median_filter_size=medfilt_size[filtername])
                    member['expname'] = outname


                asn_file_each = asn_file.replace("_asn.json", f"_{module}_asn.json")
                with open(asn_file_each, 'w') as fh:
                    json.dump(asn_data, fh)

                image3 = calwebb_image3.Image3Pipeline()

                image3.output_dir = output_dir
                image3.save_results = True
                for par in tweakreg_parameters:
                    setattr(image3.tweakreg, par, tweakreg_parameters[par])

                image3.tweakreg.fit_geometry = 'general'
                image3.tweakreg.brightest = 10000
                image3.tweakreg.snr_threshold = 5
                image3.tweakreg.nclip = 1

                image3.run(asn_file_each)
                print(f"DONE running {asn_file_each}")

                realign_to_vvv(filtername=filtername.lower(), module=module)

        print("Merging already-combined nrca + nrcb modules")
        merge_a_plus_b(filtername)
        print("DONE Merging already-combined nrca + nrcb modules")

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
        print(f"Mapping: {mapping.todict()['selections']}")
        print(f"Filtername: {filtername}")
        filter_match = [x for x in mapping.todict()['selections'] if filtername in x]
        print(filter_match)
        print(len(filter_match))
        tweakreg_asdf_filename = filter_match[0][4]
        tweakreg_asdf = asdf.open(f'https://jwst-crds.stsci.edu/unchecked_get/references/jwst/{tweakreg_asdf_filename}')
        tweakreg_parameters = tweakreg_asdf.tree['parameters']
        print(f'Filter {filtername}: {tweakreg_parameters}')

        with open(asn_file) as f_obj:
            asn_data = json.load(f_obj)

        for member in asn_data['products'][0]['members']:
            outname = destreak(member['expname'],
                               use_background_map=True,
                               median_filter_size=2048)  # median_filter_size=medfilt_size[filtername])
            member['expname'] = outname

        asn_data['products'][0]['name'] = f'jw02221-o001_t001_nircam_clear-{filtername.lower()}-merged'
        asn_file_merged = asn_file.replace("_asn.json", f"_merged_asn.json")
        with open(asn_file_merged, 'w') as fh:
            json.dump(asn_data, fh)

        image3 = calwebb_image3.Image3Pipeline()

        image3.output_dir = output_dir
        image3.save_results = True
        for par in tweakreg_parameters:
            setattr(image3.tweakreg, par, tweakreg_parameters[par])


        vvvdr2fn = (f'{basepath}/{filtername.upper()}/pipeline/jw02221-o001_t001_nircam_clear-{filtername}-{module}_vvvcat.ecsv')
        print(vvvdr2fn)
        if os.path.exists(vvvdr2fn):
            image3.tweakreg.abs_refcat = vvvdr2fn
            image3.tweakreg.abs_searchrad = 1
        else:
            print(f"Did not find VVV catalog {vvvdr2fn}")

        image3.tweakreg.fit_geometry = 'general'
        image3.tweakreg.brightest = 10000
        image3.tweakreg.snr_threshold = 5
        image3.tweakreg.nclip = 1

        image3.run(asn_file_merged)
        print(f"DONE running {asn_file_merged}")

        realign_to_vvv(filtername=filtername.lower(), module='merged')



    globals().update(locals())
    return locals()

if __name__ == "__main__":
    results = main()

    from run_notebook import run_notebook
    basepath = '/orange/adamginsburg/jwst/brick/'
    run_notebook(f'{basepath}/notebooks/BrA_Separation_nrca.ipynb')
    run_notebook(f'{basepath}/notebooks/BrA_Separation_nrcb.ipynb')
    run_notebook(f'{basepath}/notebooks/F466_separation_nrca.ipynb')
    run_notebook(f'{basepath}/notebooks/F466_separation_nrcb.ipynb')
    run_notebook(f'{basepath}/notebooks/StarDestroyer_nrca.ipynb')
    run_notebook(f'{basepath}/notebooks/StarDestroyer_nrcb.ipynb')
    run_notebook(f'{basepath}/notebooks/Stitch_A_to_B.ipynb')


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