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

import crds

import pprint

import jwst
print(jwst.__version__)


def download_files(files, output_directory, force=False):
    """Given a tuple or list of tuples containing (URL, filename),
    download the given files into the current working directory.
    Downloading is done via astropy's download_file. A symbolic link
    is created in the specified output dirctory that points to the
    downloaded file.

    Parameters
    ----------
    files : tuple or list of tuples
        Each 2-tuple should contain (URL, filename), where
        URL is the URL from which to download the file, and
        filename will be the name of the symlink pointing to
        the downloaded file.

    output_directory : str
        Name of the directory in which to create the symbolic
        links to the downloaded files

    force : bool
        If True, the file will be downloaded regarless of whether
        it is already present or not.

    Returns
    -------
    filenames : list
        List of filenames corresponding to the symbolic links
        of the downloaded files
    """
    # In the case of a single input tuple, make it a
    # 1 element list, for consistency.
    filenames = []
    if isinstance(files, tuple):
        files = [files]

    for file in files:
        filenames.append(file[1])
        if force:
            print('Downloading {}...'.format(file[1]))
            demo_file = download_file(file[0], cache='update')
            # Make a symbolic link using a local name for convenience
            if not os.path.islink(os.path.join(output_directory, file[1])):
                os.symlink(demo_file, os.path.join(output_directory, file[1]))
        else:
            if not os.path.isfile(os.path.join(output_directory, file[1])):
                print('Downloading {}...'.format(file[1]))
                demo_file = download_file(file[0], cache=True)
                # Make a symbolic link using a local name for convenience
                os.symlink(demo_file, os.path.join(output_directory, file[1]))
            else:
                print('{} already exists, skipping download...'.format(file[1]))
                continue
    return filenames


# In[11]:


def find_bad_pix_types(dq_value):
    """Given an integer representation of a series of bad pixel flags,
    identify which types of bad pixels the flags indicate.

    Parameters
    ----------
    dq_value : uint16
        Value associated with a set of bad pixel flags

    Returns
    -------
    bad_nums : list
        List of integers representing the bad pixel types

    bad_types : list
        List of bad pixel type names corresponding to bad_nums
    """
    # Change integer into a byte array
    bitarr = np.binary_repr(dq_value)

    # Find the bad pixel type associated with each bit where
    # the flag is set
    bad_nums = []
    bad_types = []
    for i, elem in enumerate(bitarr[::-1]):
        if elem == str(1):
            badval = 2**i
            bad_nums.append(badval)
            key = next(key for key, value in datamodels.dqflags.pixel.items() if value == badval)
            bad_types.append(key)
    return bad_nums, bad_types


# In[12]:


def overlay_catalog(data_2d, catalog, flux_limit=0, vmin=0, vmax=10,
                    title=None, units='MJy/str'):
    """Function to generate a 2D image of the data,
    with sources overlaid.

    data_2d : numpy.ndarray
        2D image to be displayed

    catalog : astropy.table.Table
        Table of sources

    flux_limit : float
        Minimum signal threshold to overplot sources from catalog.
        Sources below this limit will not be shown on the image.

    vmin : float
        Minimum signal value to use for scaling

    vmax : float
        Maximum signal value to use for scaling

    title : str
        String to use for the plot title

    units : str
        Units of the data. Used for the annotation in the
        color bar
    """
    norm = ImageNormalize(data_2d, interval=ManualInterval(vmin=vmin, vmax=vmax),
                              stretch=LogStretch())
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(data_2d, origin='lower', norm=norm)

    for row in catalog:
        if row['aper_total_flux'].value > flux_limit:
            plt.plot(row['xcentroid'], row['ycentroid'], marker='o',
                     markersize='3', color='red')

    plt.xlabel('Pixel column')
    plt.ylabel('Pixel row')

    fig.colorbar(im, label=units)
    fig.tight_layout()
    plt.subplots_adjust(left=0.15)

    if title:
        plt.title(title)


# In[13]:


def show_image(data_2d, vmin, vmax, xpixel=None, ypixel=None, title=None,
               scale='log', units='MJy/str'):
    """Function to generate a 2D, log-scaled image of the data,
    with an option to highlight a specific pixel.

    data_2d : numpy.ndarray
        2D image to be displayed

    vmin : float
        Minimum signal value to use for scaling

    vmax : float
        Maximum signal value to use for scaling

    xpixel : int
        X-coordinate of pixel to highlight

    ypixel : int
        Y-coordinate of pixel to highlight

    title : str
        String to use for the plot title

    scale : str
        Specify scaling of the image. Can be 'log' or 'linear'

    units : str
        Units of the data. Used for the annotation in the
        color bar
    """
    if scale == 'log':
        norm = ImageNormalize(data_2d, interval=ManualInterval(vmin=vmin, vmax=vmax),
                              stretch=LogStretch())
    elif scale == 'linear':
        norm = ImageNormalize(data_2d, interval=ManualInterval(vmin=vmin, vmax=vmax),
                              stretch=LinearStretch())
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(data_2d, origin='lower', norm=norm)

    if xpixel and ypixel:
        plt.plot(xpixel, ypixel, marker='o', color='red', label='Selected Pixel')

    fig.colorbar(im, label=units)
    plt.xlabel('Pixel column')
    plt.ylabel('Pixel row')
    if title:
        plt.title(title)



def main():


    os.environ["CRDS_PATH"] = "/orange/adamginsburg/jwst/brick/crds/"
    os.environ["CRDS_SERVER_URL"] = "https://jwst-crds-pub.stsci.edu"
    mpl.rcParams['savefig.dpi'] = 80
    mpl.rcParams['figure.dpi'] = 80
    
    with open(os.path.expanduser('/home/adamginsburg/.mast_api_token'), 'r') as fh:
        api_token = fh.read().strip()
    Mast.login(api_token.strip())
    Observations.login(api_token)


    for filtername in ('F187N', 'F182M', 'F212N'):
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
        print(len(obs_table))

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


        if False:
            for module in ('nrca', 'nrcb'):
                for detector in range(1, 5):
                    print(f"Filter {filtername} module {module} detector {detector}")
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
                    asn_data['products'][0]['name'] = f'jw02221-o001_t001_nircam_clear-{filtername.lower()}-{module}{detector}'
                    asn_data['products'][0]['members'] = [row for row in asn_data['products'][0]['members']
                                                        if f'{module}{detector}' in row['expname']]
                    asn_file_each = asn_file.replace("_asn.json", f"_{module}{detector}_asn.json")
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
        asn_file_merged = asn_file.replace("_asn.json", f"_merged_asn.json")
        with open(asn_file_merged, 'w') as fh:
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

        image3.run(asn_file_merged)
        print(f"DONE running {asn_file_merged}")


    globals().update(locals())
    return locals()

if __name__ == "__main__":
    results = main()
