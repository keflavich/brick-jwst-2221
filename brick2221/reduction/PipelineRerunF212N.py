#!/usr/bin/env python
from glob import glob
import os
os.environ["CRDS_PATH"] = "/orange/adamginsburg/jwst/brick/crds/"
os.environ["CRDS_SERVER_URL"] = "https://jwst-crds-pub.stsci.edu"
import shutil
import numpy as np
import json
import requests
import asdf
from astropy.io import ascii, fits
from astropy.utils.data import download_file
from astropy.visualization import ImageNormalize, ManualInterval, LogStretch, LinearStretch
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['savefig.dpi'] = 80
mpl.rcParams['figure.dpi'] = 80
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



import jwst
print(jwst.__version__)
# Files created in this notebook will be saved
# in a subdirectory of the base directory called `Stage3`
output_dir = '/orange/adamginsburg/jwst/brick/MAST_2022-08-29T1447/JWST/jw02221-o001_t001_nircam_clear-f212n/pipeline/'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

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



os.chdir(output_dir)


asn_file = os.path.join(output_dir, 'jw02221-o001_20220828t044842_image3_006_asn.json')


# Open the association file and load into a json object
with open(asn_file) as f_obj:
    asn_data = json.load(f_obj)
    f_obj.seek(0)
    asn_nrca4 = json.load(f_obj)


# In[18]:


asn_nrca4['products'][0]['members'] = [row for row in asn_nrca4['products'][0]['members']
                                       if 'nrca4' in row['expname']]
asn_file_nrca4 = os.path.join(output_dir, 'jw02221-o001_20220828t044842_image3_006_nrca4_asn.json')
with open(asn_file_nrca4, 'w') as fh:
    json.dump(asn_nrca4, fh)
len(asn_nrca4['products'][0]['members']), len(asn_data['products'][0]['members'])


# In[19]:


tweak_files = ['level3_lw_asn_0_tweakregstep.fits',
               'level3_lw_asn_1_tweakregstep.fits',
               'level3_lw_asn_2_tweakregstep.fits']
tweak_product = 'manual_asn_file'


# In[20]:


tweakreg_asn = asn_from_list.asn_from_list(tweak_files, rule=DMS_Level3_Base, product_name=tweak_product)


output_test = 'manual_tweakreg_asn.json'
with open(output_test, 'w') as outfile:
    name, serialized = tweakreg_asn.dump(format='json')
    outfile.write(serialized)



import crds


# In[24]:


rslt = download_file('https://jwst-crds.stsci.edu/unchecked_get/references/jwst/jwst_nircam_pars-tweakregstep_0037.asdf', cache=False)
shutil.move(rslt, './jwst_nircam_pars-tweakregstep_0037.asdf')


# In[25]:


tweak_param_reffile = 'jwst_nircam_pars-tweakregstep_0037.asdf'


# In[26]:


import pprint


# In[27]:


with asdf.open(tweak_param_reffile) as tweak_params:
    pprint.pprint(tweak_params.tree)




for module in ('nrca','nrcb'):
    for detector in range(1,5):
        with open(asn_file) as f_obj:
            asn_data = json.load(f_obj)
        asn_data['products'][0]['name'] = f'jw02221-o001_t001_nircam_clear-f212n-{module}{detector}'
        asn_data['products'][0]['members'] = [row for row in asn_data['products'][0]['members']
                                               if f'{module}{detector}' in row['expname']]
        asn_file_each = os.path.join(output_dir, f'jw02221-o001_20220828t044842_image3_006_{module}{detector}_asn.json')
        with open(asn_file_each, 'w') as fh:
            json.dump(asn_data, fh)
        len(asn_nrca4['products'][0]['members']), len(asn_data['products'][0]['members'])

        image3 = calwebb_image3.Image3Pipeline()

        image3.output_dir = output_dir
        image3.save_results = True

        image3.run(asn_file_each)
        print(f"DONE running {asn_file_each}")


input_files = [item['expname'] for item in asn_data_nrca4['products'][0]['members']]       


# In[ ]:


input_files


# Define the names of the other output files.

# In[ ]:


mosaic_file = os.path.join(output_dir, 'l3_lw_results_i2d.fits')
source_cat_file = os.path.join(output_dir, 'l3_lw_results_cat.ecsv')
segmentation_map_file = os.path.join(output_dir, 'l3_lw_results_segm.fits')
cr_flagged_files = [item.replace('cal.fits', 'crf.fits') for item in input_files]


# Read in the final mosaic image and display

# In[ ]:


mosaic = datamodels.open(mosaic_file)


# In[ ]:


show_image(mosaic.data, vmin=0, vmax=5)


# In[ ]:





# Let's look at the segmentation map that was created by the `source_catalog` step. This shows which pixels are associated with the identified sources.

# In[ ]:


seg_map = fits.getdata(segmentation_map_file)


# In[ ]:


show_image(seg_map, vmin=0, vmax=5, scale='linear')


# And now examine the actual source catalog. For each source, the catalog lists the location, along with flux and AB/Vega magnitude values in three different apertures, as well as calculated values for an infinite aperture. Within the documentation, you can see the [full list of column definitions](https://jwst-pipeline.readthedocs.io/en/stable/jwst/source_catalog/main.html#source-catalog-table).

# In[ ]:


source_cat = ascii.read(source_cat_file)


# In[ ]:


source_cat


# Finally, let's overlay the source catalog on top of the mosaic image. In order to cut down on the number of spurious detections, we only show sources above a minimum flux limit. Another way to cut down on the number of spurious detections would be to change some of the `source_catalog` parameter values when calling the pipeline above.

# In[ ]:


overlay_catalog(mosaic.data, source_cat, flux_limit=5e-7, vmin=0, vmax=10,
                title='Final mosaic with source catalog')


# [Top of Notebook](#top)
