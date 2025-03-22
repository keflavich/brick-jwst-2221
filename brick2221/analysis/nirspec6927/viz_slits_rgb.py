"""
Show the NIRSpec slits on 3-color images
"""
import os
import glob
import warnings
import PIL
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import simple_norm
import regions
from pyavm import AVM
import pyavm
from tqdm.auto import tqdm

plt.ioff()

basepath = '/Users/adam/Dropbox/brick2221'
filenames = [('BrickJWST_merged_longwave_narrowband_lighter.png', True), 
             ('BrickJWST_merged_longwave_narrowband_rotated_withstars.png', True),
             ('BrickJWST_merged_longwave_narrowband_withstars.png', True),
             ('BrickJWST_1182p2221_*png', False),
             #('Spitzer_RGB_*png', False),
             ]

def imshow_png(image_file, swap=False, fig=None, reg=None, margin=5):
    # Read the image
    # alternative: img = plt.imread(image_file)
    img = np.array(PIL.Image.open(image_file))
    
    # Get WCS from AVM metadata
    avm = AVM.from_image(image_file)
    wcs = avm.to_wcs()

    # alternative wcs = WCS(fits.Header.fromstring(avm.Spatial.FITSheader))
    if swap:
        wcs = wcs.sub((2,1))
    else:
        img = img[::-1, :].swapaxes(0, 1)
        wcs = wcs.sub((2,1))

    if reg is not None:
        shape = img.shape[:2]
        cutout_mask = reg.to_pixel(wcs).to_mask()
        slc_big, slc_small = cutout_mask.get_overlap_slices(shape)
        if slc_big is None or any(x is None for x in slc_big):
            raise ValueError(f"No overlap between region and image for {image_file}")
        slc_bigger = tuple([slice(max(0, s.start - margin), min(s.stop + margin, d)) for s, d in zip(slc_big, shape)])
        ww = wcs[slc_bigger]
        cutout_image = img[slc_bigger[0], slc_bigger[1], :]
    else:
        ww = wcs
        cutout_image = img
    
    # Create figure
    if fig is None:
        fig = plt.figure(figsize=(15, 7.5))
    ax = fig.add_subplot(111, projection=ww)
    
    # Show the image
    ax.imshow(cutout_image)

    ax.coords['ra'].set_axislabel('Right Ascension')
    ax.coords['dec'].set_axislabel('Declination')

    ra = lon = ax.coords['ra']
    dec = lat = ax.coords['dec']
    ra.set_major_formatter('hh:mm:ss.ss')
    dec.set_major_formatter('dd:mm:ss.ss')

    ax.set_ylim(ax.get_ylim()[::-1])

    if swap:
        ra.set_ticks_position('l')
        ra.set_ticklabel_position('l')
        ra.set_axislabel_position('l')
        dec.set_ticks_position('b')
        dec.set_ticklabel_position('b')
        dec.set_axislabel_position('b')

    return ax, img, ww, fig


def main():
    # Get all region files
    region_files = glob.glob('*_regions.reg')
    regions_dict = {os.path.basename(fn).replace('_regions.reg', ''): regions.Regions.read(fn) 
                    for fn in region_files}

    # Process each RGB image
    for filename, swap in filenames:
        # Handle glob patterns
        if '*' in filename:
            image_files = glob.glob(os.path.join(basepath, filename))
        else:
            image_files = [os.path.join(basepath, filename)]
        
        for image_file in tqdm(image_files, desc=f"Processing {filename}"):
            
            ax, img, wcs, fig = imshow_png(image_file, swap=swap)
            
            # Overlay all regions
            for reg_name, reg_list in regions_dict.items():
                for reg in reg_list:
                    # Convert region to pixel coordinates
                    pixel_reg = reg.to_pixel(wcs)
                    pixel_reg.plot(ax=ax, edgecolor='red', linewidth=1, alpha=0.5)
            
            # Save the result
            output_name = os.path.basename(os.path.splitext(image_file)[0] + '_with_slits.png')
            plt.savefig(f'{basepath}/BrickNirspec/{output_name}', dpi=300, bbox_inches='tight')
            plt.close(fig)

if __name__ == "__main__":
    main()