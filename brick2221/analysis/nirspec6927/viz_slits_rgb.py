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
from astropy import units as u
from astropy.visualization import simple_norm, quantity_support
from astropy.coordinates import SkyCoord
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

            for show_numbers in (True, False):

                ax, img, wcs, fig = imshow_png(image_file, swap=swap)

                shown_regs = []
                nregs_added = 0
                pbar = tqdm(desc="Adding regions", leave=False)

                # Overlay all regions
                for jj, (reg_name, reg_list) in enumerate(regions_dict.items()):
                    for ii, reg in enumerate(reg_list):
                        if len(shown_regs) > 0:
                            crd = SkyCoord([r.center for r in shown_regs], unit='deg')

                        if len(shown_regs) == 0 or reg.center.match_to_catalog_sky(crd)[1] > 1*u.arcsec:
                            shown_regs.append(reg)
                            # Convert region to pixel coordinates
                            pixel_reg = reg.to_pixel(wcs)
                            pixel_reg.visual['edgecolor'] = 'red'
                            pixel_reg.visual['fill'] = False
                            rect = pixel_reg.as_artist(edgecolor='red', linewidth=0.2, alpha=0.75, facecolor='none')
                            rect.set_facecolor('none')
                            rect.set_edgecolor('red')
                            ax.add_artist(rect)
                            if show_numbers:
                                # Get pixel coordinates using the WCS
                                # because the coords are flipped, we don't want to use ax.get_transform('world') unless we can be sure the coordinate is in the right order
                                ax.text_coord(reg.center, str(ii), color='red', fontsize=12)
                    pbar.set_description(f"Added {len(shown_regs)-nregs_added} regions ({len(shown_regs)} total) in {reg_name}. ({jj} of {len(regions_dict)})")
                    nregs_added = len(shown_regs)
                pbar.close()

                # Save the result
                output_name = os.path.basename(os.path.splitext(image_file)[0] + f'_with_slits{"_numbered" if show_numbers else ""}.png')
                plt.savefig(f'{basepath}/BrickNirspec/{output_name}', dpi=500, bbox_inches='tight')
                plt.close(fig)

if __name__ == "__main__":
    main()