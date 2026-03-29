import os
import warnings
import glob
import socket
import astropy.io.fits as fits
import regions
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO
from tqdm.auto import tqdm
from astropy.visualization import simple_norm
from pyavm import AVM

from viz_slits_rgb import imshow_png

plt.ioff()

computer_name = socket.gethostname()
if 'cyg' in computer_name:
    imagepath = '/Users/adam/Dropbox/brick2221/20250322_FITS/'
else:
    imagepath = '/orange/adamginsburg/jwst/brick/images/'

def imshow_fits_cutout(hdu, reg, wcs, fig=None, margin=5):
    """Create a cutout from a FITS file for a given region"""
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        wcs = WCS(hdu['SCI'].header)

    cutout_mask = reg.to_pixel(wcs).to_mask()
    slc_big, slc_small = cutout_mask.get_overlap_slices(hdu['SCI'].data.shape)
    if slc_big is None or any(x is None for x in slc_big):
        return None, None
    slc_bigger = tuple([slice(max(0, s.start - margin), min(s.stop + margin, d)) for s, d in zip(slc_big, hdu['SCI'].data.shape)])
    ww = wcs[slc_bigger]
    cutout_image = hdu['SCI'].data[slc_bigger]

    if fig is None:
        fig = plt.figure(figsize=(1, 1))
    ax = fig.add_axes([x_start, 0.05, cutout_size/available_width, 0.9], projection=ww)
    ax.imshow(cutout_image, origin='lower', cmap='gray_r', aspect='equal',
                norm=simple_norm(cutout_image, stretch='asinh'))

    return ww, cutout_image, ax

def create_region_cutouts(fits_files, regs, swaps=None, suffix=''):
    # Number of cutouts per row
    n_cutouts_per_row = len(fits_files)

    # Page dimensions and margins in inches
    page_width_inches = 8.5  # Letter width in inches
    page_height_inches = 11.0  # Letter height in inches
    margin_inches = 0.5  # Increased margin to prevent cutoff

    # Size calculations in inches
    label_size = 0.6  # Label width
    padding = 0.05  # Padding between cutouts
    total_padding = padding * (n_cutouts_per_row - 1)
    available_width = page_width_inches - 2*margin_inches
    cutout_size = (available_width - label_size - total_padding) / n_cutouts_per_row
    row_height = cutout_size * 1.1  # Slightly more compact vertical spacing
    header_height = 0.3  # Height for the header row

    dpi = 150

    # Print debug information
    print(f"Page dimensions: {page_width_inches}x{page_height_inches} inches")
    print(f"Available width: {available_width} inches")
    print(f"Label size: {label_size} inches")
    print(f"Cutout size: {cutout_size} inches")
    print(f"Row height: {row_height} inches")
    print(f"Total width used: {label_size + n_cutouts_per_row*cutout_size + total_padding} inches")

    # Calculate how many rows we can fit per page
    usable_height = page_height_inches - 2*margin_inches - header_height
    rows_per_page = int(usable_height / row_height)
    print(f"Rows per page: {rows_per_page}")
    print(f"Usable height: {usable_height} inches")

    # Process each region file
    for reg_name, reg_list in regs.items():
        # Create PDF for this region file
        pdf_filename = f"region_cutouts_{reg_name}{suffix}.pdf"
        print(f"Creating {pdf_filename}")
        c = canvas.Canvas(pdf_filename, pagesize=letter)

        # Create and add header row
        header_fig = plt.figure(figsize=(available_width, header_height))

        # Add empty space for the "Reg X" column
        ax_empty = header_fig.add_axes([0.01, 0.1, label_size/available_width, 0.8])
        ax_empty.axis('off')

        # Add column labels
        for idx, name in enumerate(fits_files.keys()):
            x_start = (label_size + idx * (cutout_size + padding)) / available_width
            ax = header_fig.add_axes([x_start, 0.1, cutout_size/available_width, 0.8])
            ax.text(0.5, 0.5, name,
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=10,
                    rotation=45)
            ax.axis('off')

        # Save header to PDF
        buf = BytesIO()
        header_fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        img = ImageReader(buf)

        # Position header at top of page
        y_pos = page_height_inches - margin_inches - header_height
        y_pos_points = y_pos * 72
        x_pos_points = margin_inches * 72
        width_points = available_width * 72
        height_points = header_height * 72

        c.drawImage(img, x_pos_points, y_pos_points, width=width_points, height=height_points)
        plt.close(header_fig)

        current_page = 0
        for regidx, reg in tqdm(enumerate(reg_list), total=len(reg_list)):
            # Calculate row position within current page
            row_within_page = regidx % rows_per_page
            if row_within_page == 0 and regidx > 0:
                c.showPage()
                current_page += 1

                # Add header to new page
                buf = BytesIO()
                header_fig = plt.figure(figsize=(available_width, header_height))
                ax_empty = header_fig.add_axes([0.01, 0.1, label_size/available_width, 0.8])
                ax_empty.axis('off')
                for idx, name in enumerate(fits_files.keys()):
                    x_start = (label_size + idx * (cutout_size + padding)) / available_width
                    ax = header_fig.add_axes([x_start, 0.1, cutout_size/available_width, 0.8])
                    ax.text(0.5, 0.5, name,
                            horizontalalignment='center',
                            verticalalignment='center',
                            fontsize=10,
                            rotation=45)
                    ax.axis('off')
                header_fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0)
                buf.seek(0)
                img = ImageReader(buf)
                c.drawImage(img, x_pos_points, y_pos_points, width=width_points, height=height_points)
                plt.close(header_fig)

            # Create figure for this row
            fig = plt.figure(figsize=(available_width, cutout_size))

            # Create label subplot
            ax_label = fig.add_axes([0.01, 0.1, label_size/available_width, 0.8])
            ax_label.text(0.5, 0.5, f'Reg {regidx+1}',
                         horizontalalignment='center',
                         verticalalignment='center',
                         fontsize=12)
            ax_label.axis('off')

            for idx, (name, data) in enumerate(fits_files.items()):
                x_start = (label_size + idx * (cutout_size + padding)) / available_width
                # not used? ax = fig.add_axes([x_start, 0.05, cutout_size/available_width, 0.9])

                if isinstance(data, fits.HDUList):
                    # Handle FITS file
                    margin = 5 if int(name[1:-1]) > 278 else 10
                    ww, cutout_image, ax = imshow_fits_cutout(data, reg, None, margin=margin, fig=fig)
                    if ww is None:
                        ax.text(0.5, 0.5, 'No overlap',
                               horizontalalignment='center',
                               verticalalignment='center')
                        ax.axis('off')
                        continue
                else:
                    # Handle PNG file
                    try:
                        ax, img, ww, fig = imshow_png(data, swap=swaps[data], fig=fig, reg=reg)
                        ax.set_position([x_start, 0.05, cutout_size/available_width, 0.9])
                    except ValueError as ex:
                        ax = fig.add_axes([x_start, 0.05, cutout_size/available_width, 0.9])
                        ax.text(0.5, 0.5, 'No Overlap',
                               horizontalalignment='center',
                               verticalalignment='center')
                        ax.axis('off')
                        continue
                    except Exception as ex:
                        raise ex

                # Plot region
                pixel_reg = reg.to_pixel(ww)
                pixel_reg.plot(ax=ax, edgecolor='red', linewidth=1)
                # DEBUG print(reg, pixel_reg)

                # Hide coordinates
                ax.coords[0].set_ticks_visible(False)
                ax.coords[1].set_ticks_visible(False)
                ax.coords[0].set_ticklabel_visible(False)
                ax.coords[1].set_ticklabel_visible(False)

            # Save the row to PDF
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            img = ImageReader(buf)

            # Calculate y position from top of page, accounting for header
            y_pos = page_height_inches - margin_inches - header_height - (row_within_page * row_height) - cutout_size
            y_pos_points = y_pos * 72
            x_pos_points = margin_inches * 72
            width_points = available_width * 72
            height_points = cutout_size * 72

            c.drawImage(img, x_pos_points, y_pos_points, width=width_points, height=height_points)
            plt.close(fig)

            # DEBUG if regidx > 10:
            # DEBUG     break

        # Save and close this region's PDF
        c.save()
        print(f"Saved {pdf_filename}")

if __name__ == "__main__":

    regs = {regname.split("_")[0]: regions.Regions.read(regname) for regname in glob.glob("*regions.reg")}
    if False: # temporary
        fits_files = {os.path.basename(fn).split("-")[2]: fits.open(fn) for fn in glob.glob(f'{imagepath}/jw*merged_i2d.fits')}
        fits_files = {k: v for k, v in sorted(fits_files.items())}

        create_region_cutouts(fits_files, regs)

    imgpath = '/Users/adam/Dropbox/brick2221/'
    avm_files = [('starless', f'{imgpath}/BrickJWST_merged_longwave_narrowband_lighter.png', True),
                 ('405-410-466', f'{imgpath}/BrickJWST_merged_longwave_narrowband_rotated_withstars.png', True),
                 #('405-410-466', f'{imgpath}/BrickJWST_merged_longwave_narrowband_withstars.png', True),
                 ('200-187-115', f'{imgpath}/BrickJWST_1182p2221_200_187_115.png', False),
                 ('356-212-200', f'{imgpath}/BrickJWST_1182p2221_356_212_200.png', False),
                 ('410-200-182', f'{imgpath}/BrickJWST_1182p2221_410_200_182.png', False)]

    swaps = {fn: swap for name, fn, swap in avm_files}
    avm_filenames = {name: fn for name, fn, _ in avm_files}

    create_region_cutouts(avm_filenames, regs, swaps=swaps, suffix='_color')