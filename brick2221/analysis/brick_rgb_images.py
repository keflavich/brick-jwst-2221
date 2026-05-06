#!/usr/bin/env python
# coding: utf-8

from astropy.io import fits
import numpy as np
from astropy.visualization import simple_norm
import pylab as plt
from astropy import wcs
import os
from reproject import reproject_interp
import reproject
import PIL
import shutil
from astropy.wcs import WCS
import pyavm

def save_rgb(img, filename, avm=None, flip=-1, alma_data=None, alma_level=None):
    img = (img*256)
    img[img<0] = 0
    img[img>255] = 255
    img = img.astype('uint8')

    if alma_data is not None and alma_level is not None:
        contour_mask = np.zeros_like(alma_data, dtype=bool)
        contour_mask[alma_data >= alma_level] = True
        from scipy.ndimage import binary_dilation
        contour_mask1 = binary_dilation(contour_mask)
        contour_mask = contour_mask1 ^ contour_mask

        for i in range(3):
            img[contour_mask, i] = 255 - img[contour_mask, i]

    img = PIL.Image.fromarray(img[::flip,:,:])
    img.save(filename)

    if avm is not None:
        base = os.path.basename(filename)
        dir = os.path.dirname(filename)
        avmname = os.path.join(dir, 'avm_'+base)
        avm.embed(filename, avmname)
        shutil.move(avmname, filename)

    filename = filename.replace('.png', '.jpg')

    img.save(filename, format='JPEG',
             quality=95,
             progressive=True)

    return img

# Updated image filenames for Brick data with correct project codes
image_filenames_pipe = {
    "f115w": "/orange/adamginsburg/jwst/brick/images/jw01182-o004_t001_nircam_clear-f115w-merged_i2d.fits",
    "f182m": "/orange/adamginsburg/jwst/brick/mastDownload/jw02221-o002_t001_nircam_clear-f182m_i2d.fits",
    "f187n": "/orange/adamginsburg/jwst/brick/mastDownload/jw02221-o001_t001_nircam_clear-f187n_i2d.fits",
    "f200w": "/orange/adamginsburg/jwst/brick/images/jw01182-o004_t001_nircam_clear-f200w-merged_i2d.fits",
    "f212n": "/orange/adamginsburg/jwst/brick/mastDownload/jw02221-o001_t001_nircam_clear-f212n_i2d.fits",
    "f356w": "/orange/adamginsburg/jwst/brick/images/jw01182-o004_t001_nircam_clear-f356w-merged_i2d.fits",
    "f405n": "/orange/adamginsburg/jwst/brick/mastDownload/jw02221-o001_t001_nircam_f405n-f444w_i2d.fits",
    "f410m": "/orange/adamginsburg/jwst/brick/mastDownload/jw02221-o001_t001_nircam_clear-f410m_i2d.fits",
    "f444w": "/orange/adamginsburg/jwst/brick/images/jw01182-o004_t001_nircam_clear-f444w-merged_i2d.fits",
    "f466n": "/orange/adamginsburg/jwst/brick/mastDownload/jw02221-o001_t001_nircam_f444w-f466n_i2d.fits",
}

# Commenting out subtracted images for now to get basic functionality working
# image_sub_filenames_pipe = {
#     "f405n-f410m": "/orange/adamginsburg/jwst/brick/F405_minus_F410cont_pipeline_v0.1.fits",
#     "f410m-f405n": "/orange/adamginsburg/jwst/brick/F410_minus_F405_fractional_bandwidth_pipeline_v0.1.fits",
#     "f212n-f210m": "/orange/adamginsburg/jwst/brick/F212_minus_F210cont_pipeline_v0.1.fits",
#     "f187n-f182m": "/orange/adamginsburg/jwst/brick/F187_minus_F182cont.fits",
# }

image_sub_filenames_pipe = {}

def make_pngs(target_filter='f466n', new_basepath='/orange/adamginsburg/jwst/brick/data_reprojected/'):
    print(f"Making PNGs for {target_filter}")

    png_path = f'/orange/adamginsburg/jwst/brick/pngs_{target_filter[1:-1]}'
    os.makedirs(png_path, exist_ok=True)

    tgt_header = fits.getheader(image_filenames_pipe[target_filter], ext=('SCI', 1))
    AVM = pyavm.AVM.from_header(tgt_header)

    repr_image_filenames = {x: y.replace("i2d", f"i2d_pipeline_v0.1_reprj_{target_filter[:-1]}") for x,y in image_filenames_pipe.items()}
    repr_image_filenames = {x: (new_basepath+os.path.basename(y)) for x,y in repr_image_filenames.items()}
    repr_image_sub_filenames = {x: y.replace(".fits", f"reprj_{target_filter[:-1]}.fits") for x,y in image_sub_filenames_pipe.items()}
    repr_image_sub_filenames = {x: (new_basepath+os.path.basename(y)) for x,y in repr_image_sub_filenames.items()}

    for filtername in image_filenames_pipe:
        if not os.path.exists(repr_image_filenames[filtername]):
            print(f"Reprojecting {filtername} {image_filenames_pipe[filtername]} to {repr_image_filenames[filtername]}")
            result,_ = reproject.reproject_interp(image_filenames_pipe[filtername], tgt_header, hdu_in='SCI')
            hdu = fits.PrimaryHDU(data=result, header=tgt_header)
            hdu.writeto(repr_image_filenames[filtername], overwrite=True)

    for filtername in image_sub_filenames_pipe:
        if not os.path.exists(repr_image_sub_filenames[filtername]):
            print(f"Reprojecting {filtername} {image_sub_filenames_pipe[filtername]} to {repr_image_sub_filenames[filtername]}")
            result,_ = reproject.reproject_interp(image_sub_filenames_pipe[filtername], tgt_header, hdu_in='SCI')
            hdu = fits.PrimaryHDU(data=result, header=tgt_header)
            hdu.writeto(repr_image_sub_filenames[filtername], overwrite=True)

    # Updated ALMA file path for Brick data
    alma_brick_3mm = "/orange/adamginsburg/brick/alma/rathborne/brick.cont.alma.image.fits"
    alma_level = 3e-4

    alma_reproj_fn = f'/orange/adamginsburg/jwst/brick/data_reprojected/alma_brick_reprojected_jwst_{target_filter[:-1]}.fits'
    if os.path.exists(alma_reproj_fn):
        alma_brick_reprojected_jwst = fits.getdata(alma_reproj_fn)
    else:
        print(f"Reprojecting ALMA data to {alma_reproj_fn}")
        fh = fits.open(alma_brick_3mm)
        data = fh[0].data.squeeze()
        hdr = WCS(fh[0].header).celestial
        alma_brick_reprojected_jwst, footprint = reproject.reproject_interp((data, hdr), tgt_header)

        alma_brick_reprojected_jwst = np.nan_to_num(alma_brick_reprojected_jwst) / footprint
        del footprint
        fits.writeto(alma_reproj_fn, alma_brick_reprojected_jwst, tgt_header, overwrite=True)




    filternames = sorted(list(image_filenames_pipe.keys()),
                        key=lambda x: int(''.join(filter(str.isdigit, x))))[::-1]
    print(f"Sorted list of filters: {filternames}")

    for f1, f2, f3 in zip(filternames, filternames[1:], filternames[2:]):
        print(f1,f2,f3)
        rgb = np.array([
            fits.getdata(repr_image_filenames[f1]),
            fits.getdata(repr_image_filenames[f2]),
            fits.getdata(repr_image_filenames[f3]),
        ]).swapaxes(0,2).swapaxes(0,1)
        rgb_scaled = np.array([simple_norm(rgb[:,:,0], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,0]),
                            simple_norm(rgb[:,:,1], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,1]),
                            simple_norm(rgb[:,:,2], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,2])]).swapaxes(0,2).swapaxes(0,1)

        f1n, f2n, f3n = ''.join(filter(str.isdigit, f1)), ''.join(filter(str.isdigit, f2)), ''.join(filter(str.isdigit, f3))
        save_rgb(rgb_scaled, f'{png_path}/Brick_RGB_{f1n}-{f2n}-{f3n}.png', avm=AVM)
        save_rgb(rgb_scaled, f'{png_path}/Brick_RGB_{f1n}-{f2n}-{f3n}_alma.png', avm=AVM, alma_data=alma_brick_reprojected_jwst, alma_level=alma_level)

        rgb_scaled = np.array([simple_norm(rgb[:,:,0], stretch='log', min_percent=1.5, max_percent=99.5)(rgb[:,:,0]),
                            simple_norm(rgb[:,:,1], stretch='log', min_percent=1.5, max_percent=99.5)(rgb[:,:,1]),
                            simple_norm(rgb[:,:,2], stretch='log', min_percent=1.5, max_percent=99.5)(rgb[:,:,2])]).swapaxes(0,2).swapaxes(0,1)

        save_rgb(rgb_scaled, f'{png_path}/Brick_RGB_{f1n}-{f2n}-{f3n}_log.png', avm=AVM)
        save_rgb(rgb_scaled, f'{png_path}/Brick_RGB_{f1n}-{f2n}-{f3n}_log_alma.png', avm=AVM, alma_data=alma_brick_reprojected_jwst, alma_level=alma_level)

    filternames_sub = sorted(list(image_sub_filenames_pipe.keys()),
                           key=lambda x: int(''.join(filter(str.isdigit, x))))[::-1]
    print(f"Sorted list of subtracted-filters: {filternames_sub}")

    for f1, f2, f3 in zip(filternames_sub, filternames_sub[1:], filternames_sub[2:]):
        print(f1,f2,f3)
        try:
            rgb = np.array([
                fits.getdata(repr_image_sub_filenames[f1]),
                fits.getdata(repr_image_sub_filenames[f2]),
                fits.getdata(repr_image_sub_filenames[f3]),
            ]).swapaxes(0,2).swapaxes(0,1)
        except Exception as ex:
            print(ex)
            print(f"Shape of {f1} is {fits.getdata(repr_image_sub_filenames[f1]).shape}")
            print(f"Shape of {f2} is {fits.getdata(repr_image_sub_filenames[f2]).shape}")
            print(f"Shape of {f3} is {fits.getdata(repr_image_sub_filenames[f3]).shape}")
            raise ex

        rgb_scaled = np.array([simple_norm(rgb[:,:,0], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,0]),
                            simple_norm(rgb[:,:,1], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,1]),
                            simple_norm(rgb[:,:,2], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,2])]).swapaxes(0,2).swapaxes(0,1)

        f1n, f2n, f3n = ''.join(filter(str.isdigit, f1)), ''.join(filter(str.isdigit, f2)), ''.join(filter(str.isdigit, f3))
        save_rgb(rgb_scaled, f'{png_path}/Brick_RGB_{f1n}-{f2n}-{f3n}_sub.png', avm=AVM)
        try:
            save_rgb(rgb_scaled, f'{png_path}/Brick_RGB_{f1n}-{f2n}-{f3n}_sub_alma.png', avm=AVM, alma_data=alma_brick_reprojected_jwst, alma_level=alma_level)
        except Exception as ex:
            print(ex)
            print(f"ALMA data shape = {fits.getdata(alma_reproj_fn).shape}")
            print(f"RGB data shape = {rgb.shape}")
            raise ex

        rgb_scaled = np.array([simple_norm(rgb[:,:,0], stretch='log', min_percent=1.0, max_percent=99.5)(rgb[:,:,0]),
                            simple_norm(rgb[:,:,1], stretch='log', min_percent=1.0, max_percent=99.5)(rgb[:,:,1]),
                            simple_norm(rgb[:,:,2], stretch='log', min_percent=1.0, max_percent=99.5)(rgb[:,:,2])]).swapaxes(0,2).swapaxes(0,1)

        save_rgb(rgb_scaled, f'{png_path}/Brick_RGB_{f1n}-{f2n}-{f3n}_sub_log.png', avm=AVM)
        try:
            save_rgb(rgb_scaled, f'{png_path}/Brick_RGB_{f1n}-{f2n}-{f3n}_sub_log_alma.png', avm=AVM, alma_data=alma_brick_reprojected_jwst, alma_level=alma_level)
        except Exception as ex:
            print(ex)
            print(f"ALMA data shape = {fits.getdata(alma_reproj_fn).shape}")
            print(f"RGB data shape = {rgb.shape}")
            raise ex

    # Special BGR combinations as requested
    print("Creating special BGR combinations:")

    # BGR = 405, 405+466, 466
    print("Creating BGR: 405, 405+466, 466")
    f405_data = fits.getdata(repr_image_filenames['f405n'])
    f466_data = fits.getdata(repr_image_filenames['f466n'])

    # Create composite 405+466 channel
    f405_466_data = f405_data + f466_data

    # BGR arrangement: Blue=405, Green=405+466, Red=466
    bgr_405_405466_466 = np.array([
        f405_data,      # Blue
        f405_466_data,  # Green
        f466_data       # Red
    ]).swapaxes(0,2).swapaxes(0,1)

    bgr_scaled = np.array([
        simple_norm(bgr_405_405466_466[:,:,0], stretch='asinh', min_percent=1, max_percent=99.5)(bgr_405_405466_466[:,:,0]),
        simple_norm(bgr_405_405466_466[:,:,1], stretch='asinh', min_percent=1, max_percent=99.5)(bgr_405_405466_466[:,:,1]),
        simple_norm(bgr_405_405466_466[:,:,2], stretch='asinh', min_percent=1, max_percent=99.5)(bgr_405_405466_466[:,:,2])
    ]).swapaxes(0,2).swapaxes(0,1)

    save_rgb(bgr_scaled, f'{png_path}/Brick_BGR_405-405466-466.png', avm=AVM)
    save_rgb(bgr_scaled, f'{png_path}/Brick_BGR_405-405466-466_alma.png', avm=AVM, alma_data=alma_brick_reprojected_jwst, alma_level=alma_level)

    # Log version
    bgr_scaled_log = np.array([
        simple_norm(bgr_405_405466_466[:,:,0], stretch='log', min_percent=1.5, max_percent=99.5)(bgr_405_405466_466[:,:,0]),
        simple_norm(bgr_405_405466_466[:,:,1], stretch='log', min_percent=1.5, max_percent=99.5)(bgr_405_405466_466[:,:,1]),
        simple_norm(bgr_405_405466_466[:,:,2], stretch='log', min_percent=1.5, max_percent=99.5)(bgr_405_405466_466[:,:,2])
    ]).swapaxes(0,2).swapaxes(0,1)

    save_rgb(bgr_scaled_log, f'{png_path}/Brick_BGR_405-405466-466_log.png', avm=AVM)
    save_rgb(bgr_scaled_log, f'{png_path}/Brick_BGR_405-405466-466_log_alma.png', avm=AVM, alma_data=alma_brick_reprojected_jwst, alma_level=alma_level)

    # BGR = 410, 410+466, 466
    print("Creating BGR: 410, 410+466, 466")
    f410_data = fits.getdata(repr_image_filenames['f410m'])

    # Create composite 410+466 channel
    f410_466_data = f410_data + f466_data

    # BGR arrangement: Blue=410, Green=410+466, Red=466
    bgr_410_410466_466 = np.array([
        f410_data,      # Blue
        f410_466_data,  # Green
        f466_data       # Red
    ]).swapaxes(0,2).swapaxes(0,1)

    bgr_scaled = np.array([
        simple_norm(bgr_410_410466_466[:,:,0], stretch='asinh', min_percent=1, max_percent=99.5)(bgr_410_410466_466[:,:,0]),
        simple_norm(bgr_410_410466_466[:,:,1], stretch='asinh', min_percent=1, max_percent=99.5)(bgr_410_410466_466[:,:,1]),
        simple_norm(bgr_410_410466_466[:,:,2], stretch='asinh', min_percent=1, max_percent=99.5)(bgr_410_410466_466[:,:,2])
    ]).swapaxes(0,2).swapaxes(0,1)

    save_rgb(bgr_scaled, f'{png_path}/Brick_BGR_410-410466-466.png', avm=AVM)
    save_rgb(bgr_scaled, f'{png_path}/Brick_BGR_410-410466-466_alma.png', avm=AVM, alma_data=alma_brick_reprojected_jwst, alma_level=alma_level)

    # Log version
    bgr_scaled_log = np.array([
        simple_norm(bgr_410_410466_466[:,:,0], stretch='log', min_percent=1.5, max_percent=99.5)(bgr_410_410466_466[:,:,0]),
        simple_norm(bgr_410_410466_466[:,:,1], stretch='log', min_percent=1.5, max_percent=99.5)(bgr_410_410466_466[:,:,1]),
        simple_norm(bgr_410_410466_466[:,:,2], stretch='log', min_percent=1.5, max_percent=99.5)(bgr_410_410466_466[:,:,2])
    ]).swapaxes(0,2).swapaxes(0,1)

    save_rgb(bgr_scaled_log, f'{png_path}/Brick_BGR_410-410466-466_log.png', avm=AVM)
    save_rgb(bgr_scaled_log, f'{png_path}/Brick_BGR_410-410466-466_log_alma.png', avm=AVM, alma_data=alma_brick_reprojected_jwst, alma_level=alma_level)

    # BGR = 212, 405, 466
    print("Creating BGR: 212, 405, 466")
    f212_data = fits.getdata(repr_image_filenames['f212n'])

    # BGR arrangement: Blue=212, Green=405, Red=466
    bgr_212_405_466 = np.array([
        f212_data,  # Blue
        f405_data,  # Green
        f466_data   # Red
    ]).swapaxes(0,2).swapaxes(0,1)

    bgr_scaled = np.array([
        simple_norm(bgr_212_405_466[:,:,0], stretch='asinh', min_percent=1, max_percent=99.5)(bgr_212_405_466[:,:,0]),
        simple_norm(bgr_212_405_466[:,:,1], stretch='asinh', min_percent=1, max_percent=99.5)(bgr_212_405_466[:,:,1]),
        simple_norm(bgr_212_405_466[:,:,2], stretch='asinh', min_percent=1, max_percent=99.5)(bgr_212_405_466[:,:,2])
    ]).swapaxes(0,2).swapaxes(0,1)

    save_rgb(bgr_scaled, f'{png_path}/Brick_BGR_212-405-466.png', avm=AVM)
    save_rgb(bgr_scaled, f'{png_path}/Brick_BGR_212-405-466_alma.png', avm=AVM, alma_data=alma_brick_reprojected_jwst, alma_level=alma_level)

    # Log version
    bgr_scaled_log = np.array([
        simple_norm(bgr_212_405_466[:,:,0], stretch='log', min_percent=1.5, max_percent=99.5)(bgr_212_405_466[:,:,0]),
        simple_norm(bgr_212_405_466[:,:,1], stretch='log', min_percent=1.5, max_percent=99.5)(bgr_212_405_466[:,:,1]),
        simple_norm(bgr_212_405_466[:,:,2], stretch='log', min_percent=1.5, max_percent=99.5)(bgr_212_405_466[:,:,2])
    ]).swapaxes(0,2).swapaxes(0,1)

    save_rgb(bgr_scaled_log, f'{png_path}/Brick_BGR_212-405-466_log.png', avm=AVM)
    save_rgb(bgr_scaled_log, f'{png_path}/Brick_BGR_212-405-466_log_alma.png', avm=AVM, alma_data=alma_brick_reprojected_jwst, alma_level=alma_level)

def main():
    # Updated to use f466n as default since f150w is not available in Brick dataset
    for target_filter in ('f466n', 'f200w'):
        make_pngs(target_filter)

if __name__ == '__main__':
    main()
