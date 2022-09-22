import os
from astropy.io import fits
from scipy.ndimage import median_filter, map_coordinates
import numpy as np
from astropy.wcs import WCS

basepath = '/orange/adamginsburg/jwst/brick/'

background_mapping = {
    'f212n': 'jw02221-o001_t001_nircam_clear-f212n_i2d_medfilt256.fits',
    'f187n': 'jw02221-o001_t001_nircam_clear-f187n_i2d_medfilt256.fits',
    'f410m': 'jw02221-o001_t001_nircam_clear-f410m_i2d_medfilt128.fits',
    'f405n': 'jw02221-o001_t001_nircam_f405n-f444w_i2d_medfilt128.fits',
    'f182m': 'jw02221-o001_t001_nircam_clear-f182m_i2d_medfilt256.fits',
    'f466n': 'jw02221-o001_t001_nircam_f444w-f466n_i2d_medfilt128.fits',
}


def compute_zero_spacing_approximation(filename, ext=('SCI', 1), dx=128, percentile=10):
    """
    Use a local, large-scale percentile to estimate the "zero spacing"
    background level across the image.

    We'll then use this to replace the missing zero-spacing lost from
    the destreaking process.
    """
    img = fits.getdata(filename, ext=ext)

    # the bottom-left pixel will now be centered at (dx/2 + 1) in FITS coordinates
    chunks = [[img[(slice(sty, sty+dx), slice(stx, stx+dx))]
            for stx in range(0, img.shape[1], dx//2)]
            for sty in range(0, img.shape[0], dx//2)
            ]

    # only include positive values
    arr = np.array(
        [[np.percentile(ch[ch > 0], percentile) if np.any(ch > 0) else 0
          for ch in row]
         for row in chunks]
    )

    header = fits.getheader(filename, ext=ext)
    ww = WCS(header)

    # I can never remember how to do this, but I'm *certain* this is wrong (independent of what this next line says:)
    # but empirically I'm _pretty_ sure dx/4 + 0.5 looks like it matches maybe
    wwsl = ww[dx/4+0.5::dx//2, dx/4+0.5::dx//2]

    return fits.PrimaryHDU(data=arr, header=wwsl.to_header())


def nozero_percentile(arr, pct, **kwargs):
    arr = arr.copy()
    arr[arr == 0] = np.nan
    rslt = np.nanpercentile(arr, pct, **kwargs)

    # sometimes whole rows are zero.  We want to retain these as zero.
    return np.nan_to_num(rslt)


def destreak_data(data, percentile=10, median_filter_size=256, add_smoothed=True):

    for start in range(0, 2048, 512):
        chunk = data[:, slice(start, start + 512)]
        pct = nozero_percentile(chunk, percentile, axis=1)
        if add_smoothed:
            if median_filter_size >= 2048:
                smoothed_pct = median_filter(pct, median_filter_size)
            else:
                smoothed_pct = np.ones(2048) * np.median(pct)
            data[:, slice(start, start + 512)] = chunk - pct[:, None] + smoothed_pct[:, None]
        else:
            data[:, slice(start, start + 512)] = chunk - pct[:, None]

    return data


def add_background_map(data, hdu, background_mapping=background_mapping,
                       bgmap_path=f'{basepath}/images/',
                       verbose=False,
                       ext=('SCI', 1),
                       return_background=False):
    filtername = hdu[0].header['PUPIL']
    if filtername in ('CLEAR', 'F444W'):
        filtername = hdu[0].header['FILTER']

    bgfile = os.path.join(bgmap_path, background_mapping[filtername.lower()])
    if verbose:
        print(f'Background filename: {bgfile}')

    ww = WCS(hdu[ext].header)

    bg = fits.getdata(bgfile)

    # we want the middles of the columns
    for start in range(0, 2048, 512):
        # pixel coordinates (px)
        pxy = np.arange(2048)
        pxx = np.ones(2048) * (start + 512) / 2
        crds = ww.pixel_to_world(pxx, pxy)

        wwbg = WCS(fits.getheader(bgfile))
        bgx, bgy = wwbg.world_to_pixel(crds)

        bg_sampled = map_coordinates(bg, [bgy, bgx], order=1)
        if verbose:
            print(f'bg_sampled shape: {bg_sampled.shape}, nanmedian: {np.nanmedian(bg_sampled)}')

        data[:, slice(start, start + 512)] += bg_sampled[:, None]

    return data


def destreak(frame, percentile=10, median_filter_size=256, overwrite=True, write=True,
             background_folder='/orange/adamginsburg/jwst/brick/images/',
             background_mapping=background_mapping,
             use_background_map=False
             ):
    """
    "Massimo's Destreaker" - subtract off the median (or percentile)
    of each row, but add back the median of the percentiles so you're not
    changing the total flux.

    For some filters, there are zeros, so we use a 'nozero percentile' to
    mask out the zeros before calculating the percentile

    Also add back in the smoothed version of the streaks so we don't lose
    large angular scales.

    Upgraded to add in the zero-spacing though
    """
    assert frame.endswith('_cal.fits')
    print(f"Destreaking {frame}")
    hdu = fits.open(frame)

    data = hdu[('SCI', 1)].data

    hdu[('SCI', 1)].data = destreak_data(data, percentile=percentile,
                                         median_filter_size=median_filter_size,
                                         add_smoothed=not use_background_map
                                         )

    if use_background_map:
        hdu[('SCI', 1)].data = add_background_map(data, hdu, background_mapping=background_mapping)

    if write:
        outname = frame.replace("_cal.fits", "_destreak.fits")
        hdu.writeto(outname, overwrite=overwrite)

        return outname

    else:
        return hdu
