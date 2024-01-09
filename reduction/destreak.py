import os
from astropy.io import fits
from scipy.ndimage import median_filter, map_coordinates
import numpy as np
from astropy.wcs import WCS
import scipy
import scipy.ndimage

basepath = '/orange/adamginsburg/jwst/brick/'

# these were created in notebooks/MedianFilterBackground.ipynb
background_mapping = { '2221':
                      { '001':
                       {
                        'regionname': 'brick',
                        'f212n': 'jw02221-o001_t001_nircam_clear-f212n_i2d_medfilt256.fits',
                        'f187n': 'jw02221-o001_t001_nircam_clear-f187n_i2d_medfilt256.fits',
                        'f410m': 'jw02221-o001_t001_nircam_clear-f410m_i2d_medfilt128.fits',
                        'f405n': 'jw02221-o001_t001_nircam_f405n-f444w_i2d_medfilt128.fits',
                        'f182m': 'jw02221-o001_t001_nircam_clear-f182m_i2d_medfilt256.fits',
                        'f466n': 'jw02221-o001_t001_nircam_f444w-f466n_i2d_medfilt128.fits',
                        'f444w': 'jw01182-o004_t001_nircam_clear-f444w-merged_nodestreak_realigned-to-refcat_background.fits',
                        'f356w': 'jw01182-o004_t001_nircam_clear-f356w-merged_nodestreak_realigned-to-refcat_background.fits',
                        'f200w': 'jw01182-o004_t001_nircam_clear-f200w-merged_nodestreak_realigned-to-refcat_background.fits',
                        #'f115w': 'jw01182-o004_t001_nircam_clear-f115w-merged_nodestreak_realigned-to-refcat_background.fits',
                       },
                        '002':
                       {
                        'regionname': 'cloudc',
                        'f405n': 'jw02221-o002_t001_nircam_clear-f405n-merged_realigned-to-vvv_i2d_medfilt128.fits',
                       }
                      }
                     }


def compute_zero_spacing_approximation(filename, ext=('SCI', 1), dx=128,
                                       smooth=True,
                                       percentile=10, regs=None, progressbar=lambda x: x):
    """
    Use a local, large-scale percentile to estimate the "zero spacing"
    background level across the image.

    We'll then use this to replace the missing zero-spacing lost from
    the destreaking process.


    smooth: use percentile_Filter
    """
    img = fits.getdata(filename, ext=ext)
    header = fits.getheader(filename, ext=ext)
    ww = WCS(header)

    img[img == 0] = np.nan

    if regs is not None:
        for reg in regs:
            preg = reg.to_pixel(ww)
            mask = preg.to_mask()
            slcs,smslcs = mask.get_overlap_slices(img.shape)
            img[slcs][mask.data.astype('bool')[smslcs]] = np.nan


    if smooth:
        y, x = np.mgrid[:dx, :dx]
        circle = ((x-dx/2)**2 + (y-dx/2)**2) < (dx/2)**2
        arr = scipy.ndimage.percentile_filter(img, percentile,
                                              #size=(dx, dx),
                                              footprint=circle,
                                              mode='reflect',
                                             )
        return fits.PrimaryHDU(data=arr, header=header)
    else:
        # the bottom-left pixel will be centered at (dx/2 + 1) in FITS coordinates if we start at 0
        # so we start at -dx/4 so that the bottom-left pixel is centered at 1,1
        # (BLC of image is at -0.5, -0.5 in FITS, pixel size is dx/2, so offset is dx/4)
        # we don't want to wrap, so we use max(pixel, 0)
        # the percentile will be over a smaller region, but that should be OK
        chunks = [[img[(slice(max(sty, 0), sty+dx), slice(max(stx, 0), stx+dx))]
                for stx in range(-dx//4, img.shape[1]+dx//2, dx//2)]
                for sty in range(-dx//4, img.shape[0]+dx//2, dx//2)
                ]

        # only include positive values (actually no that didn't work)
        arr = np.array(
            [[np.nanpercentile(ch, percentile)  # if np.any(ch > 0) else 0
            for ch in row]
            for row in progressbar(chunks)]
        )

        # I can never remember how to do this, but I'm *certain* this is wrong (independent of what this next line says:)
        # but empirically I'm _pretty_ sure dx/4 + 0.5 looks like it matches maybe
        # with revised version, we drop the shift
        wwsl = ww[::dx//2, ::dx//2]

        return fits.PrimaryHDU(data=arr, header=wwsl.to_header())


def nozero_percentile(arr, pct, **kwargs):
    """
    nanpercentile([nan, nan, nan]) gives nan, but we want zero, so this function
    returns zero if everything is nan
    """
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
    if filtername in ('CLEAR', 'F444W') and hdu[0].header['FILTER'] in ('F405N', 'F466N', 'F410M'):
        filtername = hdu[0].header['FILTER']

    proposal_id = hdu[0].header['PROGRAM'][1:5]
    obsid = hdu[0].header['OBSERVTN'].strip()
    visit = hdu[0].header['VISIT'].strip()

    if (proposal_id not in background_mapping or obsid not in background_mapping[proposal_id]
        or filtername.lower() not in background_mapping[proposal_id][obsid]):
        print(f"WARNING: filter {filtername} is not in background mapping {background_mapping}.  "
              "This likely means you haven't made it yet!")
        return data

    bgm = background_mapping[proposal_id][obsid]
    bgfile = os.path.join(bgmap_path, bgm[filtername.lower()])
    if verbose:
        print(f'Background filename: {bgfile}')

    ww = WCS(hdu[ext].header)

    bg = fits.getdata(bgfile)

    # we want the middles of the columns
    for start in range(0, 2048, 512):
        # pixel coordinates (px)
        pxy = np.arange(2048)
        pxx = np.ones(2048) * (start + 256)
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

    data = destreak_data(data, percentile=percentile,
                         median_filter_size=median_filter_size,
                         add_smoothed=not use_background_map
                                         )

    proposal_id = hdu[0].header['PROGRAM'][1:5]
    obsid = hdu[0].header['OBSERVTN'].strip()
    if use_background_map and not (proposal_id not in background_mapping or obsid not in background_mapping[proposal_id]):
        regionname = background_mapping[proposal_id][obsid]['regionname']
        basepath = f'/orange/adamginsburg/jwst/{regionname}/'
        bgmap_path=f'{basepath}/images/'
        data = add_background_map(data, hdu, background_mapping=background_mapping, bgmap_path=bgmap_path)

    hdu[('SCI', 1)].data = data

    if write:
        outname = frame.replace("_cal.fits", "_destreak.fits")
        hdu.writeto(outname, overwrite=overwrite)

        return outname

    else:
        return hdu
