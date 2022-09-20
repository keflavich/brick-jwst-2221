from astropy.io import fits
from scipy.ndimage import median_filter
import numpy as np


def nozero_percentile(arr, pct, **kwargs):
    arr = arr.copy()
    arr[arr == 0] = np.nan
    rslt = np.nanpercentile(arr, pct, **kwargs)

    # sometimes whole rows are zero.  We want to retain these as zero.
    return np.nan_to_num(rslt)


def destreak_data(data, percentile=10, median_filter_size=256):

    for start in range(0, 2048, 512):
        chunk = data[:, slice(start, start + 512)]
        pct = nozero_percentile(chunk, percentile, axis=1)
        smoothed_pct = median_filter(pct, median_filter_size)
        data[:, slice(start, start + 512)] = chunk - pct[:, None] + smoothed_pct[:, None]

    return data


def destreak(frame, percentile=10, median_filter_size=256, overwrite=True, write=True):
    """
    "Massimo's Destreaker" - subtract off the median (or percentile)
    of each row, but add back the median of the percentiles so you're not
    changing the total flux.

    For some filters, there are zeros, so we use a 'nozero percentile' to
    mask out the zeros before calculating the percentile

    Also add back in the smoothed version of the streaks so we don't lose
    large angular scales
    """
    assert frame.endswith('_cal.fits')
    print(f"Destreaking {frame}")
    hdu = fits.open(frame)

    data = hdu[('SCI', 1)].data

    hdu[('SCI', 1)].data = destreak_data(data, percentile=percentile,
                                         median_filter_size=median_filter_size)

    if write:
        outname = frame.replace("_cal.fits", "_destreak.fits")
        hdu.writeto(outname, overwrite=overwrite)

        return outname

    else:
        return hdu