#!/usr/bin/env python
"""Build per-frame background images from the iter3 photometry residual.

Input:  ``<frame>_iter3_daophot_iterative_residual.fits`` (per-frame
        iter3 residual; the data minus iter3 satstar model minus iter3
        photutils model).

Output: ``<frame>_iter3_daophot_iterative_residual_smoothed_bg.fits``
        (3x3 median-filter of the input).  Same WCS/header.

This file is consumed by the iter4-residbg / iter5-residbg photometry
runs, which use it as the background estimate (in place of running
``Background2D`` on the data itself).  The rationale: the iter3 model
already accounts for every fittable point source, so what remains is
either real diffuse background or per-pixel noise; a 3x3 median filter
suppresses the noise while preserving genuine background structure on
the PSF scale.

Usage:
    python make_iter3_residual_bgmaps.py --target sickle
    python make_iter3_residual_bgmaps.py --target brick --filter F410M
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
from astropy.io import fits
from scipy.ndimage import median_filter

TARGET_BASEPATHS = {
    'sickle': '/orange/adamginsburg/jwst/sickle',
    'brick':  '/blue/adamginsburg/adamginsburg/jwst/brick',
    'cloudc': '/blue/adamginsburg/adamginsburg/jwst/cloudc',
    'sgrb2':  '/orange/adamginsburg/jwst/sgrb2',
}

# Suffix appended to the per-frame residual filename to produce the
# smoothed background filename.  Mirror this token in the consumer
# (crowdsource_catalogs_long.py --use-iter3-residual-bg path).
SMOOTHED_BG_SUFFIX = '_smoothed_bg'


def smooth_one(in_path, out_path, median_size=3, overwrite=False):
    if os.path.exists(out_path) and not overwrite:
        return False
    with fits.open(in_path) as hdul:
        # Some residual files have only PrimaryHDU; others have SCI extension.
        if 'SCI' in [h.name for h in hdul]:
            sci_idx = [h.name for h in hdul].index('SCI')
            data = hdul[sci_idx].data.astype(np.float32)
            header = hdul[sci_idx].header
            primary_header = hdul[0].header
        else:
            data = hdul[0].data.astype(np.float32)
            header = hdul[0].header
            primary_header = None

    finite = np.isfinite(data)
    work = np.where(finite, data, 0.0)
    smoothed = median_filter(work, size=median_size, mode='nearest')
    smoothed = np.where(finite, smoothed, np.nan).astype(np.float32)

    out = fits.HDUList()
    if primary_header is not None:
        out.append(fits.PrimaryHDU(header=primary_header))
        out.append(fits.ImageHDU(data=smoothed, header=header, name='SCI'))
    else:
        out.append(fits.PrimaryHDU(data=smoothed, header=header))
    out[0].header['HISTORY'] = (
        f'Built by make_iter3_residual_bgmaps.py: 3x3 median filter of '
        f'{os.path.basename(in_path)}'
    )
    out.writeto(out_path, overwrite=True)
    return True


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target', required=True, choices=sorted(TARGET_BASEPATHS))
    p.add_argument('--filter', '--filtername', dest='filtername', default=None,
                   help='Limit to one filter (e.g. F410M).  Default: all filters.')
    p.add_argument('--median-size', type=int, default=3,
                   help='Median filter footprint (square, in pixels).  Default 3.')
    p.add_argument('--overwrite', action='store_true',
                   help='Re-smooth even if output exists.')
    args = p.parse_args(argv)

    base = TARGET_BASEPATHS[args.target]
    if args.filtername:
        filt_dirs = [os.path.join(base, args.filtername.upper())]
    else:
        filt_dirs = sorted(glob.glob(os.path.join(base, 'F*')))

    n_done = 0
    n_skipped = 0
    for filt_dir in filt_dirs:
        if not os.path.isdir(filt_dir):
            continue
        # Per-frame iter3 residuals live in <filt>/pipeline/
        pat = os.path.join(filt_dir, 'pipeline',
                           '*_iter3_daophot_iterative_residual.fits')
        infiles = sorted(glob.glob(pat))
        for infile in infiles:
            outfile = infile.replace('.fits', f'{SMOOTHED_BG_SUFFIX}.fits')
            try:
                wrote = smooth_one(infile, outfile,
                                   median_size=args.median_size,
                                   overwrite=args.overwrite)
            except (OSError, ValueError) as ex:
                print(f'  ERROR: {infile}: {ex}', file=sys.stderr)
                continue
            if wrote:
                n_done += 1
                if n_done % 20 == 0:
                    print(f'  ... wrote {n_done} smoothed bg files')
            else:
                n_skipped += 1
    print(f'[{args.target}] done.  wrote={n_done}  skipped (existed)={n_skipped}')


if __name__ == '__main__':
    sys.exit(main())
