#!/usr/bin/env python
"""Build the MERGED background image from the iter3 photometry residual.

Input:  ``...-merged_iter3_daophot_iterative_residual_i2d.fits`` (the
        whole-field merged iter3 residual mosaic produced by
        ``mosaic_each_exposure_residuals`` -- the co-add of every
        exposure's residual: data minus iter3 satstar model minus iter3
        photutils model).

Output: ``...-merged_iter3_daophot_iterative_residual_smoothed_bg_i2d.fits``
        (median-filter of the input).  Same WCS/header.

This file is consumed by the iter*-residbg photometry runs
(``catalog_long.py --use-iter3-residual-bg``), which
reproject it onto each exposure's pixel grid and use it as the
background estimate (in place of running ``Background2D`` on the data
itself).  The rationale: the iter3 model already accounts for every
fittable point source, so what remains is either real diffuse
background or per-pixel noise; a median filter suppresses the noise
while preserving genuine background structure on the PSF scale.

2026-06-06: switched from per-frame residuals to the MERGED residual
mosaic.  The merged residual co-adds all exposures, so its background
has much higher S/N than any single frame; the consumer reprojects it
back onto each exposure grid.

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

# Suffix inserted into the merged residual mosaic filename to produce the
# smoothed background filename.  Mirror this token in the consumer
# (catalog_long.py --use-iter3-residual-bg path), which
# rebuilds the same name to find this file.
SMOOTHED_BG_SUFFIX = '_smoothed_bg'

# Glob for whole-field iter3 residual mosaics (iterative kind, no flag
# tokens), for ANY module token.  Targets whose whole-field co-add is a
# single detector name it after that detector (e.g. sickle LW = 'nrcb');
# multi-detector co-adds use 'merged'.  The trailing ``_residual_i2d.fits``
# excludes the ``_infilled_i2d`` / ``_i2d_smoothed`` / ``_smoothed_bg_i2d``
# variants.  The consumer (catalog_long.py
# --use-iter3-residual-bg) selects which module's smoothed bg to read via
# --resbg-mosaic-module.
RESIDUAL_MOSAIC_GLOB = '*_iter3_daophot_iterative_residual_i2d.fits'


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
    n_missing = 0
    for filt_dir in filt_dirs:
        if not os.path.isdir(filt_dir):
            continue
        # The iter3 residual mosaic(s) live in <filt>/pipeline/.
        pat = os.path.join(filt_dir, 'pipeline', RESIDUAL_MOSAIC_GLOB)
        infiles = sorted(glob.glob(pat))
        if not infiles:
            n_missing += 1
            print(f'  no iter3 residual mosaic in {filt_dir}/pipeline '
                  f'(expected {RESIDUAL_MOSAIC_GLOB}); run '
                  f'mosaic_each_exposure_residuals first',
                  file=sys.stderr)
            continue
        for infile in infiles:
            # ..._residual_i2d.fits -> ..._residual_smoothed_bg_i2d.fits
            outfile = infile.replace('_residual_i2d.fits',
                                     f'_residual{SMOOTHED_BG_SUFFIX}_i2d.fits')
            try:
                wrote = smooth_one(infile, outfile,
                                   median_size=args.median_size,
                                   overwrite=args.overwrite)
            except (OSError, ValueError) as ex:
                print(f'  ERROR: {infile}: {ex}', file=sys.stderr)
                continue
            if wrote:
                n_done += 1
                print(f'  wrote {outfile}')
            else:
                n_skipped += 1
    print(f'[{args.target}] done.  wrote={n_done}  skipped (existed)={n_skipped}'
          f'  filters_missing_merged={n_missing}')


if __name__ == '__main__':
    sys.exit(main())
