#!/usr/bin/env python
"""
Mosaic per-frame *_satstar_residual.fits files onto the same WCS as an
existing daophot residual i2d, to give a satstar-only diagnostic image.

Usage:
    make_satstar_residual_mosaic.py --target sickle --filter F480M --module nrcb
"""
import argparse
import glob
import os
import sys

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
from reproject.mosaicking import reproject_and_coadd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--target', required=True)
    p.add_argument('--filter', required=True)
    p.add_argument('--module', default='nrcb')
    p.add_argument('--iteration-label', default='',
                   help='empty=iter1 (no token), or "iter2"/"iter3"')
    p.add_argument('--each-suffix', default='destreak_o007_crf')
    args = p.parse_args()

    if args.target in ('sickle', 'cloudef', 'sgrc', 'sgrb2', 'arches',
                       'quintuplet', 'sgra', 'gc2211'):
        basepath = f'/orange/adamginsburg/jwst/{args.target}'
    else:
        basepath = f'/blue/adamginsburg/adamginsburg/jwst/{args.target}'
    pipedir = f'{basepath}/{args.filter}/pipeline'

    iter_token = f'_{args.iteration_label}' if args.iteration_label else ''
    satstar_glob = (f'{pipedir}/jw*_{args.each_suffix}{iter_token}'
                    f'_satstar_residual.fits')
    files = sorted(glob.glob(satstar_glob))
    if not files:
        print(f"No satstar_residual files found: {satstar_glob}",
              file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(files)} satstar_residual files")

    # Target WCS: use the existing i2d data mosaic (always present)
    ref_path = (f'{pipedir}/jw03958-o007_t001_nircam_clear-'
                f'{args.filter.lower()}-{args.module}_i2d.fits')
    if not os.path.exists(ref_path):
        print(f"Reference i2d not found at {ref_path}", file=sys.stderr)
        sys.exit(1)
    with fits.open(ref_path) as ref_hdul:
        target_wcs = WCS(ref_hdul['SCI'].header)
        target_shape = ref_hdul['SCI'].data.shape

    inputs = []
    for f in files:
        hdul = fits.open(f)
        data = hdul[0].data
        w = WCS(hdul[0].header)
        inputs.append((data, w))

    mosaic, footprint = reproject_and_coadd(
        inputs, target_wcs, shape_out=target_shape,
        reproject_function=reproject_interp,
        combine_function='mean',
    )

    iter_suffix = f'_{args.iteration_label}' if args.iteration_label else ''
    out_path = (f'{pipedir}/jw03958-o007_t001_nircam_clear-'
                f'{args.filter.lower()}-{args.module}{iter_suffix}'
                f'_satstar_residual_i2d.fits')
    hdu = fits.PrimaryHDU(data=mosaic.astype(np.float32),
                          header=target_wcs.to_header())
    hdu.writeto(out_path, overwrite=True)
    print(f"Wrote {out_path}  shape={mosaic.shape}")


if __name__ == '__main__':
    main()
