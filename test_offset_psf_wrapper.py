#!/usr/bin/env python3
"""
test_offset_psf_wrapper.py

Sanity check that ``OffsetGriddedPSF`` evaluated in cutout pixel
coordinates produces exactly the same image as the underlying
``GriddedPSFModel`` evaluated in detector pixel coordinates (up to the
constant cutout→detector shift).

Run with the photutils-allframe fork on PYTHONPATH::

    PYTHONPATH=/blue/adamginsburg/adamginsburg/repos/photutils-allframe \\
        python test_offset_psf_wrapper.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from allframe_cutout_demo import OffsetGriddedPSF  # noqa: E402

from photutils.psf import GriddedPSFModel  # noqa: E402


def _check(grid, det_x, det_y, dx, dy, npix=21, label=''):
    """Compare wrapped(cutout coords) vs direct(detector coords)."""
    half = npix // 2
    # detector-coord eval grid centred on the source
    yy, xx = np.mgrid[int(round(det_y)) - half:int(round(det_y)) + half + 1,
                      int(round(det_x)) - half:int(round(det_x)) + half + 1]
    grid.x_0 = float(det_x)
    grid.y_0 = float(det_y)
    grid.flux = 1.0
    direct = grid(xx, yy)

    # cutout-coord eval grid: same physical pixels, just shifted by (dx, dy)
    cxx = xx - dx
    cyy = yy - dy

    wrapper = OffsetGriddedPSF(grid, dx=dx, dy=dy)
    wrapped = wrapper.evaluate(
        cxx.astype(float), cyy.astype(float),
        x_0=det_x - dx, y_0=det_y - dy, flux=1.0,
    )

    diff = np.abs(direct - wrapped).max()
    print(f'{label:>20s}: direct sum={direct.sum():.6f}  '
          f'wrapped sum={wrapped.sum():.6f}  max|diff|={diff:.3e}')
    if diff > 1e-12:
        raise AssertionError(
            f'{label}: OffsetGriddedPSF disagrees with direct GriddedPSFModel '
            f'at det=({det_x}, {det_y}); max |diff|={diff}'
        )


def main():
    grid = GriddedPSFModel.read(
        '/orange/adamginsburg/jwst/sickle/'
        'nircam_nrcb1_f187n_fovp101_samp4_npsf16.fits'
    )

    # Integer offset (e.g. cutout from a subarray with integer SUBSTRT and
    # integer cutout origin): trivial bit-exact match expected.
    _check(grid, det_x=1235, det_y=1567, dx=1225, dy=1557, label='integer offset')
    _check(grid, det_x=400, det_y=1800, dx=395, dy=1790, label='another integer')

    # Non-integer source position with integer offset (typical case where
    # SUBSTRT-1 is integer but the source falls between pixels):
    _check(grid, det_x=1234.7, det_y=1567.3, dx=1225, dy=1557,
           label='source mid-pixel')

    # Non-integer offset (Cutout2D position_original - position_cutout can
    # be fractional if the centre wasn't on a pixel — e.g. partial mode):
    _check(grid, det_x=1234.7, det_y=1567.3, dx=1224.7, dy=1557.3,
           label='fractional offset')

    print('OK: wrapper reproduces direct evaluation to machine precision.')


if __name__ == '__main__':
    main()
