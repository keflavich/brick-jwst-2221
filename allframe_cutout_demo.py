#!/usr/bin/env python3
"""
allframe_cutout_demo.py

Simultaneous all-frame PSF photometry across every Sickle-field exposure
(NIRCam SW + LW, all filters configured below) within a small benchmark
cutout, using ``photutils.psf.AllFramePhotometry`` from the ``allframe``
branch of photutils.

Each input frame is a per-exposure subarray ``*_crf.fits`` calibrated
image. We extract a small ``Cutout2D`` covering the benchmark region from
every frame that overlaps it, then hand the per-frame
(data, error, wcs, psf_model) tuples to ``AllFramePhotometry`` in joint
mode with shared sky positions.

Correct PSF location
--------------------
The stpsf grids are computed on the FULL detector. Every frame here is
twice-offset from the detector frame:

    detector_pixel = cutout_pixel
                   + cutout_origin_in_subarray
                   + (SUBSTRT - 1)

So a naive call with cutout-frame (x_0, y_0) would interpolate the
gridded PSF at the wrong place. To prevent this we wrap each
``GriddedPSFModel`` in :class:`OffsetGriddedPSF`, which carries a frame-
specific ``(dx, dy)`` cutout-to-detector shift and applies it inside
``evaluate`` so the underlying grid is always sampled at the true
detector location. No fallback or "near enough" PSF is ever used.

Outputs
-------
``allframe_cutout_results.fits`` next to this script:
    one row per source with shared (ra_fit, dec_fit) and
    ``flux_fit_<i>``/``flux_err_<i>`` for every input frame.
``allframe_cutout_frames.csv``:
    bookkeeping table mapping frame index ``i`` →
    (filter, detector, exposure, dx, dy).

Usage
-----
    PYTHONPATH=/blue/adamginsburg/adamginsburg/repos/photutils-allframe \\
        python allframe_cutout_demo.py
"""

import glob
import os
import re
import sys

import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.modeling import Fittable2DModel, Parameter
from astropy.nddata import Cutout2D, NoOverlapError
from astropy.table import QTable, Table
from astropy.wcs import WCS

# ── photutils-allframe imports (PYTHONPATH must point at the fork) ──────────
import photutils
from photutils.psf import AllFramePhotometry, GriddedPSFModel
from photutils.psf.groupers import SourceGrouper

if 'photutils-allframe' not in photutils.__file__:
    raise RuntimeError(
        f'photutils is being imported from {photutils.__file__!r}; '
        'set PYTHONPATH=/blue/adamginsburg/adamginsburg/repos/photutils-allframe '
        'so the allframe branch is picked up.'
    )

# ── paths ────────────────────────────────────────────────────────────────────

SICKLE = '/orange/adamginsburg/jwst/sickle'
PSF_DIR = SICKLE  # nircam_<det>_<filt>_fovp101_samp4_npsf16.fits live here
REGION_FILE = os.path.join(SICKLE, 'regions_/benchmark_cutout.reg')
UNION_CAT = os.path.join(SICKLE, 'catalogs/seed_union_iter3_sickle.fits')

OUT_PHOT = os.path.join(os.path.dirname(__file__), 'allframe_cutout_results.fits')
OUT_FRAMES = os.path.join(os.path.dirname(__file__), 'allframe_cutout_frames.csv')

DET_TOKEN_RE = re.compile(r'_(nrca[1-5]|nrcalong|nrcb[1-4]|nrcblong)_')

# Sickle NIRCam filters (skip MIRI for this demo — different PSF infrastructure)
# and which detector tokens contribute per filter.
FILTER_CFG = [
    # (filter, [detector_tokens], is_long_wavelength)
    ('f187n', ['nrcb1', 'nrcb2', 'nrcb3', 'nrcb4'], False),
    ('f210m', ['nrcb1', 'nrcb2', 'nrcb3', 'nrcb4'], False),
    ('f335m', ['nrcblong'], True),
    ('f470n', ['nrcblong'], True),
    ('f480m', ['nrcblong'], True),
]


# ─── region parsing ──────────────────────────────────────────────────────────

def parse_box_region(path):
    """Parse a single ICRS box() DS9 region. Returns (SkyCoord center,
    (height, width) Quantity).

    Expected format:
        icrs
        box(ra_deg, dec_deg, w_arcsec", h_arcsec", angle)
    """
    text = open(path).read()
    m = re.search(
        r'box\(\s*([\d\.+-eE]+)\s*,\s*([\d\.+-eE]+)\s*,'
        r'\s*([\d\.+-eE]+)"\s*,\s*([\d\.+-eE]+)"\s*,\s*([\d\.+-eE]+)\s*\)',
        text,
    )
    if m is None:
        raise ValueError(f'No box(...) region found in {path}')
    ra, dec, w_as, h_as, angle = (float(x) for x in m.groups())
    if angle != 0.0:
        raise ValueError(
            f'Rotated box (angle={angle}) not supported by Cutout2D path'
        )
    center = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')
    # Cutout2D 'size' takes (ny, nx); pass (h, w) in arcsec.
    size = (h_as * u.arcsec, w_as * u.arcsec)
    return center, size


# ─── PSF wrapper ─────────────────────────────────────────────────────────────

class OffsetGriddedPSF(Fittable2DModel):
    """Wrap a ``GriddedPSFModel`` with a fixed cutout→detector pixel offset.

    The wrapper exposes ``(x_0, y_0, flux)`` parameters in the *cutout*
    coordinate system (so it composes naturally with ``AllFramePhotometry``
    and a cutout-frame WCS). Inside ``evaluate`` we add ``(dx, dy)`` to
    every input before delegating to the underlying ``GriddedPSFModel``,
    so the grid is interpolated at the true detector pixel — no fallback,
    no nearest-neighbour, no average PSF.
    """

    n_inputs = 2
    n_outputs = 1

    x_0 = Parameter(default=0.0, description='source x in cutout pixels')
    y_0 = Parameter(default=0.0, description='source y in cutout pixels')
    flux = Parameter(default=1.0, description='source flux')

    def __init__(self, base_psf, dx, dy, **kwargs):
        if not isinstance(base_psf, GriddedPSFModel):
            raise TypeError(
                f'base_psf must be a GriddedPSFModel; got {type(base_psf).__name__}'
            )
        self._base = base_psf
        self._dx = float(dx)
        self._dy = float(dy)
        super().__init__(**kwargs)

    @property
    def detector_offset(self):
        return (self._dx, self._dy)

    def evaluate(self, x, y, x_0, y_0, flux):
        b = self._base
        b.x_0 = float(x_0) + self._dx
        b.y_0 = float(y_0) + self._dy
        b.flux = float(flux)
        return b(np.asarray(x) + self._dx, np.asarray(y) + self._dy)


# ─── frame loading ───────────────────────────────────────────────────────────

def _det_token(fname):
    m = DET_TOKEN_RE.search(fname)
    if m is None:
        raise ValueError(f'No detector token in {fname}')
    return m.group(1)


def load_frame_cutout(crf_path, center, size):
    """Read one ``*_crf.fits`` exposure, build the WCS, and cut out the
    benchmark region.

    Returns
    -------
    None if the cutout has no overlap with this frame, otherwise a dict
    with keys: data, error, wcs, dx, dy, filter, detector, exposure_id,
    substrt1, substrt2, origin_x, origin_y.
    """
    with fits.open(crf_path, memmap=False) as hdul:
        prim = hdul[0].header
        sci_hdu = hdul['SCI']
        sci = sci_hdu.data.astype(np.float64)
        sci_hdr = sci_hdu.header
        err_hdu = hdul['ERR']
        err = err_hdu.data.astype(np.float64)

    # WCS lives on the SCI extension for JWST cal/crf products
    wcs = WCS(sci_hdr, naxis=2)
    substrt1 = int(prim.get('SUBSTRT1', sci_hdr.get('SUBSTRT1', 1)))
    substrt2 = int(prim.get('SUBSTRT2', sci_hdr.get('SUBSTRT2', 1)))
    filt = prim.get('FILTER') or sci_hdr.get('FILTER')
    detector = (prim.get('DETECTOR') or sci_hdr.get('DETECTOR')).lower()

    try:
        sci_cut = Cutout2D(sci, center, size, wcs=wcs, mode='partial',
                           fill_value=np.nan, copy=True)
    except NoOverlapError:
        return None
    err_cut = Cutout2D(err, center, size, wcs=wcs, mode='partial',
                       fill_value=np.nan, copy=True)

    # The shift from cutout pixel coordinates to original (subarray) pixel
    # coordinates is constant: original = cutout + shift. We get it from
    # the user-supplied centre, which Cutout2D records in both frames.
    # This is correct in 'partial' mode too, where origin_original gets
    # clipped at 0 and cannot be used as the shift.
    pos_orig = np.asarray(sci_cut.input_position_original, dtype=float)
    pos_cut = np.asarray(sci_cut.input_position_cutout, dtype=float)
    shift_x, shift_y = pos_orig - pos_cut

    # cutout pixel → full detector pixel
    dx = shift_x + (substrt1 - 1)
    dy = shift_y + (substrt2 - 1)

    err_arr = err_cut.data.copy()
    err_arr[~np.isfinite(err_arr)] = np.inf  # downweight rather than crash
    err_arr[err_arr <= 0] = np.inf

    exposure_id = os.path.basename(crf_path).replace('.fits', '')

    mask = ~np.isfinite(sci_cut.data)

    return dict(
        data=sci_cut.data,
        error=err_arr,
        mask=mask,
        wcs=sci_cut.wcs,
        dx=float(dx),
        dy=float(dy),
        filter=filt.lower() if filt else None,
        detector=detector,
        exposure_id=exposure_id,
        substrt1=substrt1,
        substrt2=substrt2,
        shift_x=float(shift_x),
        shift_y=float(shift_y),
        shape=sci_cut.data.shape,
    )


def load_psf_grid(filt, det_token):
    """Load the stpsf gridded PSF for (filter, detector)."""
    psf_path = os.path.join(PSF_DIR,
                            f'nircam_{det_token}_{filt}_fovp101_samp4_npsf16.fits')
    if not os.path.exists(psf_path):
        raise FileNotFoundError(f'Missing stpsf grid: {psf_path}')
    return GriddedPSFModel.read(psf_path)


# ─── init catalog ────────────────────────────────────────────────────────────

def init_sources_in_cutout(center, size):
    """Return (SkyCoord, ids) of union-catalog sources inside the box.

    The union catalog is the iter3 cross-band master list. We restrict
    to a slightly enlarged sky box so sources near the cutout edge that
    still contribute PSF flux are retained.
    """
    if not os.path.exists(UNION_CAT):
        raise FileNotFoundError(
            f'Union catalog not found at {UNION_CAT} — required for init '
            'positions; AllFramePhotometry has no built-in finder.'
        )
    cat = Table.read(UNION_CAT)
    cat_sc = SkyCoord(ra=np.asarray(cat['ra'], dtype=float) * u.deg,
                      dec=np.asarray(cat['dec'], dtype=float) * u.deg)
    ny, nx = size
    # 1.1x oversize tolerates edge sources whose PSF wing leaks into the
    # cutout but whose centre lies just outside. Use the box half-diagonal
    # as the angular cut, then refine in tangent plane.
    half_diag = 1.1 * np.hypot(nx.to(u.arcsec).value,
                               ny.to(u.arcsec).value) / 2.0 * u.arcsec
    sep = cat_sc.separation(center)
    keep = sep < half_diag
    sub = cat[keep]
    if len(sub) == 0:
        raise RuntimeError(
            f'No union-catalog sources fall inside {nx}×{ny} box at '
            f'({center.ra.deg}, {center.dec.deg})'
        )
    sc = SkyCoord(ra=sub['ra'] * u.deg, dec=sub['dec'] * u.deg)
    return sc, np.asarray(sub['source_id_union'])


# ─── main ────────────────────────────────────────────────────────────────────

def collect_frames(center, size):
    """Walk every configured (filter, detector) and return the per-frame
    payload list ready for AllFramePhotometry.

    Returns
    -------
    frames : list of dict
        Entries from ``load_frame_cutout`` augmented with ``psf_model``
        (an ``OffsetGriddedPSF`` carrying this frame's dx/dy).
    """
    frames = []
    grid_cache = {}

    for filt, det_tokens, _ in FILTER_CFG:
        fdir = os.path.join(SICKLE, filt.upper(), 'pipeline')
        if not os.path.isdir(fdir):
            print(f'[skip] {fdir} does not exist', flush=True)
            continue
        crfs = sorted(
            f for f in glob.glob(os.path.join(fdir, '*_crf.fits'))
            if '_bgsub_' not in os.path.basename(f)
            and '_iter3_' not in os.path.basename(f)
        )
        print(f'[{filt}] {len(crfs)} crf files', flush=True)
        for crf in crfs:
            det = _det_token(os.path.basename(crf))
            if det not in det_tokens:
                continue
            payload = load_frame_cutout(crf, center, size)
            if payload is None:
                continue
            if payload['filter'] != filt:
                raise ValueError(
                    f'Header FILTER={payload["filter"]!r} disagrees with '
                    f'directory filter {filt!r} for {crf}'
                )
            if payload['detector'] != det:
                raise ValueError(
                    f'Header DETECTOR={payload["detector"]!r} disagrees '
                    f'with filename detector token {det!r} for {crf}'
                )
            key = (filt, det)
            if key not in grid_cache:
                grid_cache[key] = load_psf_grid(filt, det)
            base_psf = grid_cache[key]
            payload['psf_model'] = OffsetGriddedPSF(
                base_psf, dx=payload['dx'], dy=payload['dy'],
            )
            frames.append(payload)
    return frames


def write_frame_log(frames, path):
    rows = [
        dict(
            i=i,
            filter=f['filter'],
            detector=f['detector'],
            exposure_id=f['exposure_id'],
            ny=f['shape'][0],
            nx=f['shape'][1],
            substrt1=f['substrt1'],
            substrt2=f['substrt2'],
            shift_x=f['shift_x'],
            shift_y=f['shift_y'],
            dx=f['dx'],
            dy=f['dy'],
        )
        for i, f in enumerate(frames)
    ]
    Table(rows).write(path, format='csv', overwrite=True)


def main():
    print(f'Reading region {REGION_FILE}', flush=True)
    center, size = parse_box_region(REGION_FILE)
    print(f'  center = {center.to_string("hmsdms")}, '
          f'size = {size[1].to(u.arcsec):.3f} × {size[0].to(u.arcsec):.3f}',
          flush=True)

    print('Collecting frames + PSFs ...', flush=True)
    frames = collect_frames(center, size)
    if not frames:
        raise RuntimeError('No frames overlap the benchmark cutout.')
    print(f'  → {len(frames)} frames overlap the cutout', flush=True)
    write_frame_log(frames, OUT_FRAMES)
    print(f'  wrote frame log → {OUT_FRAMES}', flush=True)

    sc_init, ids = init_sources_in_cutout(center, size)
    print(f'Union-catalog sources in cutout: {len(sc_init)}', flush=True)

    init = QTable()
    init['id'] = ids
    init['ra'] = sc_init.ra.deg
    init['dec'] = sc_init.dec.deg

    psf_models = [f['psf_model'] for f in frames]
    data_list = [f['data'] for f in frames]
    err_list = [f['error'] for f in frames]
    mask_list = [f['mask'] for f in frames]
    wcs_list = [f['wcs'] for f in frames]

    fit_shape_per_frame = [
        (21, 21) if f['detector'].endswith('long') else (15, 15)
        for f in frames
    ]
    aperture_per_frame = [
        6.0 if f['detector'].endswith('long') else 4.0
        for f in frames
    ]

    print('Building AllFramePhotometry (joint, shared sky positions) ...',
          flush=True)
    aff = AllFramePhotometry(
        psf_models=psf_models,
        fit_shape=fit_shape_per_frame,
        aperture_radius=aperture_per_frame,
        mode='joint',
        position_frame='sky',
        share_position=True,
        grouper=SourceGrouper(min_separation=8),
        progress_bar=True,
    )

    print('Fitting ...', flush=True)
    results = aff(data_list, wcs=wcs_list, errors=err_list,
                  masks=mask_list, init_params=init)
    print(f'  → {len(results)} sources fit; columns: {results.colnames}',
          flush=True)

    results.write(OUT_PHOT, overwrite=True)
    print(f'Wrote {OUT_PHOT}', flush=True)


if __name__ == '__main__':
    sys.exit(main())
