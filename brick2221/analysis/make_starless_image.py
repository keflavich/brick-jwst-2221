#!/usr/bin/env python3
"""
make_starless_image.py

Produce "starless" versions of the iter3 residual mosaics:

  1. MEASURE  — for every catalog source, measure how far its residual
                extends above the background noise in a local cutout.
                The cutout size scales with source brightness so that
                bright/saturated stars get large search radii.

  2. MASK     — stamp a circle of the measured radius onto a boolean mask.
                Run scipy.ndimage.distance_transform_edt to get a "scale map"
                (how far each masked pixel is from the nearest real pixel).

  3. INFILL   — set masked pixels to NaN, then repeatedly apply a NaN-aware
                Gaussian filter with progressively larger σ.  A pixel is
                filled at the smallest σ for which σ ≥ scale_map value,
                ensuring the kernel always reaches real (unmasked) data.

  4. NOISE    — add per-pixel Gaussian noise drawn from the ERR plane of the
                original i2d so the infilled region has realistic texture.

Usage:
  python make_starless_image.py --target sickle [--filter F187N ...] [--dry-run]

The "dry-run" mode runs only step 1 and writes the radius table, which is
useful for inspecting the flux–radius relationship before committing to a
full run.
"""

import argparse
import glob
import os
import warnings

import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.stats import sigma_clipped_stats


# ── target configuration ──────────────────────────────────────────────────────
# catalog_rel : path to the union seed catalog, relative to basepath
# catalog_flux_prefix : column name stub, e.g. 'flux_f187n'
TARGETS = {
    'sickle': dict(
        basepath='/orange/adamginsburg/jwst/sickle',
        filters=['F187N', 'F210M', 'F335M', 'F470N', 'F480M'],
        catalog_rel='catalogs/seed_union_iter3_sickle.fits',
        # Region files with manually-identified residuals to force-mask,
        # regardless of catalog SNR.  Paths relative to basepath.
        force_mask_regs=[
            'regions_/sickle_starlike_residuals_colorcoded.reg',
            'regions_/undersubtracted_stars_20260426.reg',
        ],
    ),
    'brick': dict(
        basepath='/blue/adamginsburg/adamginsburg/jwst/brick',
        filters=['F115W', 'F182M', 'F187N', 'F200W', 'F212N',
                 'F356W', 'F405N', 'F410M', 'F444W', 'F466N'],
        catalog_rel='catalogs/seed_union_iter3_brick.fits',
    ),
    'cloudc': dict(
        basepath='/blue/adamginsburg/adamginsburg/jwst/cloudc',
        filters=['F182M', 'F187N', 'F212N', 'F405N', 'F410M', 'F466N'],
        catalog_rel='catalogs/seed_union_iter3_cloudc.fits',
    ),
    'sgrb2': dict(
        basepath='/orange/adamginsburg/jwst/sgrb2',
        filters=['F150W', 'F182M', 'F187N', 'F210M', 'F212N',
                 'F300M', 'F360M', 'F405N', 'F410M', 'F466N', 'F480M'],
        catalog_rel='catalogs/seed_union_iter3_sgrb2.fits',
    ),
}

# ── algorithm parameters ──────────────────────────────────────────────────────
# Mask radius measurement
BKG_ANNULUS_INNER_FRAC = 0.70   # background annulus starts at this fraction of max_r
BKG_ANNULUS_OUTER_FRAC = 0.90   # background annulus ends at this fraction of max_r
BKG_ANNULUS_MIN_INNER  = 6      # minimum inner radius regardless of max_r (px)
RESIDUAL_NSIGMA        = 2.5    # mask out to where |median profile| < N * sigma_bkg
SPIKE_NSIGMA           = 10.0   # mask if any ring pixel deviates > N * sigma_bkg (catches spikes)
SPIKE_MAX_R            = 8      # px — apply spike test only inside this radius to avoid picking up
                                 #   neighboring stars' PSF peaks in the outer rings of crowded fields
RADIAL_BIN_WIDTH       = 2      # px
MIN_MASK_RADIUS        = 2      # px — skip masking for anything smaller
MAX_MASK_RADIUS        = 600    # px — absolute cap

# Maximum search radius per brightness tier (px)
MAX_R_SINGULAR  = 80     # the single brightest star only (hardcoded position)
MAX_R_SATURATED = 25     # all other is_saturated sources
MAX_R_BRIGHT    = 25     # SNR > 200 (non-saturated)
MAX_R_MEDIUM    = 25     # SNR > 100 — must be >= BKG_ANNULUS_MIN_INNER/INNER_FRAC for annulus to fit
MAX_R_FAINT     = 10     # SNR > 50  — must be >= BKG_ANNULUS_MIN_INNER/INNER_FRAC for annulus to fit
MAX_R_VERYFAINT = 20     # used for --force-mask-reg positions (not catalog SNR-based)
SNR_SKIP        = 50.    # below this SNR: skip catalog-based measurement

# The single saturated supergiant that needs a 70 px mask — identified by
# visual inspection.  Matched with a 1" tolerance.
_SINGULAR_STAR_RA  = 266.572927778   # 17:46:17.5026267824
_SINGULAR_STAR_DEC = -28.803434417   # -28:48:12.3643944825
_SINGULAR_STAR_TOL = 1.0 / 3600      # 1 arcsec in degrees

# Iterative boundary-propagation infill
# Each pass uses this fixed kernel; σ grows automatically only if the kernel
# can't reach any real data (large isolated holes).  Small σ preserves
# background structure; larger values give smoother fills for very deep holes.
INFILL_SIGMA     = 3      # px — starting (and preferred) kernel size
INFILL_MAX_ITER  = 1000   # safety cap on iterations
INFILL_SIGMA_MAX = 128    # px — never go wider than this

NOISE_SEED  = 42
NOISE_SCALE = 1.0 / 3.0   # fraction of ERR to use as noise sigma


# ── file path helpers ─────────────────────────────────────────────────────────

def _find_residual_mosaic(basepath, filt, method='iterative', bgsub=False,
                          infilled=True):
    """
    Glob for the iter3 residual i2d.  Returns the first match or None.
    Prefers infilled over non-infilled, iterative over basic.

    bgsub=False explicitly excludes files whose basename contains 'bgsub'
    (which would otherwise sort ahead of the non-bgsub version).
    """
    fdir = os.path.join(basepath, filt.upper(), 'pipeline')
    infill_tok = '_infilled' if infilled else ''
    pattern = os.path.join(
        fdir,
        f'*{filt.lower()}*iter3_daophot_{method}_residual{infill_tok}_i2d.fits'
    )
    import re as _re
    def _combined_first(p):
        # Prefer combined-module files (e.g. nrcb_iter3) over per-detector
        # (nrcb1_iter3, nrcb2_iter3, ...).  ASCII sort puts digits before '_',
        # so without this key nrcb1 would be returned before nrcb.
        return (1 if _re.search(r'nrcb\d', os.path.basename(p)) else 0,
                p)

    matches = [p for p in sorted(glob.glob(pattern))
               if ('bgsub' in os.path.basename(p)) == bgsub]
    matches.sort(key=_combined_first)
    if matches:
        return matches[0]
    # fall back: any residual i2d for this filter/method, respecting bgsub flag
    fallback = os.path.join(fdir, f'*{filt.lower()}*iter3*{method}*residual*i2d.fits')
    matches = [p for p in sorted(glob.glob(fallback))
               if ('bgsub' in os.path.basename(p)) == bgsub]
    matches.sort(key=_combined_first)
    return matches[0] if matches else None


def _find_original_i2d(basepath, filt):
    """Return path to the original (non-residual) i2d mosaic for the ERR plane."""
    fdir = os.path.join(basepath, filt.upper(), 'pipeline')
    pattern = os.path.join(fdir, f'*{filt.lower()}*_i2d.fits')
    # Exclude residual files
    matches = [p for p in sorted(glob.glob(pattern)) if 'residual' not in p]
    return matches[0] if matches else None


# ── FITS I/O ──────────────────────────────────────────────────────────────────

def load_sci(path):
    """Return (data_float64, wcs, sci_header)."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        hdul = fits.open(path, memmap=False)
    ext_names = [e.name for e in hdul]
    ext = 'SCI' if 'SCI' in ext_names else 0
    hdr  = hdul[ext].header
    data = hdul[ext].data.astype(np.float64)
    hdul.close()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        wcs = WCS(hdr, naxis=2)
    return data, wcs, hdr


def load_err(path):
    """Return ERR array (float64) or None."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        hdul = fits.open(path, memmap=False)
    if 'ERR' not in [e.name for e in hdul]:
        hdul.close()
        return None
    err = hdul['ERR'].data.astype(np.float64)
    hdul.close()
    return err


# ── NaN-aware Gaussian filter ─────────────────────────────────────────────────

def nan_gaussian(data, sigma, truncate=4.0):
    """
    Gaussian filter that interpolates across NaN pixels.

    Uses the kernel-normalisation trick: smooth (data with NaN→0) and
    smooth (finite mask), then divide.  Pixels where the normalisation
    weight < 0.01 (too far from real data) are returned as NaN.
    """
    finite  = np.isfinite(data).astype(np.float64)
    filled  = np.where(finite.astype(bool), data, 0.0)
    smooth  = gaussian_filter(filled, sigma=sigma, truncate=truncate)
    norm    = gaussian_filter(finite, sigma=sigma, truncate=truncate)
    with np.errstate(invalid='ignore'):
        result = smooth / norm
    result[norm < 0.01] = np.nan
    return result


# ── STEP 1: measure mask radii ────────────────────────────────────────────────

def _max_r_for_source(flux, snr, is_saturated):
    """Return the maximum search radius (px) appropriate for this source."""
    if is_saturated:
        return MAX_R_SATURATED   # 25 px; singular star handled upstream by coords
    if snr > 200:
        return MAX_R_BRIGHT      # 25 px
    if snr > 100:
        return MAX_R_MEDIUM      # 25 px
    if snr > SNR_SKIP:
        return MAX_R_FAINT       # 10 px
    return 0                     # too faint — skip (use force_mask_regs for exceptions)


def measure_mask_radius_cutout(data, cy, cx, max_r):
    """
    Measure the radius (px) out to which the residual around (cy, cx) is
    elevated above the background noise.

    Strategy:
      - Estimate background median and sigma from an outer annulus.
      - Scan radial bins; extend mask_r whenever the ring is elevated above
        background by either the median test (RESIDUAL_NSIGMA * sigma) or
        by any single pixel exceeding SPIKE_NSIGMA * sigma (catches diffraction
        spikes).  Both tests use background-referenced deviations so the
        function works correctly in non-zero background regions.

    Returns 0.0 if the source is undetectable.
    """
    ny, nx = data.shape
    # Clamp cutout to image bounds
    r = max_r
    y0 = max(0, cy - r);  y1 = min(ny, cy + r + 1)
    x0 = max(0, cx - r);  x1 = min(nx, cx + r + 1)

    cut = data[y0:y1, x0:x1]
    yy, xx = np.ogrid[y0:y1, x0:x1]
    rr = np.sqrt((yy - cy)**2 + (xx - cx)**2)

    # Background annulus
    ann_inner = max(BKG_ANNULUS_MIN_INNER, r * BKG_ANNULUS_INNER_FRAC)
    ann_outer = r * BKG_ANNULUS_OUTER_FRAC
    ann = (rr >= ann_inner) & (rr <= ann_outer) & np.isfinite(cut)
    bkg_pix = cut[ann]
    if bkg_pix.size < 10:
        return 0.0
    _, bkg_med, sigma_bkg = sigma_clipped_stats(bkg_pix, sigma=3.0, maxiters=5)
    if sigma_bkg <= 0:
        return 0.0
    threshold      = RESIDUAL_NSIGMA * sigma_bkg
    spike_threshold = SPIKE_NSIGMA   * sigma_bkg

    # Radial profile — scan out from centre, record outermost elevated bin.
    # Test 1 (median): catches symmetric residuals even in non-zero background.
    # Test 2 (max-deviation): catches diffraction spikes (single bright pixels).
    bins = np.arange(0, ann_inner, RADIAL_BIN_WIDTH)
    mask_r = 0.0
    for r0, r1 in zip(bins[:-1], bins[1:]):
        ring = (rr >= r0) & (rr < r1) & np.isfinite(cut)
        ring_pix = cut[ring]
        if ring_pix.size == 0:
            continue
        ring_dev = np.abs(ring_pix - bkg_med)
        spike_ok = (r1 <= SPIKE_MAX_R) and (np.max(ring_dev) > spike_threshold)
        if np.median(ring_dev) > threshold or spike_ok:
            mask_r = r1

    return float(mask_r)


def measure_all_mask_radii(data, wcs, catalog, filt_lower):
    """
    Loop over catalog sources and measure each one's mask radius.

    Returns (px_arr, py_arr, radii) — pixel coords and radii parallel to
    the catalog rows.  Sources off the image or below SNR_SKIP get radius=0.
    """
    sc = SkyCoord(catalog['ra'] * u.deg, catalog['dec'] * u.deg)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pix = wcs.world_to_pixel(sc)
    px_arr = np.array(pix[0], dtype=float)
    py_arr = np.array(pix[1], dtype=float)

    ny, nx = data.shape
    radii  = np.zeros(len(catalog), dtype=float)

    # Pre-compute which source (if any) is the singular supergiant
    _singular_coord = SkyCoord(_SINGULAR_STAR_RA * u.deg, _SINGULAR_STAR_DEC * u.deg)
    _sep_deg = sc.separation(_singular_coord).deg
    is_singular = _sep_deg < _SINGULAR_STAR_TOL

    # Per-filter flux & SNR columns
    flux_col    = f'flux_{filt_lower}'
    fluxerr_col = f'fluxerr_{filt_lower}'
    has_flux    = flux_col in catalog.colnames
    has_err     = fluxerr_col in catalog.colnames

    # All-filter flux/err column pairs for computing max-band SNR
    all_filt_pairs = [(f'flux_{f}', f'fluxerr_{f}')
                      for f in ['f187n', 'f210m', 'f335m', 'f470n', 'f480m']
                      if f'flux_{f}' in catalog.colnames and f'fluxerr_{f}' in catalog.colnames]

    # Global noise estimate for SNR computation
    finite_pix = data[np.isfinite(data)]
    _, _, global_sigma = sigma_clipped_stats(finite_pix, sigma=3.0, maxiters=3)

    for i in range(len(catalog)):
        px, py = px_arr[i], py_arr[i]
        cx, cy = int(round(px)), int(round(py))
        if cx < 0 or cx >= nx or cy < 0 or cy >= ny:
            continue

        is_sat = bool(catalog['is_saturated'][i]) if 'is_saturated' in catalog.colnames else False

        # Per-filter SNR (used for masking this filter's residual)
        if has_flux:
            flux = float(catalog[flux_col][i]) if not np.ma.is_masked(catalog[flux_col][i]) else 0.0
            err  = float(catalog[fluxerr_col][i]) if (has_err and not np.ma.is_masked(catalog[fluxerr_col][i])) else global_sigma
            snr  = flux / err if err > 0 else 0.0
        else:
            flux, snr = 0.0, 0.0

        # Max SNR across all filters: ensures sources bright in other bands
        # (but missed/faint in this one) still get an adequate search window.
        max_snr = snr
        for fc, ec in all_filt_pairs:
            try:
                if np.ma.is_masked(catalog[fc][i]) or np.ma.is_masked(catalog[ec][i]):
                    continue
                f_val = float(catalog[fc][i])
                e_val = float(catalog[ec][i])
                if e_val > 0:
                    max_snr = max(max_snr, f_val / e_val)
            except Exception:
                pass

        if is_singular[i]:
            max_r = MAX_R_SINGULAR
        else:
            max_r = _max_r_for_source(flux, max_snr, is_sat)
        if max_r == 0:
            continue

        r = measure_mask_radius_cutout(data, cy, cx, max_r)
        radii[i] = r

        if (i + 1) % 1000 == 0:
            n_done = (radii[:i+1] >= MIN_MASK_RADIUS).sum()
            print(f'    {i+1}/{len(catalog)} sources measured; '
                  f'{n_done} will be masked')

    return px_arr, py_arr, radii


# ── STEP 2: build mask and scale map ─────────────────────────────────────────

def build_mask_and_scale_map(shape, px_arr, py_arr, radii):
    """
    Stamp circles onto a boolean mask, then compute the distance transform
    so that scale_map[y, x] = distance in pixels to the nearest unmasked pixel.
    """
    mask = np.zeros(shape, dtype=bool)
    ny, nx = shape

    for px, py, r in zip(px_arr, py_arr, radii):
        if r < MIN_MASK_RADIUS:
            continue
        cx, cy  = int(round(px)), int(round(py))
        r_ceil  = int(np.ceil(r))
        y0 = max(0, cy - r_ceil);  y1 = min(ny, cy + r_ceil + 1)
        x0 = max(0, cx - r_ceil);  x1 = min(nx, cx + r_ceil + 1)
        yy, xx  = np.ogrid[y0:y1, x0:x1]
        circle  = (yy - cy)**2 + (xx - cx)**2 <= r**2
        mask[y0:y1, x0:x1] |= circle

    # Pixels that are already NaN in the image don't count as real data
    # (they're outside the mosaic footprint) — treat them as masked too so
    # the distance transform measures to real, finite pixels.
    scale_map = distance_transform_edt(mask)
    return mask, scale_map


# ── STEP 3: iterative boundary-propagation infill ─────────────────────────────

def progressive_infill(data, mask, scale_map,
                       sigma=INFILL_SIGMA, max_iter=INFILL_MAX_ITER,
                       sigma_max=INFILL_SIGMA_MAX):
    """
    Fill masked pixels by propagating the boundary inward with a fixed small
    Gaussian kernel, so the infilled texture always matches the local background
    at the scale of `sigma` pixels rather than blurring from far away.

    Algorithm:
      - Set masked pixels to NaN.
      - Each iteration: apply NaN-aware Gaussian(σ), fill any still-NaN pixel
        where the kernel reached real data.  The newly filled pixels become
        real data for the next pass, so the fill front advances ~σ px inward
        per iteration.
      - If an iteration fills nothing (kernel too small to reach real data,
        i.e. a very large isolated hole), double σ and retry.  σ never exceeds
        sigma_max.

    This preserves background structure much better than filling with a single
    large kernel: a pixel 40 px inside a mask is filled from its ~3 px-away
    neighbour (which was itself filled from its neighbour, etc.) rather than
    being interpolated directly from 160 px away.
    """
    work = data.copy()
    work[mask] = np.nan
    still_nan = mask.copy()

    current_sigma = float(sigma)
    stall_count   = 0

    for iteration in range(max_iter):
        if not still_nan.any():
            break

        smoothed  = nan_gaussian(work, current_sigma)
        fill_here = still_nan & np.isfinite(smoothed)

        if not fill_here.any():
            # Kernel can't reach real data — widen and retry
            stall_count += 1
            current_sigma = min(current_sigma * 2, sigma_max)
            if stall_count > 10:
                print(f'  [warn] stalled at σ={current_sigma:.0f}; '
                      f'{still_nan.sum():,} px remaining')
                break
            continue

        stall_count = 0
        work[fill_here] = smoothed[fill_here]
        still_nan[fill_here] = False

        if (iteration + 1) % 20 == 0:
            print(f'    iter {iteration+1:4d} σ={current_sigma:.0f}: '
                  f'{still_nan.sum():,} px remaining')

    n_left = still_nan.sum()
    if n_left:
        print(f'  [warn] {n_left:,} px could not be filled — left as NaN')
    else:
        print(f'    done in {iteration+1} iterations (max σ used: {current_sigma:.0f} px)')

    return work


# ── STEP 4: add noise ─────────────────────────────────────────────────────────

def add_noise(data, infill_mask, err, rng):
    """
    Add Gaussian noise with σ = ERR * NOISE_SCALE to infilled pixels.

    If ERR is unavailable, estimate a single global noise level from the
    sigma-clipped background of the unmasked pixels.
    """
    if err is not None and err.shape != data.shape:
        # Original i2d and residual mosaic can differ by a pixel along
        # an axis due to different ResampleStep output grids.
        if err.shape[0] >= data.shape[0] and err.shape[1] >= data.shape[1]:
            err = err[:data.shape[0], :data.shape[1]]
        else:
            # err is smaller than data in at least one axis; can't trim
            # data without shrinking the output, so fall back to global
            # noise estimate.
            err = None

    if err is not None:
        noise_map = np.abs(err) * NOISE_SCALE
    else:
        bkg = data[~infill_mask & np.isfinite(data)]
        _, _, sigma = sigma_clipped_stats(bkg, sigma=3.0, maxiters=5)
        noise_map = np.full(data.shape, sigma * NOISE_SCALE)

    result = data.copy()
    where  = infill_mask & np.isfinite(data)
    result[where] += rng.normal(0.0, noise_map[where])
    return result


# ── force-mask: user-specified region files ───────────────────────────────────

def _parse_ds9_points(reg_path):
    """Return list of (ra_deg, dec_deg) from DS9/CARTA point region file."""
    import re as _re
    coords = []
    with open(reg_path) as f:
        for line in f:
            m = _re.match(r'\s*point\(\s*([0-9.+-]+)\s*,\s*([0-9.+-]+)\s*\)', line)
            if m:
                coords.append((float(m.group(1)), float(m.group(2))))
    return coords


def measure_force_mask_positions(data, wcs, coord_list, max_r=MAX_R_VERYFAINT):
    """
    Measure mask radii at a list of (ra, dec) positions, bypassing catalog SNR.
    The ring scan is centered on the given coordinate, so it works even when the
    catalog position is offset from the actual stellar peak.

    Returns (px_arr, py_arr, radii) to be appended to the catalog-based results.
    """
    if not coord_list:
        return np.array([]), np.array([]), np.array([])
    sc = SkyCoord(ra=[c[0] for c in coord_list] * u.deg,
                  dec=[c[1] for c in coord_list] * u.deg)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pix = wcs.world_to_pixel(sc)
    px_arr = np.array(pix[0], dtype=float)
    py_arr = np.array(pix[1], dtype=float)
    ny, nx = data.shape
    radii = np.zeros(len(coord_list), dtype=float)
    for i, (px, py) in enumerate(zip(px_arr, py_arr)):
        cx, cy = int(round(px)), int(round(py))
        if cx < 0 or cx >= nx or cy < 0 or cy >= ny:
            continue
        radii[i] = measure_mask_radius_cutout(data, cy, cx, max_r)
    return px_arr, py_arr, radii


# ── main per-filter function ──────────────────────────────────────────────────

def make_starless_filter(filt, basepath, catalog_path, output_dir,
                         dry_run=False, method='iterative', bgsub=False,
                         force_mask_regs=None):
    """Full pipeline for one filter.

    Mask radii are measured and infilling is performed on the same image.
    Using a background-referenced ring test keeps the measurement correct even
    in regions of elevated diffuse emission.
    """
    print(f'\n=== {filt} ===')
    filt_lower = filt.lower()

    # ── locate input files ───────────────────────────────────────────────────
    res_path = _find_residual_mosaic(basepath, filt, method=method,
                                     bgsub=bgsub, infilled=True)
    if res_path is None:
        res_path = _find_residual_mosaic(basepath, filt, method=method,
                                         bgsub=bgsub, infilled=False)
    if res_path is None:
        print(f'  [skip] no iter3 {method} residual mosaic found')
        return

    # Measure radii from the same image used for infilling.  Using the
    # background-referenced ring test (median deviation from bkg_med) keeps
    # measurements correct even in regions of elevated diffuse emission, while
    # also detecting sources that the bgsub pipeline subtracted but the non-bgsub
    # pipeline did not (which the old bgsub-measurement approach would silently
    # miss).
    meas_path = res_path

    orig_path = _find_original_i2d(basepath, filt)

    print(f'  infill image:  {os.path.basename(res_path)}')
    print(f'  measure from:  {os.path.basename(meas_path)}')
    if orig_path:
        print(f'  noise source:  {os.path.basename(orig_path)}')
    else:
        print(f'  [warn] no original i2d found; noise will be estimated from image')

    if not os.path.exists(catalog_path):
        print(f'  [skip] catalog not found: {catalog_path}')
        return

    # ── load data ────────────────────────────────────────────────────────────
    data,      wcs,      sci_hdr = load_sci(res_path)
    meas_data, meas_wcs, _       = load_sci(meas_path)
    cat = Table.read(catalog_path)
    print(f'  image shape: {data.shape}  |  catalog: {len(cat)} sources')

    # ── step 1: measure mask radii ───────────────────────────────────────────
    print(f'  measuring mask radii …')
    px_arr, py_arr, radii = measure_all_mask_radii(meas_data, meas_wcs, cat, filt_lower)

    n_masked = (radii >= MIN_MASK_RADIUS).sum()
    r_nonzero = radii[radii >= MIN_MASK_RADIUS]
    if r_nonzero.size:
        print(f'  {n_masked}/{len(cat)} sources will be masked '
              f'(r: {r_nonzero.min():.1f}–{r_nonzero.max():.1f} px, '
              f'median {np.median(r_nonzero):.1f} px)')
    else:
        print(f'  0 sources require masking')

    # Save radius table for diagnostics / inspection (catalog sources only)
    radius_table = Table({
        'source_id_union': cat['source_id_union'] if 'source_id_union' in cat.colnames
                           else np.arange(len(cat)),
        'ra':              cat['ra'],
        'dec':             cat['dec'],
        'x_pix':           px_arr,
        'y_pix':           py_arr,
        'mask_radius_pix': radii,
    })
    rtab_path = os.path.join(output_dir,
                             f'{filt_lower}_starless_mask_radii.fits')
    radius_table.write(rtab_path, overwrite=True)
    print(f'  → radius table: {rtab_path}')

    # ── force-mask: manually-identified residuals ────────────────────────────
    # Run the ring scan centered on the region-file position (bypassing catalog
    # SNR).  Handles cases where the catalog position is wrong/offset or the
    # photometry measurement failed.
    if force_mask_regs:
        force_coords = []
        for rpath in force_mask_regs:
            if os.path.exists(rpath):
                new_coords = _parse_ds9_points(rpath)
                force_coords.extend(new_coords)
                print(f'  force-mask: {len(new_coords)} positions from {os.path.basename(rpath)}')
            else:
                print(f'  [warn] force-mask region not found: {rpath}')
        if force_coords:
            f_px, f_py, f_r = measure_force_mask_positions(
                meas_data, meas_wcs, force_coords, max_r=MAX_R_VERYFAINT)
            n_forced = (f_r >= MIN_MASK_RADIUS).sum()
            print(f'  {n_forced}/{len(force_coords)} forced positions detected and will be masked')
            px_arr = np.concatenate([px_arr, f_px])
            py_arr = np.concatenate([py_arr, f_py])
            radii  = np.concatenate([radii, f_r])

    if dry_run:
        return

    # ── step 2: build mask + scale map ───────────────────────────────────────
    print(f'  building mask …')
    mask, scale_map = build_mask_and_scale_map(data.shape, px_arr, py_arr, radii)
    frac = 100.0 * mask.mean()
    print(f'  masked pixels: {mask.sum():,} ({frac:.1f}% of image)')

    # ── step 3: iterative boundary-propagation infill ────────────────────────
    print(f'  infilling (σ={INFILL_SIGMA} px boundary propagation) …')
    filled = progressive_infill(data, mask, scale_map)

    # ── step 4: add noise ────────────────────────────────────────────────────
    print(f'  adding noise …')
    rng = np.random.default_rng(NOISE_SEED)
    err = load_err(orig_path) if orig_path else None
    filled_noisy = add_noise(filled, mask, err, rng)

    # ── write output ─────────────────────────────────────────────────────────
    bgsub_tok  = '_bgsub' if bgsub else ''
    out_name   = (f'{os.path.splitext(os.path.basename(res_path))[0]}'
                  .replace('_residual_infilled', '')
                  .replace('_residual', ''))
    out_name  += '_starless_i2d.fits'
    out_path   = os.path.join(output_dir, out_name)

    new_hdr = sci_hdr.copy()
    new_hdr['STARLESS'] = (True, 'stellar residuals masked and Gaussian-infilled')
    new_hdr['INFILLSG'] = (INFILL_SIGMA, 'boundary-propagation Gaussian sigma (px)')
    new_hdr['NMASK']    = (int(mask.sum()), 'pixels replaced by infill')
    new_hdr['NSOURCE']  = (int(n_masked), 'sources with mask radius >= MIN_MASK_RADIUS')

    hdu = fits.PrimaryHDU(data=filled_noisy.astype(np.float32), header=new_hdr)
    hdu.writeto(out_path, overwrite=True)
    print(f'  → starless image: {out_path}')


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Create starless residual mosaics via Gaussian infill.')
    parser.add_argument('--target', required=True, choices=list(TARGETS),
                        help='pipeline target name')
    parser.add_argument('--filter', dest='filters', nargs='+', metavar='FILTER',
                        help='process only these filters (default: all for target)')
    parser.add_argument('--catalog', default=None,
                        help='override union catalog path')
    parser.add_argument('--output-dir', default=None,
                        help='output directory (default: <basepath>/catalogs/starless)')
    parser.add_argument('--method', default='iterative',
                        choices=['iterative', 'basic'],
                        help='daophot residual method to use (default: iterative)')
    parser.add_argument('--bgsub', action='store_true',
                        help='use background-subtracted residual mosaic')
    parser.add_argument('--dry-run', action='store_true',
                        help='measure mask radii only; skip infill and image write')
    parser.add_argument('--force-mask-reg', dest='force_mask_regs', nargs='+',
                        metavar='REG_FILE',
                        help='additional DS9 region files with positions to force-mask '
                             '(appended to any target-specific list)')
    args = parser.parse_args()

    cfg      = TARGETS[args.target]
    basepath = cfg['basepath']
    filters  = args.filters if args.filters else cfg['filters']
    cat_path = args.catalog or os.path.join(basepath, cfg['catalog_rel'])
    out_dir  = args.output_dir or os.path.join(basepath, 'catalogs', 'starless')
    os.makedirs(out_dir, exist_ok=True)

    # Merge target-config force-mask regions with any CLI-supplied extras.
    # Target-config paths are relative to basepath.
    cfg_regs = [os.path.join(basepath, p) for p in cfg.get('force_mask_regs', [])]
    cli_regs = list(args.force_mask_regs or [])
    force_mask_regs = cfg_regs + cli_regs

    for filt in filters:
        make_starless_filter(
            filt, basepath, cat_path, out_dir,
            dry_run=args.dry_run,
            method=args.method,
            bgsub=args.bgsub,
            force_mask_regs=force_mask_regs or None,
        )

    print('\nDone.')


if __name__ == '__main__':
    main()
