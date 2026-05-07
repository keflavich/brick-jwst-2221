#!/usr/bin/env python3
"""
forced_photometry_residuals.py  –  brick-jwst-2221 pipeline

Per-frame forced PSF photometry for sources in the iter3 union seed catalog
that have detected_{filter}=False in a given filter.

For each per-frame satstar residual file (*_crf_iter3_satstar_residual.fits)
the sources that land on that frame are measured using the optimal matched filter:

    flux     = Σ [p·d / σ²] / Σ [p² / σ²]
    flux_err = 1 / sqrt( Σ [p² / σ²] )

where p = PSF stamp (GriddedPSFModel in the full-detector pixel frame, NOT
renormalised to the stamp sum — raw values correctly recover the total source
flux in the matched-filter formula), d = background-subtracted residual data,
σ = ERR extension of the matching CRF file.

PSF coordinates are in the full-detector frame:
    det_x = frame_x + (SUBSTRT1 - 1)
    det_y = frame_y + (SUBSTRT2 - 1)

This gives N measurements per source per filter (one per frame it landed on),
enabling flux+RMS consistency checks across exposures.

Outputs (in {basepath}/catalogs/):
    forced_photometry_iter3_{target}_perframe.fits
        One row per (source, frame).
        Columns: source_id_union, ra, dec, filter, exposure_id,
                 x_frame, y_frame, det_x, det_y,
                 flux_forced, flux_err_forced, snr_forced, n_good_pix,
                 forced, flag_edge, flag_allnan.

    forced_photometry_iter3_{target}_summary.fits
        One row per (source, filter): n_frames, flux_mean, flux_rms,
        flux_err_median, snr_mean.

Usage:
    python forced_photometry_residuals.py --target sickle
    python forced_photometry_residuals.py --target brick --filternames F115W,F356W
    python forced_photometry_residuals.py --target brick \\
        --union-catalog /path/to/seed.fits --output-dir /path/to/output/
"""

import os
import re
import glob
import warnings
import argparse
import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.stats import sigma_clipped_stats
from photutils.psf import GriddedPSFModel


# ── NIRCam SW / LW classification ────────────────────────────────────────────
# SW: detector pixel scale ~0.031 arcsec/pix;  LW: ~0.063 arcsec/pix
_SW_FILTERS = frozenset([
    'F070W', 'F090W', 'F115W', 'F140M', 'F150W', 'F150W2',
    'F162M', 'F164N', 'F182M', 'F187N', 'F200W', 'F210M', 'F212N',
])

def _is_sw(filt_upper):
    return filt_upper.upper() in _SW_FILTERS


# ── Stamp / annulus geometry (in native detector pixels) ─────────────────────
STAMP_NPX_SW, ANN_INNER_SW, ANN_OUTER_SW = 25, 13, 20
STAMP_NPX_LW, ANN_INNER_LW, ANN_OUTER_LW = 21, 11, 17


# ── Per-target pipeline configuration ────────────────────────────────────────
TARGETS = {
    'sickle': dict(
        basepath = '/orange/adamginsburg/jwst/sickle',
        filters  = ['F187N', 'F210M', 'F335M', 'F470N', 'F480M'],
    ),
    'brick': dict(
        basepath = '/blue/adamginsburg/adamginsburg/jwst/brick',
        filters  = ['F115W', 'F182M', 'F187N', 'F200W', 'F212N',
                    'F356W', 'F405N', 'F410M', 'F444W', 'F466N'],
    ),
    'cloudc': dict(
        basepath = '/blue/adamginsburg/adamginsburg/jwst/cloudc',
        filters  = ['F182M', 'F187N', 'F212N', 'F405N', 'F410M', 'F466N'],
    ),
    'sgrb2': dict(
        basepath = '/orange/adamginsburg/jwst/sgrb2',
        filters  = ['F150W', 'F182M', 'F187N', 'F210M', 'F212N',
                    'F300M', 'F360M', 'F405N', 'F410M', 'F466N', 'F480M'],
    ),
}

# Matches nrcb1-4, nrca1-4, nrcblong, nrcalong in per-frame filenames.
DET_TOKEN_RE = re.compile(r'_(nrc[ab][1-4]|nrc[ab]long)_')


# ═══════════════════════════════════════════════════════════════════════════════
# PSF helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _psf_filepath(basepath, det_tok, filt_lower):
    """Return path to the GriddedPSFModel FITS for this detector / filter.

    Naming convention (shared across all pipeline targets):
        SW det nrcb{1-4} / nrca{1-4}  →  nircam_{det}_{filt}_fovp101_samp4_npsf16.fits
        LW det nrcblong               →  nircam_nrcb5_{filt}_fovp101_samp4_npsf16.fits
        LW det nrcalong               →  nircam_nrca5_{filt}_fovp101_samp4_npsf16.fits
    """
    if det_tok == 'nrcblong':
        psf_det = 'nrcb5'
    elif det_tok == 'nrcalong':
        psf_det = 'nrca5'
    else:
        psf_det = det_tok    # nrcb1-4 or nrca1-4
    return os.path.join(basepath,
                        f'nircam_{psf_det}_{filt_lower}_fovp101_samp4_npsf16.fits')


def _det_token(path):
    m = DET_TOKEN_RE.search(os.path.basename(path))
    return m.group(1) if m else None


# ═══════════════════════════════════════════════════════════════════════════════
# Measurement helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _eval_psf(psf_model, det_x, det_y, npix):
    """Return raw PSF stamp (npix×npix) at full-detector position (det_x, det_y).

    The values are NOT renormalised to the stamp sum.  In the matched-filter
    formula  flux = Σ(p d/σ²) / Σ(p²/σ²)  the raw model values (which sum
    to <1 because only the central npix pixels are sampled) correctly recover
    the total source flux.  Dividing by the stamp sum would underestimate flux
    by the enclosed-energy fraction (~10 % for a 21-px LW stamp).
    """
    half = npix // 2
    x0   = int(np.round(det_x))
    y0   = int(np.round(det_y))
    yy, xx = np.mgrid[y0 - half: y0 - half + npix,
                      x0 - half: x0 - half + npix]
    psf_model.x_0  = float(det_x)
    psf_model.y_0  = float(det_y)
    psf_model.flux = 1.0
    return psf_model(xx, yy)


def _extract_stamp(data, cx, cy, npix):
    half   = npix // 2
    x0, y0 = int(np.round(cx)), int(np.round(cy))
    ny, nx  = data.shape
    ys = slice(max(y0 - half, 0),      min(y0 - half + npix, ny))
    xs = slice(max(x0 - half, 0),      min(x0 - half + npix, nx))
    stamp   = np.full((npix, npix), np.nan)
    dy0 = ys.start - (y0 - half)
    dx0 = xs.start - (x0 - half)
    dy1 = dy0 + (ys.stop - ys.start)
    dx1 = dx0 + (xs.stop - xs.start)
    if dy1 > 0 and dx1 > 0:
        stamp[dy0:dy1, dx0:dx1] = data[ys, xs]
    flag_edge = (dy0 > 0 or dx0 > 0 or dy1 < npix or dx1 < npix)
    return stamp, flag_edge


def _annulus_background(stamp, inner, outer):
    npix = stamp.shape[0]
    cy = cx = (npix - 1) / 2.0
    yy, xx   = np.indices(stamp.shape)
    r        = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    mask     = (r >= inner) & (r <= outer) & np.isfinite(stamp)
    if mask.sum() < 3:
        return 0.0
    _, bkg, _ = sigma_clipped_stats(stamp[mask], sigma=3.0, maxiters=5)
    return float(bkg)


def _psf_weighted_flux(psf_stamp, sci_stamp, err_stamp, bkg):
    d    = sci_stamp - bkg
    p, s = psf_stamp, err_stamp
    good = np.isfinite(d) & np.isfinite(p) & np.isfinite(s) & (s > 0)
    if not good.any():
        return np.nan, np.nan, 0
    p_g, d_g, inv_var = p[good], d[good], 1.0 / s[good]**2
    denom = np.dot(p_g**2, inv_var)
    if denom <= 0:
        return np.nan, np.nan, int(good.sum())
    return (np.dot(p_g * d_g, inv_var) / denom,
            1.0 / np.sqrt(denom),
            int(good.sum()))


# ═══════════════════════════════════════════════════════════════════════════════
# Per-frame processor
# ═══════════════════════════════════════════════════════════════════════════════

def process_frame(res_path, psf_model,
                  sub_sc, sub_ids, sub_ra, sub_dec,
                  npix, ann_in, ann_out):
    """Return list of measurement dicts for sources landing on this frame."""
    fname       = os.path.basename(res_path)
    exposure_id = fname.replace('_iter3_satstar_residual.fits', '')

    # Residual data lives in the PRIMARY extension of the satstar residual file.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        hdul = fits.open(res_path, ignore_missing_end=True, memmap=False)
    sci       = hdul[0].data.astype(np.float64)
    ph        = hdul[0].header
    frame_wcs = WCS(ph, naxis=2)
    x_offset  = int(ph.get('SUBSTRT1', 1)) - 1   # 1-indexed → 0-indexed
    y_offset  = int(ph.get('SUBSTRT2', 1)) - 1
    hdul.close()
    ny, nx = sci.shape

    # ERR comes from the matching CRF (strip the satstar residual suffix).
    crf_path = res_path.replace('_iter3_satstar_residual.fits', '.fits')
    if not os.path.exists(crf_path):
        print(f'    [warn] CRF not found: {os.path.basename(crf_path)}')
        return []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        hdul = fits.open(crf_path, ignore_missing_end=True, memmap=False)
    err = hdul['ERR'].data.astype(np.float64)
    hdul.close()
    err[err <= 0] = np.nan

    # Vectorised sky → subarray pixel for all undetected sources.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        px, py = frame_wcs.world_to_pixel(sub_sc)
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)

    on_frame = (np.isfinite(px) & np.isfinite(py) &
                (px >= 0) & (px < nx) & (py >= 0) & (py < ny))
    indices  = np.where(on_frame)[0]
    if len(indices) == 0:
        return []

    rows = []
    for k in indices:
        xf, yf = float(px[k]), float(py[k])
        det_x  = xf + x_offset
        det_y  = yf + y_offset

        try:
            psf_stamp = _eval_psf(psf_model, det_x, det_y, npix)
        except Exception:
            continue

        sci_stamp, flag_edge = _extract_stamp(sci, xf, yf, npix)
        err_stamp, _         = _extract_stamp(err, xf, yf, npix)

        bkg  = _annulus_background(sci_stamp, ann_in, ann_out)
        flux, flux_err, n_good = _psf_weighted_flux(
            psf_stamp, sci_stamp, err_stamp, bkg)
        flag_allnan = (n_good == 0)
        snr = (flux / flux_err
               if (np.isfinite(flux) and np.isfinite(flux_err) and flux_err > 0)
               else np.nan)

        rows.append(dict(
            source_id_union = int(sub_ids[k]),
            ra              = float(sub_ra[k]),
            dec             = float(sub_dec[k]),
            exposure_id     = exposure_id,
            x_frame         = xf,
            y_frame         = yf,
            det_x           = float(det_x),
            det_y           = float(det_y),
            flux_forced     = float(flux)     if np.isfinite(flux)     else np.nan,
            flux_err_forced = float(flux_err) if np.isfinite(flux_err) else np.nan,
            snr_forced      = float(snr)      if np.isfinite(snr)      else np.nan,
            n_good_pix      = int(n_good),
            forced          = True,
            flag_edge       = bool(flag_edge),
            flag_allnan     = bool(flag_allnan),
        ))
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# Summary: mean flux, RMS, N_frames per (source, filter)
# ═══════════════════════════════════════════════════════════════════════════════

def make_summary(out):
    print('\nComputing per-source summary …', flush=True)
    rows = []
    for filt in np.unique(out['filter']):
        t = out[out['filter'] == filt]
        for grp in t.group_by('source_id_union').groups:
            sid    = int(grp['source_id_union'][0])
            fluxes = np.array(grp['flux_forced'],     dtype=float)
            errs   = np.array(grp['flux_err_forced'], dtype=float)
            finite = np.isfinite(fluxes)
            n_frames     = int(finite.sum())
            flux_mean    = float(np.nanmean(fluxes)) if n_frames      else np.nan
            flux_rms     = float(np.nanstd(fluxes))  if n_frames > 1  else np.nan
            flux_err_med = float(np.nanmedian(errs))  if n_frames      else np.nan
            snr_mean     = (flux_mean / flux_err_med
                            if (np.isfinite(flux_mean) and np.isfinite(flux_err_med)
                                and flux_err_med > 0) else np.nan)
            rows.append(dict(
                source_id_union = sid,
                ra              = float(grp['ra'][0]),
                dec             = float(grp['dec'][0]),
                filter          = str(filt),
                n_frames        = n_frames,
                flux_mean       = flux_mean,
                flux_rms        = flux_rms,
                flux_err_median = flux_err_med,
                snr_mean        = snr_mean,
                forced          = True,
            ))
    return Table(rows) if rows else None


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def run(args):
    cfg      = TARGETS[args.target]
    basepath = cfg['basepath']
    catdir   = args.output_dir or os.path.join(basepath, 'catalogs')
    os.makedirs(catdir, exist_ok=True)

    filters = ([f.strip().upper() for f in args.filternames.split(',') if f.strip()]
               if args.filternames else cfg['filters'])

    seed_path = (args.union_catalog
                 or os.path.join(basepath, 'catalogs',
                                 f'seed_union_iter3_{args.target}.fits'))
    print(f'Loading union catalog: {seed_path}')
    union   = Table.read(seed_path)
    ra      = np.array(union['ra'],  dtype=float)
    dec     = np.array(union['dec'], dtype=float)
    ids     = np.array(union['source_id_union'])
    all_sky = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)

    all_tables = []

    for filt in filters:
        filt_lower = filt.lower()
        sw = _is_sw(filt)
        npix, ann_in, ann_out = (
            (STAMP_NPX_SW, ANN_INNER_SW, ANN_OUTER_SW) if sw else
            (STAMP_NPX_LW, ANN_INNER_LW, ANN_OUTER_LW))

        det_col = f'detected_{filt_lower}'
        if det_col not in union.colnames:
            print(f'[{filt}] column {det_col!r} absent from union catalog — skipping')
            continue
        needs_forced = ~np.array(union[det_col], dtype=bool)
        n_forced     = int(needs_forced.sum())

        filt_dir  = os.path.join(basepath, filt)
        all_res   = sorted(glob.glob(
            os.path.join(filt_dir, 'pipeline', '*_iter3_satstar_residual.fits')))
        res_files = [f for f in all_res if '_bgsub_' not in os.path.basename(f)]
        n_frames  = len(res_files)

        print(f'\n[{filt}]  {n_forced}/{len(union)} sources need forced photometry'
              f'  |  {n_frames} frames', flush=True)
        if n_forced == 0 or n_frames == 0:
            continue

        sub_sc  = all_sky[needs_forced]
        sub_ids = ids[needs_forced]
        sub_ra  = ra[needs_forced]
        sub_dec = dec[needs_forced]

        # Cache PSF models per detector token (loaded on first use).
        psf_cache: dict = {}

        filt_rows = []
        for i, res_path in enumerate(res_files):
            det_tok = _det_token(res_path)
            if det_tok is None:
                print(f'  [warn] cannot parse detector from {os.path.basename(res_path)}')
                continue

            if det_tok not in psf_cache:
                psf_file = _psf_filepath(basepath, det_tok, filt_lower)
                if not os.path.exists(psf_file):
                    print(f'  [warn] PSF not found: {os.path.basename(psf_file)} — '
                          f'skipping all {det_tok} frames for {filt}')
                    psf_cache[det_tok] = None
                else:
                    psf_cache[det_tok] = GriddedPSFModel.read(psf_file)

            psf_model = psf_cache[det_tok]
            if psf_model is None:
                continue

            frame_rows = process_frame(
                res_path, psf_model,
                sub_sc, sub_ids, sub_ra, sub_dec,
                npix, ann_in, ann_out)
            filt_rows.extend(frame_rows)

            if (i + 1) % 10 == 0 or (i + 1) == n_frames:
                print(f'  frame {i+1:3d}/{n_frames}  '
                      f'accumulated {len(filt_rows)} rows', flush=True)

        if not filt_rows:
            continue

        t = Table(filt_rows)
        t['filter'] = filt_lower
        finite = np.isfinite(t['flux_forced'])
        snr    = t['snr_forced'][finite]
        print(f'  → {len(t)} rows  finite: {finite.sum()}'
              f'  SNR>3: {(snr > 3).sum()}  SNR>5: {(snr > 5).sum()}')
        all_tables.append(t)

    if not all_tables:
        print('No forced measurements produced.')
        return

    out = vstack(all_tables, metadata_conflicts='silent')
    for col in ('forced', 'flag_edge', 'flag_allnan'):
        out[col] = out[col].astype(bool)

    outfile_pf  = os.path.join(catdir,
                               f'forced_photometry_iter3_{args.target}_perframe.fits')
    outfile_sum = os.path.join(catdir,
                               f'forced_photometry_iter3_{args.target}_summary.fits')

    out.write(outfile_pf, overwrite=True)
    print(f'\nWrote {len(out)} rows → {outfile_pf}')

    summ = make_summary(out)
    if summ is not None:
        summ['forced'] = summ['forced'].astype(bool)
        summ.write(outfile_sum, overwrite=True)
        print(f'Wrote {len(summ)} rows → {outfile_sum}')

    print('\nPer-frame table columns:')
    for col in out.colnames:
        print(f'  {col:22s}  {out[col].dtype}')


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--target', required=True,
                   choices=sorted(TARGETS.keys()),
                   help='Pipeline target (sickle | brick | cloudc | sgrb2)')
    p.add_argument('--filternames', default='',
                   help='Comma-separated filters to process '
                        '(default: all filters for the target)')
    p.add_argument('--union-catalog', default='',
                   help='Path to iter3 union seed catalog '
                        '(default: {basepath}/catalogs/seed_union_iter3_{target}.fits)')
    p.add_argument('--output-dir', default='',
                   help='Directory for output FITS files '
                        '(default: {basepath}/catalogs/)')
    args = p.parse_args(argv)
    run(args)


if __name__ == '__main__':
    main()
