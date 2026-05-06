#!/usr/bin/env python

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import regions
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.convolution import Gaussian2DKernel, convolve_fft, interpolate_replace_nans
from astropy.io import fits
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats import mad_std, sigma_clipped_stats
from astropy.table import Table
from astropy.visualization import simple_norm
from astropy.wcs import WCS
from matplotlib.backends.backend_pdf import PdfPages
from photutils.background import LocalBackground
from photutils.detection import DAOStarFinder
from photutils.psf import IterativePSFPhotometry, PSFPhotometry
from stpsf.utils import to_griddedpsfmodel

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from astropy.wcs import FITSFixedWarning
import warnings
warnings.simplefilter('ignore', category=FITSFixedWarning)


SATURATED_DQ_BIT = 2


def load_fits_data_and_wcs(filename: Path):
    with fits.open(filename) as hdul:
        if 'SCI' in hdul:
            data = np.asarray(hdul['SCI'].data, dtype=float)
            wcs = WCS(hdul['SCI'].header)
        elif len(hdul) > 1 and hdul[1].data is not None:
            data = np.asarray(hdul[1].data, dtype=float)
            wcs = WCS(hdul[1].header)
        else:
            data = np.asarray(hdul[0].data, dtype=float)
            wcs = WCS(hdul[0].header)
    return data, wcs


def load_fits_bundle(filename: Path):
    with fits.open(filename) as hdul:
        if 'SCI' in hdul:
            data = np.asarray(hdul['SCI'].data, dtype=float)
            wcs = WCS(hdul['SCI'].header)
        else:
            data = np.asarray(hdul[1].data, dtype=float)
            wcs = WCS(hdul[1].header)

        if 'ERR' in hdul:
            err = np.asarray(hdul['ERR'].data, dtype=float)
        elif len(hdul) > 2 and hdul[2].data is not None:
            err = np.asarray(hdul[2].data, dtype=float)
        else:
            err = None

        if 'DQ' in hdul:
            dq = np.asarray(hdul['DQ'].data)
        elif len(hdul) > 3 and hdul[3].data is not None:
            dq = np.asarray(hdul[3].data)
        else:
            dq = None

        if 'WHT' in hdul:
            wht = np.asarray(hdul['WHT'].data, dtype=float)
        else:
            wht = None

    return data, wcs, err, dq, wht


def read_point_regions(region_file: Path):
    regs = regions.Regions.read(region_file)
    points = [
        reg for reg in regs
        if hasattr(reg, 'center') and reg.__class__.__name__.endswith('PointSkyRegion')
    ]
    return points


def detect_negative_residual_stars(residual_data, fwhm_pix, sigma_threshold, roundlo, roundhi, sharplo, sharphi):
    inv_residual = -residual_data
    finite = np.isfinite(inv_residual)
    if finite.sum() == 0:
        raise ValueError('Residual image has no finite data.')

    med, _, std = sigma_clipped_stats(inv_residual[finite], sigma=3.0)
    robust_std = mad_std(inv_residual[finite], ignore_nan=True)
    noise = robust_std if np.isfinite(robust_std) and robust_std > 0 else std

    threshold = sigma_threshold * noise
    finder = DAOStarFinder(
        threshold=threshold,
        fwhm=float(fwhm_pix),
        roundlo=float(roundlo),
        roundhi=float(roundhi),
        sharplo=float(sharplo),
        sharphi=float(sharphi),
    )
    det = finder(inv_residual - med)
    if det is None:
        det = Table(names=('id', 'xcentroid', 'ycentroid', 'peak', 'flux'))

    return det, med, noise, threshold


def add_skycoords(tbl: Table, wcs: WCS):
    if len(tbl) == 0:
        tbl['ra_deg'] = np.array([], dtype=float)
        tbl['dec_deg'] = np.array([], dtype=float)
        return tbl

    x = np.asarray(tbl['xcentroid'], dtype=float)
    y = np.asarray(tbl['ycentroid'], dtype=float)
    sky = wcs.pixel_to_world(x, y)
    tbl['ra_deg'] = sky.ra.to_value(u.deg)
    tbl['dec_deg'] = sky.dec.to_value(u.deg)
    return tbl


def match_regions_to_detections(point_regions, det_tbl: Table, max_sep_arcsec):
    from astropy.coordinates import SkyCoord

    region_coords = SkyCoord(
        ra=np.array([reg.center.ra.to_value(u.deg) for reg in point_regions]) * u.deg,
        dec=np.array([reg.center.dec.to_value(u.deg) for reg in point_regions]) * u.deg,
    )

    if len(det_tbl) == 0:
        out = Table()
        out['region_index'] = np.arange(len(point_regions), dtype=int)
        out['matched'] = np.zeros(len(point_regions), dtype=bool)
        out['separation_arcsec'] = np.full(len(point_regions), np.nan)
        out['matched_detection_index'] = np.full(len(point_regions), -1, dtype=int)
        out['region_ra_deg'] = region_coords.ra.to_value(u.deg)
        out['region_dec_deg'] = region_coords.dec.to_value(u.deg)
        return out

    det_coords = SkyCoord(ra=det_tbl['ra_deg'] * u.deg, dec=det_tbl['dec_deg'] * u.deg)
    idx, sep2d, _ = region_coords.match_to_catalog_sky(det_coords)
    sep_arcsec = sep2d.to_value(u.arcsec)
    matched = sep_arcsec <= max_sep_arcsec

    out = Table()
    out['region_index'] = np.arange(len(point_regions), dtype=int)
    out['matched'] = matched
    out['separation_arcsec'] = sep_arcsec
    out['matched_detection_index'] = idx.astype(int)
    out['region_ra_deg'] = region_coords.ra.to_value(u.deg)
    out['region_dec_deg'] = region_coords.dec.to_value(u.deg)
    return out


def write_ds9_points(path: Path, ras_deg, dec_deg, color='#2EE6D6'):
    lines = [
        '# Region file format: DS9 CARTA 5.0.1',
        'global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1',
        'icrs',
    ]
    for ra_deg, dec_deg in zip(ras_deg, dec_deg):
        lines.append(f'point({ra_deg:.9f}, {dec_deg:.9f}) # color={color} width=2')
    path.write_text('\n'.join(lines) + '\n')


def cutout_slices(xc, yc, halfsize, shape):
    x0 = max(0, int(np.floor(xc)) - halfsize)
    x1 = min(shape[1], int(np.floor(xc)) + halfsize + 1)
    y0 = max(0, int(np.floor(yc)) - halfsize)
    y1 = min(shape[0], int(np.floor(yc)) + halfsize + 1)
    return slice(y0, y1), slice(x0, x1)


def estimate_flux_init(data_cutout, x0, y0, radius_pix):
    yy, xx = np.indices(data_cutout.shape)
    rr = np.hypot(xx - x0, yy - y0)
    core = data_cutout[rr <= radius_pix]
    ann = data_cutout[(rr >= radius_pix * 1.5) & (rr <= radius_pix * 2.5)]

    if core.size == 0:
        return 1.0
    ann_med = float(np.nanmedian(ann)) if ann.size > 0 else 0.0
    flux = np.nansum(core - ann_med)
    if not np.isfinite(flux) or flux <= 0:
        flux = max(np.nanmax(core), 1.0)
    return float(flux)


def core_metrics(residual_cutout, xfit, yfit, core_r=2.0, ring_in=3.0, ring_out=6.0):
    yy, xx = np.indices(residual_cutout.shape)
    rr = np.hypot(xx - xfit, yy - yfit)
    core = residual_cutout[rr <= core_r]
    ring = residual_cutout[(rr >= ring_in) & (rr <= ring_out)]

    core_median = float(np.nanmedian(core)) if core.size > 0 else np.nan
    core_min = float(np.nanmin(core)) if core.size > 0 else np.nan
    ring_median = float(np.nanmedian(ring)) if ring.size > 0 else np.nan
    center_value = float(residual_cutout[int(np.rint(yfit)), int(np.rint(xfit))])

    return {
        'core_median_resid': core_median,
        'core_min_resid': core_min,
        'ring_median_resid': ring_median,
        'center_resid': center_value,
    }


def radial_profile_median(image, x0, y0, max_radius=12.0, dr=0.5):
    yy, xx = np.indices(image.shape)
    rr = np.hypot(xx - x0, yy - y0)
    r_edges = np.arange(0.0, max_radius + dr, dr)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    med = np.full(len(r_centers), np.nan, dtype=float)
    npix = np.zeros(len(r_centers), dtype=int)

    for ii in range(len(r_centers)):
        sel = (rr >= r_edges[ii]) & (rr < r_edges[ii + 1])
        vals = image[sel]
        finite = np.isfinite(vals)
        npix[ii] = int(np.sum(finite))
        if npix[ii] > 0:
            med[ii] = float(np.nanmedian(vals[finite]))

    return r_centers, med, npix


def make_fit_weight_map(model_image, x0, y0, fit_shape=None):
    weights = np.zeros(model_image.shape, dtype=float)
    model_pos = np.clip(np.asarray(model_image, dtype=float), 0.0, None)

    if fit_shape is None:
        stamp = model_pos
        norm = np.nansum(stamp)
        if np.isfinite(norm) and norm > 0:
            weights = stamp / norm
        return weights

    hy = int(fit_shape[0]) // 2
    hx = int(fit_shape[1]) // 2
    xc = int(np.rint(x0))
    yc = int(np.rint(y0))

    x0i = max(0, xc - hx)
    x1i = min(model_image.shape[1], xc + hx + 1)
    y0i = max(0, yc - hy)
    y1i = min(model_image.shape[0], yc + hy + 1)

    stamp = model_pos[y0i:y1i, x0i:x1i]
    norm = np.nansum(stamp)
    if np.isfinite(norm) and norm > 0:
        weights[y0i:y1i, x0i:x1i] = stamp / norm

    return weights


def _coerce_to_data_shape(arr, data_shape, name, bitwise_or=False):
    out = np.asarray(arr)
    if out.shape == data_shape:
        return out

    squeezed = np.squeeze(out)
    if squeezed.shape == data_shape:
        return squeezed

    # Some products include an extra integration/group axis; collapse it.
    if out.ndim >= 3 and out.shape[-2:] == data_shape:
        if bitwise_or:
            reduced = np.bitwise_or.reduce(out.astype(np.int64), axis=tuple(range(out.ndim - 2)))
        else:
            reduced = np.nanmedian(out, axis=tuple(range(out.ndim - 2)))
        if reduced.shape == data_shape:
            return reduced

    raise ValueError(f'{name} shape {out.shape} is incompatible with data shape {data_shape}')


def compute_crowdsource_error_map(data, err, dq=None, wht=None):
    if err is None:
        return np.full(data.shape, np.inf, dtype=float)

    err = np.asarray(_coerce_to_data_shape(err, data.shape, 'ERR', bitwise_or=False), dtype=float)
    if wht is not None:
        wht = np.asarray(_coerce_to_data_shape(wht, data.shape, 'WHT', bitwise_or=False), dtype=float)
    if dq is not None:
        dq = np.asarray(_coerce_to_data_shape(dq, data.shape, 'DQ', bitwise_or=True))

    weight = err ** -1
    bad = np.isnan(weight) | (data == 0) | np.isnan(data) | (weight == 0) | (err == 0)
    if wht is not None:
        bad |= (wht == 0)

    mask = np.isnan(data)
    if dq is not None:
        is_saturated = (dq & SATURATED_DQ_BIT) != 0
        mask |= is_saturated

    bad |= mask
    err_eff = np.array(err, copy=True, dtype=float)
    err_eff[bad] = np.inf
    return err_eff


def compute_crowdsource_weight_map(data, err, dq=None, wht=None):
    err_eff = compute_crowdsource_error_map(data, err, dq=dq, wht=wht)
    weight = np.zeros(err_eff.shape, dtype=float)
    finite = np.isfinite(err_eff) & (err_eff > 0)
    weight[finite] = 1.0 / err_eff[finite]
    return weight


def radial_shape_metrics(data_cutout, model_cutout, xfit, yfit):
    data_minus_model = data_cutout - model_cutout
    rr, dm, npix = radial_profile_median(data_minus_model, xfit, yfit, max_radius=10.0, dr=0.5)
    core = (rr <= 1.5) & (npix > 0)
    wing = (rr >= 3.0) & (rr <= 6.0) & (npix > 0)

    core_delta = float(np.nanmedian(dm[core])) if np.any(core) else np.nan
    wing_delta = float(np.nanmedian(dm[wing])) if np.any(wing) else np.nan
    return {
        'core_data_minus_model': core_delta,
        'wing_data_minus_model': wing_delta,
    }


def build_configurations(fwhm_pix):
    return [
        {
            'name': 'basic_local2_5_fit5',
            'mode': 'basic',
            'localbkg': (2, 5),
            'fit_shape': (5, 5),
            'sub_shape': (15, 15),
            'finder_sigma': 4.0,
            'maxiters': 3,
        },
        {
            'name': 'basic_local6_10_fit7',
            'mode': 'basic',
            'localbkg': (6, 10),
            'fit_shape': (7, 7),
            'sub_shape': (17, 17),
            'finder_sigma': 4.0,
            'maxiters': 3,
        },
        {
            'name': 'basic_nolocal_fit7',
            'mode': 'basic',
            'localbkg': None,
            'fit_shape': (7, 7),
            'sub_shape': (17, 17),
            'finder_sigma': 4.0,
            'maxiters': 3,
        },
        {
            'name': 'iter_local2_5_fit5_sub15',
            'mode': 'iterative',
            'localbkg': (2, 5),
            'fit_shape': (5, 5),
            'sub_shape': (15, 15),
            'finder_sigma': 4.0,
            'maxiters': 3,
        },
        {
            'name': 'iter_local6_10_fit7_sub21',
            'mode': 'iterative',
            'localbkg': (6, 10),
            'fit_shape': (7, 7),
            'sub_shape': (21, 21),
            'finder_sigma': 4.0,
            'maxiters': 5,
        },
        {
            'name': 'iter_nolocal_fit7_sub21',
            'mode': 'iterative',
            'localbkg': None,
            'fit_shape': (7, 7),
            'sub_shape': (21, 21),
            'finder_sigma': 4.0,
            'maxiters': 5,
        },
    ]


def replace_nan_pixels_for_fitting(data, fwhm_pix):
    processed = np.asarray(data, dtype=float, copy=True)
    if np.any(np.isnan(processed)):
        kernel = Gaussian2DKernel(x_stddev=float(fwhm_pix) / 2.355)
        processed = interpolate_replace_nans(processed, kernel, convolve=convolve_fft)
    return processed


def load_stpsf_psf_model(psf_grid_file: Path):
    if not psf_grid_file.exists():
        raise FileNotFoundError(f'STPSF grid file not found: {psf_grid_file}')
    return to_griddedpsfmodel(str(psf_grid_file))


def make_photometry(config, psf_model, fwhm_pix, local_noise):
    localbkg = None
    if config['localbkg'] is not None:
        localbkg = LocalBackground(config['localbkg'][0], config['localbkg'][1])

    if config['mode'] == 'basic':
        return PSFPhotometry(
            finder=None,
            localbkg_estimator=localbkg,
            psf_model=psf_model,
            fitter=LevMarLSQFitter(),
            fit_shape=config['fit_shape'],
            aperture_radius=2.0 * fwhm_pix,
            progress_bar=False,
        )

    finder = DAOStarFinder(threshold=config['finder_sigma'] * local_noise, fwhm=fwhm_pix)
    return IterativePSFPhotometry(
        finder=finder,
        localbkg_estimator=localbkg,
        psf_model=psf_model,
        fitter=LevMarLSQFitter(),
        maxiters=config['maxiters'],
        fit_shape=config['fit_shape'],
        sub_shape=config['sub_shape'],
        aperture_radius=2.0 * fwhm_pix,
        progress_bar=False,
    )


def run_cutout_sweep(
    science_data,
    science_error,
    residual_data,
    residual_wcs,
    det_tbl,
    match_tbl,
    nstars,
    fwhm_pix,
    cutout_halfsize,
    outdir,
    stpsf_psf_model,
    stpsf_label,
    det_coverage_tbl,
    crowdsource_weight_map,
):
    matched_rows = match_tbl[match_tbl['matched']]
    if len(matched_rows) == 0:
        return Table(rows=[]), Table(rows=[]), Table(rows=[])

    det_indices = np.unique(np.asarray(matched_rows['matched_detection_index'], dtype=int))

    if len(det_coverage_tbl) > 0:
        coverage_index = np.asarray(det_coverage_tbl['detection_index'], dtype=int)
        good = np.asarray(det_coverage_tbl['covered_all8_in_best_vgroup'], dtype=bool)
        coverage_lookup = {int(idx): bool(ok) for idx, ok in zip(coverage_index, good)}
        det_indices = np.array([idx for idx in det_indices if coverage_lookup.get(int(idx), False)], dtype=int)

    if len(det_indices) == 0:
        raise ValueError('No matched detections have full 8/8 coverage in any vgroup.')

    selected_det = det_tbl[det_indices]

    det_x = np.asarray(selected_det['xcentroid'], dtype=float)
    det_y = np.asarray(selected_det['ycentroid'], dtype=float)
    residual_vals = residual_data[np.rint(det_y).astype(int), np.rint(det_x).astype(int)]
    order = np.argsort(residual_vals)
    selected_indices = det_indices[order][: min(nstars, len(selected_det))]
    selected_det = selected_det[order][: min(nstars, len(selected_det))]

    psf_model = stpsf_psf_model

    configs = build_configurations(fwhm_pix)
    rows = []
    stars_rows = []
    profile_rows = []

    for i, detrow in enumerate(selected_det):
        xc = float(detrow['xcentroid'])
        yc = float(detrow['ycentroid'])

        ysl, xsl = cutout_slices(xc, yc, cutout_halfsize, science_data.shape)
        sci_cut = np.asarray(science_data[ysl, xsl], dtype=float)
        res_cut = np.asarray(residual_data[ysl, xsl], dtype=float)
        sci_fit_cut = replace_nan_pixels_for_fitting(sci_cut, fwhm_pix=fwhm_pix)
        sci_err_cut = np.asarray(science_error[ysl, xsl], dtype=float)
        weight_cut = np.asarray(crowdsource_weight_map[ysl, xsl], dtype=float)

        x0 = xc - xsl.start
        y0 = yc - ysl.start

        local_noise = mad_std(sci_fit_cut[np.isfinite(sci_fit_cut)], ignore_nan=True)
        if not np.isfinite(local_noise) or local_noise <= 0:
            local_noise = np.nanstd(sci_fit_cut[np.isfinite(sci_fit_cut)])
        if not np.isfinite(local_noise) or local_noise <= 0:
            local_noise = 1.0

        flux0 = estimate_flux_init(sci_fit_cut, x0, y0, radius_pix=max(1.5, fwhm_pix / 2.0))

        init_tbl = Table()
        init_tbl['x_0'] = [x0]
        init_tbl['y_0'] = [y0]
        init_tbl['flux_0'] = [flux0]

        sky = residual_wcs.pixel_to_world(xc, yc)
        stars_rows.append(
            {
                'star_id': i,
                'xpix': xc,
                'ypix': yc,
                'ra_deg': float(sky.ra.to_value(u.deg)),
                'dec_deg': float(sky.dec.to_value(u.deg)),
                'residual_value': float(residual_data[int(np.rint(yc)), int(np.rint(xc))]),
            }
        )

        if len(det_coverage_tbl) > 0:
            det_match = det_coverage_tbl[det_coverage_tbl['detection_index'] == int(selected_indices[i])]
            if len(det_match) == 1:
                stars_rows[-1]['best_visit'] = str(det_match['best_visit'][0])
                stars_rows[-1]['best_vgroup'] = str(det_match['best_vgroup'][0])
                stars_rows[-1]['best_vgroup_total_exposures'] = int(det_match['best_vgroup_total_exposures'][0])
                stars_rows[-1]['best_vgroup_covered_exposures'] = int(det_match['best_vgroup_covered_exposures'][0])

        for config in configs:
            phot = make_photometry(config, psf_model, fwhm_pix, local_noise)
            result = phot(sci_fit_cut, init_params=init_tbl, error=np.where(np.isfinite(sci_err_cut), sci_err_cut, 1e11))

            if len(result) == 0:
                continue

            xfit = float(result['x_fit'][0]) if 'x_fit' in result.colnames else float(result['x_0'][0])
            yfit = float(result['y_fit'][0]) if 'y_fit' in result.colnames else float(result['y_0'][0])
            flux_fit = float(result['flux_fit'][0]) if 'flux_fit' in result.colnames else np.nan

            model = phot.make_model_image(sci_fit_cut.shape, psf_shape=(21, 21), include_localbkg=False)
            resid = sci_fit_cut - model
            metrics = core_metrics(resid, xfit, yfit)
            shape_metrics = radial_shape_metrics(sci_fit_cut, model, xfit, yfit)

            r_prof, data_prof, npix_prof = radial_profile_median(sci_fit_cut, xfit, yfit, max_radius=12.0, dr=0.5)
            _, model_prof, _ = radial_profile_median(model, xfit, yfit, max_radius=12.0, dr=0.5)
            _, resid_prof, _ = radial_profile_median(resid, xfit, yfit, max_radius=12.0, dr=0.5)
            fit_weights = weight_cut
            _, w_prof, _ = radial_profile_median(fit_weights, xfit, yfit, max_radius=12.0, dr=0.5)

            for rbin, dval, mval, rval, wval, nbin in zip(r_prof, data_prof, model_prof, resid_prof, w_prof, npix_prof):
                profile_rows.append(
                    {
                        'star_id': i,
                        'config_name': config['name'],
                        'r_pix': float(rbin),
                        'npix': int(nbin),
                        'data_median': float(dval) if np.isfinite(dval) else np.nan,
                        'model_median': float(mval) if np.isfinite(mval) else np.nan,
                        'residual_median': float(rval) if np.isfinite(rval) else np.nan,
                        'weight_median': float(wval) if np.isfinite(wval) else np.nan,
                    }
                )

            rows.append(
                {
                    'star_id': i,
                    'config_name': config['name'],
                    'mode': config['mode'],
                    'localbkg_inner': -1 if config['localbkg'] is None else config['localbkg'][0],
                    'localbkg_outer': -1 if config['localbkg'] is None else config['localbkg'][1],
                    'fit_shape_y': config['fit_shape'][0],
                    'fit_shape_x': config['fit_shape'][1],
                    'sub_shape_y': config['sub_shape'][0],
                    'sub_shape_x': config['sub_shape'][1],
                    'xfit_cutout': xfit,
                    'yfit_cutout': yfit,
                    'xfit_full': xfit + xsl.start,
                    'yfit_full': yfit + ysl.start,
                    'flux_fit': flux_fit,
                    'residual_value_at_detection': float(residual_data[int(np.rint(yc)), int(np.rint(xc))]),
                    'core_median_resid': metrics['core_median_resid'],
                    'core_min_resid': metrics['core_min_resid'],
                    'ring_median_resid': metrics['ring_median_resid'],
                    'center_resid': metrics['center_resid'],
                    'core_data_minus_model': shape_metrics['core_data_minus_model'],
                    'wing_data_minus_model': shape_metrics['wing_data_minus_model'],
                }
            )

            fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.4), constrained_layout=True)
            sci_norm = simple_norm(sci_cut, stretch='log', percent=99.5)
            im0 = axes[0, 0].imshow(sci_cut, origin='lower', cmap='gray', norm=sci_norm)
            axes[0, 0].set_title('science')
            model_norm = simple_norm(model, stretch='log', percent=99.5)
            im1 = axes[0, 1].imshow(model, origin='lower', cmap='gray', norm=model_norm)
            axes[0, 1].set_title('model')
            rabs = np.nanpercentile(np.abs(resid[np.isfinite(resid)]), 99.0)
            if not np.isfinite(rabs) or rabs <= 0:
                rabs = 1.0
            im2 = axes[0, 2].imshow(resid, origin='lower', cmap='coolwarm', vmin=-rabs, vmax=rabs)
            axes[0, 2].set_title('residual')

            wmax = np.nanmax(fit_weights[np.isfinite(fit_weights)]) if np.any(np.isfinite(fit_weights)) else 1.0
            if not np.isfinite(wmax) or wmax <= 0:
                wmax = 1.0
            imw = axes[1, 0].imshow(fit_weights, origin='lower', cmap='viridis', vmin=0.0, vmax=wmax)
            axes[1, 0].set_title('fit weights')

            for ax in (axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0]):
                ax.plot([xfit], [yfit], marker='+', color='yellow', ms=9, mew=1.5)

            plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
            plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
            plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
            plt.colorbar(imw, ax=axes[1, 0], fraction=0.046, pad=0.04)

            axp = axes[1, 1]
            axp.plot(r_prof, data_prof, color='black', lw=1.8, label='data')
            axp.plot(r_prof, model_prof, color='tab:blue', lw=1.6, label='model')
            axp.plot(r_prof, resid_prof, color='tab:red', lw=1.4, label='residual (data-model)')
            axp.axhline(0.0, color='0.3', ls='--', lw=0.8)
            axp.set_xlabel('radius [pix]')
            axp.set_ylabel('median value')
            axp2 = axp.twinx()
            axp2.plot(r_prof, w_prof, color='tab:purple', lw=1.4, ls='--', label='fit weight')
            axp2.set_ylabel('median fit weight')
            axp.set_title('radial profile')
            h1, l1 = axp.get_legend_handles_labels()
            h2, l2 = axp2.get_legend_handles_labels()
            axp.legend(h1 + h2, l1 + l2, loc='best', fontsize=8)

            axes[1, 2].axis('off')

            fig.suptitle(f"star {i} {config['name']} ({stpsf_label})")
            fig.savefig(outdir / f'star{i:02d}_{config["name"]}.png', dpi=130)
            plt.close(fig)

    result_tbl = Table(rows=rows)
    stars_tbl = Table(rows=stars_rows)
    profile_tbl = Table(rows=profile_rows)
    return result_tbl, stars_tbl, profile_tbl


def summarize_configs(result_tbl: Table):
    if len(result_tbl) == 0:
        return Table(rows=[])

    summary_rows = []
    for cfg in np.unique(np.asarray(result_tbl['config_name'])):
        sub = result_tbl[result_tbl['config_name'] == cfg]
        summary_rows.append(
            {
                'config_name': cfg,
                'nfits': len(sub),
                'median_core_median_resid': float(np.nanmedian(sub['core_median_resid'])),
                'median_core_min_resid': float(np.nanmedian(sub['core_min_resid'])),
                'median_center_resid': float(np.nanmedian(sub['center_resid'])),
                'median_ring_median_resid': float(np.nanmedian(sub['ring_median_resid'])),
                'median_core_data_minus_model': float(np.nanmedian(sub['core_data_minus_model'])),
                'median_wing_data_minus_model': float(np.nanmedian(sub['wing_data_minus_model'])),
                'median_flux_fit': float(np.nanmedian(sub['flux_fit'])),
            }
        )

    return Table(rows=summary_rows)


def _parse_crf_key(filename: str):
    name = Path(filename).name
    match = re.search(r'jw\d+_(\w+?)_(\d{5})_.*_destreak_.*_crf\.fits$', name)
    if match is None:
        return None
    visit_match = re.search(r'jw\d{8}(\d{3})_', name)
    visit = visit_match.group(1) if visit_match is not None else '001'
    vgroup = match.group(1)
    exp = match.group(2)
    return visit, vgroup, exp


def _parse_visit_vgroup_exp(filename: str):
    name = Path(filename).name
    match = re.search(r'visit(\d+)_vgroup([0-9a-zA-Z]+)_exp(\d+)', name)
    if match is None:
        return None
    return match.group(1), match.group(2), match.group(3)


def _build_exposure_bundle_maps(science_image: Path):
    pipeline_dir = science_image.parent
    f480m_root = pipeline_dir.parent

    crf_files = sorted(glob.glob(str(pipeline_dir / '*destreak_o007_crf.fits')))
    residual_files = sorted(glob.glob(str(pipeline_dir / '*iter2_daophot_basic_residual.fits')))
    catalog_files = sorted(glob.glob(str(f480m_root / '*iter2_daophot_basic.fits')))

    sci_map = {k: Path(fn) for fn in crf_files if (k := _parse_crf_key(fn)) is not None}
    res_map = {k: Path(fn) for fn in residual_files if (k := _parse_visit_vgroup_exp(fn)) is not None}
    cat_map = {k: Path(fn) for fn in catalog_files if (k := _parse_visit_vgroup_exp(fn)) is not None}

    keys = sorted(set(sci_map).intersection(res_map))
    return keys, sci_map, res_map, cat_map


def _compute_detection_vgroup_coverage(det_tbl: Table, science_image: Path):
    if len(det_tbl) == 0:
        return Table(rows=[])

    keys, sci_map, _, _ = _build_exposure_bundle_maps(science_image)
    if len(keys) == 0:
        return Table(rows=[])

    grouped_keys = {}
    for key in keys:
        grouped_keys.setdefault((key[0], key[1]), []).append(key)

    # Load each SCI WCS once; reuse for all detections.
    wcs_shape_map = {}
    for key in keys:
        data, wcs = load_fits_data_and_wcs(sci_map[key])
        wcs_shape_map[key] = (wcs, data.shape)

    det_coords = SkyCoord(ra=np.asarray(det_tbl['ra_deg'], dtype=float) * u.deg,
                          dec=np.asarray(det_tbl['dec_deg'], dtype=float) * u.deg)

    rows = []
    for didx, sc in enumerate(det_coords):
        best_group = None
        best_total = -1
        best_covered = -1

        for group_key, group_exposures in grouped_keys.items():
            covered = 0
            for exp_key in group_exposures:
                wcs, shape = wcs_shape_map[exp_key]
                xpix, ypix = wcs.world_to_pixel(sc)
                in_bounds = (0 <= xpix < shape[1]) and (0 <= ypix < shape[0])
                if in_bounds:
                    covered += 1

            total = len(group_exposures)
            if (covered > best_covered) or (covered == best_covered and total > best_total):
                best_group = group_key
                best_total = total
                best_covered = covered

        full_in_best = bool(best_total > 0 and best_covered == best_total)
        full8_in_best = bool(full_in_best and best_total == 8)

        rows.append(
            {
                'detection_index': int(didx),
                'best_visit': str(best_group[0]) if best_group is not None else '',
                'best_vgroup': str(best_group[1]) if best_group is not None else '',
                'best_vgroup_total_exposures': int(best_total if best_total >= 0 else 0),
                'best_vgroup_covered_exposures': int(best_covered if best_covered >= 0 else 0),
                'covered_all_in_best_vgroup': full_in_best,
                'covered_all8_in_best_vgroup': full8_in_best,
            }
        )

    return Table(rows=rows)


def _centered_cutout(data, xpix, ypix, halfsize):
    ysl, xsl = cutout_slices(xpix, ypix, halfsize, data.shape)
    cut = np.asarray(data[ysl, xsl], dtype=float)
    x0 = float(xpix - xsl.start)
    y0 = float(ypix - ysl.start)
    return cut, x0, y0


def _plot_gallery_page(sci_cut, model_cut, res_cut, weight_cut, x0, y0, title):
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.4), constrained_layout=True)

    sci_norm = simple_norm(sci_cut, stretch='asinh', percent=99.5)
    model_norm = simple_norm(model_cut, stretch='asinh', percent=99.5)
    rabs = np.nanpercentile(np.abs(res_cut[np.isfinite(res_cut)]), 99.0)
    if not np.isfinite(rabs) or rabs <= 0:
        rabs = 1.0

    im0 = axes[0, 0].imshow(sci_cut, origin='lower', cmap='gray', norm=sci_norm)
    axes[0, 0].set_title('science (asinh)')
    im1 = axes[0, 1].imshow(model_cut, origin='lower', cmap='gray', norm=model_norm)
    axes[0, 1].set_title('model (asinh)')
    im2 = axes[0, 2].imshow(res_cut, origin='lower', cmap='coolwarm', vmin=-rabs, vmax=rabs)
    axes[0, 2].set_title('residual')

    fit_weights = np.asarray(weight_cut, dtype=float)
    wmax = np.nanmax(fit_weights[np.isfinite(fit_weights)]) if np.any(np.isfinite(fit_weights)) else 1.0
    if not np.isfinite(wmax) or wmax <= 0:
        wmax = 1.0
    imw = axes[1, 0].imshow(fit_weights, origin='lower', cmap='viridis', vmin=0.0, vmax=wmax)
    axes[1, 0].set_title('fit weights')

    for ax in (axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0]):
        ax.plot([x0], [y0], marker='+', color='yellow', ms=9, mew=1.5)

    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
    plt.colorbar(imw, ax=axes[1, 0], fraction=0.046, pad=0.04)

    rr, dprof, _ = radial_profile_median(sci_cut, x0, y0, max_radius=12.0, dr=0.5)
    _, mprof, _ = radial_profile_median(model_cut, x0, y0, max_radius=12.0, dr=0.5)
    _, rprof, _ = radial_profile_median(res_cut, x0, y0, max_radius=12.0, dr=0.5)
    _, wprof, _ = radial_profile_median(fit_weights, x0, y0, max_radius=12.0, dr=0.5)
    axp = axes[1, 1]
    axp.plot(rr, dprof, color='black', lw=1.8, label='science')
    axp.plot(rr, mprof, color='tab:blue', lw=1.6, label='model')
    axp.plot(rr, rprof, color='tab:red', lw=1.4, label='residual')
    axp.axhline(0.0, color='0.3', ls='--', lw=0.8)
    axp.set_xlabel('radius [pix]')
    axp.set_ylabel('median value')
    axp.set_title('radial profile')
    axp2 = axp.twinx()
    axp2.plot(rr, wprof, color='tab:purple', lw=1.4, ls='--', label='fit weight')
    axp2.set_ylabel('median fit weight')
    h1, l1 = axp.get_legend_handles_labels()
    h2, l2 = axp2.get_legend_handles_labels()
    axp.legend(h1 + h2, l1 + l2, loc='best', fontsize=8)

    axes[1, 2].axis('off')

    fig.suptitle(title)
    return fig


def _build_catalog_measurements(stars_tbl, keys, sci_map, cat_map, match_radius_arcsec=0.2):
    if len(stars_tbl) == 0:
        return Table(rows=[])

    star_coords = SkyCoord(ra=np.asarray(stars_tbl['ra_deg'], dtype=float) * u.deg,
                           dec=np.asarray(stars_tbl['dec_deg'], dtype=float) * u.deg)
    rows = []

    for key in keys:
        if key not in cat_map:
            continue
        sci_data, sci_wcs = load_fits_data_and_wcs(sci_map[key])
        _ = sci_data
        cat = Table.read(cat_map[key])
        if 'x_fit' not in cat.colnames or 'y_fit' not in cat.colnames:
            continue
        csky = sci_wcs.pixel_to_world(np.asarray(cat['x_fit'], dtype=float),
                                      np.asarray(cat['y_fit'], dtype=float))
        idx, sep2d, _ = star_coords.match_to_catalog_sky(csky)
        for sid in range(len(stars_tbl)):
            sep_arcsec = float(sep2d[sid].to_value(u.arcsec))
            if sep_arcsec > match_radius_arcsec:
                continue
            crow = cat[int(idx[sid])]
            rows.append({
                'star_id': int(stars_tbl['star_id'][sid]),
                'visit': key[0],
                'vgroup': key[1],
                'exp': key[2],
                'ra_fit_deg': float(csky[idx[sid]].ra.to_value(u.deg)),
                'dec_fit_deg': float(csky[idx[sid]].dec.to_value(u.deg)),
                'sep_arcsec': sep_arcsec,
                'flux_fit': float(crow['flux_fit']) if 'flux_fit' in cat.colnames else np.nan,
                'qfit': float(crow['qfit']) if 'qfit' in cat.colnames else np.nan,
                'cfit': float(crow['cfit']) if 'cfit' in cat.colnames else np.nan,
            })

    return Table(rows=rows)


def _make_summary_stats_figure(meas_tbl, out_png):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.8), constrained_layout=True)
    if len(meas_tbl) == 0:
        for ax in axes.ravel():
            ax.text(0.5, 0.5, 'No matched per-exposure catalog measurements', ha='center', va='center')
            ax.set_axis_off()
        fig.savefig(out_png, dpi=150)
        return fig

    ra = np.asarray(meas_tbl['ra_fit_deg'], dtype=float)
    dec = np.asarray(meas_tbl['dec_fit_deg'], dtype=float)
    sid = np.asarray(meas_tbl['star_id'], dtype=int)
    flux = np.asarray(meas_tbl['flux_fit'], dtype=float)
    qfit = np.asarray(meas_tbl['qfit'], dtype=float)
    cfit = np.asarray(meas_tbl['cfit'], dtype=float)

    dra_mas = np.full(len(meas_tbl), np.nan)
    ddec_mas = np.full(len(meas_tbl), np.nan)
    for star in np.unique(sid):
        ss = sid == star
        mean_ra = np.nanmean(ra[ss])
        mean_dec = np.nanmean(dec[ss])
        dra_mas[ss] = (ra[ss] - mean_ra) * np.cos(np.deg2rad(mean_dec)) * 3.6e6
        ddec_mas[ss] = (dec[ss] - mean_dec) * 3.6e6

    ax = axes[0, 0]
    for star in np.unique(sid):
        ss = sid == star
        ax.scatter(dra_mas[ss], ddec_mas[ss], s=16, alpha=0.8, label=f'star {star}')
    ax.axhline(0, color='0.4', ls='--', lw=0.8)
    ax.axvline(0, color='0.4', ls='--', lw=0.8)
    ax.set_xlabel('dRA cos(dec) [mas]')
    ax.set_ylabel('dDec [mas]')
    ax.set_title('Per-star fitted coordinate offsets from mean')
    ax.legend(fontsize=7, ncol=2)

    ax = axes[0, 1]
    ax.hist(flux[np.isfinite(flux)], bins=30, color='tab:blue', alpha=0.85)
    ax.set_title('Histogram of flux_fit')
    ax.set_xlabel('flux_fit')
    ax.set_ylabel('N')

    ax = axes[1, 0]
    ax.hist(cfit[np.isfinite(cfit)], bins=30, color='tab:orange', alpha=0.85)
    ax.set_title('Histogram of cfit')
    ax.set_xlabel('cfit')
    ax.set_ylabel('N')

    ax = axes[1, 1]
    ax.hist(qfit[np.isfinite(qfit)], bins=30, color='tab:green', alpha=0.85)
    ax.set_title('Histogram of qfit')
    ax.set_xlabel('qfit')
    ax.set_ylabel('N')

    fig.savefig(out_png, dpi=150)
    return fig


def generate_exposure_gallery_and_summary(stars_tbl, science_image: Path, outdir: Path, cutout_halfsize=18):
    keys, sci_map, res_map, cat_map = _build_exposure_bundle_maps(science_image)
    if len(keys) == 0 or len(stars_tbl) == 0:
        return Table(rows=[]), 0

    gallery_pdf = outdir / 'selected_stars_exposure_gallery.pdf'
    pdf = PdfPages(gallery_pdf)
    coverage_rows = []

    for row in stars_tbl:
        sid = int(row['star_id'])
        star_dir = outdir / f'star{sid:02d}_gallery'
        star_dir.mkdir(parents=True, exist_ok=True)
        sc = SkyCoord(ra=float(row['ra_deg']) * u.deg, dec=float(row['dec_deg']) * u.deg)
        star_page_count = 0

        for key in keys:
            sci_data, sci_wcs, sci_err, sci_dq, sci_wht = load_fits_bundle(sci_map[key])
            res_data, res_wcs = load_fits_data_and_wcs(res_map[key])
            crowd_wht_map = compute_crowdsource_weight_map(sci_data, sci_err, dq=sci_dq, wht=sci_wht)

            x_sci, y_sci = sci_wcs.world_to_pixel(sc)
            # Residual images are generated from the same exposure as science; when
            # dimensions match, use identical pixel centers to keep cutout shapes aligned.
            if res_data.shape == sci_data.shape:
                x_res, y_res = x_sci, y_sci
            else:
                x_res, y_res = res_wcs.world_to_pixel(sc)
            in_sci = (0 <= x_sci < sci_data.shape[1]) and (0 <= y_sci < sci_data.shape[0])
            in_res = (0 <= x_res < res_data.shape[1]) and (0 <= y_res < res_data.shape[0])
            if not (in_sci and in_res):
                continue

            sci_cut, x0, y0 = _centered_cutout(sci_data, x_sci, y_sci, cutout_halfsize)
            res_cut, _, _ = _centered_cutout(res_data, x_res, y_res, cutout_halfsize)
            wht_cut, _, _ = _centered_cutout(crowd_wht_map, x_sci, y_sci, cutout_halfsize)
            model_cut = sci_cut - res_cut

            title = (
                f'star {sid} visit{key[0]} vgroup{key[1]} exp{key[2]}\n'
                f'RA={float(row["ra_deg"]):.8f} deg  Dec={float(row["dec_deg"]):.8f} deg\n'
                f'{Path(sci_map[key]).name}'
            )
            fig = _plot_gallery_page(sci_cut, model_cut, res_cut, wht_cut, x0, y0, title)
            png_name = star_dir / f'star{sid:02d}_visit{key[0]}_vgroup{key[1]}_exp{key[2]}.png'
            fig.savefig(png_name, dpi=140)
            plt.close(fig)

            # Use a PNG->PDF embedding step for robustness across matplotlib/PDF backends.
            page_fig = plt.figure(figsize=(11.0, 8.4), constrained_layout=True)
            ax = page_fig.add_subplot(111)
            ax.imshow(plt.imread(png_name))
            ax.set_axis_off()
            pdf.savefig(page_fig)
            plt.close(page_fig)
            star_page_count += 1

        coverage_rows.append({'star_id': sid, 'gallery_page_count': star_page_count})

    coverage_tbl = Table(rows=coverage_rows)
    coverage_tbl.write(outdir / 'selected_star_exposure_coverage.ecsv', overwrite=True)

    meas_tbl = _build_catalog_measurements(stars_tbl, keys, sci_map, cat_map, match_radius_arcsec=0.2)
    meas_tbl.write(outdir / 'selected_star_measurements_all_exposures.ecsv', overwrite=True)
    summary_png = outdir / 'selected_stars_summary_statistics.png'
    fig = _make_summary_stats_figure(meas_tbl, summary_png)
    summary_page = plt.figure(figsize=(11.0, 8.4), constrained_layout=True)
    sax = summary_page.add_subplot(111)
    sax.imshow(plt.imread(summary_png))
    sax.set_axis_off()
    pdf.savefig(summary_page)
    plt.close(summary_page)
    plt.close(fig)
    pdf.close()

    return meas_tbl, len(keys)


def _resolve_deep_dive_target(star_id, ra_deg, dec_deg, stars_table_path: Path):
    if ra_deg is not None and dec_deg is not None:
        return SkyCoord(float(ra_deg) * u.deg, float(dec_deg) * u.deg), {
            'target_star_id': -1,
            'target_ra_deg': float(ra_deg),
            'target_dec_deg': float(dec_deg),
            'source': 'cli-ra-dec',
        }

    if star_id is None:
        raise ValueError('Deep-dive mode requires either --deep-dive-star-id or both --deep-dive-ra/--deep-dive-dec.')

    if not stars_table_path.exists():
        raise FileNotFoundError(f'Deep-dive stars table not found: {stars_table_path}')

    stars_tbl = Table.read(stars_table_path)
    sel = stars_tbl[stars_tbl['star_id'] == int(star_id)]
    if len(sel) != 1:
        raise ValueError(f'Star ID {star_id} not found exactly once in {stars_table_path}')

    row = sel[0]
    return SkyCoord(float(row['ra_deg']) * u.deg, float(row['dec_deg']) * u.deg), {
        'target_star_id': int(star_id),
        'target_ra_deg': float(row['ra_deg']),
        'target_dec_deg': float(row['dec_deg']),
        'source': str(stars_table_path),
    }


def _deep_dive_fitinfo_dict(phot):
    out = {
        'fit_ierr': np.nan,
        'fit_nfev': np.nan,
        'fit_message': '',
    }
    if hasattr(phot, 'fitter') and hasattr(phot.fitter, 'fit_info') and isinstance(phot.fitter.fit_info, dict):
        info = phot.fitter.fit_info
        if 'ierr' in info and info['ierr'] is not None:
            out['fit_ierr'] = float(info['ierr'])
        if 'nfev' in info and info['nfev'] is not None:
            out['fit_nfev'] = float(info['nfev'])
        if 'message' in info and info['message'] is not None:
            out['fit_message'] = str(info['message'])
    return out


def run_deep_dive_single_star(
    outdir: Path,
    science_image: Path,
    residual_image: Path,
    stpsf_psf_model,
    stpsf_label,
    fwhm_pix,
    cutout_halfsize,
    deep_dive_star_id,
    deep_dive_ra,
    deep_dive_dec,
    deep_dive_stars_table: Path,
    deep_dive_prefix,
):
    deep_dir = outdir / f'{deep_dive_prefix}_outputs'
    deep_dir.mkdir(parents=True, exist_ok=True)

    sc, target_meta = _resolve_deep_dive_target(
        star_id=deep_dive_star_id,
        ra_deg=deep_dive_ra,
        dec_deg=deep_dive_dec,
        stars_table_path=deep_dive_stars_table,
    )

    sci_data, sci_wcs, sci_err, sci_dq, sci_wht = load_fits_bundle(science_image)
    res_data, res_wcs = load_fits_data_and_wcs(residual_image)
    crowd_wht_map = compute_crowdsource_weight_map(sci_data, sci_err, dq=sci_dq, wht=sci_wht)

    x_sci, y_sci = sci_wcs.world_to_pixel(sc)
    if res_data.shape == sci_data.shape:
        x_res, y_res = x_sci, y_sci
    else:
        x_res, y_res = res_wcs.world_to_pixel(sc)

    if not ((0 <= x_sci < sci_data.shape[1]) and (0 <= y_sci < sci_data.shape[0])):
        raise ValueError('Deep-dive target is outside science image bounds.')
    if not ((0 <= x_res < res_data.shape[1]) and (0 <= y_res < res_data.shape[0])):
        raise ValueError('Deep-dive target is outside residual image bounds.')

    sci_cut, x0, y0 = _centered_cutout(sci_data, x_sci, y_sci, cutout_halfsize)
    res_cut, _, _ = _centered_cutout(res_data, x_res, y_res, cutout_halfsize)
    wht_cut, _, _ = _centered_cutout(crowd_wht_map, x_sci, y_sci, cutout_halfsize)
    sci_fit_cut = replace_nan_pixels_for_fitting(sci_cut, fwhm_pix=fwhm_pix)
    sci_err_cut, _, _ = _centered_cutout(sci_err, x_sci, y_sci, cutout_halfsize)

    local_noise = mad_std(sci_fit_cut[np.isfinite(sci_fit_cut)], ignore_nan=True)
    if not np.isfinite(local_noise) or local_noise <= 0:
        local_noise = np.nanstd(sci_fit_cut[np.isfinite(sci_fit_cut)])
    if not np.isfinite(local_noise) or local_noise <= 0:
        local_noise = 1.0

    flux0 = estimate_flux_init(sci_fit_cut, x0, y0, radius_pix=max(1.5, fwhm_pix / 2.0))
    init_tbl = Table()
    init_tbl['x_0'] = [x0]
    init_tbl['y_0'] = [y0]
    init_tbl['flux_0'] = [flux0]

    configs = build_configurations(fwhm_pix)
    rows = []

    for config in configs:
        phot = make_photometry(config, stpsf_psf_model, fwhm_pix, local_noise)
        result = phot(sci_fit_cut, init_params=init_tbl, error=np.where(np.isfinite(sci_err_cut), sci_err_cut, 1e11))
        if len(result) == 0:
            continue

        xfit = float(result['x_fit'][0]) if 'x_fit' in result.colnames else float(result['x_0'][0])
        yfit = float(result['y_fit'][0]) if 'y_fit' in result.colnames else float(result['y_0'][0])
        flux_fit = float(result['flux_fit'][0]) if 'flux_fit' in result.colnames else np.nan

        model = phot.make_model_image(sci_fit_cut.shape, psf_shape=(21, 21), include_localbkg=False)
        resid = sci_fit_cut - model
        resid2 = phot.make_residual_image(sci_fit_cut, psf_shape=(21, 21), include_localbkg=False)
        print(f'diff between resid and resid2: {np.nanmax(np.abs(resid - resid2)):.3e}')
        fit_weights = np.asarray(wht_cut, dtype=float)

        weighted_sse_fit = float(np.nansum(fit_weights * resid**2))
        weighted_sse_null = float(np.nansum(fit_weights * sci_fit_cut**2))
        weighted_delta = float(weighted_sse_fit - weighted_sse_null)

        rr, dprof, _ = radial_profile_median(sci_fit_cut, xfit, yfit, max_radius=12.0, dr=0.5)
        _, mprof, _ = radial_profile_median(model, xfit, yfit, max_radius=12.0, dr=0.5)
        _, rprof, _ = radial_profile_median(resid, xfit, yfit, max_radius=12.0, dr=0.5)
        _, wprof, _ = radial_profile_median(fit_weights, xfit, yfit, max_radius=12.0, dr=0.5)

        info = _deep_dive_fitinfo_dict(phot)
        err_cols = [cn for cn in result.colnames if cn.endswith('_err') or 'flag' in cn.lower()]
        err_report = '; '.join([f"{cn}={result[cn][0]}" for cn in err_cols])

        fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.4), constrained_layout=True)
        sci_norm = simple_norm(sci_fit_cut, stretch='asinh', percent=99.5)
        model_norm = simple_norm(model, stretch='asinh', percent=99.5)
        rabs = np.nanpercentile(np.abs(resid[np.isfinite(resid)]), 99.0)
        if not np.isfinite(rabs) or rabs <= 0:
            rabs = 1.0
        wmax = np.nanmax(fit_weights[np.isfinite(fit_weights)]) if np.any(np.isfinite(fit_weights)) else 1.0
        if not np.isfinite(wmax) or wmax <= 0:
            wmax = 1.0

        im0 = axes[0, 0].imshow(sci_fit_cut, origin='lower', cmap='gray', norm=sci_norm)
        axes[0, 0].set_title('science (fit input)')
        im1 = axes[0, 1].imshow(model, origin='lower', cmap='gray', norm=model_norm)
        axes[0, 1].set_title('model')
        im2 = axes[0, 2].imshow(resid, origin='lower', cmap='coolwarm', vmin=-rabs, vmax=rabs)
        axes[0, 2].set_title('residual = data-model')
        imw = axes[1, 0].imshow(fit_weights, origin='lower', cmap='viridis', vmin=0.0, vmax=wmax)
        axes[1, 0].set_title('fit weights')

        for ax in (axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0]):
            ax.plot([xfit], [yfit], marker='+', color='yellow', ms=9, mew=1.5)

        plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
        plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
        plt.colorbar(imw, ax=axes[1, 0], fraction=0.046, pad=0.04)

        axp = axes[1, 1]
        axp.plot(rr, dprof, color='black', lw=1.8, label='data')
        axp.plot(rr, mprof, color='tab:blue', lw=1.6, label='model')
        axp.plot(rr, rprof, color='tab:red', lw=1.4, label='residual')
        axp.axhline(0.0, color='0.3', ls='--', lw=0.8)
        axp.set_xlabel('radius [pix]')
        axp.set_ylabel('median value')
        axp.set_title('radial profile')
        axp2 = axp.twinx()
        axp2.plot(rr, wprof, color='tab:purple', lw=1.4, ls='--', label='fit weight')
        axp2.set_ylabel('median fit weight')
        h1, l1 = axp.get_legend_handles_labels()
        h2, l2 = axp2.get_legend_handles_labels()
        axp.legend(h1 + h2, l1 + l2, loc='best', fontsize=8)

        axes[1, 2].axis('off')
        text_lines = [
            f"config={config['name']}",
            f"mode={config['mode']}",
            f"xfit={xfit:.3f}, yfit={yfit:.3f}",
            f"flux_fit={flux_fit:.6g}",
            f"fit_ierr={info['fit_ierr']}",
            f"fit_nfev={info['fit_nfev']}",
            f"msg={info['fit_message']:.30}",
            f"weighted_sse_fit={weighted_sse_fit:.6g}",
            f"weighted_sse_null={weighted_sse_null:.6g}",
            f"weighted_delta(fit-null)={weighted_delta:.6g}",
            f"result_cols={','.join(result.colnames):.30}",
        ] + err_report.split("; ")
        axes[1, 2].text(0.01, 0.99, '\n'.join(text_lines), va='top', ha='left', fontsize=8, family='monospace')

        fig.suptitle(
            f"deep_dive star_id={target_meta['target_star_id']} "
            f"RA={target_meta['target_ra_deg']:.8f} Dec={target_meta['target_dec_deg']:.8f}\n"
            f"{science_image.name} | {config['name']} ({stpsf_label})"
        )
        fig.savefig(deep_dir / f'{deep_dive_prefix}_{config["name"]}.png', dpi=150)
        plt.close(fig)

        row = {
            'config_name': config['name'],
            'mode': config['mode'],
            'xfit_cutout': xfit,
            'yfit_cutout': yfit,
            'flux_fit': flux_fit,
            'fit_ierr': info['fit_ierr'],
            'fit_nfev': info['fit_nfev'],
            'fit_message': info['fit_message'],
            'weighted_sse_fit': weighted_sse_fit,
            'weighted_sse_null': weighted_sse_null,
            'weighted_delta_fit_minus_null': weighted_delta,
            'target_star_id': target_meta['target_star_id'],
            'target_ra_deg': target_meta['target_ra_deg'],
            'target_dec_deg': target_meta['target_dec_deg'],
            'science_image': str(science_image),
            'residual_image': str(residual_image),
            'target_source': target_meta['source'],
        }
        for cn in result.colnames:
            row[f'result_{cn}'] = result[cn][0]
        rows.append(row)

    deep_tbl = Table(rows=rows)
    deep_tbl.write(deep_dir / f'{deep_dive_prefix}_fit_results.ecsv', overwrite=True)

    lines = [
        'Deep-dive single-star fit summary',
        f"target_star_id: {target_meta['target_star_id']}",
        f"target_ra_deg: {target_meta['target_ra_deg']:.10f}",
        f"target_dec_deg: {target_meta['target_dec_deg']:.10f}",
        f'target_source: {target_meta["source"]}',
        f'science_image: {science_image}',
        f'residual_image: {residual_image}',
        f'cutout_halfsize: {cutout_halfsize}',
        f'output_table: {deep_dir / (deep_dive_prefix + "_fit_results.ecsv")}',
        f'n_configs_with_results: {len(deep_tbl)}',
    ]
    if len(deep_tbl) > 0:
        for row in deep_tbl:
            lines.append(
                f"{row['config_name']}: weighted_delta_fit_minus_null={row['weighted_delta_fit_minus_null']:.6g}, "
                f"fit_ierr={row['fit_ierr']}, fit_message={row['fit_message']}"
            )
    (deep_dir / f'{deep_dive_prefix}_summary.txt').write_text('\n'.join(lines) + '\n')

    globals().update(locals())
    return deep_tbl


def main():
    parser = argparse.ArgumentParser(description='Diagnose F480M oversubtraction and run cutout photutils parameter sweeps.')
    parser.add_argument('--science-image', default='/orange/adamginsburg/jwst/sickle/F480M/pipeline/jw03958007001_03104_00001_nrcblong_destreak_o007_crf.fits')
    parser.add_argument('--residual-image', default='/orange/adamginsburg/jwst/sickle/F480M/pipeline/jw03958007001_03104_00001_nrcblong_destreak_o007_crf_satstar_residual.fits')
    parser.add_argument('--region-file', default='/orange/adamginsburg/jwst/sickle/regions_/diagnostic_oversubtracted_stars_bigger.reg')
    parser.add_argument('--outdir', default='/orange/adamginsburg/jwst/sickle/overfitting_experiments')
    parser.add_argument('--fwhm-pix', type=float, default=2.574)
    parser.add_argument('--detect-sigma', type=float, default=3.0)
    parser.add_argument('--roundlo', type=float, default=-0.3)
    parser.add_argument('--roundhi', type=float, default=0.3)
    parser.add_argument('--sharplo', type=float, default=0.5)
    parser.add_argument('--sharphi', type=float, default=1.0)
    parser.add_argument('--match-radius-arcsec', type=float, default=0.12)
    parser.add_argument('--nstars', type=int, default=8)
    parser.add_argument('--cutout-halfsize', type=int, default=18)
    parser.add_argument('--stpsf-grid-file',
                        default='/orange/adamginsburg/jwst/sickle/psfs/nircam_nrcb5_f480m_fovp512_samp2_npsf16.fits',
                        help='STPSF grid FITS file used in cutout PSF fitting')
    parser.add_argument('--deep-dive-only', action='store_true',
                        help='Run only the standalone deep-dive single-star fitting experiment.')
    parser.add_argument('--deep-dive-star-id', type=int, default=None,
                        help='Star ID from cutout_selected_stars.ecsv to deep-dive.')
    parser.add_argument('--deep-dive-ra', type=float, default=None,
                        help='RA in deg for deep-dive target (overrides star-id lookup if Dec is also given).')
    parser.add_argument('--deep-dive-dec', type=float, default=None,
                        help='Dec in deg for deep-dive target (overrides star-id lookup if RA is also given).')
    parser.add_argument('--deep-dive-science-image', default=None,
                        help='Science image FITS for deep-dive (defaults to --science-image).')
    parser.add_argument('--deep-dive-residual-image', default=None,
                        help='Residual image FITS for deep-dive (defaults to --residual-image).')
    parser.add_argument('--deep-dive-stars-table', default=None,
                        help='Path to stars table for deep-dive star-id lookup; defaults to {outdir}/cutout_selected_stars.ecsv')
    parser.add_argument('--deep-dive-prefix', default='deep_dive',
                        help='Output filename prefix for deep-dive products.')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stpsf_grid_path = Path(args.stpsf_grid_file)
    stpsf_psf_model = load_stpsf_psf_model(stpsf_grid_path)

    deep_dive_requested = (
        args.deep_dive_only
        or args.deep_dive_star_id is not None
        or (args.deep_dive_ra is not None and args.deep_dive_dec is not None)
    )

    if args.deep_dive_only:
        deep_science = Path(args.deep_dive_science_image) if args.deep_dive_science_image is not None else Path(args.science_image)
        deep_residual = Path(args.deep_dive_residual_image) if args.deep_dive_residual_image is not None else Path(args.residual_image)
        deep_stars_table = Path(args.deep_dive_stars_table) if args.deep_dive_stars_table is not None else (outdir / 'cutout_selected_stars.ecsv')
        deep_tbl = run_deep_dive_single_star(
            outdir=outdir,
            science_image=deep_science,
            residual_image=deep_residual,
            stpsf_psf_model=stpsf_psf_model,
            stpsf_label=stpsf_grid_path.name,
            fwhm_pix=args.fwhm_pix,
            cutout_halfsize=args.cutout_halfsize,
            deep_dive_star_id=args.deep_dive_star_id,
            deep_dive_ra=args.deep_dive_ra,
            deep_dive_dec=args.deep_dive_dec,
            deep_dive_stars_table=deep_stars_table,
            deep_dive_prefix=args.deep_dive_prefix,
        )
        print(f'Deep-dive completed with {len(deep_tbl)} successful config fits.')
        return

    science_data, _, science_err, science_dq, science_wht = load_fits_bundle(Path(args.science_image))
    crowdsource_weight_map = compute_crowdsource_weight_map(science_data, science_err, dq=science_dq, wht=science_wht)
    residual_data, residual_wcs = load_fits_data_and_wcs(Path(args.residual_image))

    point_regions = read_point_regions(Path(args.region_file))
    if len(point_regions) == 0:
        raise ValueError('No point regions found in the hand-selected region file.')

    det_tbl, bkg_median, noise, threshold = detect_negative_residual_stars(
        residual_data=residual_data,
        fwhm_pix=args.fwhm_pix,
        sigma_threshold=args.detect_sigma,
        roundlo=args.roundlo,
        roundhi=args.roundhi,
        sharplo=args.sharplo,
        sharphi=args.sharphi,
    )
    det_tbl = add_skycoords(det_tbl, residual_wcs)

    match_tbl = match_regions_to_detections(
        point_regions=point_regions,
        det_tbl=det_tbl,
        max_sep_arcsec=args.match_radius_arcsec,
    )
    det_coverage_tbl = _compute_detection_vgroup_coverage(det_tbl, Path(args.science_image))

    det_tbl.write(outdir / 'negative_residual_detections.ecsv', overwrite=True)
    match_tbl.write(outdir / 'negative_residual_match_to_hand_selected.ecsv', overwrite=True)
    det_coverage_tbl.write(outdir / 'negative_residual_detection_vgroup_coverage.ecsv', overwrite=True)

    write_ds9_points(
        outdir / 'negative_residual_detections.reg',
        det_tbl['ra_deg'] if len(det_tbl) > 0 else np.array([], dtype=float),
        det_tbl['dec_deg'] if len(det_tbl) > 0 else np.array([], dtype=float),
        color='#E6A01A',
    )

    matched_idx = np.asarray(match_tbl['matched_detection_index'][match_tbl['matched']], dtype=int)
    matched_idx = np.unique(matched_idx)
    write_ds9_points(
        outdir / 'negative_residual_detections_matched_to_hand_selected.reg',
        det_tbl['ra_deg'][matched_idx] if len(matched_idx) > 0 else np.array([], dtype=float),
        det_tbl['dec_deg'][matched_idx] if len(matched_idx) > 0 else np.array([], dtype=float),
        color='#2EE6D6',
    )

    sweep_tbl, stars_tbl, radial_profile_tbl = run_cutout_sweep(
        science_data=science_data,
        science_error=science_err,
        residual_data=residual_data,
        residual_wcs=residual_wcs,
        det_tbl=det_tbl,
        match_tbl=match_tbl,
        nstars=args.nstars,
        fwhm_pix=args.fwhm_pix,
        cutout_halfsize=args.cutout_halfsize,
        outdir=outdir,
        stpsf_psf_model=stpsf_psf_model,
        stpsf_label=stpsf_grid_path.name,
        det_coverage_tbl=det_coverage_tbl,
        crowdsource_weight_map=crowdsource_weight_map,
    )

    sweep_tbl.write(outdir / 'cutout_parameter_sweep_results.ecsv', overwrite=True)
    stars_tbl.write(outdir / 'cutout_selected_stars.ecsv', overwrite=True)
    radial_profile_tbl.write(outdir / 'cutout_radial_profiles.ecsv', overwrite=True)

    gallery_measurements, n_gallery_exposures = generate_exposure_gallery_and_summary(
        stars_tbl=stars_tbl,
        science_image=Path(args.science_image),
        outdir=outdir,
        cutout_halfsize=args.cutout_halfsize,
    )

    summary_tbl = summarize_configs(sweep_tbl)
    summary_tbl.write(outdir / 'cutout_parameter_sweep_summary.ecsv', overwrite=True)

    if deep_dive_requested:
        deep_science = Path(args.deep_dive_science_image) if args.deep_dive_science_image is not None else Path(args.science_image)
        deep_residual = Path(args.deep_dive_residual_image) if args.deep_dive_residual_image is not None else Path(args.residual_image)
        deep_stars_table = Path(args.deep_dive_stars_table) if args.deep_dive_stars_table is not None else (outdir / 'cutout_selected_stars.ecsv')
        run_deep_dive_single_star(
            outdir=outdir,
            science_image=deep_science,
            residual_image=deep_residual,
            stpsf_psf_model=stpsf_psf_model,
            stpsf_label=stpsf_grid_path.name,
            fwhm_pix=args.fwhm_pix,
            cutout_halfsize=args.cutout_halfsize,
            deep_dive_star_id=args.deep_dive_star_id,
            deep_dive_ra=args.deep_dive_ra,
            deep_dive_dec=args.deep_dive_dec,
            deep_dive_stars_table=deep_stars_table,
            deep_dive_prefix=args.deep_dive_prefix,
        )

    nhand = len(point_regions)
    nmatched = int(np.sum(match_tbl['matched']))
    frac = nmatched / nhand if nhand > 0 else np.nan

    lines = [
        'F480M overfitting experiment summary',
        f'science_image: {args.science_image}',
        f'residual_image: {args.residual_image}',
        f'region_file: {args.region_file}',
        f'stpsf_grid_file: {args.stpsf_grid_file}',
        f'fwhm_pix: {args.fwhm_pix:.3f}',
        f'detection_sigma: {args.detect_sigma:.2f}',
        f'noise_estimate: {noise:.6g}',
        f'detection_threshold: {threshold:.6g}',
        f'background_median_in_minus_residual: {bkg_median:.6g}',
        f'negative_residual_detections: {len(det_tbl)}',
        f'hand_selected_points: {nhand}',
        f'matched_hand_selected_within_{args.match_radius_arcsec:.3f}arcsec: {nmatched}',
        f'matched_fraction: {frac:.3f}',
        f'cutout_stars_used: {len(stars_tbl)}',
        f'cutout_fit_rows: {len(sweep_tbl)}',
        f'cutout_radial_profile_rows: {len(radial_profile_tbl)}',
        f'gallery_exposure_count: {n_gallery_exposures}',
        f'gallery_measurements_rows: {len(gallery_measurements)}',
        '',
        'Configuration summary (median residual metrics):',
    ]

    if len(summary_tbl) == 0:
        lines.append('No cutout fit results were produced.')
    else:
        for row in summary_tbl:
            lines.append(
                f"{row['config_name']}: n={row['nfits']}, "
                f"median_core_median_resid={row['median_core_median_resid']:.5g}, "
                f"median_core_min_resid={row['median_core_min_resid']:.5g}, "
                f"median_center_resid={row['median_center_resid']:.5g}, "
                f"median_ring_median_resid={row['median_ring_median_resid']:.5g}, "
                f"median_core_data_minus_model={row['median_core_data_minus_model']:.5g}, "
                f"median_wing_data_minus_model={row['median_wing_data_minus_model']:.5g}"
            )

    (outdir / 'summary.txt').write_text('\n'.join(lines) + '\n')

    print('\n'.join(lines))


if __name__ == '__main__':
    main()
