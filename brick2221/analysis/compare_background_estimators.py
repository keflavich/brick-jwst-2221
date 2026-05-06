#!/usr/bin/env python

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.table import Table
from astropy import units as u
from astropy.wcs import WCS
from photutils.background import (
    Background2D,
    BiweightLocationBackground,
    LocalBackground,
    MMMBackground,
    MedianBackground,
    SExtractorBackground,
)


def _read_image(path):
    with fits.open(path) as hdul:
        if 'SCI' in hdul:
            data = np.asarray(hdul['SCI'].data, dtype=float)
            wcs = WCS(hdul['SCI'].header)
        else:
            data = np.asarray(hdul[1].data, dtype=float)
            wcs = WCS(hdul[1].header)
    return data, wcs


def _load_seed_table(path):
    table = Table.read(path)
    if 'refined_ra_deg' not in table.colnames or 'refined_dec_deg' not in table.colnames:
        raise KeyError('Seed table must contain refined_ra_deg and refined_dec_deg columns')
    return table


def _nonempty_strings(values):
    result = []
    for value in values:
        text = str(value)
        if text not in ('', '--', 'masked'):
            result.append(text)
    return result


def _choose_source_group(seed_table, source_filename=None):
    source_filenames = _nonempty_strings(seed_table['source_filename']) if 'source_filename' in seed_table.colnames else []
    if source_filename is None:
        if len(source_filenames) == 0:
            raise ValueError('Seed table does not include source_filename values and no --source-filename was provided')
        source_filename = Counter(source_filenames).most_common(1)[0][0]

    selected = seed_table[np.asarray([str(value) == source_filename for value in seed_table['source_filename']])]
    if len(selected) == 0:
        raise ValueError(f'No rows in seed table matched source_filename={source_filename}')

    return source_filename, selected


def _source_pixel_coordinates(table, wcs):
    skycoord = SkyCoord(ra=np.asarray(table['refined_ra_deg'], dtype=float) * u.deg,
                       dec=np.asarray(table['refined_dec_deg'], dtype=float) * u.deg)
    xpix, ypix = wcs.world_to_pixel(skycoord)
    return np.asarray(xpix, dtype=float), np.asarray(ypix, dtype=float)


def _baseline_5x5(image, xpix, ypix):
    xi = int(np.rint(xpix))
    yi = int(np.rint(ypix))
    y0 = max(0, yi - 2)
    y1 = min(image.shape[0], yi + 3)
    x0 = max(0, xi - 2)
    x1 = min(image.shape[1], xi + 3)
    patch = image[y0:y1, x0:x1]
    if patch.size == 0:
        return np.nan
    return float(np.nanmedian(patch))


def _cutout_and_center(data, wcs, xpix, ypix, cutout_size):
    cutout = Cutout2D(
        data,
        (xpix, ypix),
        (cutout_size, cutout_size),
        wcs=wcs,
        mode='partial',
        fill_value=np.nan,
    )
    return cutout.data, cutout.wcs


def _center_mask(shape, radius_pix):
    yy, xx = np.indices(shape)
    cy = (shape[0] - 1) / 2.0
    cx = (shape[1] - 1) / 2.0
    return np.hypot(xx - cx, yy - cy) <= radius_pix


def _local_background_estimate(cutout, xpix, ypix, inner_radius, outer_radius, mask_radius_pix):
    mask = np.isnan(cutout) | _center_mask(cutout.shape, mask_radius_pix)
    estimator = LocalBackground(inner_radius, outer_radius, bkg_estimator=MedianBackground())
    return float(estimator(cutout, xpix, ypix, mask=mask))


def _background2d_center(cutout, estimator, box_size, mask_radius_pix):
    mask = np.isnan(cutout) | _center_mask(cutout.shape, mask_radius_pix)
    bkg = Background2D(
        cutout,
        box_size=box_size,
        mask=mask,
        bkg_estimator=estimator,
        exclude_percentile=10.0,
    )
    cy = cutout.shape[0] // 2
    cx = cutout.shape[1] // 2
    return float(bkg.background[cy, cx])


def main():
    parser = argparse.ArgumentParser(description='Compare photutils background estimators against the residual-infilled baseline.')
    parser.add_argument('--seed-table', required=True, help='ECSV table with refined star positions and source_filename values')
    parser.add_argument('--source-filename', default='', help='Optional explicit source image to analyze')
    parser.add_argument('--cutout-size', type=int, default=41, help='Square cutout size around each seed star')
    parser.add_argument('--mask-radius-pix', type=float, default=3.0, help='Central mask radius for background estimators')
    parser.add_argument('--local-inner', type=float, default=5.0, help='LocalBackground inner radius')
    parser.add_argument('--local-outer', type=float, default=15.0, help='LocalBackground outer radius')
    parser.add_argument('--local-inner-alt', type=float, default=2.0, help='Alternative LocalBackground inner radius')
    parser.add_argument('--local-outer-alt', type=float, default=5.0, help='Alternative LocalBackground outer radius')
    parser.add_argument('--background2d-box-size', type=int, default=7, help='Box size for Background2D on cutouts')
    parser.add_argument('--output', default='background_estimator_comparison.ecsv', help='Output ECSV table')
    args = parser.parse_args()

    seed_table = _load_seed_table(args.seed_table)
    source_filename, selected = _choose_source_group(seed_table, source_filename=args.source_filename or None)

    source_data, source_wcs = _read_image(source_filename)
    infilled_filenames = _nonempty_strings(selected['infilled_filename']) if 'infilled_filename' in selected.colnames else []
    if len(infilled_filenames) == 0:
        raise ValueError('Seed table does not include infilled_filename values')
    infilled_filename = Counter(infilled_filenames).most_common(1)[0][0]
    infilled_data, infilled_wcs = _read_image(infilled_filename)

    xpix, ypix = _source_pixel_coordinates(selected, source_wcs)
    infilled_xpix, infilled_ypix = _source_pixel_coordinates(selected, infilled_wcs)

    rows = []
    estimator_names = [
        'localbackground_5_15',
        'localbackground_2_5',
        'background2d_median',
        'background2d_biweight',
        'background2d_mmm',
        'background2d_sextractor',
    ]

    estimators = {
        'localbackground_5_15': lambda cutout, x, y: _local_background_estimate(cutout, x, y, args.local_inner, args.local_outer, args.mask_radius_pix),
        'localbackground_2_5': lambda cutout, x, y: _local_background_estimate(cutout, x, y, args.local_inner_alt, args.local_outer_alt, args.mask_radius_pix),
        'background2d_median': lambda cutout, x, y: _background2d_center(cutout, MedianBackground(), args.background2d_box_size, args.mask_radius_pix),
        'background2d_biweight': lambda cutout, x, y: _background2d_center(cutout, BiweightLocationBackground(), args.background2d_box_size, args.mask_radius_pix),
        'background2d_mmm': lambda cutout, x, y: _background2d_center(cutout, MMMBackground(), args.background2d_box_size, args.mask_radius_pix),
        'background2d_sextractor': lambda cutout, x, y: _background2d_center(cutout, SExtractorBackground(), args.background2d_box_size, args.mask_radius_pix),
    }

    for index, row in enumerate(selected):
        src_x = float(xpix[index])
        src_y = float(ypix[index])
        inf_x = float(infilled_xpix[index])
        inf_y = float(infilled_ypix[index])
        if not (np.isfinite(src_x) and np.isfinite(src_y) and np.isfinite(inf_x) and np.isfinite(inf_y)):
            continue

        source_cutout, source_cutout_wcs = _cutout_and_center(source_data, source_wcs, src_x, src_y, args.cutout_size)
        infilled_cutout, _ = _cutout_and_center(infilled_data, infilled_wcs, inf_x, inf_y, args.cutout_size)
        center_x = (source_cutout.shape[1] - 1) / 2.0
        center_y = (source_cutout.shape[0] - 1) / 2.0

        baseline = _baseline_5x5(infilled_cutout, center_x, center_y)

        record = {
            'row_index': index,
            'source_filename': source_filename,
            'infilled_filename': infilled_filename,
            'ra_deg': float(row['refined_ra_deg']),
            'dec_deg': float(row['refined_dec_deg']),
            'x_pix': src_x,
            'y_pix': src_y,
            'baseline_infilled_5x5_median': baseline,
            'catalog_local_bkg': float(row['catalog_local_bkg']) if 'catalog_local_bkg' in row.colnames else np.nan,
        }

        for name, estimator in estimators.items():
            estimate = estimator(source_cutout, center_x, center_y)
            record[name] = estimate
            record[f'{name}_minus_baseline'] = estimate - baseline
            record[f'abs_{name}_minus_baseline'] = abs(estimate - baseline)

        rows.append(record)

    result = Table(rows=rows)
    output_path = Path(args.output)
    result.write(output_path, overwrite=True)

    print(f'Analyzed source image: {source_filename}')
    print(f'Compared against residual baseline image: {infilled_filename}')
    print(f'Used {len(result)} seed stars out of {len(selected)} rows in the selected exposure')
    print(f'Wrote {output_path}')

    if len(result) == 0:
        return

    print('Summary vs baseline (median absolute error / median bias):')
    for name in estimator_names:
        mae = float(np.nanmedian(result[f'abs_{name}_minus_baseline']))
        bias = float(np.nanmedian(result[f'{name}_minus_baseline']))
        print(f'  {name}: mae={mae:.3f}, bias={bias:.3f}')


if __name__ == '__main__':
    main()