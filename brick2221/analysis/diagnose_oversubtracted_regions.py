#!/usr/bin/env python

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table, vstack
from astropy import units as u
from astropy.wcs import WCS
from photutils.background import LocalBackground
import regions


def _write_ds9_region_file(path, point_coords, include_box_regions=None):
    lines = [
        '# Region file format: DS9 CARTA 5.0.1',
        'global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1',
        'icrs',
    ]
    if include_box_regions is not None:
        for reg in include_box_regions:
            lines.append(reg.serialize(format='ds9').strip())
    for coord in point_coords:
        lines.append(f'point({coord.ra.deg:.9f}, {coord.dec.deg:.9f}) # color=#2EE6D6 width=2')
    Path(path).write_text('\n'.join(lines) + '\n')


def _source_xy(table):
    if 'x_0' in table.colnames and 'y_0' in table.colnames:
        return np.asarray(table['x_0'], dtype=float), np.asarray(table['y_0'], dtype=float)
    if 'x_fit' in table.colnames and 'y_fit' in table.colnames:
        return np.asarray(table['x_fit'], dtype=float), np.asarray(table['y_fit'], dtype=float)
    if 'xcentroid' in table.colnames and 'ycentroid' in table.colnames:
        return np.asarray(table['xcentroid'], dtype=float), np.asarray(table['ycentroid'], dtype=float)
    if 'x_init' in table.colnames and 'y_init' in table.colnames:
        return np.asarray(table['x_init'], dtype=float), np.asarray(table['y_init'], dtype=float)
    if 'x' in table.colnames and 'y' in table.colnames:
        return np.asarray(table['x'], dtype=float), np.asarray(table['y'], dtype=float)
    raise KeyError(f'No recognized x/y columns in {table.colnames}')


def _read_wcs(filename):
    with fits.open(filename) as hdul:
        if 'SCI' in hdul:
            return WCS(hdul['SCI'].header)
        return WCS(hdul[1].header)


def _normalize_path(path_text):
    return Path(path_text.replace('//', '/'))


def _catalog_skycoord(catalog, catalog_path):
    if 'skycoord_centroid' in catalog.colnames:
        return catalog['skycoord_centroid'], _normalize_path(catalog_path)
    if 'skycoord_fit' in catalog.colnames:
        return catalog['skycoord_fit'], _normalize_path(catalog_path)
    source_filename = catalog.meta['FILENAME']
    source_path = _normalize_path(source_filename)
    if not source_path.is_absolute() or not source_path.exists():
        source_path = _normalize_path(str(Path(catalog_path).parent / source_filename))
    wcs = _read_wcs(source_path)
    xvals, yvals = _source_xy(catalog)
    skycoord = wcs.pixel_to_world(xvals, yvals)
    return skycoord, source_path


def _load_residual_image(path):
    with fits.open(path) as hdul:
        if 'SCI' in hdul:
            data = np.asarray(hdul['SCI'].data, dtype=float)
            wcs = WCS(hdul['SCI'].header)
        else:
            data = np.asarray(hdul[1].data, dtype=float)
            wcs = WCS(hdul[1].header)
    return data, wcs


def _extract_row_xy(row, colnames):
    if 'x_0' in colnames and 'y_0' in colnames:
        return float(row['x_0']), float(row['y_0'])
    if 'x_fit' in colnames and 'y_fit' in colnames:
        return float(row['x_fit']), float(row['y_fit'])
    if 'xcentroid' in colnames and 'ycentroid' in colnames:
        return float(row['xcentroid']), float(row['ycentroid'])
    if 'x_init' in colnames and 'y_init' in colnames:
        return float(row['x_init']), float(row['y_init'])
    if 'x' in colnames and 'y' in colnames:
        return float(row['x']), float(row['y'])
    return np.nan, np.nan


def _extract_visit_vgroup_exp(catalog_path):
    match = re.search(r'visit(\d+)_vgroup([0-9a-zA-Z]+)_exp(\d+)', Path(catalog_path).name)
    if match is None:
        return None, None, None
    return match.group(1), match.group(2), match.group(3)


def _extract_iteration_token(catalog_path):
    match = re.search(r'_(iter[^_]+)_daophot_', Path(catalog_path).name)
    if match is None:
        return None
    return match.group(1)


def _find_infilled_image_for_row(row, colnames, cache):
    source_filename = row['_source_filename'] if '_source_filename' in colnames else ''
    catalog_path = row['_catalog_path'] if '_catalog_path' in colnames else ''
    if not source_filename or not catalog_path:
        return None

    visit_id, vgroup_id, exp_id = _extract_visit_vgroup_exp(catalog_path)
    iteration_token = _extract_iteration_token(catalog_path)
    if visit_id is None or iteration_token is None:
        return None

    pipeline_dir = str(_normalize_path(source_filename).parent)
    filter_name = str(row['_filter']).lower() if '_filter' in colnames else 'f480m'
    module_name = str(row['_module']) if '_module' in colnames else 'nrcb'
    pattern = (
        f"{pipeline_dir}/jw0*-o*_t001_nircam_*-{filter_name}-{module_name}"
        f"_visit{visit_id}_vgroup{vgroup_id}_exp{exp_id}*_{iteration_token}_residual_infilled.fits"
    )
    matches = sorted(glob.glob(pattern))
    if len(matches) == 0:
        return None

    infilled_path = matches[0]
    if infilled_path not in cache:
        with fits.open(infilled_path) as hdul:
            if 'SCI' in hdul:
                cache[infilled_path] = np.asarray(hdul['SCI'].data, dtype=float)
            else:
                cache[infilled_path] = np.asarray(hdul[1].data, dtype=float)
    return infilled_path


def _sample_5x5_median(image, xpix, ypix):
    if image is None or not np.isfinite(xpix) or not np.isfinite(ypix):
        return np.nan
    xi = int(np.rint(xpix))
    yi = int(np.rint(ypix))
    y0 = max(0, yi - 2)
    y1 = min(image.shape[0], yi + 3)
    x0 = max(0, xi - 2)
    x1 = min(image.shape[1], xi + 3)
    if y0 >= y1 or x0 >= x1:
        return np.nan
    patch = image[y0:y1, x0:x1]
    return float(np.nanmedian(patch))


def _sample_5x5_median_at_skycoord(image, wcs, skycoord):
    if image is None or wcs is None or skycoord is None:
        return np.nan
    xpix, ypix = wcs.world_to_pixel(skycoord)
    return _sample_5x5_median(image, xpix, ypix)


def _sample_image_at_skycoord(data, wcs, skycoord):
    xpix, ypix = wcs.world_to_pixel(skycoord)
    xi = np.rint(xpix).astype(int)
    yi = np.rint(ypix).astype(int)
    sample = np.full(len(xi), np.nan, dtype=float)
    valid = ((xi >= 0) & (yi >= 0) & (yi < data.shape[0]) & (xi < data.shape[1]))
    sample[valid] = data[yi[valid], xi[valid]]
    return sample


def _region_center(region):
    if hasattr(region, 'center'):
        return region.center
    return None


def _summarize_box(region, residual_data, residual_wcs):
    pixel_region = region.to_pixel(residual_wcs)
    mask = pixel_region.to_mask(mode='center')
    if mask is None:
        return None
    cutout = mask.cutout(residual_data)
    if cutout is None:
        return None
    masked = mask.data.astype(bool)
    values = cutout[masked]
    if values.size == 0:
        return None
    return {
        'box_label': getattr(region.meta, 'text', '') if hasattr(region, 'meta') else '',
        'npix': int(values.size),
        'median': float(np.nanmedian(values)),
        'mean': float(np.nanmean(values)),
        'min': float(np.nanmin(values)),
        'max': float(np.nanmax(values)),
    }


def main():
    parser = argparse.ArgumentParser(description='Crossmatch oversubtracted diagnostic regions against iter2 catalogs.')
    parser.add_argument('--region-file', required=True, help='DS9 region file with oversubtracted stars')
    parser.add_argument('--catalog-glob', required=True, help='Glob for iter2 catalog FITS files')
    parser.add_argument('--residual-image', required=True, help='Residual mosaic FITS image, e.g. *_residual_i2d.fits')
    parser.add_argument('--baseline-image', default='', help='Optional mosaic infilled baseline image to compare against')
    parser.add_argument('--match-radius-arcsec', type=float, default=0.1, help='Matching radius in arcsec')
    parser.add_argument('--output', default='diagnostic_oversubtracted_matches.ecsv', help='Output summary table path')
    parser.add_argument('--refined-region-output', default='', help='Optional DS9 region output with refined point positions')
    args = parser.parse_args()

    region_list = regions.Regions.read(args.region_file)
    point_regions = [reg for reg in region_list if hasattr(reg, 'center') and reg.__class__.__name__.endswith('PointSkyRegion')]
    box_regions = [reg for reg in region_list if reg.__class__.__name__.endswith('RectangleSkyRegion')]

    catalog_paths = sorted(glob.glob(args.catalog_glob))
    if len(catalog_paths) == 0:
        raise FileNotFoundError(f'No catalogs matched {args.catalog_glob}')

    residual_data, residual_wcs = _load_residual_image(args.residual_image)
    baseline_data = None
    baseline_wcs = None
    if args.baseline_image:
        baseline_data, baseline_wcs = _load_residual_image(args.baseline_image)

    catalog_tables = []
    catalog_coords = []
    catalog_meta = []
    for catalog_path in catalog_paths:
        table = Table.read(catalog_path)
        table['_catalog_path'] = np.array([catalog_path] * len(table), dtype='U512')
        table['_source_filename'] = np.array([str(table.meta.get('FILENAME', ''))] * len(table), dtype='U512')
        table['_filter'] = np.array([str(table.meta.get('FILTER', ''))] * len(table), dtype='U32')
        table['_module'] = np.array([str(table.meta.get('MODULE', ''))] * len(table), dtype='U32')
        skycoord, source_path = _catalog_skycoord(table, catalog_path)
        catalog_tables.append(table)
        catalog_coords.append(skycoord)
        catalog_meta.append((catalog_path, source_path))

    all_coords = SkyCoord(
        ra=np.concatenate([coord.ra.to_value(u.deg) for coord in catalog_coords]) * u.deg,
        dec=np.concatenate([coord.dec.to_value(u.deg) for coord in catalog_coords]) * u.deg,
    )
    all_tables = vstack(catalog_tables, metadata_conflicts='silent')

    point_rows = []
    point_coords = SkyCoord(
        ra=np.array([region.center.ra.to_value(u.deg) for region in point_regions]) * u.deg,
        dec=np.array([region.center.dec.to_value(u.deg) for region in point_regions]) * u.deg,
    ) if point_regions else None

    refined_point_coords = []
    if point_coords is not None and len(all_coords) > 0:
        idx, sep2d, _ = point_coords.match_to_catalog_sky(all_coords)
        residual_at_source = _sample_image_at_skycoord(residual_data, residual_wcs, all_coords[idx])
        localbkg_estimator = LocalBackground(5, 15)
        infilled_cache = {}
        for region_index, region in enumerate(point_regions):
            source_index = int(idx[region_index])
            matched = all_tables[source_index]
            sep_arcsec = float(sep2d[region_index].to_value(u.arcsec))
            matched_within_radius = sep_arcsec <= args.match_radius_arcsec
            refined_coord = all_coords[source_index] if matched_within_radius else region.center
            refined_point_coords.append(refined_coord)
            row_xpix, row_ypix = _extract_row_xy(matched, all_tables.colnames)
            if baseline_data is not None and baseline_wcs is not None:
                infilled_median_5x5 = _sample_5x5_median_at_skycoord(baseline_data, baseline_wcs, all_coords[source_index])
                infilled_path = args.baseline_image
                xres, yres = residual_wcs.world_to_pixel(all_coords[source_index])
                localbkg_residual_mosaic = float(localbkg_estimator(residual_data, float(xres), float(yres)))
                xinfill, yinfill = baseline_wcs.world_to_pixel(all_coords[source_index])
                localbkg_infilled_mosaic = float(localbkg_estimator(baseline_data, float(xinfill), float(yinfill)))
            else:
                infilled_path = _find_infilled_image_for_row(matched, all_tables.colnames, infilled_cache)
                infilled_image = infilled_cache[infilled_path] if infilled_path is not None else None
                infilled_median_5x5 = _sample_5x5_median(infilled_image, row_xpix, row_ypix)
                localbkg_residual_mosaic = np.nan
                localbkg_infilled_mosaic = np.nan
            catalog_local_bkg = float(matched['local_bkg']) if 'local_bkg' in matched.colnames else np.nan
            point_rows.append({
                'region_index': region_index,
                'region_ra_deg': float(region.center.ra.to_value(u.deg)),
                'region_dec_deg': float(region.center.dec.to_value(u.deg)),
                'catalog_source_index': source_index,
                'separation_arcsec': sep_arcsec,
                'matched_within_radius': matched_within_radius,
                'catalog_flux_fit': float(matched['flux_fit']) if 'flux_fit' in matched.colnames else np.nan,
                'catalog_local_bkg': catalog_local_bkg,
                'infilled_bkg_5x5_median': infilled_median_5x5,
                'local_bkg_minus_infilled5x5': catalog_local_bkg - infilled_median_5x5,
                'abs_local_bkg_minus_infilled5x5': abs(catalog_local_bkg - infilled_median_5x5),
                'localbkg_residual_mosaic': localbkg_residual_mosaic,
                'localbkg_infilled_mosaic': localbkg_infilled_mosaic,
                'localbkg_infilled_minus_residual': localbkg_infilled_mosaic - localbkg_residual_mosaic,
                'catalog_minus_localbkg_residual_mosaic': catalog_local_bkg - localbkg_residual_mosaic,
                'catalog_minus_localbkg_infilled_mosaic': catalog_local_bkg - localbkg_infilled_mosaic,
                'catalog_flags': int(matched['flags']) if 'flags' in matched.colnames else -1,
                'catalog_qfit': float(matched['qfit']) if 'qfit' in matched.colnames else np.nan,
                'catalog_cfit': float(matched['cfit']) if 'cfit' in matched.colnames else np.nan,
                'catalog_is_saturated': bool(matched['is_saturated']) if 'is_saturated' in matched.colnames else False,
                'source_filename': str(matched['_source_filename']) if '_source_filename' in all_tables.colnames else '',
                'infilled_filename': infilled_path if infilled_path is not None else '',
                'refined_ra_deg': float(refined_coord.ra.to_value(u.deg)),
                'refined_dec_deg': float(refined_coord.dec.to_value(u.deg)),
                'residual_value': float(residual_at_source[region_index]),
            })

    point_table = Table(rows=point_rows)

    box_rows = []
    for box_index, box_region in enumerate(box_regions):
        summary = _summarize_box(box_region, residual_data, residual_wcs)
        if summary is not None:
            summary['box_index'] = box_index
            box_rows.append(summary)

    box_table = Table(rows=box_rows)

    output_path = Path(args.output)
    point_output = output_path.with_name(output_path.stem + '_points.ecsv')
    box_output = output_path.with_name(output_path.stem + '_boxes.ecsv')
    point_table.write(point_output, overwrite=True)
    box_table.write(box_output, overwrite=True)

    if args.refined_region_output:
        _write_ds9_region_file(args.refined_region_output, refined_point_coords, include_box_regions=box_regions)

    print(f'Read {len(catalog_paths)} catalogs with {len(all_tables)} total sources')
    print(f'Matched {len(point_table)} point regions')
    print(f'Wrote {point_output}')
    print(f'Wrote {box_output}')

    if len(point_table) > 0:
        print('Point-region summary:')
        print(f"  median separation arcsec = {np.nanmedian(point_table['separation_arcsec']):.3f}")
        recovered = int(np.sum(point_table['matched_within_radius']))
        print(f"  recovered within {args.match_radius_arcsec:.3f} arcsec = {recovered}/{len(point_table)}")
        print(f"  median local_bkg = {np.nanmedian(point_table['catalog_local_bkg']):.3f}")
        if 'local_bkg_minus_infilled5x5' in point_table.colnames:
            print(f"  median (local_bkg - infilled_5x5) = {np.nanmedian(point_table['local_bkg_minus_infilled5x5']):.3f}")
            print(f"  median |local_bkg - infilled_5x5| = {np.nanmedian(point_table['abs_local_bkg_minus_infilled5x5']):.3f}")
        print(f"  median residual_value = {np.nanmedian(point_table['residual_value']):.3f}")
        negative = np.sum(point_table['residual_value'] < 0)
        print(f"  negative residual matches = {negative}/{len(point_table)}")


if __name__ == '__main__':
    main()