#!/usr/bin/env python
"""
Foreground extinction analysis via red-clump surface density maps and diagonal Av slices.

This script performs multiple extinction-based CMD analyses on JWST Brick and Cloud C data:
1. Simple narrow color-bin cuts on f182m-f212n
2. F115W-F200W color range analyses
3. F200W-F444W color range analyses
4. F115W-F200W diagonal Av slices (G23)
5. F200W-F444W diagonal Av slices (CT06)
6. Cloud C F182M-F212N diagonal Av slices (G23)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from matplotlib.path import Path
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
from dust_extinction.parameter_averages import G23
from dust_extinction.averages import CT06_MWGC

# Configuration
BASEPATH = '/orange/adamginsburg/jwst/brick'
BRICK_CATALOG_PATH = f'{BASEPATH}/catalogs/basic_merged_indivexp_photometry_tables_merged_ok2221or1182_20251211.fits'
CLOUDC_JWSTPLOTS_PATH = '/home/savannahgramze/orange_link/adamginsburg/jwst/cloudc/lactea-filament/lactea-filament'
OUTDIR = f'{BASEPATH}/distance'
NTH_NEIGHBOR = 10
MAP_PIXELS = 220

y_intercept_f200w = 12.5
y_intercept_f182m = 12.0
y_intercept_f115w = 11.5

os.makedirs(OUTDIR, exist_ok=True)


# ============================================================================
# Utility Functions
# ============================================================================

def load_brick_catalog():
    """Load the Brick catalog."""
    return Table.read(BRICK_CATALOG_PATH)


def load_cloudc_catalog():
    """Load the Cloud C catalog."""
    if CLOUDC_JWSTPLOTS_PATH not in sys.path:
        sys.path.append(CLOUDC_JWSTPLOTS_PATH)
    import jwst_plots
    return jwst_plots.make_cat_use().catalog


def filled_float(column):
    """Convert column to float array, handling masked arrays."""
    if hasattr(column, 'filled'):
        return np.asarray(column.filled(np.nan), dtype=float)
    return np.asarray(column, dtype=float)


def get_skycoord(table):
    """Extract SkyCoord from table, checking multiple column names."""
    if 'skycoord_ref' in table.colnames:
        return SkyCoord(table['skycoord_ref'])
    if 'skycoord_f410m' in table.colnames:
        return SkyCoord(table['skycoord_f410m'])
    if 'ra' in table.colnames and 'dec' in table.colnames:
        return SkyCoord(table['ra'], table['dec'], unit='deg')
    skycoord_cols = [name for name in table.colnames if name.startswith('skycoord_')]
    if len(skycoord_cols) > 0:
        return SkyCoord(table[skycoord_cols[0]])
    raise KeyError('No coordinate column found in table.')


def rc_cut_mask(table):
    """Simple red-clump cut mask: 0.5 < f182m-f212n < 1.0, f182m < 16."""
    mag182 = filled_float(table['mag_ab_f182m'])
    mag212 = filled_float(table['mag_ab_f212n'])
    color = mag182 - mag212
    return np.isfinite(mag182) & np.isfinite(mag212) & (color > 0.5) & (color < 1.0) & (mag182 < 16)


def knn_density_map(table, nth_neighbor=10, map_pixels=220):
    """Compute n-th nearest neighbor surface density map."""
    skycoord = get_skycoord(table)
    x = skycoord.ra.deg
    y = skycoord.dec.deg

    x_edges = np.linspace(np.nanpercentile(x, 0.5), np.nanpercentile(x, 99.5), map_pixels + 1)
    y_edges = np.linspace(np.nanpercentile(y, 0.5), np.nanpercentile(y, 99.5), map_pixels + 1)
    x_cent = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_cent = 0.5 * (y_edges[:-1] + y_edges[1:])
    gx, gy = np.meshgrid(x_cent, y_cent, indexing='xy')
    grid_points = np.column_stack([gx.ravel(), gy.ravel()])

    points = np.column_stack([x, y])
    tree = cKDTree(points)
    dists, _ = tree.query(grid_points, k=nth_neighbor)
    r_n = np.asarray(dists[:, -1], dtype=float)
    r_n = np.maximum(r_n, 1e-6)

    density_deg2 = nth_neighbor / (np.pi * r_n**2)
    density_map = density_deg2.reshape(gx.shape)
    return x_edges, y_edges, density_map


def create_field_figure(
    table_rc,
    table_all,
    x_edges,
    y_edges,
    density_map,
    field_name,
    nth_neighbor,
    vmin,
    vmax,
    color_num_col='mag_ab_f182m',
    color_den_col='mag_ab_f212n',
    mag_col='mag_ab_f182m',
    color_bounds=(0.5, 1.0),
    mag_cut=16,
    cmd_xlim=(0, 2),
    color_label='F182M - F212N (mag)',
    mag_label='F182M (mag)',
    selection_label='Selected stars',
):
    """Create 1x2 figure with density map and CMD."""
    fig, (ax_density, ax_cmd) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    im = ax_density.pcolormesh(x_edges, y_edges, np.log10(density_map), shading='auto', 
                               cmap='viridis', vmin=vmin, vmax=vmax)
    ax_density.set_xlim(np.max(x_edges), np.min(x_edges))
    ax_density.set_aspect('equal')
    ax_density.set_title(f'{field_name} {nth_neighbor}-NN surface density\nN={len(table_rc):,}', fontsize=12)
    ax_density.set_xlabel('RA (deg)')
    ax_density.set_ylabel('Dec (deg)')
    cbar = fig.colorbar(im, ax=ax_density)
    cbar.set_label(r'log10(surface density [deg$^{-2}$])')

    mag_num_all = filled_float(table_all[color_num_col])
    mag_den_all = filled_float(table_all[color_den_col])
    mag_all = filled_float(table_all[mag_col])
    color_all = mag_num_all - mag_den_all
    valid_all = np.isfinite(color_all) & np.isfinite(mag_all)

    mag_num_rc = filled_float(table_rc[color_num_col])
    mag_den_rc = filled_float(table_rc[color_den_col])
    mag_rc = filled_float(table_rc[mag_col])
    color_rc = mag_num_rc - mag_den_rc
    valid_rc = np.isfinite(color_rc) & np.isfinite(mag_rc)

    ax_cmd.scatter(color_all[valid_all], mag_all[valid_all], s=1, alpha=0.1, color='gray', label='All stars')
    ax_cmd.scatter(color_rc[valid_rc], mag_rc[valid_rc], s=3, alpha=0.6, color='blue', label=selection_label)

    ax_cmd.axvline(color_bounds[0], color='red', linestyle='--', linewidth=1.5,
                   label=f'Color cut: {color_bounds[0]:.3g} < color < {color_bounds[1]:.3g}')
    ax_cmd.axvline(color_bounds[1], color='red', linestyle='--', linewidth=1.5)
    ax_cmd.axhline(mag_cut, color='orange', linestyle='--', linewidth=1.5,
                   label=f'Magnitude cut: < {mag_cut}')

    ax_cmd.set_xlabel(color_label, fontsize=11)
    ax_cmd.set_ylabel(mag_label, fontsize=11)
    ax_cmd.set_title(f'{field_name} Color-Magnitude Diagram', fontsize=12)
    ax_cmd.invert_yaxis()
    ax_cmd.legend(loc='upper right', fontsize=9)
    ax_cmd.grid(alpha=0.3)
    ax_cmd.set_xlim(*cmd_xlim)

    return fig


def av_tag(value):
    """Convert float to filename-safe tag."""
    return f'{value:.1f}'.replace('.', 'p')


def ordered_closed_polygon(vertices):
    """Return vertices ordered by polar angle and explicitly closed."""
    centroid = vertices.mean(axis=0)
    angles = np.arctan2(vertices[:, 1] - centroid[1], vertices[:, 0] - centroid[0])
    order = np.argsort(angles)
    ordered = vertices[order]
    return np.vstack([ordered, ordered[0]])


def run_polygon_av_slices(
    table,
    color,
    mag,
    base_mask,
    k_num,
    k_den,
    y_intercept,
    width,
    av_step,
    av_max_potential,
    nth_neighbor,
    outdir,
    outprefix,
    field_label,
    cmd_xlabel,
    cmd_ylabel,
    cmd_xlim,
    cmd_ylim=None,
):
    """Run Av-slice selection using ordered, closed polygon containment."""
    k_color = k_num - k_den
    k_mag = k_num
    vv = k_color**2 + k_mag**2
    av_coord = ((color * k_color) + ((mag - y_intercept) * k_mag)) / vv
    points = np.column_stack((color, mag))
    av_max_data = np.nanmax(av_coord[base_mask])

    av_bins = []
    av_lo = 0.0
    while av_lo < av_max_potential:
        av_hi = av_lo + av_step

        blc = av_lo * (k_num - k_den), y_intercept + width + av_lo * k_num
        tlc = av_lo * (k_num - k_den), y_intercept - width + av_lo * k_num
        brc = av_hi * (k_num - k_den), y_intercept + width + av_hi * k_num
        trc = av_hi * (k_num - k_den), y_intercept - width + av_hi * k_num

        polygon_vertices = np.array([blc, tlc, trc, brc], dtype=float)
        polygon_vertices_closed = ordered_closed_polygon(polygon_vertices)
        poly_path = Path(polygon_vertices_closed)
        in_bin = base_mask & poly_path.contains_points(points)
        n_selected = int(in_bin.sum())

        av_bins.append((av_lo, av_hi, n_selected, polygon_vertices_closed))

        if av_lo > av_max_data:
            break
        av_lo = av_hi

    if len(av_bins) == 0:
        print('WARNING: No Av bins found in range!')
        return

    av_bins_with_data = [
        (av_lo, av_hi, polygon_vertices)
        for av_lo, av_hi, n, polygon_vertices in av_bins
        if n >= nth_neighbor
    ]

    print(f'Av bins (step {av_step} mag): Av={av_bins[0][0]:.0f} to {av_bins[-1][1]:.0f}')
    print(f'Total bins: {len(av_bins)}, Bins with data (N≥{nth_neighbor}): {len(av_bins_with_data)}')

    print(f'\nDiagnostic: Av bin statistics')
    print(f'{"Av_lo":>6} {"Av_hi":>6} {"N_stars":>10}')
    print('-' * 30)
    for av_lo, av_hi, n, _ in av_bins[::3]:
        print(f'{av_lo:6.0f} {av_hi:6.0f} {n:10,d}')
    print(f'(showing every 3rd bin; empty bins indicated by N=0)')

    for av_lo, av_hi, polygon_vertices in av_bins_with_data:
        poly_path = Path(polygon_vertices)
        in_bin = base_mask & poly_path.contains_points(points)
        table_slice = table[in_bin]

        if len(table_slice) < nth_neighbor:
            continue

        xe, ye, dens = knn_density_map(table_slice, nth_neighbor=nth_neighbor, map_pixels=MAP_PIXELS)
        log_dens = np.log10(dens[np.isfinite(dens)])
        vmin = np.nanpercentile(log_dens, 5)
        vmax = np.nanpercentile(log_dens, 99)

        fig, (ax_dens, ax_cmd) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

        im = ax_dens.pcolormesh(xe, ye, np.log10(dens), shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
        ax_dens.set_xlim(np.max(xe), np.min(xe))
        ax_dens.set_aspect('equal')
        ax_dens.set_title(f'{field_label} Av [{av_lo:.0f}, {av_hi:.0f})\nN={len(table_slice):,}')
        ax_dens.set_xlabel('RA (deg)')
        ax_dens.set_ylabel('Dec (deg)')
        cbar = fig.colorbar(im, ax=ax_dens)
        cbar.set_label(r'log10(surface density [deg$^{-2}$])')

        ax_cmd.scatter(color[base_mask], mag[base_mask], s=1, alpha=0.1, color='gray', label='All stars')
        ax_cmd.scatter(color[in_bin], mag[in_bin], s=3, alpha=0.8, color='blue', label=f'Av [{av_lo:.0f}, {av_hi:.0f}) selection')

        poly_x = polygon_vertices[:, 0]
        poly_y = polygon_vertices[:, 1]
        ax_cmd.plot(poly_x, poly_y, color='red', linestyle='--', linewidth=1.8, label=f'Av bin ±{width:.2f} mag')

        ax_cmd.set_xlabel(cmd_xlabel)
        ax_cmd.set_ylabel(cmd_ylabel)
        ax_cmd.set_title(f'{field_label} Color-Magnitude Diagram')
        ax_cmd.invert_yaxis()
        ax_cmd.grid(alpha=0.3)
        ax_cmd.set_xlim(*cmd_xlim)
        if cmd_ylim is not None:
            ax_cmd.set_ylim(*cmd_ylim)
        ax_cmd.legend(loc='upper right', fontsize=9)

        outfile = os.path.join(outdir, f'{outprefix}_av_{av_tag(av_lo)}_{av_tag(av_hi)}_knn{nth_neighbor}.png')
        fig.savefig(outfile, dpi=220)
        plt.close(fig)
        print(f'{field_label} Av=[{av_lo:.0f}, {av_hi:.0f}): {len(table_slice):,} stars → {outfile}')

    fig_cmd, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(color[base_mask], mag[base_mask], s=1, alpha=0.05, color='gray', label='All stars')

    color_range = np.array([cmd_xlim[0], cmd_xlim[1]])
    mag_range = y_intercept + (k_mag / k_color) * color_range
    ax.plot(color_range, mag_range, 'k--', lw=2.0, label='Extinction-aligned reference')

    for av_lo, av_hi, n, polygon_vertices in av_bins[::2]:
        poly_x = polygon_vertices[:, 0]
        poly_y = polygon_vertices[:, 1]
        poly_color = 'red' if n >= nth_neighbor else 'lightgray'
        linewidth = 1.5 if n >= nth_neighbor else 0.8
        ax.plot(poly_x, poly_y, color=poly_color, alpha=0.4, lw=linewidth)

        mid_color = np.mean(polygon_vertices[:-1, 0])
        mid_mag = np.mean(polygon_vertices[:-1, 1])
        ax.text(
            mid_color,
            mid_mag,
            f'{av_lo:.0f}\n({n})',
            ha='center',
            fontsize=7,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7 if n >= nth_neighbor else 0.3),
        )

    ax.set_xlabel(cmd_xlabel, fontsize=11)
    ax.set_ylabel(cmd_ylabel, fontsize=11)
    ax.set_title(f'{field_label}: Av Bins (Red=has data, Gray=empty)', fontsize=12)
    ax.invert_yaxis()
    ax.grid(alpha=0.3)
    ax.set_xlim(*cmd_xlim)
    if cmd_ylim is not None:
        ax.set_ylim(*cmd_ylim)
    ax.legend(loc='upper right', fontsize=10)

    outfile_cmd = os.path.join(outdir, f'{outprefix}_av_slices_summary_y{av_tag(y_intercept)}.png')
    fig_cmd.savefig(outfile_cmd, dpi=220)
    plt.close(fig_cmd)
    print(f'\nSummary CMD plot: {outfile_cmd}')


# ============================================================================
# Analysis Functions
# ============================================================================

def analyze_simple_narrow_color_bins(brick_all, cloudc_all, nth_neighbor=NTH_NEIGHBOR, outdir=OUTDIR):
    """
    Analyze narrow F182M-F212N color bins for both Brick and Cloud C.
    
    Uses simple magnitude cuts on fixed color ranges without extinction modeling.
    """
    print("\n" + "="*80)
    print("ANALYSIS 1: Simple Narrow Color Bins (F182M-F212N)")
    print("="*80)
    
    color_bins = [(0.5, 0.625, '0.5-0.625'), (0.625, 0.75, '0.625-0.75')]
    mag_cuts = [16, 18]

    for mag_cut in mag_cuts:
        for color_lo, color_hi, color_label in color_bins:
            # Brick
            mag182_b = filled_float(brick_all['mag_ab_f182m'])
            mag212_b = filled_float(brick_all['mag_ab_f212n'])
            color_b = mag182_b - mag212_b
            brick_mask_narrow = (np.isfinite(mag182_b) & np.isfinite(mag212_b) & 
                                 (color_b >= color_lo) & (color_b < color_hi) & 
                                 (mag182_b < mag_cut))
            brick_narrow = brick_all[brick_mask_narrow]
            
            # Cloud C
            mag182_c = filled_float(cloudc_all['mag_ab_f182m'])
            mag212_c = filled_float(cloudc_all['mag_ab_f212n'])
            color_c = mag182_c - mag212_c
            cloudc_mask_narrow = (np.isfinite(mag182_c) & np.isfinite(mag212_c) & 
                                  (color_c >= color_lo) & (color_c < color_hi) & 
                                  (mag182_c < mag_cut))
            cloudc_narrow = cloudc_all[cloudc_mask_narrow]
            
            # Compute density maps
            if len(brick_narrow) >= nth_neighbor:
                brick_xe_n, brick_ye_n, brick_density_n = knn_density_map(brick_narrow, 
                                                                           nth_neighbor=nth_neighbor, 
                                                                           map_pixels=MAP_PIXELS)
            else:
                brick_xe_n = brick_ye_n = brick_density_n = None
                
            if len(cloudc_narrow) >= nth_neighbor:
                cloudc_xe_n, cloudc_ye_n, cloudc_density_n = knn_density_map(cloudc_narrow, 
                                                                              nth_neighbor=nth_neighbor, 
                                                                              map_pixels=MAP_PIXELS)
            else:
                cloudc_xe_n = cloudc_ye_n = cloudc_density_n = None
            
            # Compute shared color scale
            if brick_density_n is not None and cloudc_density_n is not None:
                all_log_n = np.concatenate([
                    np.log10(brick_density_n[np.isfinite(brick_density_n)]),
                    np.log10(cloudc_density_n[np.isfinite(cloudc_density_n)]),
                ])
                vmin_n = np.nanpercentile(all_log_n, 5)
                vmax_n = np.nanpercentile(all_log_n, 99)
            else:
                vmin_n = vmax_n = 0
            
            # Brick figure
            if brick_density_n is not None and len(brick_narrow) > 0:
                fig_b = create_field_figure(brick_narrow, brick_all, brick_xe_n, brick_ye_n, 
                                            brick_density_n, f'Brick (f182m < {mag_cut})', 
                                            nth_neighbor, vmin_n, vmax_n)
                outpath_b = f'{outdir}/rc_colorcut_{color_label}_mag{mag_cut}_knn{nth_neighbor}_brick.png'
                fig_b.savefig(outpath_b, dpi=220)
                plt.close(fig_b)
                print(f'Brick f182m-f212n=[{color_lo}, {color_hi}], f182m<{mag_cut}: {len(brick_narrow):,} stars → {outpath_b}')
            else:
                print(f'Brick f182m-f212n=[{color_lo}, {color_hi}], f182m<{mag_cut}: insufficient stars')
            
            # Cloud C figure
            if cloudc_density_n is not None and len(cloudc_narrow) > 0:
                fig_c = create_field_figure(cloudc_narrow, cloudc_all, cloudc_xe_n, cloudc_ye_n, 
                                            cloudc_density_n, f'Cloud C (f182m < {mag_cut})', 
                                            nth_neighbor, vmin_n, vmax_n)
                outpath_c = f'{outdir}/rc_colorcut_{color_label}_mag{mag_cut}_knn{nth_neighbor}_cloudc.png'
                fig_c.savefig(outpath_c, dpi=220)
                plt.close(fig_c)
                print(f'Cloud C f182m-f212n=[{color_lo}, {color_hi}], f182m<{mag_cut}: {len(cloudc_narrow):,} stars → {outpath_c}')
            else:
                print(f'Cloud C f182m-f212n=[{color_lo}, {color_hi}], f182m<{mag_cut}: insufficient stars')


def analyze_f115w_f200w_color_range(brick_all, nth_neighbor=NTH_NEIGHBOR, outdir=OUTDIR):
    """
    Analyze F115W-F200W color range for Brick: full range and narrow bins.
    """
    print("\n" + "="*80)
    print("ANALYSIS 2: F115W-F200W Color Range Analysis (Brick)")
    print("="*80)
    
    mag115w = filled_float(brick_all['mag_ab_f115w'])
    mag200w = filled_float(brick_all['mag_ab_f200w'])
    color_115_200 = mag115w - mag200w

    brick_mask_115_200_full = (
        np.isfinite(mag115w)
        & np.isfinite(mag200w)
        & (color_115_200 > 2.0)
        & (color_115_200 < 8.0)
        & (mag115w < 18)
    )

    brick_115_200_full = brick_all[brick_mask_115_200_full]

    # Full-range figure
    if len(brick_115_200_full) >= nth_neighbor:
        brick_xe_115_200, brick_ye_115_200, brick_density_115_200 = knn_density_map(
            brick_115_200_full, nth_neighbor=nth_neighbor, map_pixels=MAP_PIXELS
        )

        log_dens_115_200 = np.log10(brick_density_115_200[np.isfinite(brick_density_115_200)])
        vmin_115_200 = np.nanpercentile(log_dens_115_200, 5)
        vmax_115_200 = np.nanpercentile(log_dens_115_200, 99)

        fig_115_200 = create_field_figure(
            brick_115_200_full, brick_all, brick_xe_115_200, brick_ye_115_200, 
            brick_density_115_200, 'Brick F115W-F200W (f115w < 18)', nth_neighbor, 
            vmin_115_200, vmax_115_200, color_num_col='mag_ab_f115w', 
            color_den_col='mag_ab_f200w', mag_col='mag_ab_f115w', 
            color_bounds=(2.0, 8.0), mag_cut=18, cmd_xlim=(-1, 10),
            color_label='F115W - F200W (mag)', mag_label='F115W (mag)',
            selection_label='F115W-F200W selected',
        )

        outpath_115_200 = f'{outdir}/rc_115w200w_mag18_knn{nth_neighbor}_brick.png'
        fig_115_200.savefig(outpath_115_200, dpi=220)
        plt.close(fig_115_200)
        print(f'Brick F115W-F200W [2.0, 8.0], f115w<18: {len(brick_115_200_full):,} stars → {outpath_115_200}')
    else:
        print(f'Brick F115W-F200W [2.0, 8.0], f115w<18: insufficient stars')

    # Narrow bins
    color_bins_115_200 = [(4.0, 5.0, '4.0-5.0'), (5.0, 6.0, '5.0-6.0')]
    mag_cuts_115w = [20, 21]

    for mag_cut in mag_cuts_115w:
        for color_lo, color_hi, color_label in color_bins_115_200:
            mag115w_b = filled_float(brick_all['mag_ab_f115w'])
            mag200w_b = filled_float(brick_all['mag_ab_f200w'])
            color_b = mag115w_b - mag200w_b
            brick_mask_narrow_115_200 = (
                np.isfinite(mag115w_b) & np.isfinite(mag200w_b) & 
                (color_b >= color_lo) & (color_b < color_hi) & 
                (mag115w_b < mag_cut)
            )
            brick_narrow_115_200 = brick_all[brick_mask_narrow_115_200]

            if len(brick_narrow_115_200) >= nth_neighbor:
                brick_xe_n, brick_ye_n, brick_density_n = knn_density_map(
                    brick_narrow_115_200, nth_neighbor=nth_neighbor, map_pixels=MAP_PIXELS,
                )
                log_dens_n = np.log10(brick_density_n[np.isfinite(brick_density_n)])
                vmin_n = np.nanpercentile(log_dens_n, 5)
                vmax_n = np.nanpercentile(log_dens_n, 99)

                fig_b = create_field_figure(
                    brick_narrow_115_200, brick_all, brick_xe_n, brick_ye_n, brick_density_n,
                    f'Brick F115W-F200W (f115w < {mag_cut})', nth_neighbor, vmin_n, vmax_n,
                    color_num_col='mag_ab_f115w', color_den_col='mag_ab_f200w', 
                    mag_col='mag_ab_f115w', color_bounds=(color_lo, color_hi), 
                    mag_cut=mag_cut, cmd_xlim=(-1, 10),
                    color_label='F115W - F200W (mag)', mag_label='F115W (mag)',
                    selection_label='F115W-F200W selected',
                )

                outpath_b = f'{outdir}/rc_115w200w_{color_label}_mag{mag_cut}_knn{nth_neighbor}_brick.png'
                fig_b.savefig(outpath_b, dpi=220)
                plt.close(fig_b)
                print(f'Brick F115W-F200W=[{color_lo}, {color_hi}], f115w<{mag_cut}: {len(brick_narrow_115_200):,} stars → {outpath_b}')
            else:
                print(f'Brick F115W-F200W=[{color_lo}, {color_hi}], f115w<{mag_cut}: insufficient stars')


def analyze_f200w_f444w_color_range(brick_all, nth_neighbor=NTH_NEIGHBOR, outdir=OUTDIR):
    """
    Analyze F200W-F444W color range for Brick: full range and narrow bins.
    """
    print("\n" + "="*80)
    print("ANALYSIS 3: F200W-F444W Color Range Analysis (Brick)")
    print("="*80)
    
    mag200w = filled_float(brick_all['mag_ab_f200w'])
    mag444w = filled_float(brick_all['mag_ab_f444w'])
    color_200_444 = mag200w - mag444w

    brick_mask_200_444_full = (
        np.isfinite(mag200w) & np.isfinite(mag444w) & 
        (color_200_444 > 0.5) & (color_200_444 < 4.0) & 
        (mag200w < 17)
    )

    brick_200_444_full = brick_all[brick_mask_200_444_full]

    # Full-range figure
    if len(brick_200_444_full) >= nth_neighbor:
        brick_xe_200_444, brick_ye_200_444, brick_density_200_444 = knn_density_map(
            brick_200_444_full, nth_neighbor=nth_neighbor, map_pixels=MAP_PIXELS
        )

        log_dens_200_444 = np.log10(brick_density_200_444[np.isfinite(brick_density_200_444)])
        vmin_200_444 = np.nanpercentile(log_dens_200_444, 5)
        vmax_200_444 = np.nanpercentile(log_dens_200_444, 99)

        fig_200_444 = create_field_figure(
            brick_200_444_full, brick_all, brick_xe_200_444, brick_ye_200_444, 
            brick_density_200_444, 'Brick F200W-F444W (f200w < 18)', nth_neighbor, 
            vmin_200_444, vmax_200_444, color_num_col='mag_ab_f200w', 
            color_den_col='mag_ab_f444w', mag_col='mag_ab_f200w', 
            color_bounds=(0.5, 4.0), mag_cut=18, cmd_xlim=(-0.5, 5),
            color_label='F200W - F444W (mag)', mag_label='F200W (mag)',
            selection_label='F200W-F444W selected',
        )

        outpath_200_444 = f'{outdir}/rc_200w444w_mag18_knn{nth_neighbor}_brick.png'
        fig_200_444.savefig(outpath_200_444, dpi=220)
        plt.close(fig_200_444)
        print(f'Brick F200W-F444W [0.5, 4.0], f200w<18: {len(brick_200_444_full):,} stars → {outpath_200_444}')
    else:
        print(f'Brick F200W-F444W [0.5, 4.0], f200w<18: insufficient stars')

    # Narrow bins
    color_bins_200_444 = [(1.2, 1.5, '1.2-1.5'), (1.5, 1.9, '1.5-1.9'), (1.9, 3.0, '1.9-3.0')]
    mag_cuts_200w = [16, 17]

    for mag_cut in mag_cuts_200w:
        for color_lo, color_hi, color_label in color_bins_200_444:
            mag200w_b = filled_float(brick_all['mag_ab_f200w'])
            mag444w_b = filled_float(brick_all['mag_ab_f444w'])
            color_b = mag200w_b - mag444w_b
            brick_mask_narrow_200_444 = (
                np.isfinite(mag200w_b) & np.isfinite(mag444w_b) & 
                (color_b >= color_lo) & (color_b < color_hi) & 
                (mag200w_b < mag_cut)
            )
            brick_narrow_200_444 = brick_all[brick_mask_narrow_200_444]

            if len(brick_narrow_200_444) >= nth_neighbor:
                brick_xe_n, brick_ye_n, brick_density_n = knn_density_map(
                    brick_narrow_200_444, nth_neighbor=nth_neighbor, map_pixels=MAP_PIXELS,
                )
                log_dens_n = np.log10(brick_density_n[np.isfinite(brick_density_n)])
                vmin_n = np.nanpercentile(log_dens_n, 5)
                vmax_n = np.nanpercentile(log_dens_n, 99)

                fig_b = create_field_figure(
                    brick_narrow_200_444, brick_all, brick_xe_n, brick_ye_n, brick_density_n,
                    f'Brick F200W-F444W (f200w < {mag_cut})', nth_neighbor, vmin_n, vmax_n,
                    color_num_col='mag_ab_f200w', color_den_col='mag_ab_f444w', 
                    mag_col='mag_ab_f200w', color_bounds=(color_lo, color_hi), 
                    mag_cut=mag_cut, cmd_xlim=(-0.5, 5),
                    color_label='F200W - F444W (mag)', mag_label='F200W (mag)',
                    selection_label='F200W-F444W selected',
                )

                outpath_b = f'{outdir}/rc_200w444w_{color_label}_mag{mag_cut}_knn{nth_neighbor}_brick.png'
                fig_b.savefig(outpath_b, dpi=220)
                plt.close(fig_b)
                print(f'Brick F200W-F444W=[{color_lo}, {color_hi}], f200w<{mag_cut}: {len(brick_narrow_200_444):,} stars → {outpath_b}')
            else:
                print(f'Brick F200W-F444W=[{color_lo}, {color_hi}], f200w<{mag_cut}: insufficient stars')


def analyze_f115w_f200w_diagonal_av(brick_all, nth_neighbor=NTH_NEIGHBOR, outdir=OUTDIR):
    """
    Analyze F115W-F200W with diagonal Av slices using G23 (Rv=3.1) extinction law.
    """
    print("\n" + "="*80)
    print("ANALYSIS 4: F115W-F200W Diagonal Av Slices (G23, Rv=3.1, Brick)")
    print("="*80)
    
    mag115w = filled_float(brick_all['mag_ab_f115w'])
    mag200w = filled_float(brick_all['mag_ab_f200w'])
    color_115_200 = mag115w - mag200w

    base_mask = np.isfinite(mag115w) & np.isfinite(mag200w) & np.isfinite(color_115_200) & (color_115_200 > 3.0) & (color_115_200 < 10.0)

    g23 = G23(Rv=3.1)
    k115 = float(g23(1.15 * u.um))
    k200 = float(g23(2.00 * u.um))

    k_color = k115 - k200
    k_mag = k115
    y_intercept = y_intercept_f115w # hard-coded by user, don't mess with it

    print(f'G23 (Rv=3.1): A115={k115:.4f}, A200={k200:.4f}, dcolor/dAv={k_color:.4f}, dmag/dAv={k_mag:.4f}')
    print(f'Hard-coded y-intercept at color=0: {y_intercept:.3f}')

    run_polygon_av_slices(
        table=brick_all,
        color=color_115_200,
        mag=mag115w,
        base_mask=base_mask,
        k_num=k115,
        k_den=k200,
        y_intercept=y_intercept,
        width=0.5,
        av_step=2.0,
        av_max_potential=50.0,
        nth_neighbor=nth_neighbor,
        outdir=outdir,
        outprefix='rc_115w200w_g23_brick',
        field_label='Brick F115W-F200W (G23)',
        cmd_xlabel='F115W - F200W (mag)',
        cmd_ylabel='F115W (mag)',
        cmd_xlim=(3, 10),
    )


def analyze_f182m_f212n_av_slices_ct06(brick_all, cloudc_all, nth_neighbor=NTH_NEIGHBOR, outdir=OUTDIR):
    """
    Analyze F182M-F212N with Av slices using CT06 for both Brick and Cloud C.
    """
    print("\n" + "="*80)
    print("ANALYSIS 6: F182M-F212N Av Slices (CT06, Brick + Cloud C)")
    print("="*80)

    extinction_ct06 = CT06_MWGC()
    k182 = float(extinction_ct06(1.82 * u.um))
    k212 = float(extinction_ct06(2.12 * u.um))
    k_color = k182 - k212
    k_mag = k182

    print(f'CT06: A182={k182:.4f}, A212={k212:.4f}, dcolor/dAv={k_color:.4f}, dmag/dAv={k_mag:.4f}')

    mag182_b = filled_float(brick_all['mag_ab_f182m'])
    mag212_b = filled_float(brick_all['mag_ab_f212n'])
    color_b = mag182_b - mag212_b
    base_mask_b = np.isfinite(mag182_b) & np.isfinite(mag212_b) & np.isfinite(color_b) & (color_b > 0.2) & (color_b < 1.8) & (mag182_b > 13.0) & (mag182_b < 20.0)
    y_intercept_b = float(np.nanmedian(mag182_b[base_mask_b] - (k_mag / k_color) * color_b[base_mask_b]))
    y_intercept_b = y_intercept_f182m
    print(f'Brick hard-coded y-intercept at color=0: {y_intercept_b:.3f}')

    run_polygon_av_slices(
        table=brick_all,
        color=color_b,
        mag=mag182_b,
        base_mask=base_mask_b,
        k_num=k182,
        k_den=k212,
        y_intercept=y_intercept_b,
        width=0.75,
        av_step=2.0,
        av_max_potential=50.0,
        nth_neighbor=nth_neighbor,
        outdir=outdir,
        outprefix='rc_182m212n_ct06_brick',
        field_label='Brick F182M-F212N (CT06)',
        cmd_xlabel='F182M - F212N (mag)',
        cmd_ylabel='F182M (mag)',
        cmd_xlim=(0.0, 1.8),
        cmd_ylim=(20.0, 13.0),
    )

    mag182_c = filled_float(cloudc_all['mag_ab_f182m'])
    mag212_c = filled_float(cloudc_all['mag_ab_f212n'])
    color_c = mag182_c - mag212_c
    base_mask_c = np.isfinite(mag182_c) & np.isfinite(mag212_c) & np.isfinite(color_c) & (color_c > 0.2) & (color_c < 1.8) & (mag182_c > 13.0) & (mag182_c < 20.0)
    y_intercept_c = y_intercept_f182m
    print(f'Cloud C hard-coded y-intercept at color=0: {y_intercept_c:.3f}')

    run_polygon_av_slices(
        table=cloudc_all,
        color=color_c,
        mag=mag182_c,
        base_mask=base_mask_c,
        k_num=k182,
        k_den=k212,
        y_intercept=y_intercept_c,
        width=0.75,
        av_step=2.0,
        av_max_potential=50.0,
        nth_neighbor=nth_neighbor,
        outdir=outdir,
        outprefix='rc_182m212n_ct06_cloudc',
        field_label='Cloud C F182M-F212N (CT06)',
        cmd_xlabel='F182M - F212N (mag)',
        cmd_ylabel='F182M (mag)',
        cmd_xlim=(0.0, 1.8),
        cmd_ylim=(20.0, 13.0),
    )


def analyze_f200w_f444w_av_slices_ct06(brick_all, nth_neighbor=NTH_NEIGHBOR, outdir=OUTDIR):
    """
    Analyze F200W-F444W with Av slices using CT06 extinction curve.
    
    The reference line (1.25, 14.5) -> (4.0, 20.0) defines the red-clump locus.
    Starting from Y-intercept ≈ 12.0 (where F200W-F444W=0), we apply CT06 to create
    Av-dependent selection strips from Av=0 until we run out of stars, in steps of 2 mag.
    
    Each bin selects stars in the range [Av_lo, Av_hi) along the extinction vector,
    constrained to be within ±0.25 mag perpendicular to the vector.
    
    Validation: the extinction vector (k_mag/k_color) should match the reference line slope.
    
    Output: Individual 1x2 figures (density map + CMD) for each Av slice.
    """
    print("\n" + "="*80)
    print("ANALYSIS 5: F200W-F444W Av-Slice Selection (CT06, Brick)")
    print("="*80)

    mag200w = filled_float(brick_all['mag_ab_f200w'])
    mag444w = filled_float(brick_all['mag_ab_f444w'])
    color_200_444 = mag200w - mag444w

    base_mask = np.isfinite(mag200w) & np.isfinite(mag444w) & np.isfinite(color_200_444)

    # Reference line: (1.25, 14.5) -> (4.0, 20.0) in (F200W-F444W, F200W) space
    # all data that are relevant lie within these lines, but they are ONLY a reference, they're note meant to be used.
    p1_color, p1_mag = 1.25, 14.5
    p2_color, p2_mag = 4.0, 20.0
    ref_slope = (p2_mag - p1_mag) / (p2_color - p1_color)
    y_intercept = p1_mag - p1_color * ref_slope  # F200W at color=0
    y_intercept = y_intercept_f200w

    # CT06 extinction model
    extinction_ct06 = CT06_MWGC()
    # Get k-values (A_lambda per unit A_V)
    k_200w = float(extinction_ct06(2.00 * u.um))
    k_444w = float(extinction_ct06(4.44 * u.um))
    k_color = k_200w - k_444w
    k_mag = k_200w
    vv = k_color**2 + k_mag**2
    av_coord = ((color_200_444 * k_color) + ((mag200w - y_intercept) * k_mag)) / vv

    print(f'Reference line Y-intercept: {y_intercept:.2f} mag (at F200W-F444W=0)')
    print(f'Reference line slope: {ref_slope:.4f}')
    print(f'CT06 k-values: k_color={k_color:.4f}, k_mag={k_mag:.4f}')
    print(f'Validation (k_mag/k_color): {k_mag/k_color:.4f} (should ≈ {ref_slope:.4f})')

    points = np.column_stack((color_200_444, mag200w))

    # Identify Av bins by scanning from Av=0 upward
    av_step = 2.0
    av_max_potential = 50.0
    width = 0.5
    av_max_data = np.nanmax(av_coord[base_mask])

    # List of (av_lo, av_hi, n_selected, polygon_vertices)
    av_bins = []
    av_lo = 0.0
    while av_lo < av_max_potential:
        av_hi = av_lo + av_step

        blc = av_lo * (k_200w - k_444w), y_intercept + width + av_lo * k_200w
        tlc = av_lo * (k_200w - k_444w), y_intercept - width + av_lo * k_200w
        brc = av_hi * (k_200w - k_444w), y_intercept + width + av_hi * k_200w
        trc = av_hi * (k_200w - k_444w), y_intercept - width + av_hi * k_200w

        polygon_vertices = np.array([blc, tlc, trc, brc], dtype=float)
        polygon_vertices_closed = ordered_closed_polygon(polygon_vertices)
        poly_path = Path(polygon_vertices_closed)
        in_bin = base_mask & poly_path.contains_points(points)
        n_selected = int(in_bin.sum())

        # Include bin even if empty - we'll skip it when creating figures, but keep contiguity
        av_bins.append((av_lo, av_hi, n_selected, polygon_vertices_closed))

        if av_lo > av_max_data:
            break

        av_lo = av_hi
    
    if len(av_bins) == 0:
        print('WARNING: No Av bins found in range!')
        return
    
    # Filter to bins with data
    av_bins_with_data = [
        (av_lo, av_hi, polygon_vertices)
        for av_lo, av_hi, n, polygon_vertices in av_bins
        if n >= nth_neighbor
    ]
    
    n_av_slices = len(av_bins_with_data)
    print(f'Av bins (step {av_step} mag): Av={av_bins[0][0]:.0f} to {av_bins[-1][1]:.0f}')
    print(f'Total bins: {len(av_bins)}, Bins with data (N≥{nth_neighbor}): {n_av_slices}')
    
    # Diagnostic: show bin statistics for all bins
    print(f'\nDiagnostic: Av bin statistics')
    print(f'{"Av_lo":>6} {"Av_hi":>6} {"N_stars":>10}')
    print('-' * 30)
    for av_lo, av_hi, n, _ in av_bins[::3]:  # Every 3rd bin for brevity
        print(f'{av_lo:6.0f} {av_hi:6.0f} {n:10,d}')
    print(f'(showing every 3rd bin; empty bins indicated by N=0)')

    # Create individual 1x2 figures for each Av bin with data
    for av_lo, av_hi, polygon_vertices in av_bins_with_data:
        poly_path = Path(polygon_vertices)
        in_bin = base_mask & poly_path.contains_points(points)
        brick_av_slice = brick_all[in_bin]

        if len(brick_av_slice) < nth_neighbor:
            continue

        # Density map
        brick_xe, brick_ye, brick_dens = knn_density_map(
            brick_av_slice,
            nth_neighbor=nth_neighbor,
            map_pixels=MAP_PIXELS,
        )
        log_dens = np.log10(brick_dens[np.isfinite(brick_dens)])
        vmin = np.nanpercentile(log_dens, 5)
        vmax = np.nanpercentile(log_dens, 99)

        # Create 1x2 figure
        fig, (ax_dens, ax_cmd) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

        # Left: Density map
        im = ax_dens.pcolormesh(
            brick_xe, brick_ye, np.log10(brick_dens),
            shading='auto', cmap='viridis', vmin=vmin, vmax=vmax,
        )
        ax_dens.set_xlim(np.max(brick_xe), np.min(brick_xe))
        ax_dens.set_aspect('equal')
        ax_dens.set_title(f'Brick F200W-F444W Av [{av_lo:.0f}, {av_hi:.0f})\nN={len(brick_av_slice):,}')
        ax_dens.set_xlabel('RA (deg)')
        ax_dens.set_ylabel('Dec (deg)')
        cbar = fig.colorbar(im, ax=ax_dens)
        cbar.set_label(r'log10(surface density [deg$^{-2}$])')

        # Right: CMD
        cmd_mask_all = base_mask
        ax_cmd.scatter(color_200_444[cmd_mask_all], mag200w[cmd_mask_all], s=1, alpha=0.1,
                      color='gray', label='All stars')
        ax_cmd.scatter(color_200_444[in_bin], mag200w[in_bin], s=3, alpha=0.8,
                      color='blue', label=f'Av [{av_lo:.0f}, {av_hi:.0f}) selection')

        # Draw polygon showing Av-bin boundaries in CMD space
        poly_x = polygon_vertices[:, 0]
        poly_y = polygon_vertices[:, 1]
        ax_cmd.plot(poly_x, poly_y, color='red', linestyle='--', linewidth=1.8, label=f'Av bin ±{width:.2f} mag')

        # Also draw reference line for context
        color_range = np.array([0, 4.5])
        mag_range = y_intercept + ref_slope * color_range
        ax_cmd.plot(color_range, mag_range, 'k:', lw=1.5, alpha=0.5, label='Reference line')

        ax_cmd.set_xlabel('F200W - F444W (mag)')
        ax_cmd.set_ylabel('F200W (mag)')
        ax_cmd.set_title('Brick Color-Magnitude Diagram')
        ax_cmd.invert_yaxis()
        ax_cmd.grid(alpha=0.3)
        ax_cmd.set_xlim(-0.5, 4.5)
        ax_cmd.legend(loc='upper right', fontsize=9)

        outfile = os.path.join(outdir, f'rc_200w444w_av_{av_tag(av_lo)}_{av_tag(av_hi)}_ct06_knn{nth_neighbor}_brick.png')
        fig.savefig(outfile, dpi=220)
        plt.close(fig)
        print(f'Brick F200W-F444W Av=[{av_lo:.0f}, {av_hi:.0f}): {len(brick_av_slice):,} stars → {outfile}')

    # Create summary CMD plot showing all Av bins
    fig_cmd, ax = plt.subplots(figsize=(10, 8))
    
    # Plot all stars in gray
    ax.scatter(color_200_444[base_mask], mag200w[base_mask], s=1, alpha=0.05, color='gray', label='All stars')
    
    # Plot reference line
    color_range = np.array([0, 4.5])
    mag_range = y_intercept + ref_slope * color_range
    ax.plot(color_range, mag_range, 'k--', lw=2.5, label='Reference line')
    
    # Draw all Av bin rectangles (show contiguity, color by whether they have data)
    for av_lo, av_hi, n, polygon_vertices in av_bins[::2]:  # Every other bin for readability
        poly_x = polygon_vertices[:, 0]
        poly_y = polygon_vertices[:, 1]
        
        # Color: red if has data, light gray if empty
        color = 'red' if n >= nth_neighbor else 'lightgray'
        linewidth = 1.5 if n >= nth_neighbor else 0.8
        ax.plot(poly_x, poly_y, color=color, alpha=0.4, lw=linewidth)
        
        # Label with star count
        mid_color = np.mean(polygon_vertices[:-1, 0])
        mid_mag = np.mean(polygon_vertices[:-1, 1])
        label_text = f'{av_lo:.0f}\n({n})'
        ax.text(mid_color, mid_mag, label_text, ha='center', fontsize=7,
               bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7 if n >= nth_neighbor else 0.3))
    
    ax.set_xlabel('F200W - F444W (mag)', fontsize=11)
    ax.set_ylabel('F200W (mag)', fontsize=11)
    ax.set_title('Brick F200W-F444W: Av Bins vs Reference Line (CT06)\n(Red=has data, Gray=empty)', fontsize=12)
    ax.invert_yaxis()
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(22, 10)
    ax.legend(loc='upper right', fontsize=10)
    
    outfile_cmd = os.path.join(outdir, f'rc_200w444w_av_slices_summary_y{av_tag(y_intercept)}_ct06_brick.png')
    fig_cmd.savefig(outfile_cmd, dpi=220)
    plt.close(fig_cmd)
    
    print(f'\nSummary CMD plot: {outfile_cmd}')


def analyze_cloudc_f182m_f212n_diagonal_av(cloudc_all, nth_neighbor=NTH_NEIGHBOR, outdir=OUTDIR):
    """
    Analyze Cloud C F182M-F212N with diagonal Av slices using G23 (Rv=3.1) extinction law.
    """
    print("\n" + "="*80)
    print("ANALYSIS 6: Cloud C F182M-F212N Diagonal Av Slices (G23, Rv=3.1)")
    print("="*80)
    
    extinction_ct06 = CT06_MWGC()
    k182 = float(extinction_ct06(1.82 * u.um))
    k212 = float(extinction_ct06(2.12 * u.um))
    k_color = k182 - k212
    k_mag = k182

    mag182 = filled_float(cloudc_all['mag_ab_f182m'])
    mag212 = filled_float(cloudc_all['mag_ab_f212n'])
    color_182_212 = mag182 - mag212

    base_mask_c = np.isfinite(mag182) & np.isfinite(mag212) & np.isfinite(color_182_212) & (color_182_212 > 0.2) & (color_182_212 < 1.8) & (mag182 > 13.0) & (mag182 < 20.0)
    #y_intercept_c = float(np.nanmedian(mag182[base_mask_c] - (k_mag / k_color) * color_182_212[base_mask_c]))
    y_intercept_c = y_intercept_f182m # this is hard-coded by the user and should not be overwritten

    print(f'CT06: A182={k182:.4f}, A212={k212:.4f}, dcolor/dAv={k_color:.4f}, dmag/dAv={k_mag:.4f}')
    print(f'Cloud C estimated y-intercept at color=0: {y_intercept_c:.3f}')

    run_polygon_av_slices(
        table=cloudc_all,
        color=color_182_212,
        mag=mag182,
        base_mask=base_mask_c,
        k_num=k182,
        k_den=k212,
        y_intercept=y_intercept_c,
        width=0.75,
        av_step=2.0,
        av_max_potential=50.0,
        nth_neighbor=nth_neighbor,
        outdir=outdir,
        outprefix='cloudc_rc_182m212n_ct06',
        field_label='Cloud C F182M-F212N (CT06)',
        cmd_xlabel='F182M - F212N (mag)',
        cmd_ylabel='F182M (mag)',
        cmd_xlim=(0.0, 1.8),
        cmd_ylim=(20.0, 13.0),
    )


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Load catalogs and run all analyses."""
    print("Loading catalogs...")
    brick_all = load_brick_catalog()
    cloudc_all = load_cloudc_catalog()
    print(f"Brick: {len(brick_all):,} stars")
    print(f"Cloud C: {len(cloudc_all):,} stars")

    # Run analyses
    analyze_f200w_f444w_av_slices_ct06(brick_all, nth_neighbor=NTH_NEIGHBOR, outdir=OUTDIR)
    print("Done with F200W-F444W Av-slice analysis. Moving on...")
    analyze_simple_narrow_color_bins(brick_all, cloudc_all, nth_neighbor=NTH_NEIGHBOR, outdir=OUTDIR)
    analyze_f115w_f200w_color_range(brick_all, nth_neighbor=NTH_NEIGHBOR, outdir=OUTDIR)
    analyze_f200w_f444w_color_range(brick_all, nth_neighbor=NTH_NEIGHBOR, outdir=OUTDIR)
    analyze_f115w_f200w_diagonal_av(brick_all, nth_neighbor=NTH_NEIGHBOR, outdir=OUTDIR)
    analyze_f182m_f212n_av_slices_ct06(brick_all, cloudc_all, nth_neighbor=NTH_NEIGHBOR, outdir=OUTDIR)

    print("\n" + "="*80)
    print("All analyses complete!")
    print("="*80)


if __name__ == '__main__':
    main()
