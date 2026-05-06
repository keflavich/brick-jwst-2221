#!/usr/bin/env python
"""
Hexbin difference image: filament vs rest of Cloud C RC stars.

Purpose: Visual diagnostic to identify which stars are in front/behind 
the filament relative to the rest of Cloud C field.

Uses RC-selected Cloud C catalog and filament region definitions.
"""

import glob
import numpy as np
import pylab as pl
from astropy.table import Table
from astropy.coordinates import SkyCoord
import regions as reg

from brick2221.analysis.analysis_setup import basepath
from brick2221.analysis.distance_analysis import (
    load_cloudc_catalog,
    rc_peak_fit_slope,
    _filled_float,
    _region_contains_skycoord,
)


def select_stars_in_regions(catalog, region_list):
    """Return boolean mask of catalog stars inside any region in region_list."""
    if len(region_list) == 0:
        return np.zeros(len(catalog), dtype=bool)
    
    # Get skycoord from catalog
    if 'skycoord_ref' in catalog.colnames:
        skycoord = SkyCoord(catalog['skycoord_ref'])
    elif 'ra' in catalog.colnames and 'dec' in catalog.colnames:
        skycoord = SkyCoord(catalog['ra'], catalog['dec'], unit='deg')
    else:
        # Extract from individual filter skycoord columns
        filt_cols = [c for c in catalog.colnames if c.startswith('skycoord_')]
        if filt_cols:
            skycoord = SkyCoord(catalog[filt_cols[0]])
        else:
            raise KeyError(f"Could not find coordinate columns in table")
    
    mask = np.zeros(len(catalog), dtype=bool)
    
    for region_obj in region_list:
        # Use the distance_analysis function that properly handles SkyRegion containment
        in_region = _region_contains_skycoord(region_obj, skycoord)
        mask = mask | in_region
    
    return mask


def load_filament_regions():
    """Load filament_box region specifically (not parent_cloud which covers whole field)."""
    # Use only filament_box to get the actual filament structure
    # (filament_parent_cloud covers ~96% of Cloud C and is not useful for comparison)
    fpath = '/home/savannahgramze/orange_link/adamginsburg/jwst/cloudc/regions_/filament.reg'
    
    regs = reg.Regions.read(fpath)
    print(f"Loaded {len(regs)} filament_parent_cloud region(s)")
    return list(regs)


def main():
    # Load filament regions
    print("Loading filament regions...")
    filament_regions = load_filament_regions()
    print(f"Loaded {len(filament_regions)} region objects")

    print("Loading Cloud C catalog...")
    cloudc_table = (load_cloudc_catalog())
    print(f"Loaded {len(cloudc_table)} total Cloud C stars")
    
    # Select filament vs non-filament stars from ALL stars (not just RC-selected)
    in_filament = select_stars_in_regions(cloudc_table, filament_regions)
    in_rest = ~in_filament
    
    print(f"ALL stars in filament: {in_filament.sum()}")
    print(f"ALL stars in rest of Cloud C: {in_rest.sum()}")
    
    # For comparison, also show RC star counts
    print("\nFitting RC peak for reference...")
    rc_peak_result = rc_peak_fit_slope(cloudc_table, 'Cloud C')
    sel_mask = rc_peak_result['sel_mask_full']
    print(f"RC-selected stars total: {sel_mask.sum()}")
    print(f"RC-selected in filament: {(in_filament & sel_mask).sum()}")
    print(f"RC-selected in rest of Cloud C: {(in_rest & sel_mask).sum()}")
    
    # Extract coordinates and magnitudes
    # Use skycoord_ref or skycoord_ref_filtername mapping
    if 'skycoord_ref' in cloudc_table.colnames:
        skycoord = SkyCoord(cloudc_table['skycoord_ref'])
    elif 'ra' in cloudc_table.colnames and 'dec' in cloudc_table.colnames:
        skycoord = SkyCoord(cloudc_table['ra'], cloudc_table['dec'], unit='deg')
    else:
        # Extract from individual filter skycoord columns
        filt_cols = [c for c in cloudc_table.colnames if c.startswith('skycoord_')]
        if filt_cols:
            skycoord = SkyCoord(cloudc_table[filt_cols[0]])
        else:
            raise KeyError(f"Could not find coordinate columns in table. Available: {cloudc_table.colnames[:10]}")
    
    ra = skycoord.ra.deg
    dec = skycoord.dec.deg
    
    # Use dereddened magnitudes if available, otherwise use raw magnitudes
    if 'mag_ab_f187n_dered' in cloudc_table.colnames:
        mag_f187n_dered = _filled_float(cloudc_table['mag_ab_f187n_dered'])
        mag_f212n_dered = _filled_float(cloudc_table['mag_ab_f212n_dered'])
    else:
        # Fallback: use raw magnitudes
        mag_f187n_dered = _filled_float(cloudc_table['mag_ab_f187n'])
        mag_f212n_dered = _filled_float(cloudc_table['mag_ab_f212n'])
    
    color_f187n_f212n = mag_f187n_dered - mag_f212n_dered
    
    # Extract distance if available
    distance_kpc = None
    if 'distance_kpc' in cloudc_table.colnames:
        distance_kpc = _filled_float(cloudc_table['distance_kpc'])
    
    print("\nCreating diagnostic figures...")
    
    # ─────────────────────────────────────────────────
    # Spatial distribution hexbins
    # ─────────────────────────────────────────────────
    fig, (ax1, ax2, ax3) = pl.subplots(1, 3, figsize=(18, 5))
    
    # Shared extent for consistency
    ra_min, ra_max = ra.min(), ra.max()
    dec_min, dec_max = dec.min(), dec.max()
    extent = [ra_min, ra_max, dec_min, dec_max]
    
    # Filament hexbin
    hb1 = ax1.hexbin(ra[in_filament], dec[in_filament], mincnt=1, gridsize=40,
                     extent=extent, cmap='YlOrRd', edgecolors='none')
    ax1.set_xlabel('RA (deg)')
    ax1.set_ylabel('Dec (deg)')
    ax1.set_title(f'Filament RC stars (N={in_filament.sum()})')
    pl.colorbar(hb1, ax=ax1, label='Count')
    
    # Rest of Cloud C hexbin
    hb2 = ax2.hexbin(ra[in_rest], dec[in_rest], mincnt=1, gridsize=40,
                     extent=extent, cmap='Blues', edgecolors='none')
    ax2.set_xlabel('RA (deg)')
    ax2.set_ylabel('Dec (deg)')
    ax2.set_title(f'Rest of Cloud C RC stars (N={in_rest.sum()})')
    pl.colorbar(hb2, ax=ax2, label='Count')
    
    # Difference hexbin (filament fraction)
    h_fila, xedges, yedges = np.histogram2d(ra[in_filament], dec[in_filament], 
                                             bins=40, range=[(ra_min, ra_max), (dec_min, dec_max)])
    h_rest, _, _ = np.histogram2d(ra[in_rest], dec[in_rest], 
                                   bins=40, range=[(ra_min, ra_max), (dec_min, dec_max)])
    h_total = h_fila + h_rest
    h_total[h_total == 0] = 1
    frac_filament = h_fila / h_total
    
    im = ax3.pcolormesh(xedges, yedges, frac_filament.T, cmap='RdBu_r', 
                        vmin=0, vmax=1, shading='auto')
    ax3.set_xlabel('RA (deg)')
    ax3.set_ylabel('Dec (deg)')
    ax3.set_title('Filament fraction (filament / total)')
    pl.colorbar(im, ax=ax3, label='Filament fraction')
    
    fig.suptitle('Cloud C RC: Filament vs Rest Spatial Distribution', fontsize=14, y=1.00)
    fig.tight_layout()
    outpath = f'{basepath}/distance/cloudc_filament_spatial_diagnostic.png'
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"Wrote spatial diagnostic to {outpath}")
    pl.close(fig)
    
    # ─────────────────────────────────────────────────
    # Normalized difference CMD hexbin
    # ─────────────────────────────────────────────────
    print("Creating normalized difference CMD hexbin...")
    
    # Valid CMD points for both populations
    valid_all = np.isfinite(color_f187n_f212n) & np.isfinite(mag_f187n_dered)
    valid_fil = valid_all & in_filament
    valid_rest = valid_all & in_rest
    
    x_fil = color_f187n_f212n[valid_fil]
    y_fil = mag_f187n_dered[valid_fil]
    x_rest = color_f187n_f212n[valid_rest]
    y_rest = mag_f187n_dered[valid_rest]
    
    print(f"  Filament stars with valid CMD: {len(x_fil)}")
    print(f"  Rest stars with valid CMD: {len(x_rest)}")
    
    # Build weights for normalized difference: filament minus rest
    x_all = np.concatenate([x_fil, x_rest])
    y_all = np.concatenate([y_fil, y_rest])
    weights = np.concatenate([
        np.full(len(x_fil), 1.0 / max(len(x_fil), 1), dtype=float),
        np.full(len(x_rest), -1.0 / max(len(x_rest), 1), dtype=float),
    ])
    
    fig_cmd = pl.figure(figsize=(10, 7))
    ax_cmd = fig_cmd.gca()
    
    gridsize = 50
    extent = (0.3, 1.5, min(13, 21), max(13, 21))
    
    hb_diff = ax_cmd.hexbin(
        x_all,
        y_all,
        C=weights,
        reduce_C_function=np.sum,
        gridsize=gridsize,
        extent=extent,
        mincnt=1,
        cmap='RdBu_r',
    )
    
    # Get per-sample counts on the same hex grid for noise masking
    fig_tmp1 = pl.figure(figsize=(1, 1))
    ax_tmp1 = fig_tmp1.gca()
    hb_fil = ax_tmp1.hexbin(x_fil, y_fil, gridsize=gridsize, extent=extent, mincnt=1)
    fil_offsets = hb_fil.get_offsets()
    fil_counts = hb_fil.get_array()
    pl.close(fig_tmp1)
    
    fig_tmp2 = pl.figure(figsize=(1, 1))
    ax_tmp2 = fig_tmp2.gca()
    hb_rest = ax_tmp2.hexbin(x_rest, y_rest, gridsize=gridsize, extent=extent, mincnt=1)
    rest_offsets = hb_rest.get_offsets()
    rest_counts = hb_rest.get_array()
    pl.close(fig_tmp2)
    
    # Match bins by center coordinates
    def _keyify(offsets):
        return [f"{xv:.7f},{yv:.7f}" for xv, yv in offsets]
    
    fil_map = dict(zip(_keyify(fil_offsets), fil_counts))
    rest_map = dict(zip(_keyify(rest_offsets), rest_counts))
    
    offsets = hb_diff.get_offsets()
    keys = _keyify(offsets)
    fil_in_bin = np.array([fil_map.get(key, 0.0) for key in keys], dtype=float)
    rest_in_bin = np.array([rest_map.get(key, 0.0) for key in keys], dtype=float)
    
    # Mask bins where both populations have <3 stars
    noise_mask = (fil_in_bin < 3) & (rest_in_bin < 3)
    diff_vals = np.array(hb_diff.get_array(), dtype=float)
    diff_vals_masked = np.ma.array(diff_vals, mask=noise_mask)
    
    vlim = np.nanmax(np.abs(diff_vals_masked))
    if not np.isfinite(vlim) or vlim == 0:
        vlim = 1e-4
    
    hb_diff.set_array(diff_vals_masked)
    hb_diff.set_clim(-vlim, vlim)
    
    cbar = fig_cmd.colorbar(hb_diff, ax=ax_cmd, label='Normalized density (Filament - Rest)')
    ax_cmd.set_xlabel('F187N - F212N (mag)')
    ax_cmd.set_ylabel('F187N dereddened (mag)')
    ax_cmd.set_title(f'Cloud C: Filament vs Rest CMD (All {len(valid_all)} stars)')
    ax_cmd.invert_yaxis()
    ax_cmd.grid(alpha=0.2)
    
    fig_cmd.tight_layout()
    outpath_cmd = f'{basepath}/distance/cloudc_filament_cmd_normalized_difference.png'
    fig_cmd.savefig(outpath_cmd, dpi=150, bbox_inches='tight')
    print(f"Wrote CMD normalized difference to {outpath_cmd}")
    pl.close(fig_cmd)
    
    # ─────────────────────────────────────────────────
    # Distance diagnostic (if available)
    # ─────────────────────────────────────────────────
    if distance_kpc is not None:
        fig3, (ax7, ax8, ax9) = pl.subplots(1, 3, figsize=(18, 5))
        
        # Filament distance histogram
        ax7.hist(distance_kpc[in_filament], bins=20, alpha=0.7, color='red', edgecolor='black', label='Filament')
        ax7.set_xlabel('Distance (kpc)')
        ax7.set_ylabel('Count')
        ax7.set_title(f'Filament RC distance distribution')
        ax7.legend()
        
        # Rest distance histogram
        ax8.hist(distance_kpc[in_rest], bins=20, alpha=0.7, color='blue', edgecolor='black', label='Rest')
        ax8.set_xlabel('Distance (kpc)')
        ax8.set_ylabel('Count')
        ax8.set_title(f'Rest Cloud C RC distance distribution')
        ax8.legend()
        
        # Overlaid for comparison
        ax9.hist(distance_kpc[in_filament], bins=20, alpha=0.5, color='red', label='Filament', density=True)
        ax9.hist(distance_kpc[in_rest], bins=20, alpha=0.5, color='blue', label='Rest', density=True)
        ax9.set_xlabel('Distance (kpc)')
        ax9.set_ylabel('Normalized count')
        ax9.set_title('Distance distributions (normalized)')
        ax9.legend()
        
        fig3.suptitle('Cloud C RC: Distance Distributions', fontsize=14, y=1.00)
        fig3.tight_layout()
        outpath3 = f'{basepath}/distance/cloudc_filament_distance_diagnostic.png'
        fig3.savefig(outpath3, dpi=150, bbox_inches='tight')
        print(f"Wrote distance diagnostic to {outpath3}")
        pl.close(fig3)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
