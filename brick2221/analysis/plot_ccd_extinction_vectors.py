#!/usr/bin/env python3
"""
Plot color-color diagrams with extinction vectors for G23(3.1), G23(5.5), and CT06.
Uses hexbin plotting with different colormaps for ice-free and ice-dominated data.
Ice-dominated criteria: f405n-f466n < -0.1
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from astropy.table import Table
from astropy import units as u

from astroquery.svo_fps import SvoFps

# Import extinction models
from dust_extinction.averages import CT06_MWGC, G21_MWAvg, F11_MWGC
from dust_extinction.parameter_averages import G23
from dust_extinction.grain_models import P24 as KP5

# Define basepath directly
basepath = '/blue/adamginsburg/adamginsburg/jwst/brick/'


plt.rcParams['figure.facecolor'] = 'w'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def wavelength_of_filter(filtername, telescope='JWST', instrument='NIRCAM'):
    wavelength_table = SvoFps.get_transmission_data(f'{telescope}/{instrument}.{filtername}')
    return wavelength_table


def kp5(wavelength):
    tb = Table.read('/orange/adamginsburg/ice/pontoppidan2025/kp5.fits')
    ref_wav = 550*u.nm
    ref_kabs = np.interp(ref_wav, tb['wavelength'], tb['kabs'] + tb['ksca'])
    return np.interp(wavelength, tb['wavelength'], tb['kabs'] + tb['ksca']) / ref_kabs


def plot_extinction_vector(ax, color1, color2, ext_model, av_scale=30,
                          start=(0, 0), color='black', label=None, alpha=0.8):
    """Plot extinction vector for given extinction model and color combination"""

    # Get wavelengths for each filter
    w1 = wavelength_of_filter(color1[0])
    w2 = wavelength_of_filter(color1[1])
    w3 = wavelength_of_filter(color2[0])
    w4 = wavelength_of_filter(color2[1])

    # Calculate extinction at each wavelength
    e1 = (ext_model(w1['Wavelength'].quantity) * w1['Transmission']).sum() / w1['Transmission'].sum() * av_scale
    e2 = (ext_model(w2['Wavelength'].quantity) * w2['Transmission']).sum() / w2['Transmission'].sum() * av_scale
    e3 = (ext_model(w3['Wavelength'].quantity) * w3['Transmission']).sum() / w3['Transmission'].sum() * av_scale
    e4 = (ext_model(w4['Wavelength'].quantity) * w4['Transmission']).sum() / w4['Transmission'].sum() * av_scale

    # Vector components (color differences)
    dx = e1 - e2
    dy = e3 - e4

    # print("start", start)
    # print(label, w1, w2, w3, w4)
    # print(label, e1, e2, e3, e4, dx, dy)

    ax.plot([start[0], start[0] + dx], [start[1], start[1] + dy],
            color=color,
            alpha=alpha, linewidth=2, label=label, linestyle='-')

    # Plot arrow
    ax.annotate('', xy=(start[0] + dx, start[1] + dy),
                xytext=(start[0], start[1]),
                arrowprops=dict(arrowstyle='->', color=color, alpha=alpha,
                                shrinkB=0,
                                linewidth=2, mutation_scale=15))


def plot_ccd_with_extinction_vectors(basetable, color1, color2,
                                   ice_criterion_col1='f405n', ice_criterion_col2='f466n',
                                   ice_threshold=-0.1,
                                   axlims=(-1, 3, -2, 1),
                                   av_scale=30,
                                   n_bins=100,
                                   figsize=(5, 4),
                                   plot_ice_dominated=True,
                                   savepath=None):
    """
    Plot color-color diagram with hexbin for ice-free/ice-dominated data and extinction vectors

    Parameters:
    -----------
    basetable : astropy.table.Table
        Photometry table
    color1 : tuple
        (filter1, filter2) for x-axis color
    color2 : tuple
        (filter1, filter2) for y-axis color
    ice_criterion_col1, ice_criterion_col2 : str
        Filters for ice selection criterion
    ice_threshold : float
        Threshold for ice selection (ice-dominated if col1-col2 < threshold)
    axlims : tuple
        Axis limits (xmin, xmax, ymin, ymax)
    av_scale : float
        Scale factor for extinction vectors (Av value)
    n_bins : int
        Number of hexbin bins
    figsize : tuple
        Figure size
    savepath : str or None
        Path to save figure
    """

    fig, ax = plt.subplots(figsize=figsize)

    # Calculate colors
    x_color = basetable[f'mag_ab_{color1[0]}'] - basetable[f'mag_ab_{color1[1]}']
    y_color = basetable[f'mag_ab_{color2[0]}'] - basetable[f'mag_ab_{color2[1]}']
    ice_color = basetable[f'mag_ab_{ice_criterion_col1}'] - basetable[f'mag_ab_{ice_criterion_col2}']

    # Remove invalid data (NaN, inf)
    valid = (np.isfinite(x_color) & np.isfinite(y_color) & np.isfinite(ice_color))
    x_color = x_color[valid]
    y_color = y_color[valid]
    ice_color = ice_color[valid]

    # Apply ice selection criterion
    ice_dominated = ice_color < ice_threshold
    ice_free = ~ice_dominated

    print(f"Total sources: {len(x_color)}", end='. ')
    print(f"Ice-dominated sources: {ice_dominated.sum()}", end='. ')
    print(f"Ice-free sources: {ice_free.sum()}", end='. ')

    # Plot ice-dominated data with red colormap
    if plot_ice_dominated:
        hb2 = ax.hexbin(x_color[ice_dominated], y_color[ice_dominated],
                        gridsize=n_bins, extent=axlims, cmap='Reds_r',
                        linewidths=0.1,
                        mincnt=1, alpha=0.8)#, label='Ice-dominated')

    elif plot_ice_dominated is None:
        hb1 = ax.hexbin(x_color, y_color,
                        gridsize=n_bins, extent=axlims, cmap='Greys_r',
                        linewidths=0.1,
                        mincnt=1, alpha=0.8)

    # Plot ice-free data with grayscale colormap
    else:
        hb1 = ax.hexbin(x_color[ice_free], y_color[ice_free],
                        gridsize=n_bins, extent=axlims, cmap='Greys_r',
                        linewidths=0.1,
                        mincnt=1, alpha=0.8)#, label='Ice-free')

    # Set up extinction models
    ext_models = {
        'G23(3.1)': G23(Rv=3.1),
        'G23(5.5)': G23(Rv=5.5),
        'CT06': CT06_MWGC(),
        'F11': F11_MWGC(),
        'KP5': KP5(),
    }

    colors = {'G23(3.1)': 'blue', 'G23(5.5)': 'green', 'CT06': 'red', 'F11': 'purple', 'KP5': 'orange'}

    # Plot extinction vectors
    #vector_start = (axlims[0] + 0.1*(axlims[1]-axlims[0]),
    #                axlims[2] + 0.8*(axlims[3]-axlims[2]))

    for name, ext_model in ext_models.items():
        #print(name, colors[name], color1, color2, av_scale, ext_model(2.12*u.um)*30)
        plot_extinction_vector(ax, color1, color2, ext_model,
                               av_scale=av_scale, #start=vector_start,
                               color=colors[name],
                               label=f'{name} (Av={av_scale})')

    # Format plot
    ax.set_xlabel(f'{color1[0]} - {color1[1]}')
    ax.set_ylabel(f'{color2[0]} - {color2[1]}')
    ax.set_xlim(axlims[0], axlims[1])
    ax.set_ylim(axlims[2], axlims[3])

    # Add legend
    ax.legend(loc='best')

    # Add title
    #ice_criterion = f'{ice_criterion_col1}-{ice_criterion_col2} < {ice_threshold}'
    #ax.set_title(f'Color-Color Diagram\nIce criterion: {ice_criterion}')

    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=250, bbox_inches='tight')
        print(f"Saved figure to {savepath}")

    return fig, ax



def main(basetable=None):
    """Main function to create multiple CCD plots"""
    from brick2221.analysis.selections import load_table
    from brick2221.analysis.analysis_setup import fh_merged as fh, ww410_merged as ww410, ww410_merged as ww

    if basetable is None:
        basetable_merged1182_daophot = Table.read(f'{basepath}/catalogs/basic_merged_indivexp_photometry_tables_merged.fits')
        result = load_table(basetable_merged1182_daophot, ww=ww)
        ok2221 = result['ok2221']
        ok1182 = result['ok1182'][ok2221]
        oksep_noJ = result['oksep_noJ']
        bad = result['bad']
        basetable = basetable_merged1182_daophot[ok2221]
        # there are several bad data points in F182M that are brighter than 15.5 mag
        print("Loaded merged1182_daophot_basic_indivexp")

        sel = ok = oksep_noJ[ok2221] & ~bad[ok2221] & (basetable['mag_ab_f182m'] > 15.5)
        basetable = basetable[sel]

    print(f"Using {len(basetable)} good sources for analysis")

    # Define color combinations to plot (using filters from 2221 project)
    # Note: using lowercase filter names to match table columns
    color_combinations = [
        # Narrow-band combinations
        (['f182m', 'f212n'], ['f405n', 'f466n']),
        (['f182m', 'f212n'], ['f410m', 'f466n']),
        (['f405n', 'f410m'], ['f466n', 'f410m']),
        (['f182m', 'f212n'], ['f212n', 'f405n']),
        (['f182m', 'f212n'], ['f405n', 'f410m']),

        # Wide-band combinations
        (['f200w', 'f356w'], ['f356w', 'f444w']),
        #(['f115w', 'f200w'], ['f356w', 'f444w']),
        (['f182m', 'f212n'], ['f356w', 'f444w']),
        (['f182m', 'f212n'], ['f212n', 'f444w']),
        (['f182m', 'f212n'], ['f212n', 'f356w']),
    ]

        # Define axis limits for each color combination to match make_ccd_with_icemodels.py
    axis_limits = {
        ('f182m', 'f212n', 'f405n', 'f466n'): (0, 3, -1.5, 1.0),
        ('f182m', 'f212n', 'f410m', 'f466n'): (0, 3, -1.5, 1.0),
        ('f182m', 'f212n', 'f212n', 'f405n'): (0, 1.5, -0.5, 3.5),
        ('f405n', 'f410m', 'f466n', 'f410m'): (-0.5, 0.5, -0.5, 1.5),  # Based on f405n-f410m vs f356w-f444w
        ('f182m', 'f212n', 'f405n', 'f410m'): (0, 3, -0.5, 0.2),
        ('f200w', 'f356w', 'f356w', 'f444w'): (0, 5, -0.5, 1.5),
        ('f115w', 'f200w', 'f356w', 'f444w'): (0, 20, -0.5, 1.5),
        ('f182m', 'f212n', 'f356w', 'f444w'): (0, 3, -0.5, 1.5),  # Interpolated from similar combinations
    }

    # Plot each color combination
    for i, (color1, color2) in enumerate(color_combinations):
        print(f"\nCreating plot {i+1}/{len(color_combinations)}: {color1} vs {color2}")

        # Get axis limits for this specific combination
        key = (color1[0], color1[1], color2[0], color2[1])
        axlims = axis_limits.get(key, (0, 2, -0.5, 2.5))  # Default fallback
        av_scale = 20

        for plot_ice_dominated, suffix in zip([True, False, None], ['_ice', '_noice', '']):
            # Create filename
            filename = f'CCD_extinction_vectors_{color1[0]}-{color1[1]}_{color2[0]}-{color2[1]}{suffix}.pdf'
            savepath = f'{basepath}/figures/{filename}'

            # Create plot
            fig, ax = plot_ccd_with_extinction_vectors(
                basetable, color1, color2,
                ice_criterion_col1='f405n', ice_criterion_col2='f466n',
                ice_threshold=-0.1,
                axlims=axlims,
                av_scale=av_scale,
                n_bins=100,
                savepath=savepath,
                plot_ice_dominated=plot_ice_dominated,
            )
            plt.close(fig)  # Close to save memory

    return basetable


if __name__ == "__main__":
    if 'basetable' in globals():
        main(basetable=basetable)
    else:
        basetable = main()