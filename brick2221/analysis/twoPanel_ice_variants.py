"""
Two-panel "all variants" figures for a single ice species.

Left panel: a single ice + dust color (default F405N - F466N) versus N(H2),
with secondary axes for A_V and N(species). Mirrors the dmag_vs_color
plots produced by colorcolordiagrams.plot_color_vs_column.

Right panel: opacity vs wavelength for every valid optical-constants table
(curves grouped by author and phase with min/max envelope shading and a
median curve), mirroring opacities_on_f466_f410_f405_water_versions.

Both panels use the same author-color and phase-linestyle convention so
the curves correspond visually across the figure.

Generates one PDF per species in the overleaf figures directory:
    twopanel_iceVariants_{H2O,CO,CO2}.pdf
"""

import os
import glob
import warnings
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as pl
from matplotlib.lines import Line2D
import numpy as np
from astropy import units as u
from astropy.table import Table

from icemodels.core import (read_ocdb_file, optical_constants_cache_dir,
                            absorbed_spectrum, composition_to_molweight,
                            read_ehrenfreund_NK_file)
from icemodels.colorcolordiagrams import (
    compute_dmag_from_column, _resolve_single_mol_id, ext as ccd_ext,
    carbon_abundance, oxygen_abundance, _fmt_sci, mixes2,
)
from astroquery.svo_fps import SvoFps


# ---------------- shared helpers ---------------- #

PHASE_LINESTYLE = {
    'amorphous':   '-',
    'crystalline': '--',
    'liquid':      ':',
    'unknown':     '-.',
}

AUTHOR_COLOR = {
    'Mastrapa':    '#1f77b4',
    'Hudgins':     '#ff7f0e',
    'Bertie':      '#2ca02c',
    'Clapp':       '#d62728',
    'Kitta':       '#9467bd',
    'Léger':       '#8c564b',
    'Leger':       '#8c564b',
    'Mukai':       '#e377c2',
    'Curtis':      '#7f7f7f',
    'Rajaram':     '#bcbd22',
    'Zhang':       '#17becf',
    'Gerakines':   '#1f77b4',
    'Ehrenfreund': '#ff7f0e',
    'Elsila':      '#2ca02c',
    'Baratta':     '#d62728',
    'Palumbo':     '#9467bd',
}


def _classify_phase(species, author, T):
    """Phase label per (species, author, T). For species other than H2O the
    deposit-temperature distinction between amorphous and crystalline is
    less clean; use 'amorphous' as the default cold label and 'crystalline'
    above ~30 K for CO/CO2 to give legend grouping a meaningful split."""
    a = str(author).strip()
    if species == 'H2O':
        if a == 'Mastrapa':  return 'amorphous' if T < 110 else 'crystalline'
        if a == 'Hudgins':   return 'amorphous' if T <= 100 else 'crystalline'
        if a in ('Kitta', 'Léger', 'Leger', 'Mukai'):       return 'amorphous'
        if a in ('Bertie', 'Curtis', 'Rajaram', 'Clapp'):   return 'crystalline'
        if a == 'Zhang':                                    return 'liquid'
    else:
        # CO / CO2: treat low-T deposits as amorphous, warmer ones as
        # crystalline. The split is rough — none of these tables are
        # phase-labeled in the OCDB metadata.
        return 'amorphous' if T <= 30 else 'crystalline'
    return 'unknown'


def _temp_str_to_float(temperature):
    s = str(temperature).strip()
    if s.lower().endswith('k'):
        s = s[:-1].strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def _drop_clapp_spike(tb, author):
    """Clapp 1995 H2O 190K (OCDB id 5) has a single-channel digitization
    spike at 4.165 um where k jumps to ~0.18 between neighbors at
    ~0.015-0.025. Drop it."""
    if author != 'Clapp':
        return tb
    wl = np.asarray(tb['Wavelength'], dtype=float)
    kk = np.asarray(tb['k'], dtype=float)
    spike = (wl > 4.13) & (wl < 4.20) & (kk > 0.05)
    if spike.any():
        tb = tb[~spike]
    return tb


def _load_species_tables(species, require_overlap=None):
    """Return list of (table, author, T_num, phase) for every optical-constants
    file matching the species. Skips files whose wavelength range does not
    cover the species-specific minimum overlap window. H2O continuum spans
    F405N+F466N so we require [4.0, 4.7]; CO is a narrow absorber at 4.67 um
    so only F466N coverage [4.6, 4.7] is required (Ehrenfreund/Elsila/Baratta
    pure-CO tables only span ~4.5-5.0 um and would otherwise be excluded).
    CO2 absorbs around 4.27 um so we require [4.2, 4.5].
    """
    if require_overlap is None:
        require_overlap = {
            'H2O': (4.0, 4.7),
            'CO':  (4.6, 4.7),
            'CO2': (4.2, 4.5),
        }.get(species, (4.0, 4.7))

    pattern = f'{optical_constants_cache_dir}/*_{species}_(1)_*K_*.txt'
    files = sorted(glob.glob(pattern))
    out = []
    for fn in files:
        try:
            tb = read_ocdb_file(fn)
        except Exception as ex:
            print(f"  skip {os.path.basename(fn)}: read error: {ex}")
            continue
        wl = np.asarray(tb['Wavelength'], dtype=float)
        if not (np.isfinite(wl).any() and
                wl.min() <= require_overlap[0] and
                wl.max() >= require_overlap[1]):
            print(f"  skip {os.path.basename(fn)}: wl {wl.min():.2f}-{wl.max():.2f}")
            continue
        author = tb.meta.get('author', '')
        T = _temp_str_to_float(tb.meta.get('temperature', np.nan))
        tb = _drop_clapp_spike(tb, author)
        phase = _classify_phase(species, author, T)
        out.append((tb, author, T, phase))
    return out


# ---------------- panel renderers ---------------- #

def _draw_filter_overlay(ax, filternames=('F466N', 'F410M', 'F405N'),
                         linestyles=('-', ':', '--'), alpha=0.5):
    transmission_ax = ax.twinx()
    tmax = 0.0
    for filtername, ls in zip(filternames, linestyles):
        wt = SvoFps.get_transmission_data(f'JWST/NIRCam.{filtername}')
        xarr = wt['Wavelength'].quantity.to(u.um)
        transmission_ax.plot(xarr, wt['Transmission'], color='k', linewidth=2,
                             alpha=alpha, linestyle=ls)
        tmax = max(tmax, wt['Transmission'].max())
    transmission_ax.set_ylim(0, tmax * 1.10)
    transmission_ax.set_ylabel('Transmission')
    return transmission_ax, tmax


def _opacity_panel(ax, entries, species, show_phase_in_label=True,
                   left_color=None):
    """Right-panel: opacity vs wavelength, grouped by (author, phase). For
    multi-T groups, plot min/max envelope + median curve. Filter overlay
    and x-limits track ``left_color`` (the left panel's filter pair)."""
    filters, xlim = _right_panel_filters_xlim(left_color or DEFAULT_COLOR)
    grid = np.linspace(min(xlim[0], 3.0), max(xlim[1], 5.2), 1500)
    grouped = defaultdict(list)
    for tb, author, T, phase in entries:
        molwt = u.Quantity(composition_to_molweight(tb.meta['composition']),
                           u.Da)
        op_per_mol = absorbed_spectrum(
            xarr=tb['Wavelength'], ice_column=1,
            ice_model_table=tb, molecular_weight=molwt,
            return_tau=True).to(u.cm**2).value     # cm^2 per molecule
        op = op_per_mol / molwt.to(u.g).value      # cm^2 / g
        wl = np.asarray(tb['Wavelength'], dtype=float)
        so = np.argsort(wl)
        op_g = np.interp(grid, wl[so], op[so], left=np.nan, right=np.nan)
        grouped[(author, phase)].append((T, op_g))

    phase_order = {'amorphous': 0, 'crystalline': 1, 'liquid': 2, 'unknown': 3}
    legend_handles = []
    for (author, phase), eg in sorted(grouped.items(),
                                      key=lambda kv: (phase_order[kv[0][1]], kv[0][0])):
        Ts = sorted(e[0] for e in eg)
        stack = np.array([e[1] for e in eg])
        color = AUTHOR_COLOR.get(author, None)
        ls = PHASE_LINESTYLE.get(phase, '-') if show_phase_in_label else '-'
        phase_tag = f" {phase}" if show_phase_in_label else ""
        if len(eg) >= 2:
            lo = np.nanmin(stack, axis=0)
            hi = np.nanmax(stack, axis=0)
            med = np.nanmedian(stack, axis=0)
            ax.fill_between(grid, lo, hi, color=color, alpha=0.18, linewidth=0)
            ax.plot(grid, med, color=color, linestyle=ls, alpha=0.95,
                    linewidth=1.4)
            label = f"{author}{phase_tag} ({len(eg)} T: {Ts[0]:g}–{Ts[-1]:g} K)"
        else:
            ax.plot(grid, stack[0], color=color, linestyle=ls, alpha=0.95,
                    linewidth=1.4)
            label = f"{author}{phase_tag} ({Ts[0]:g} K)"
        legend_handles.append(Line2D([0], [0], color=color, linestyle=ls,
                                     linewidth=1.4, label=label))

    ax.set_xlabel(r'Wavelength ($\mu$m)')
    ax.set_ylabel(r'$\kappa$ [cm$^{2}$ g$^{-1}$]  ($\tau = \kappa\,\Sigma_{ice}$)')
    ax.semilogy()
    ax.set_ylim(1, 1e5)
    ax.set_xlim(*xlim)

    transmission_ax, tmax = _draw_filter_overlay(ax, filternames=filters)
    for fname in filters:
        wl0 = int(fname[1:-1]) / 100.0
        transmission_ax.text(wl0, tmax * 1.03, fname, ha='center', fontsize=8)

    return legend_handles


DEFAULT_COLOR = ('F405N', 'F466N')   # filter pair for the left-panel y-axis
# Right-panel filter overlay + xlim per left-panel color choice. Picked so
# the panel zooms onto the wavelength range that drives the chosen color.
RIGHT_PANEL_FILTERS = {
    ('F405N', 'F466N'): (('F405N', 'F410M', 'F466N'), (3.71, 4.75)),
    ('F405N', 'F410M'): (('F405N', 'F410M'),          (3.85, 4.25)),
    ('F356W', 'F444W'): (('F356W', 'F405N', 'F410M',
                          'F444W', 'F466N'),          (2.90, 5.10)),
}


def _right_panel_filters_xlim(color):
    return RIGHT_PANEL_FILTERS.get(
        tuple(color), (('F466N', 'F410M', 'F405N'), (3.71, 4.75)))

NH_TO_AV = 2.21e21                   # cm^-2 mag^-1
NH2_TO_NH = 2                        # N(H) = 2 * N(H2)
H2_GRID_MIN = 1e21                   # cm^-2 (~A_V ~ 0.9)
H2_GRID_MAX = 5e23                   # cm^-2 (~A_V ~ 450)
ABUNDANCE_DEFAULT = {
    'H2O': oxygen_abundance,
    'CO':  carbon_abundance,
    'CO2': carbon_abundance,
}


def _wavelength_of_filter(filtername):
    return u.Quantity(int(filtername[1:-1]) / 100, u.um).to(u.um, u.spectral())


def _color_vs_column_panel(ax, dmag_tbl, entries, species,
                           color=DEFAULT_COLOR, abundance_wrt_h2=None,
                           show_phase_in_label=True):
    """Left-panel: ice+dust color vs N(H2) (with A_V and N(species) secondary
    x-axes), grouped by (author, phase). Multi-T groups get min/max envelope
    + median curve. Uses the same author color and phase linestyle as the
    opacity panel."""
    if abundance_wrt_h2 is None:
        abundance_wrt_h2 = ABUNDANCE_DEFAULT[species]

    h2_grid = np.geomspace(H2_GRID_MIN, H2_GRID_MAX, 200)
    av_grid = h2_grid * NH2_TO_NH / NH_TO_AV
    EVc = (ccd_ext(_wavelength_of_filter(color[0])) -
           ccd_ext(_wavelength_of_filter(color[1])))
    a_color = av_grid * EVc

    # composition is fixed per pure-species table
    comp = {'H2O': 'H2O (1)', 'CO': 'CO (1)', 'CO2': 'CO2 (1)'}[species]

    grouped = defaultdict(list)
    for tb, author, T, phase in entries:
        try:
            mol_id = _resolve_single_mol_id(dmag_tbl, author, comp, T)
        except Exception as ex:
            print(f"  left-panel skip {author} {T:g}K: {ex}")
            continue
        try:
            sub = (dmag_tbl
                   .loc['mol_id', mol_id]
                   .loc['composition', comp]
                   .loc['temperature', float(T)])
        except Exception as ex:
            print(f"  left-panel slice failed for {author} {T:g}K: {ex}")
            continue
        if not len(sub):
            continue
        n_species = h2_grid * abundance_wrt_h2
        try:
            dmag = compute_dmag_from_column(
                n_species, sub, icemol=species,
                maxcol=1e22, filter1=color[0], filter2=color[1],
                verbose=False,
            )
        except Exception as ex:
            print(f"  left-panel dmag failed for {author} {T:g}K: {ex}")
            continue
        yvals = dmag + a_color
        grouped[(author, phase)].append((T, yvals))

    phase_order = {'amorphous': 0, 'crystalline': 1, 'liquid': 2, 'unknown': 3}
    legend_handles = []
    for (author, phase), eg in sorted(grouped.items(),
                                      key=lambda kv: (phase_order[kv[0][1]], kv[0][0])):
        Ts = sorted(e[0] for e in eg)
        stack = np.array([e[1] for e in eg])
        c = AUTHOR_COLOR.get(author, None)
        ls = PHASE_LINESTYLE.get(phase, '-') if show_phase_in_label else '-'
        phase_tag = f" {phase}" if show_phase_in_label else ""
        if len(eg) >= 2:
            lo = np.nanmin(stack, axis=0)
            hi = np.nanmax(stack, axis=0)
            med = np.nanmedian(stack, axis=0)
            ax.fill_between(h2_grid, lo, hi, color=c, alpha=0.18, linewidth=0)
            ax.plot(h2_grid, med, color=c, linestyle=ls, alpha=0.95, linewidth=1.4)
            label = f"{author}{phase_tag} ({len(eg)} T: {Ts[0]:g}–{Ts[-1]:g} K)"
        else:
            ax.plot(h2_grid, stack[0], color=c, linestyle=ls, alpha=0.95, linewidth=1.4)
            label = f"{author}{phase_tag} ({Ts[0]:g} K)"
        legend_handles.append(Line2D([0], [0], color=c, linestyle=ls,
                                     linewidth=1.4, label=label))

    ax.set_xscale('log')
    ax.set_xlabel(r'N(H$_2$) [cm$^{-2}$]')
    ax.set_ylabel(rf'{color[0]} - {color[1]} (mag, ice + dust)')
    ax.grid(alpha=0.3)

    # secondary x-axis: A_V (top)
    secax_av = ax.secondary_xaxis(
        'top',
        functions=(lambda x: x * NH2_TO_NH / NH_TO_AV,
                   lambda x: x * NH_TO_AV / NH2_TO_NH),
    )
    secax_av.set_xlabel('A$_V$ (mag)')
    # hard-clamp the y-axis to ±5 mag (after secondary axes / autoscale)
    ax.set_ylim(-5, 5)
    return legend_handles


# ---------------- top-level driver ---------------- #

def make_two_panel(species, dmag_tbl, savedir, color=DEFAULT_COLOR):
    print(f"=== {species} two-panel ===")
    entries = _load_species_tables(species)
    print(f"  {len(entries)} valid tables")
    if not entries:
        print(f"  no entries for {species}; skipping")
        return None

    phases_present = {ph for _, _, _, ph in entries}
    show_phase = len(phases_present) > 1

    fig, (ax_left, ax_op) = pl.subplots(1, 2, figsize=(14, 6.5))
    left_handles = _color_vs_column_panel(ax_left, dmag_tbl, entries, species,
                                          color=color,
                                          show_phase_in_label=show_phase)
    right_handles = _opacity_panel(ax_op, entries, species,
                                   show_phase_in_label=show_phase,
                                   left_color=color)


    ylim = ax_left.get_ylim()
    if ylim[0] < -5:
        ax_left.set_ylim(bottom=-5)
    if ylim[1] > 5:
        ax_left.set_ylim(top=5)

    # Single combined legend (group key — color = author, linestyle = phase
    # if multiple phases present, shaded = T range across deposits). Left and
    # right panels share the same encoding so one legend suffices.
    seen = set()
    combined = []
    for h in list(left_handles) + list(right_handles):
        if h.get_label() in seen:
            continue
        seen.add(h.get_label())
        combined.append(h)
    legend_title = ('group (shaded = T range across deposits)'
                    if not show_phase else
                    'group (color = author, linestyle = phase, shaded = T range)')
    fig.legend(handles=combined, loc='lower center', ncol=4, fontsize=8,
               frameon=True, bbox_to_anchor=(0.5, -0.02),
               title=legend_title)

    species_label = {'H2O': r'H$_2$O', 'CO': 'CO', 'CO2': r'CO$_2$'}[species]
    abundance = ABUNDANCE_DEFAULT[species]
    fig.suptitle(
        rf"Pure {species_label} ice — left: ${color[0]}-{color[1]}$ "
        rf"(ice + dust) vs $N(\mathrm{{H_2}})$ at "
        rf"$N(${species_label}$)/N(\mathrm{{H_2}}) = {_fmt_sci(abundance)}$;  "
        rf"right: opacity profile",
        fontsize=11, y=0.90,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    out = os.path.join(
        savedir,
        f'twopanel_iceVariants_{species}_{color[0]}-{color[1]}.pdf')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    pl.close(fig)
    print(f"  saved {out}")
    return out


def make_two_panel_mixes(dmag_tbl, savedir, mixes=mixes2, name='mixes2',
                         color=DEFAULT_COLOR, abundance_wrt_h2=2.5e-4,
                         icemol='CO'):
    """Two-panel for the paper's standard ice mixture set (mixes2 by default).
    Left: ice+dust color vs N(H2) for each mix composition (one curve per
    entry, color cycles by entry index). Right: opacity vs wavelength loaded
    from the precomputed mymixes ecsv tables when available; otherwise
    rebuilt on the fly.
    """
    print(f"=== {name} two-panel ===")
    h2_grid = np.geomspace(H2_GRID_MIN, H2_GRID_MAX, 200)
    av_grid = h2_grid * NH2_TO_NH / NH_TO_AV
    EVc = (ccd_ext(_wavelength_of_filter(color[0])) -
           ccd_ext(_wavelength_of_filter(color[1])))
    a_color = av_grid * EVc

    fig, (ax_left, ax_op) = pl.subplots(1, 2, figsize=(14, 6.5))

    # color cycle
    cycle = pl.rcParams['axes.prop_cycle'].by_key().get('color',
            ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])

    _f_xlim = _right_panel_filters_xlim(color)[1]
    grid = np.linspace(min(_f_xlim[0], 3.0), max(_f_xlim[1], 5.2), 1500)
    legend_handles = []
    mymix_dir = ('/blue/adamginsburg/adamginsburg/repos/icemodels/'
                 'icemodels/data/mymixes')

    for ii, (composition, T) in enumerate(mixes):
        c = cycle[ii % len(cycle)]
        # ----- left panel: color vs N(H2) -----
        try:
            sub = (dmag_tbl
                   .loc['composition', composition]
                   .loc['temperature', float(T)])
        except Exception as ex:
            print(f"  skip left {composition} {T}K: {ex}")
            sub = None
        if sub is not None and len(sub):
            n_icemol = h2_grid * abundance_wrt_h2
            try:
                dmag = compute_dmag_from_column(
                    n_icemol, sub, icemol=icemol,
                    maxcol=1e22, filter1=color[0], filter2=color[1],
                    verbose=False)
                ax_left.plot(h2_grid, dmag + a_color, color=c, linewidth=1.3,
                             label=f"{composition} {T:g}K")
            except Exception as ex:
                print(f"  left dmag failed {composition}: {ex}")

        # ----- right panel: opacity vs wavelength -----
        # Use precomputed mymix file if it exists, else skip (rebuilding on
        # the fly requires component opacity tables).
        safe_comp = composition.replace(' ', '_').replace(':', ':')
        candidates = glob.glob(f"{mymix_dir}/{safe_comp}.ecsv") + \
                     glob.glob(f"{mymix_dir}/{composition.replace(' ', '_')}.ecsv")
        op_tab = None
        for cand in candidates:
            try:
                op_tab = Table.read(cand)
                break
            except Exception:
                continue
        if op_tab is None:
            print(f"  skip right {composition}: no mymix file")
            continue
        kk = op_tab['k₁'] if 'k₁' in op_tab.colnames else op_tab['k']
        molwt = u.Quantity(composition_to_molweight(composition), u.Da)
        try:
            op_per_mol = absorbed_spectrum(
                xarr=op_tab['Wavelength'], ice_column=1,
                ice_model_table=op_tab, molecular_weight=molwt,
                return_tau=True).to(u.cm**2).value
            op = op_per_mol / molwt.to(u.g).value     # cm^2 / g
        except Exception as ex:
            print(f"  right opacity failed {composition}: {ex}")
            continue
        wl = np.asarray(op_tab['Wavelength'], dtype=float)
        so = np.argsort(wl)
        op_g = np.interp(grid, wl[so], op[so], left=np.nan, right=np.nan)
        ax_op.plot(grid, op_g, color=c, linewidth=1.3,
                   label=f"{composition} {T:g}K")

        legend_handles.append(Line2D([0], [0], color=c, linewidth=1.3,
                                     label=f"{composition} {T:g}K"))

    # left panel cosmetics
    ax_left.set_xscale('log')
    ax_left.set_xlabel(r'$N(\mathrm{H_2})$ [cm$^{-2}$]')
    ax_left.set_ylabel(rf'${color[0]}-{color[1]}$ (mag, ice + dust)')
    ax_left.grid(alpha=0.3)
    secax = ax_left.secondary_xaxis(
        'top',
        functions=(lambda x: x * NH2_TO_NH / NH_TO_AV,
                   lambda x: x * NH_TO_AV / NH2_TO_NH))
    secax.set_xlabel(r'$A_V$ (mag)')
    ax_left.set_ylim(-5, 5)  # hard-clamp color axis to +/- 5 mag

    # right panel cosmetics
    ax_op.set_xlabel(r'Wavelength ($\mu$m)')
    ax_op.set_ylabel(r'$\kappa$ [cm$^{2}$ g$^{-1}$]  ($\tau = \kappa\,\Sigma_{ice}$)')
    ax_op.semilogy()
    ax_op.set_ylim(1, 1e5)
    _filters, _xlim = _right_panel_filters_xlim(color)
    ax_op.set_xlim(*_xlim)
    transmission_ax, tmax = _draw_filter_overlay(ax_op, filternames=_filters)
    for _fname in _filters:
        _wl0 = int(_fname[1:-1]) / 100.0
        transmission_ax.text(_wl0, tmax * 1.03, _fname, ha='center', fontsize=8)

    fig.legend(handles=legend_handles, loc='lower center', ncol=3, fontsize=8,
               frameon=True, bbox_to_anchor=(0.5, -0.02),
               title=f'{name} compositions')

    fig.suptitle(
        rf"Standard mix set ({name}) — left: ${color[0]}-{color[1]}$ "
        rf"(ice + dust) vs $N(\mathrm{{H_2}})$ at "
        rf"$N(\mathrm{{{icemol}}})/N(\mathrm{{H_2}}) = "
        rf"{_fmt_sci(abundance_wrt_h2)}$;  right: opacity profile",
        fontsize=11, y=0.97,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    out = os.path.join(
        savedir,
        f'twopanel_iceVariants_{name}_{color[0]}-{color[1]}.pdf')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    pl.close(fig)
    print(f"  saved {out}")
    return out


def _load_ehrenfreund_co_tables(require_overlap=(4.62, 4.70)):
    """Find every optical-constants table whose author or filename references
    Ehrenfreund and whose composition contains CO. Includes:
      * OCDB Ehrenfreund deposits (`*Ehrenfreund*.txt`)
      * Wayback E*.NK files (`wayback_ehrenfreund_E*.NK`) — these are the
        Ehrenfreund/Schutte ISODB tables retrieved from the Internet Archive
      * Schutte Dropbox files containing CO (raw experiment data)

    Wavelength range relaxed to [4.5, 4.8] um since the archived Ehrenfreund
    pure-CO tables only span the CO fundamental."""
    candidates_ocdb = sorted(set(
        glob.glob(f'{optical_constants_cache_dir}/*Ehrenfreund*.txt') +
        glob.glob(f'{optical_constants_cache_dir}/*ehrenfreund*.txt') +
        [fn for fn in glob.glob(f'{optical_constants_cache_dir}/wayback_*')
         if 'CO' in os.path.basename(fn).upper()
         and not fn.upper().endswith('.NK')] +
        [fn for fn in glob.glob(f'{optical_constants_cache_dir}/schutte_dropbox_*')
         if 'CO' in os.path.basename(fn).upper()]
    ))
    candidates_NK = sorted(glob.glob(
        f'{optical_constants_cache_dir}/wayback_ehrenfreund_E*.NK'))

    out = []
    for fn in candidates_ocdb:
        try:
            tb = read_ocdb_file(fn)
        except Exception:
            try:
                from icemodels.core import read_lida_file
                tb = read_lida_file(fn)
            except Exception as ex:
                print(f"  skip {os.path.basename(fn)}: read error: {ex}")
                continue
        comp = tb.meta.get('composition', '')
        if 'CO' not in comp.upper():
            continue
        wl = np.asarray(tb['Wavelength'], dtype=float)
        if not (np.isfinite(wl).any() and wl.min() <= require_overlap[0]
                and wl.max() >= require_overlap[1]):
            print(f"  skip {os.path.basename(fn)} ({comp}): wl "
                  f"{wl.min():.2f}-{wl.max():.2f}")
            continue
        author = tb.meta.get('author', 'Ehrenfreund')
        T = _temp_str_to_float(tb.meta.get('temperature', np.nan))
        out.append((tb, author, T, comp, fn))

    for fn in candidates_NK:
        try:
            tb = read_ehrenfreund_NK_file(fn)
        except Exception as ex:
            print(f"  skip {os.path.basename(fn)}: NK parse error: {ex}")
            continue
        comp = tb.meta.get('composition', '')
        # Token-level CO check: skip CO2-only / pure-CO2 tables.
        comp_tokens = [tok.split('(')[0].strip()
                       for tok in comp.split(':')]
        if 'CO' not in comp_tokens:
            continue
        wl = np.asarray(tb['Wavelength'], dtype=float)
        if not (np.isfinite(wl).any() and wl.min() <= require_overlap[0]
                and wl.max() >= require_overlap[1]):
            print(f"  skip {os.path.basename(fn)} ({comp}): wl "
                  f"{wl.min():.2f}-{wl.max():.2f}")
            continue
        T = _temp_str_to_float(tb.meta.get('temperature', np.nan))
        out.append((tb, 'Ehrenfreund', T, comp, fn))

    # Deduplicate by (composition, T): prefer shortest path (raw OCDB id over
    # `ocdb_<id>_` duplicate copies).
    by_key = {}
    for entry in out:
        key = (entry[3], entry[2])
        if key not in by_key or len(entry[4]) < len(by_key[key][4]):
            by_key[key] = entry
    return list(by_key.values())


def make_two_panel_co_ehrenfreund(dmag_tbl, savedir,
                                  color=DEFAULT_COLOR,
                                  abundance_wrt_h2=2.5e-4,
                                  icemol='CO'):
    """Two-panel figure dedicated to Ehrenfreund CO opacity variants
    (pure CO + CO-bearing mixes tagged with Ehrenfreund authorship)."""
    print("=== CO_ehrenfreund two-panel ===")
    entries = _load_ehrenfreund_co_tables()
    if not entries:
        print("  no Ehrenfreund CO tables found; skipping")
        return None
    print(f"  {len(entries)} Ehrenfreund CO-bearing tables")

    h2_grid = np.geomspace(H2_GRID_MIN, H2_GRID_MAX, 200)
    av_grid = h2_grid * NH2_TO_NH / NH_TO_AV
    EVc = (ccd_ext(_wavelength_of_filter(color[0])) -
           ccd_ext(_wavelength_of_filter(color[1])))
    a_color = av_grid * EVc
    _f_xlim = _right_panel_filters_xlim(color)[1]
    grid = np.linspace(min(_f_xlim[0], 3.0), max(_f_xlim[1], 5.2), 1500)

    fig, (ax_left, ax_op) = pl.subplots(1, 2, figsize=(14, 6.5))
    cycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    legend_handles = []

    for ii, (tb, author, T, comp, fn) in enumerate(entries):
        c = cycle[ii % len(cycle)]
        # ---- left panel: color vs N(H2) ----
        try:
            sub = (dmag_tbl
                   .loc['composition', comp]
                   .loc['temperature', float(T)])
        except Exception:
            sub = None
        if sub is not None and len(sub):
            n_icemol = h2_grid * abundance_wrt_h2
            try:
                dmag = compute_dmag_from_column(
                    n_icemol, sub, icemol=icemol, maxcol=1e22,
                    filter1=color[0], filter2=color[1], verbose=False)
                ax_left.plot(h2_grid, dmag + a_color, color=c, linewidth=1.3,
                             label=f"{comp} {T:g}K")
            except Exception as ex:
                print(f"  left dmag failed {comp} {T}K: {ex}")
        else:
            print(f"  no dmag entry for {comp} {T}K (skipped left panel)")

        # ---- right panel: opacity vs wavelength ----
        try:
            molwt = u.Quantity(composition_to_molweight(comp), u.Da)
            op_per_mol = absorbed_spectrum(
                xarr=tb['Wavelength'], ice_column=1,
                ice_model_table=tb, molecular_weight=molwt,
                return_tau=True).to(u.cm**2).value
            op = op_per_mol / molwt.to(u.g).value      # cm^2 / g
        except Exception as ex:
            print(f"  right opacity failed {comp}: {ex}")
            continue
        wl = np.asarray(tb['Wavelength'], dtype=float)
        so = np.argsort(wl)
        op_g = np.interp(grid, wl[so], op[so], left=np.nan, right=np.nan)
        ax_op.plot(grid, op_g, color=c, linewidth=1.3, label=f"{comp} {T:g}K")

        legend_handles.append(Line2D([0], [0], color=c, linewidth=1.3,
                                     label=f"{comp} {T:g}K"))

    # left panel cosmetics
    ax_left.set_xscale('log')
    ax_left.set_xlabel(r'$N(\mathrm{H_2})$ [cm$^{-2}$]')
    ax_left.set_ylabel(rf'${color[0]}-{color[1]}$ (mag, ice + dust)')
    ax_left.grid(alpha=0.3)
    secax = ax_left.secondary_xaxis(
        'top',
        functions=(lambda x: x * NH2_TO_NH / NH_TO_AV,
                   lambda x: x * NH_TO_AV / NH2_TO_NH))
    secax.set_xlabel(r'$A_V$ (mag)')
    ax_left.set_ylim(-5, 5)  # hard-clamp color axis to +/- 5 mag

    # right panel cosmetics
    ax_op.set_xlabel(r'Wavelength ($\mu$m)')
    ax_op.set_ylabel(r'$\kappa$ [cm$^{2}$ g$^{-1}$]  ($\tau = \kappa\,\Sigma_{ice}$)')
    ax_op.semilogy()
    ax_op.set_ylim(1, 1e5)
    _filters, _xlim = _right_panel_filters_xlim(color)
    ax_op.set_xlim(*_xlim)
    transmission_ax, tmax = _draw_filter_overlay(ax_op, filternames=_filters)
    for _fname in _filters:
        _wl0 = int(_fname[1:-1]) / 100.0
        transmission_ax.text(_wl0, tmax * 1.03, _fname, ha='center', fontsize=8)

    fig.legend(handles=legend_handles, loc='lower center', ncol=3, fontsize=8,
               frameon=True, bbox_to_anchor=(0.5, -0.02),
               title='Ehrenfreund CO-bearing opacity tables')
    fig.suptitle(
        rf"Ehrenfreund CO opacity variants — left: ${color[0]}-{color[1]}$ "
        rf"(ice + dust) vs $N(\mathrm{{H_2}})$ at "
        rf"$N(\mathrm{{{icemol}}})/N(\mathrm{{H_2}}) = "
        rf"{_fmt_sci(abundance_wrt_h2)}$;  right: opacity profile",
        fontsize=11, y=0.97,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    out = os.path.join(
        savedir,
        f'twopanel_iceVariants_CO_ehrenfreund_{color[0]}-{color[1]}.pdf')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    pl.close(fig)
    print(f"  saved {out}")
    return out


def _filter_avg_exp_tau(tau_lambda, wl_um, filtername):
    """Filter-transmission-weighted mean of exp(-tau(λ)).
    Returns a scalar. Approximates the stellar SED as flat across the filter
    (good to <1% for narrow JWST filters near peak SED of T=4000 K star)."""
    wt = SvoFps.get_transmission_data(f'JWST/NIRCam.{filtername}')
    fwl = wt['Wavelength'].quantity.to(u.um).value
    ftr = np.asarray(wt['Transmission'])
    # interpolate tau onto filter wavelength grid
    so = np.argsort(wl_um)
    tau_on_filter = np.interp(fwl, wl_um[so], tau_lambda[so],
                              left=0.0, right=0.0)
    weights = ftr / np.trapezoid(ftr, fwl)
    return np.trapezoid(np.exp(-tau_on_filter) * weights, fwl)


def make_two_panel_bergner(savedir, color=DEFAULT_COLOR,
                           abundance_wrt_h2=2.5e-4, icemol='CO',
                           require_overlap=(3.8, 4.75)):
    """Two-panel for the Bergner & Piacentino (2024) Zenodo ice library.

    Left: ice + dust color (F405N - F466N by default) vs N(H2). For each
    Bergner file we have a bulk mixture k(λ) plus the actual deposit column
    densities (Bergner Table 3). The dmag is computed on the fly by
    integrating exp(-tau(λ;N)) through F405N and F466N transmissions and
    rescaling to N(icemol)/N(H2) = abundance_wrt_h2.

    Right: opacity vs wavelength, grouped by Bergner sample with min/max
    envelope across deposit temperature.
    """
    from icemodels.core import read_bergner_file as _read_bergner
    print("=== Bergner two-panel ===")
    files = sorted(glob.glob(
        f'{optical_constants_cache_dir}/bergner_*.txt'))
    entries = []
    for fn in files:
        try:
            tb = _read_bergner(fn)
        except Exception as ex:
            print(f"  skip {os.path.basename(fn)}: {ex}")
            continue
        if 'k' not in tb.colnames:
            continue
        wl = np.asarray(tb['Wavelength'], dtype=float)
        if not (np.isfinite(wl).any() and wl.min() <= require_overlap[0]
                and wl.max() >= require_overlap[1]):
            continue
        sample = tb.meta.get('sample', os.path.basename(fn))
        T = tb.meta['temperature']
        comp = tb.meta.get('composition', sample)
        col_dens = tb.meta.get('column_densities_1e15_per_cm2', {}) or {}
        if not col_dens:
            continue
        entries.append((tb, sample, T, comp, col_dens))
    print(f"  {len(entries)} Bergner tables")
    if not entries:
        return None

    # group by sample for envelopes
    by_sample = defaultdict(list)
    for e in entries:
        by_sample[e[1]].append(e)
    print(f"  {len(by_sample)} unique samples")

    h2_grid = np.geomspace(H2_GRID_MIN, H2_GRID_MAX, 60)
    av_grid = h2_grid * NH2_TO_NH / NH_TO_AV
    EVc = (ccd_ext(_wavelength_of_filter(color[0])) -
           ccd_ext(_wavelength_of_filter(color[1])))
    a_color = av_grid * EVc
    _f_xlim = _right_panel_filters_xlim(color)[1]
    grid = np.linspace(min(_f_xlim[0], 3.0), max(_f_xlim[1], 5.2), 1500)

    # 20-color tab20 cycle so all 13 Bergner samples render in unique colors.
    import matplotlib.cm as _cm
    _cmap = _cm.get_cmap('tab20', 20)
    cycle = [_cmap(i) for i in range(20)]

    fig, (ax_left, ax_op) = pl.subplots(1, 2, figsize=(14, 6.5))
    legend_handles = []

    # Per-species ISM abundances X(species) = N(species_in_ice)/N(H2). These
    # are used to scale a deposit's bulk-mixture opacity to a chosen N(H2).
    # We pick the deposit's dominant constituent (by column density) for the
    # scaling so every Bergner sample has a left-panel curve, not just CO-
    # bearing ones. The choice of abundance only affects the absolute
    # x-position of the curve along N(H2), not its shape.
    species_abundance_defaults = {
        'CO':    abundance_wrt_h2,   # paper default (2.5e-4)
        'H2O':   1.0e-4,             # ~half of cosmic O in water ice
        'CO2':   3.0e-5,             # CO2/H2O ~ 0.3 in dense clouds
        'CH3OH': 5.0e-6,             # CH3OH/H2O ~ 0.05
        'CH4':   2.0e-6,
        'NH3':   2.0e-6,
        'O2':    1.0e-6,
        'OCS':   1.0e-7,
        'HCOOH': 1.0e-7,
    }

    # Priority order when picking which deposit constituent sets the
    # absolute scaling. CO is preferred (paper convention; CO is the
    # dominant F466N absorber so its abundance assumption controls the
    # blueing magnitude). H2O second (since it dominates ice mass and
    # F356W-F444W reddening). CO2 / CH3OH tertiary.
    species_priority = ['CO', 'H2O', 'CO2', 'CH3OH', 'NH3', 'CH4',
                        'O2', 'OCS', 'HCOOH']

    def _color_path(tb, col_dens):
        """For one Bergner spectrum, return color(F1)-color(F2) on the h2_grid.
        Scales by the highest-priority constituent present in the deposit
        using the assumed N(species)/N(H2) from ``species_abundance_defaults``.
        Tau scales linearly with N, so we evaluate at the deposit's actual
        column once and rescale to N(species)=N(H2)*abundance.
        """
        if not col_dens:
            return None
        species_for_scaling = None
        for sp in species_priority:
            if sp in col_dens:
                species_for_scaling = sp
                break
        if species_for_scaling is None:
            species_for_scaling = max(col_dens, key=col_dens.get)
        abund = species_abundance_defaults.get(species_for_scaling)
        if abund is None:
            return None
        n_ice_ref = col_dens[species_for_scaling] * 1e15 / u.cm**2
        molwt = u.Quantity(composition_to_molweight(tb.meta['composition']),
                           u.Da)
        tau1 = absorbed_spectrum(
            xarr=tb['Wavelength'], ice_column=n_ice_ref,
            ice_model_table=tb, molecular_weight=molwt,
            return_tau=True).value
        wl = np.asarray(tb['Wavelength'], dtype=float)
        n_species = h2_grid * abund        # N(species at given H2)
        scale = n_species / (col_dens[species_for_scaling] * 1e15)
        c1 = np.empty_like(h2_grid)
        c2 = np.empty_like(h2_grid)
        for ii, s in enumerate(scale):
            tau_s = tau1 * s
            c1[ii] = -2.5 * np.log10(_filter_avg_exp_tau(tau_s, wl, color[0]))
            c2[ii] = -2.5 * np.log10(_filter_avg_exp_tau(tau_s, wl, color[1]))
        return (c1 - c2)

    for ii, (sample, group) in enumerate(sorted(by_sample.items())):
        c = cycle[ii % len(cycle)]
        Ts = sorted(g[2] for g in group)
        # left panel
        color_paths = []
        for tb, sname, T, comp, col_dens in sorted(group, key=lambda g: g[2]):
            p = _color_path(tb, col_dens)
            if p is None:
                continue
            color_paths.append(p + a_color)
        if color_paths:
            stack = np.array(color_paths)
            if len(color_paths) >= 2:
                lo = np.nanmin(stack, axis=0)
                hi = np.nanmax(stack, axis=0)
                med = np.nanmedian(stack, axis=0)
                ax_left.fill_between(h2_grid, lo, hi, color=c, alpha=0.18,
                                     linewidth=0)
                ax_left.plot(h2_grid, med, color=c, linewidth=1.3)
            else:
                ax_left.plot(h2_grid, stack[0], color=c, linewidth=1.3)

        # right panel
        op_paths = []
        for tb, sname, T, comp, col_dens in sorted(group, key=lambda g: g[2]):
            molwt = u.Quantity(composition_to_molweight(tb.meta['composition']),
                               u.Da)
            try:
                op_per_mol = absorbed_spectrum(
                    xarr=tb['Wavelength'], ice_column=1,
                    ice_model_table=tb, molecular_weight=molwt,
                    return_tau=True).to(u.cm**2).value
                op = op_per_mol / molwt.to(u.g).value     # cm^2 / g
            except Exception as ex:
                print(f"    opacity failed {sample} {T}K: {ex}")
                continue
            wl = np.asarray(tb['Wavelength'], dtype=float)
            so = np.argsort(wl)
            op_paths.append(np.interp(grid, wl[so], op[so],
                                      left=np.nan, right=np.nan))
        if op_paths:
            stk = np.array(op_paths)
            if len(op_paths) >= 2:
                lo = np.nanmin(stk, axis=0)
                hi = np.nanmax(stk, axis=0)
                med = np.nanmedian(stk, axis=0)
                ax_op.fill_between(grid, lo, hi, color=c, alpha=0.18,
                                   linewidth=0)
                ax_op.plot(grid, med, color=c, linewidth=1.3)
            else:
                ax_op.plot(grid, stk[0], color=c, linewidth=1.3)

        comp_label = group[0][3]
        T_label = (f"{Ts[0]:g}–{Ts[-1]:g} K" if len(Ts) >= 2
                   else f"{Ts[0]:g} K")
        legend_handles.append(Line2D([0], [0], color=c, linewidth=1.4,
                                     label=f"{sample} ({comp_label}) {T_label}"))

    # left panel cosmetics
    ax_left.set_xscale('log')
    ax_left.set_xlabel(r'$N(\mathrm{H_2})$ [cm$^{-2}$]')
    ax_left.set_ylabel(rf'${color[0]}-{color[1]}$ (mag, ice + dust)')
    ax_left.grid(alpha=0.3)
    secax = ax_left.secondary_xaxis(
        'top',
        functions=(lambda x: x * NH2_TO_NH / NH_TO_AV,
                   lambda x: x * NH_TO_AV / NH2_TO_NH))
    secax.set_xlabel(r'$A_V$ (mag)')
    ax_left.set_ylim(-5, 5)

    # right panel cosmetics
    ax_op.set_xlabel(r'Wavelength ($\mu$m)')
    ax_op.set_ylabel(r'$\kappa$ [cm$^{2}$ g$^{-1}$]  ($\tau = \kappa\,\Sigma_{ice}$)')
    ax_op.semilogy()
    ax_op.set_ylim(1, 1e5)
    _filters, _xlim = _right_panel_filters_xlim(color)
    ax_op.set_xlim(*_xlim)
    transmission_ax, tmax = _draw_filter_overlay(ax_op, filternames=_filters)
    for _fname in _filters:
        _wl0 = int(_fname[1:-1]) / 100.0
        transmission_ax.text(_wl0, tmax * 1.03, _fname, ha='center', fontsize=8)

    fig.legend(handles=legend_handles, loc='lower center', ncol=3, fontsize=7,
               frameon=True, bbox_to_anchor=(0.5, -0.02),
               title='Bergner & Piacentino 2024 ice samples '
                     '(shaded = T range across deposits)')
    fig.suptitle(
        rf"Bergner+Piacentino 2024 ice library — left: ${color[0]}-{color[1]}$ "
        rf"(ice + dust) vs $N(\mathrm{{H_2}})$ at "
        rf"$N(\mathrm{{{icemol}}})/N(\mathrm{{H_2}}) = "
        rf"{_fmt_sci(abundance_wrt_h2)}$;  right: opacity profile",
        fontsize=11, y=0.97,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    out = os.path.join(
        savedir,
        f'twopanel_iceVariants_bergner_{color[0]}-{color[1]}.pdf')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    pl.close(fig)
    print(f"  saved {out}")
    return out


def make_two_panel_bluest_compare(dmag_tbl, savedir,
                                  bergner_sample='Polar-10-2-2',
                                  bergner_T=30,
                                  mixes_entry=('H2O:CO:CO2 (5:1:1)', 25.0),
                                  color=DEFAULT_COLOR,
                                  abundance_wrt_h2=2.5e-4,
                                  icemol='CO'):
    """Two-panel side-by-side comparison of the bluest Bergner sample and
    the bluest mixes2 entry. Left: ice + dust color vs N(H2). Right:
    opacity vs wavelength. Useful for diagnosing why Bergner deposits show
    less blueing than the synthetic mixes2 ratios."""
    print(f"=== bluest compare: Bergner {bergner_sample}@{bergner_T}K vs "
          f"mixes2 {mixes_entry} ===")
    from icemodels.core import read_bergner_file as _read_bergner
    h2_grid = np.geomspace(H2_GRID_MIN, H2_GRID_MAX, 200)
    av_grid = h2_grid * NH2_TO_NH / NH_TO_AV
    EVc = (ccd_ext(_wavelength_of_filter(color[0])) -
           ccd_ext(_wavelength_of_filter(color[1])))
    a_color = av_grid * EVc
    grid = np.linspace(3.7, 4.8, 1500)

    fig, (ax_left, ax_op) = pl.subplots(1, 2, figsize=(13, 5.5))

    # ---- Bergner curve ----
    matches = sorted(glob.glob(
        f'{optical_constants_cache_dir}/bergner_*_'
        f'{bergner_sample}_{bergner_T}K.txt'))
    if not matches:
        print(f"  no Bergner file for {bergner_sample} {bergner_T}K")
        return None
    tb_b = _read_bergner(matches[0], baseline_subtract=False)
    cd = tb_b.meta['column_densities_1e15_per_cm2']
    sp_b = 'CO' if 'CO' in cd else max(cd, key=cd.get)
    n_ref = cd[sp_b] * 1e15 / u.cm**2
    molwt_b = u.Quantity(composition_to_molweight(tb_b.meta['composition']),
                         u.Da)
    tau_b = absorbed_spectrum(xarr=tb_b['Wavelength'], ice_column=n_ref,
                              ice_model_table=tb_b, molecular_weight=molwt_b,
                              return_tau=True).value
    wl_b = np.asarray(tb_b['Wavelength'], dtype=float)
    n_obs = h2_grid * abundance_wrt_h2
    scale_b = n_obs / (cd[sp_b] * 1e15)
    c1_b = np.empty_like(h2_grid)
    c2_b = np.empty_like(h2_grid)
    for ii, s in enumerate(scale_b):
        c1_b[ii] = -2.5*np.log10(_filter_avg_exp_tau(tau_b*s, wl_b, color[0]))
        c2_b[ii] = -2.5*np.log10(_filter_avg_exp_tau(tau_b*s, wl_b, color[1]))
    diff_b_ice = c1_b - c2_b
    ax_left.plot(h2_grid, diff_b_ice + a_color, color='C0', linewidth=1.6,
                 label=f"Bergner {bergner_sample} {bergner_T} K (ice + dust)")
    ax_left.plot(h2_grid, diff_b_ice, color='C0', linewidth=1.0, linestyle=':',
                 alpha=0.6, label="  ice-only")
    op_b_per_mol = absorbed_spectrum(
        xarr=tb_b['Wavelength'], ice_column=1,
        ice_model_table=tb_b, molecular_weight=molwt_b,
        return_tau=True).to(u.cm**2).value
    op_b = op_b_per_mol / molwt_b.to(u.g).value     # cm^2 / g
    so_b = np.argsort(wl_b)
    op_b_g = np.interp(grid, wl_b[so_b], op_b[so_b], left=np.nan, right=np.nan)
    ax_op.plot(grid, op_b_g, color='C0', linewidth=1.4,
               label=f"Bergner {bergner_sample} {bergner_T} K")

    # ---- mixes2 curve ----
    comp, T = mixes_entry
    sub = dmag_tbl.loc['composition', comp].loc['temperature', float(T)]
    # opacity for mixes2: load from mymix file. Use this same k(lambda) for
    # both panels — left dmag is computed on the fly via filter integrals
    # (matches the Bergner pipeline) so the right-panel curve and the
    # left-panel curve are guaranteed consistent.
    mymix_path = ('/blue/adamginsburg/adamginsburg/repos/icemodels/'
                  f'icemodels/data/mymixes/{comp.replace(" ", "_")}.ecsv')
    try:
        op_tab = Table.read(mymix_path)
        if 'k' not in op_tab.colnames and 'k₁' in op_tab.colnames:
            op_tab['k'] = op_tab['k₁']
        molwt_m = u.Quantity(composition_to_molweight(comp), u.Da)
        # right panel curve
        op_m_per_mol = absorbed_spectrum(
            xarr=op_tab['Wavelength'], ice_column=1,
            ice_model_table=op_tab, molecular_weight=molwt_m,
            return_tau=True).to(u.cm**2).value
        op_m = op_m_per_mol / molwt_m.to(u.g).value     # cm^2 / g
        wl_m = np.asarray(op_tab['Wavelength'], dtype=float)
        so_m = np.argsort(wl_m)
        op_m_g = np.interp(grid, wl_m[so_m], op_m[so_m],
                           left=np.nan, right=np.nan)
        ax_op.plot(grid, op_m_g, color='C3', linewidth=1.4,
                   label=f"mixes2 {comp} {T:g} K")
        # left panel: compute dmag on the fly from same k(lambda)
        # mixes2 composition keeps mol_fractions; CO mol fraction:
        from icemodels.core import molscomps
        mols, comps = molscomps(comp)
        co_frac = (comps[mols.index('CO')] / sum(comps)) if 'CO' in mols else 1.0
        # tau at deposit ref column = 1 / cm^2 -> nonsense; instead pick
        # n_ref = 1e18 cm^-2 *for CO* and back out the total ice column
        # such that N(CO_in_mix) = 1e18.
        n_co_ref = 1e18 / u.cm**2
        n_total_ref = n_co_ref / co_frac
        tau_m = absorbed_spectrum(xarr=op_tab['Wavelength'],
                                  ice_column=n_total_ref,
                                  ice_model_table=op_tab,
                                  molecular_weight=molwt_m,
                                  return_tau=True).value
        # for each N(H2): N(CO_obs) = h2_grid * abundance, scale tau by
        # N(CO_obs)/n_co_ref
        scale_m = (h2_grid * abundance_wrt_h2) / 1e18
        c1_m = np.empty_like(h2_grid)
        c2_m = np.empty_like(h2_grid)
        for ii, s in enumerate(scale_m):
            c1_m[ii] = -2.5*np.log10(_filter_avg_exp_tau(tau_m*s, wl_m, color[0]))
            c2_m[ii] = -2.5*np.log10(_filter_avg_exp_tau(tau_m*s, wl_m, color[1]))
        diff_m_ice = c1_m - c2_m
        ax_left.plot(h2_grid, diff_m_ice + a_color, color='C3', linewidth=1.6,
                     label=f"mixes2 {comp} {T:g} K (ice + dust)")
        ax_left.plot(h2_grid, diff_m_ice, color='C3', linewidth=1.0,
                     linestyle=':', alpha=0.6, label="  ice-only")
    except Exception as ex:
        print(f"  could not load/compute mixes2: {ex}")

    # left panel cosmetics
    ax_left.set_xscale('log')
    ax_left.set_xlabel(r'$N(\mathrm{H_2})$ [cm$^{-2}$]')
    ax_left.set_ylabel(rf'${color[0]}-{color[1]}$ (mag)')
    ax_left.grid(alpha=0.3)
    secax = ax_left.secondary_xaxis(
        'top',
        functions=(lambda x: x * NH2_TO_NH / NH_TO_AV,
                   lambda x: x * NH_TO_AV / NH2_TO_NH))
    secax.set_xlabel(r'$A_V$ (mag)')
    ax_left.set_ylim(-2.5, 0.5)
    ax_left.legend(fontsize=7, loc='best')

    # right panel cosmetics
    ax_op.set_xlabel(r'Wavelength ($\mu$m)')
    ax_op.set_ylabel(r'$\kappa$ [cm$^{2}$ g$^{-1}$]  ($\tau = \kappa\,\Sigma_{ice}$)')
    ax_op.semilogy()
    ax_op.set_ylim(1, 1e5)
    ax_op.set_xlim(3.71, 4.75)
    transmission_ax, tmax = _draw_filter_overlay(
        ax_op, filternames=('F405N', 'F410M', 'F466N'))
    for fname in ('F405N', 'F410M', 'F466N'):
        wl0 = int(fname[1:-1]) / 100.0
        transmission_ax.text(wl0, tmax * 1.03, fname, ha='center', fontsize=8)
    ax_op.legend(fontsize=8, loc='upper left')

    fig.suptitle(
        rf"Bluest models: Bergner {bergner_sample} vs mixes2 {comp}; "
        rf"$N(\mathrm{{{icemol}}})/N(\mathrm{{H_2}})={_fmt_sci(abundance_wrt_h2)}$",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(savedir,
                       f'twopanel_iceVariants_BLUEST_compare_{color[0]}-{color[1]}.pdf')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    pl.close(fig)
    print(f"  saved {out}")
    return out


def make_bergner_ccd(savedir, color1, color2,
                     abundance_wrt_h2=2.5e-4, icemol='CO',
                     baseline_subtract=False):
    """Plot Bergner deposits on a (color1, color2) CCD.

    color1 = (F_a, F_b) for x-axis (F_a - F_b); color2 = (F_c, F_d) for y.
    For each Bergner sample we compute on-the-fly the dmag in each filter
    via filter-transmission-weighted exp(-tau), then add foreground dust
    reddening using CT06_MWGC. Baseline subtraction is OFF by default
    because the polynomial baseline clips real CO band wings.
    """
    from icemodels.core import read_bergner_file as _read_bergner
    print(f"=== Bergner CCD: {color1} vs {color2} ===")

    h2_grid = np.geomspace(H2_GRID_MIN, H2_GRID_MAX, 200)
    av_grid = h2_grid * NH2_TO_NH / NH_TO_AV
    EVc1 = (ccd_ext(_wavelength_of_filter(color1[0])) -
            ccd_ext(_wavelength_of_filter(color1[1])))
    EVc2 = (ccd_ext(_wavelength_of_filter(color2[0])) -
            ccd_ext(_wavelength_of_filter(color2[1])))
    a1 = av_grid * EVc1
    a2 = av_grid * EVc2

    files = sorted(glob.glob(
        f'{optical_constants_cache_dir}/bergner_*.txt'))
    by_sample = defaultdict(list)
    for fn in files:
        try:
            tb = _read_bergner(fn, baseline_subtract=baseline_subtract)
        except Exception as ex:
            continue
        if 'k' not in tb.colnames:
            continue
        cd = tb.meta.get('column_densities_1e15_per_cm2', {}) or {}
        if not cd:
            continue
        wl = np.asarray(tb['Wavelength'], dtype=float)
        # require coverage from F182M (1.82 um) through F466N (4.66 um). Use
        # the requested filter wavelengths to set the bound (NIR side from
        # min(filt) - 0.05 um, IR side from max(filt) + 0.05 um).
        filter_wls = [int(f[1:-1]) / 100.0
                      for f in (color1[0], color1[1], color2[0], color2[1])]
        wl_lo = min(filter_wls) - 0.05
        wl_hi = max(filter_wls) + 0.05
        if not (np.isfinite(wl).any() and wl.min() <= wl_lo and wl.max() >= wl_hi):
            continue
        sample = tb.meta['sample']
        by_sample[sample].append((tb.meta['temperature'], tb, cd))

    if not by_sample:
        print("  no Bergner deposits cover the required wavelength range")
        return None

    species_priority = ['CO', 'H2O', 'CO2', 'CH3OH', 'NH3', 'CH4',
                        'O2', 'OCS', 'HCOOH']
    abund_defaults = {
        'CO': abundance_wrt_h2, 'H2O': 1.0e-4, 'CO2': 3.0e-5,
        'CH3OH': 5.0e-6, 'NH3': 2.0e-6, 'CH4': 2.0e-6,
        'O2': 1.0e-6, 'OCS': 1.0e-7, 'HCOOH': 1.0e-7,
    }

    import matplotlib.cm as _cm
    cmap = _cm.get_cmap('tab20', 20)

    fig, ax = pl.subplots(figsize=(9, 7))
    legend_handles = []
    for ii, (sample, eg) in enumerate(sorted(by_sample.items())):
        eg.sort(key=lambda x: x[0])    # by T
        c = cmap(ii % 20)
        c1_paths = []
        c2_paths = []
        Ts = []
        for T, tb, cd in eg:
            sp = next((s for s in species_priority if s in cd), None)
            if sp is None:
                sp = max(cd, key=cd.get)
            ab = abund_defaults.get(sp)
            if ab is None:
                continue
            n_ref = cd[sp] * 1e15 / u.cm**2
            molwt = u.Quantity(composition_to_molweight(tb.meta['composition']),
                               u.Da)
            try:
                tau1 = absorbed_spectrum(
                    xarr=tb['Wavelength'], ice_column=n_ref,
                    ice_model_table=tb, molecular_weight=molwt,
                    return_tau=True).value
            except Exception:
                continue
            wl = np.asarray(tb['Wavelength'], dtype=float)
            scale = (h2_grid * ab) / (cd[sp] * 1e15)
            c1 = np.empty_like(h2_grid)
            c2 = np.empty_like(h2_grid)
            for jj, s in enumerate(scale):
                tau_s = tau1 * s
                d_a = -2.5*np.log10(_filter_avg_exp_tau(tau_s, wl, color1[0]))
                d_b = -2.5*np.log10(_filter_avg_exp_tau(tau_s, wl, color1[1]))
                d_cc = -2.5*np.log10(_filter_avg_exp_tau(tau_s, wl, color2[0]))
                d_d = -2.5*np.log10(_filter_avg_exp_tau(tau_s, wl, color2[1]))
                c1[jj] = (d_a - d_b) + a1[jj]
                c2[jj] = (d_cc - d_d) + a2[jj]
            c1_paths.append(c1)
            c2_paths.append(c2)
            Ts.append(T)
        if not c1_paths:
            continue
        # plot the deposit at the lowest temperature for each sample
        ax.plot(c1_paths[0], c2_paths[0], color=c, linewidth=1.4,
                label=f"{sample} {Ts[0]:g}-{Ts[-1]:g}K")
        legend_handles.append(Line2D([0], [0], color=c, linewidth=1.4,
                                     label=f"{sample} ({Ts[0]:g}-{Ts[-1]:g} K)"))

    # plot a pure-dust reddening line (extinction only, no ice)
    ax.plot(a1, a2, color='k', linewidth=2, linestyle='--',
            label='pure dust (CT06 MWGC)')

    ax.set_xlabel(f'{color1[0]} - {color1[1]}')
    ax.set_ylabel(f'{color2[0]} - {color2[1]}')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=7)
    ax.set_title(
        rf"Bergner ices on CCD ({color1[0]}-{color1[1]}) vs "
        rf"({color2[0]}-{color2[1]});  "
        rf"$N(\mathrm{{{icemol}}})/N(\mathrm{{H_2}}) = "
        rf"{_fmt_sci(abundance_wrt_h2)}$, baseline_subtract={baseline_subtract}",
        fontsize=10,
    )
    fig.tight_layout()
    out = os.path.join(
        savedir,
        f'CCD_bergner_{color1[0]}-{color1[1]}_{color2[0]}-{color2[1]}.pdf')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    pl.close(fig)
    print(f"  saved {out}")
    return out


if __name__ == '__main__':
    savedir = '/orange/adamginsburg/ice/colors_of_ices_overleaf/figures'
    os.makedirs(savedir, exist_ok=True)

    dmag_tbl = Table.read(
        '/blue/adamginsburg/adamginsburg/repos/icemodels/icemodels/data/'
        'combined_ice_absorption_tables.ecsv'
    )
    for col in ('mol_id', 'composition', 'temperature', 'database', 'author'):
        dmag_tbl.add_index(col)

    color_pairs = (
        ('F405N', 'F466N'),
        ('F405N', 'F410M'),
        ('F356W', 'F444W'),
    )
    with warnings.catch_warnings():
        warnings.simplefilter('always')
        for color in color_pairs:
            print(f"\n########## color = {color[0]} - {color[1]} ##########")
            for species in ('H2O', 'CO', 'CO2'):
                make_two_panel(species, dmag_tbl, savedir, color=color)
            make_two_panel_mixes(dmag_tbl, savedir, color=color)
            make_two_panel_co_ehrenfreund(dmag_tbl, savedir, color=color)
            make_two_panel_bergner(savedir, color=color)
        # Bergner vs mixes2 at IDENTICAL component ratios (Polar-10-2-2 = 5:1:1)
        make_two_panel_bluest_compare(
            dmag_tbl, savedir,
            bergner_sample='Polar-10-2-2', bergner_T=30,
            mixes_entry=('H2O:CO:CO2 (5:1:1)', 25.0))
        # Bergner CCD plots — many color pairs. Baseline-subtract OFF: the
        # F182M & F212N filters sit on Bergner's noise floor anyway, but the
        # baseline polynomial would clip the CO band wings on F466N.
        bergner_ccd_color_pairs = [
            (('F182M', 'F212N'), ('F212N', 'F466N')),
            (('F182M', 'F212N'), ('F405N', 'F466N')),
            (('F182M', 'F212N'), ('F405N', 'F410M')),
            (('F356W', 'F444W'), ('F405N', 'F466N')),
            (('F405N', 'F410M'), ('F405N', 'F466N')),
            (('F200W', 'F356W'), ('F356W', 'F444W')),
        ]
        for c1, c2 in bergner_ccd_color_pairs:
            make_bergner_ccd(savedir, c1, c2)
        print("\n=== all panels done ===")
