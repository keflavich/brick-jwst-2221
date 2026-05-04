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
                            absorbed_spectrum, composition_to_molweight)
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


def _opacity_panel(ax, entries, species, show_phase_in_label=True):
    """Right-panel: opacity vs wavelength, grouped by (author, phase). For
    multi-T groups, plot min/max envelope + median curve."""
    grid = np.linspace(3.7, 4.8, 1100)
    grouped = defaultdict(list)
    for tb, author, T, phase in entries:
        molwt = u.Quantity(composition_to_molweight(tb.meta['composition']),
                           u.Da)
        op = absorbed_spectrum(xarr=tb['Wavelength'], ice_column=1,
                               ice_model_table=tb, molecular_weight=molwt,
                               return_tau=True).to(u.cm**2).value
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
    species_label = {'H2O': r'H$_2$O', 'CO': 'CO', 'CO2': r'CO$_2$'}[species]
    ax.set_ylabel(rf'$\kappa_{{eff}}$ [$\tau = \kappa_{{eff}}\,N$({species_label})]')
    ax.semilogy()
    ax.set_ylim(1e-21, 1e-17)
    ax.set_xlim(3.71, 4.75)

    transmission_ax, tmax = _draw_filter_overlay(ax)
    transmission_ax.text(4.66, tmax * 1.03, 'F466N', ha='center', fontsize=8)
    transmission_ax.text(4.10, tmax * 1.03, 'F410M', ha='center', fontsize=8)
    transmission_ax.text(4.05, tmax * 1.03, 'F405N', ha='center', fontsize=8)

    return legend_handles


DEFAULT_COLOR = ('F405N', 'F466N')   # filter pair for the left-panel y-axis
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

    ylim = ax.get_ylim()
    if ylim[0] < -5:
        ax.set_ylim(bottom=-5)
    if ylim[1] > 5:
        ax.set_ylim(top=5)

    # secondary x-axis: A_V (top)
    secax_av = ax.secondary_xaxis(
        'top',
        functions=(lambda x: x * NH2_TO_NH / NH_TO_AV,
                   lambda x: x * NH_TO_AV / NH2_TO_NH),
    )
    secax_av.set_xlabel('A$_V$ (mag)')
    return legend_handles


# ---------------- top-level driver ---------------- #

def make_two_panel(species, dmag_tbl, savedir):
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
                                          show_phase_in_label=show_phase)
    right_handles = _opacity_panel(ax_op, entries, species,
                                   show_phase_in_label=show_phase)

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
    color = DEFAULT_COLOR
    fig.suptitle(
        rf"Pure {species_label} ice — left: ${color[0]}-{color[1]}$ "
        rf"(ice + dust) vs $N(\mathrm{{H_2}})$ at "
        rf"$N(${species_label}$)/N(\mathrm{{H_2}}) = {_fmt_sci(abundance)}$;  "
        rf"right: opacity profile",
        fontsize=11, y=0.90,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    out = os.path.join(savedir, f'twopanel_iceVariants_{species}.pdf')
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

    grid = np.linspace(3.7, 4.8, 1100)
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
            op = absorbed_spectrum(xarr=op_tab['Wavelength'], ice_column=1,
                                   ice_model_table=op_tab,
                                   molecular_weight=molwt,
                                   return_tau=True).to(u.cm**2).value
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

    # right panel cosmetics
    ax_op.set_xlabel(r'Wavelength ($\mu$m)')
    ax_op.set_ylabel(r'$\kappa_{eff}$ [$\tau = \kappa_{eff}\,N$(mix)]')
    ax_op.semilogy()
    ax_op.set_ylim(1e-21, 1e-17)
    ax_op.set_xlim(3.71, 4.75)
    transmission_ax, tmax = _draw_filter_overlay(ax_op)
    transmission_ax.text(4.66, tmax * 1.03, 'F466N', ha='center', fontsize=8)
    transmission_ax.text(4.10, tmax * 1.03, 'F410M', ha='center', fontsize=8)
    transmission_ax.text(4.05, tmax * 0.90, 'F405N', ha='center', fontsize=8)

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
    out = os.path.join(savedir, f'twopanel_iceVariants_{name}.pdf')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    pl.close(fig)
    print(f"  saved {out}")
    return out


def _load_ehrenfreund_co_tables(require_overlap=(4.5, 4.8)):
    """Find every optical-constants table whose author or filename references
    Ehrenfreund and whose composition contains CO. Includes pure CO + mixes
    (CO:CO2 etc). Wavelength range relaxed to [4.5, 4.8] um since the
    archived Ehrenfreund pure-CO tables only span the CO fundamental."""
    candidates = sorted(set(
        glob.glob(f'{optical_constants_cache_dir}/*Ehrenfreund*.txt') +
        glob.glob(f'{optical_constants_cache_dir}/*ehrenfreund*.txt') +
        [fn for fn in glob.glob(f'{optical_constants_cache_dir}/wayback_*')
         if 'CO' in os.path.basename(fn).upper()] +
        [fn for fn in glob.glob(f'{optical_constants_cache_dir}/schutte_dropbox_*')
         if 'CO' in os.path.basename(fn).upper()]
    ))
    out = []
    for fn in candidates:
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
    grid = np.linspace(3.7, 4.8, 1100)

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
            op = absorbed_spectrum(xarr=tb['Wavelength'], ice_column=1,
                                   ice_model_table=tb,
                                   molecular_weight=molwt,
                                   return_tau=True).to(u.cm**2).value
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

    # right panel cosmetics
    ax_op.set_xlabel(r'Wavelength ($\mu$m)')
    ax_op.set_ylabel(r'$\kappa_{eff}$ [$\tau = \kappa_{eff}\,N$(mix)]')
    ax_op.semilogy()
    ax_op.set_ylim(1e-21, 1e-17)
    ax_op.set_xlim(3.71, 4.75)
    transmission_ax, tmax = _draw_filter_overlay(ax_op)
    transmission_ax.text(4.66, tmax * 1.03, 'F466N', ha='center', fontsize=8)
    transmission_ax.text(4.10, tmax * 1.03, 'F410M', ha='center', fontsize=8)
    transmission_ax.text(4.05, tmax * 0.90, 'F405N', ha='center', fontsize=8)

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
    out = os.path.join(savedir, 'twopanel_iceVariants_CO_ehrenfreund.pdf')
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

    with warnings.catch_warnings():
        warnings.simplefilter('default')
        for species in ('H2O', 'CO', 'CO2'):
            make_two_panel(species, dmag_tbl, savedir)
        make_two_panel_mixes(dmag_tbl, savedir)
        make_two_panel_co_ehrenfreund(dmag_tbl, savedir)
