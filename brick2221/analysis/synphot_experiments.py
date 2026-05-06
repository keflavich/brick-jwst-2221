"""Run synphot experiments for close-color JWST CCD uncertainty envelopes.

This script quantifies how much spread in the close colors
F405N-F410M and F187N-F182M can be produced from only:

1) stellar atmosphere diversity across available synthetic grids, and
2) CT06 Galactic Center extinction over Av=0..50.

No line emission or ice absorption terms are included.
"""

import argparse
import hashlib
import json
import os
import time
import warnings

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.constants import R_sun
from astropy.table import Table
from astropy.table import vstack
from astroquery.svo_fps import SvoFps
from dust_extinction.averages import CT06_MWGC
from stsynphot import Vega, grid_to_spec
from stsynphot.catalog import get_catalog_index
from stsynphot.exceptions import ParameterOutOfBounds
from synphot import Observation, SourceSpectrum, SpectralElement
from synphot.exceptions import SynphotError
from synphot.models import Empirical1D

try:
	from tqdm.auto import tqdm
except ImportError:
	tqdm = None

try:
	from brick2221.analysis.paths import basepath
except ImportError:
	basepath = '/blue/adamginsburg/adamginsburg/jwst/brick/'


FILTERS = ("F182M", "F187N", "F405N", "F410M")
LIGHT_SPEED_KMS = 299792.458
LINEWIDTH_KMS = 10.0
LINE_PEAK_JY_GRID = (0.01, 0.1, 1.0, 10.0)
STELLAR_CLASS_STYLE = {
	'giant': {'color': '#E15759', 'marker': 'o', 'label': 'giant'},
	'red_clump': {'color': '#4E79A7', 'marker': 's', 'label': 'red clump'},
	'dwarf': {'color': '#59A14F', 'marker': '^', 'label': 'dwarf'},
	'intermediate': {'color': '#F28E2B', 'marker': 'D', 'label': 'intermediate'},
}
LINE_MARKERS = {
	0.01: 'o',
	0.1: 's',
	1.0: '^',
	10.0: 'D',
}
LINE_DEFINITIONS = {
	'paa': {'label': 'Pa alpha', 'center_um': 1.8756},
	'bra': {'label': 'Br alpha', 'center_um': 4.0523},
}


def resolve_svo_filter_id(filter_name, default_facility='JWST/NIRCam'):
	"""Resolve short filter names to SVO IDs, while allowing full non-JWST IDs."""
	if '/' in str(filter_name):
		return str(filter_name)
	return f'{default_facility}.{filter_name}'


def model_rows_signature(model_rows):
	"""Create a stable hash signature for a model row list."""
	text = "\n".join(
		f"{row['grid']}:{row['teff']:0.3f}:{row['metallicity']:0.3f}:{row['logg']:0.3f}"
		for row in model_rows
	)
	return hashlib.sha256(text.encode('utf-8')).hexdigest()


def make_synphot_cache_key(
	model_rows,
	av_grid,
	wave_step_angstrom,
	distance_pc,
	injected_lines=None,
):
	"""Build a cache key from parameters controlling synthetic photometry output."""
	line_payload = []
	if injected_lines:
		for line in injected_lines:
			line_payload.append(
				{
					'tag': str(line['tag']),
					'center_um': float(line['center_um']),
					'peak_jy': float(line['peak_jy']),
					'fwhm_kms': float(line['fwhm_kms']),
				}
			)

	payload = {
		'filters': list(FILTERS),
		'av_grid': [float(val) for val in av_grid],
		'wave_step_angstrom': float(wave_step_angstrom),
		'distance_pc': float(distance_pc),
		'model_rows_sig': model_rows_signature(model_rows),
		'injected_lines': line_payload,
	}
	txt = json.dumps(payload, sort_keys=True)
	return hashlib.sha256(txt.encode('utf-8')).hexdigest()[:16]



def build_bandpasses(filters=FILTERS, default_facility='JWST/NIRCam'):
	"""Load passbands from SVO and convert to synphot elements."""
	bandpasses = {}
	wave_ranges_um = []
	for filt in filters:
		filter_id = resolve_svo_filter_id(filt, default_facility=default_facility)
		trans = SvoFps.get_transmission_data(filter_id)
		waves = trans['Wavelength'].quantity
		throughput = np.array(trans['Transmission'], dtype=float)
		keep = throughput > 1e-5
		wave_ranges_um.append((waves[keep].min().to_value(u.um), waves[keep].max().to_value(u.um)))
		bandpasses[filt] = SpectralElement(Empirical1D, points=waves, lookup_table=throughput)

	wave_min_um = min(item[0] for item in wave_ranges_um)
	wave_max_um = max(item[1] for item in wave_ranges_um)
	return bandpasses, wave_min_um, wave_max_um


def collect_model_rows(grids, max_models=None):
	"""Collect available model grid rows from STScI catalog indices."""
	rows = []
	for gridname in grids:
		index_rows, _base_dir = get_catalog_index(gridname)
		for row in index_rows:
			rows.append(
				{
					'grid': gridname,
					'teff': float(row[0]),
					'metallicity': float(row[1]),
					'logg': float(row[2]),
				}
			)

	rows.sort(key=lambda rr: (rr['grid'], rr['teff'], rr['metallicity'], rr['logg']))
	if max_models is not None and max_models < len(rows):
		keep_idx = np.linspace(0, len(rows) - 1, max_models).astype(int)
		rows = [rows[ii] for ii in keep_idx]

	return rows


def collect_evolved_model_rows(
	grids,
	teff_min=3000.0,
	teff_max=4000.0,
	giant_logg_max=2.0,
	red_clump_logg_min=2.3,
	red_clump_logg_max=2.8,
	max_models=24,
):
	"""Collect giant + red-clump models in the requested Teff range from synthetic grids."""
	rows = collect_model_rows(grids=grids, max_models=None)
	selected = [
		row for row in rows
		if (
			teff_min <= row['teff'] <= teff_max
			and (
				row['logg'] <= giant_logg_max
				or (red_clump_logg_min <= row['logg'] <= red_clump_logg_max)
			)
		)
	]

	selected.sort(key=lambda rr: (rr['teff'], rr['logg'], rr['metallicity'], rr['grid']))
	if max_models is not None and max_models < len(selected):
		keep_idx = np.linspace(0, len(selected) - 1, max_models).astype(int)
		selected = [selected[ii] for ii in keep_idx]

	return selected


def select_by_metallicity(models, min_metallicity=0.0):
	"""Keep only models with metallicity at or above threshold."""
	return [row for row in models if row['metallicity'] >= min_metallicity]


def filter_compute_models(model_rows, subset='all'):
	"""Filter full model grid to a science-motivated subset for compute tractability."""
	if subset == 'all':
		return model_rows

	def evolved_sel(row, teff_max, solarplus=False):
		is_evolved = (row['logg'] <= 2.0) or (2.3 <= row['logg'] <= 2.8)
		is_teff = 3000.0 <= row['teff'] <= teff_max
		is_metal = row['metallicity'] >= 0.0 if solarplus else True
		return is_evolved and is_teff and is_metal

	if subset == 'evolved_3000_4000':
		return [row for row in model_rows if evolved_sel(row, teff_max=4000.0, solarplus=False)]
	if subset == 'evolved_3000_4500':
		return [row for row in model_rows if evolved_sel(row, teff_max=4500.0, solarplus=False)]
	if subset == 'evolved_3000_4000_solarplus':
		return [row for row in model_rows if evolved_sel(row, teff_max=4000.0, solarplus=True)]
	if subset == 'solar_3000_5000_logg_lt3':
		return [
			row for row in model_rows
			if (3000.0 <= row['teff'] <= 5000.0) and (row['logg'] < 3.0) and np.isclose(row['metallicity'], 0.0)
		]

	raise ValueError(f'Unknown subset={subset}')


def stellar_class_from_logg(logg):
	if logg <= 2.0:
		return 'giant'
	if 2.3 <= logg <= 2.8:
		return 'red_clump'
	if logg >= 4.0:
		return 'dwarf'
	return 'intermediate'


def distance_modulus(distance_pc):
	"""Return distance modulus for distance in parsec."""
	return 5.0 * np.log10(float(distance_pc)) - 5.0


def add_gaussian_emission_lines(reddened_flux, wave_grid, injected_lines):
	"""Add Gaussian emission lines parameterized by peak Fnu (Jy) and velocity FWHM."""
	if not injected_lines:
		return reddened_flux

	out_flux = reddened_flux.copy()
	wave_um = wave_grid.to_value(u.um)

	for line in injected_lines:
		center_um = float(line['center_um'])
		peak_jy = float(line['peak_jy'])
		fwhm_kms = float(line['fwhm_kms'])

		sigma_um = center_um * (fwhm_kms / LIGHT_SPEED_KMS) / 2.354820045
		if sigma_um <= 0:
			continue

		profile = np.exp(-0.5 * ((wave_um - center_um) / sigma_um) ** 2)
		delta_fnu = (peak_jy * profile) * u.Jy
		delta_flam = delta_fnu.to(reddened_flux.unit, equivalencies=u.spectral_density(wave_grid))
		out_flux = out_flux + delta_flam

	return out_flux


def compute_single_model_magnitudes(
	model,
	bandpasses,
	wave_min_um,
	wave_max_um,
	wave_step_angstrom=10.0,
	av=0.0,
	distance_pc=10.0,
	radius_rsun=None,
):
	"""Compute absolute Vega magnitudes for one model with optional radius scaling."""
	wave_grid = np.arange(wave_min_um * 1e4, wave_max_um * 1e4 + wave_step_angstrom, wave_step_angstrom) * u.AA
	wave_grid_um = wave_grid.to(u.um)
	flux = grid_to_spec(model['grid'], model['teff'], model['metallicity'], model['logg'])(wave_grid)

	if radius_rsun is not None:
		distance = (float(distance_pc) * u.pc).to(u.cm)
		radius = (float(radius_rsun) * R_sun).to(u.cm)
		flux = flux * (radius / distance) ** 2

	ext_curve = CT06_MWGC()(wave_grid_um)
	flux = flux * (10 ** (-0.4 * float(av) * ext_curve))
	model_spec = SourceSpectrum(Empirical1D, points=wave_grid, lookup_table=flux)
	distmod = distance_modulus(distance_pc)

	mags = {}
	for filt, bp in bandpasses.items():
		apparent_mag = Observation(model_spec, bp, force='extrap').effstim('vegamag', vegaspec=Vega).value
		mags[filt] = apparent_mag - distmod

	return mags


def vega_like_sanity_check(
	bandpasses,
	wave_min_um,
	wave_max_um,
	grids,
	wave_step_angstrom=10.0,
	distance_pc=10.0,
	teff=10000.0,
	logg=4.0,
	metallicity=0.0,
	radius_rsun=2.5,
):
	"""Evaluate whether a Vega-like model gives near-zero Vega absolute magnitudes."""
	chosen_model = None
	for grid in grids:
		trial = {'grid': grid, 'teff': teff, 'metallicity': metallicity, 'logg': logg}
		try:
			_ = grid_to_spec(grid, teff, metallicity, logg)
		except ParameterOutOfBounds:
			continue
		chosen_model = trial
		break

	if chosen_model is None:
		raise ParameterOutOfBounds('No requested grid contains the Vega-like sanity-check model point.')

	mags = compute_single_model_magnitudes(
		model=chosen_model,
		bandpasses=bandpasses,
		wave_min_um=wave_min_um,
		wave_max_um=wave_max_um,
		wave_step_angstrom=wave_step_angstrom,
		av=0.0,
		distance_pc=distance_pc,
		radius_rsun=radius_rsun,
	)
	max_abs_delta = max(abs(float(mags[filt])) for filt in FILTERS)

	return {
		'grid': chosen_model['grid'],
		'mags_abs_vega': {filt: float(mags[filt]) for filt in FILTERS},
		'max_abs_mag_delta_from_zero': float(max_abs_delta),
		'teff': float(teff),
		'logg': float(logg),
		'metallicity': float(metallicity),
		'radius_rsun': float(radius_rsun),
		'distance_pc': float(distance_pc),
	}


def compute_synphot_grid(
	model_rows,
	bandpasses,
	av_grid,
	wave_min_um,
	wave_max_um,
	wave_step_angstrom=10.0,
	injected_lines=None,
	distance_pc=10.0,
	intrinsic_flux_cache=None,
	show_progress=True,
):
	"""Compute synthetic magnitudes and close colors for model and extinction grids."""
	wave_grid = np.arange(wave_min_um * 1e4, wave_max_um * 1e4 + wave_step_angstrom, wave_step_angstrom) * u.AA
	wave_grid_um = wave_grid.to(u.um)

	ext_model = CT06_MWGC()
	ext_curve = ext_model(wave_grid_um)

	rows = []
	n_skipped_models = 0
	n_skipped_observations = 0
	total_cases = len(model_rows) * len(av_grid)
	completed_cases = 0
	t_start = time.time()
	distmod = distance_modulus(distance_pc)

	pbar = None
	if show_progress and tqdm is not None:
		pbar = tqdm(total=total_cases, desc='synphot model x Av', unit='case')

	for model in model_rows:
		model_key = (model['grid'], model['teff'], model['metallicity'], model['logg'])
		if intrinsic_flux_cache is not None and model_key in intrinsic_flux_cache:
			intrinsic_flux = intrinsic_flux_cache[model_key]
		else:
			try:
				spectrum = grid_to_spec(model['grid'], model['teff'], model['metallicity'], model['logg'])
			except ParameterOutOfBounds:
				n_skipped_models += 1
				completed_cases += len(av_grid)
				if pbar is not None:
					pbar.update(len(av_grid))
					elapsed = time.time() - t_start
					rate = completed_cases / elapsed if elapsed > 0 else np.nan
					pbar.set_postfix(rows=len(rows), skipped_model=n_skipped_models, rate=f'{rate:0.2f}/s')
				continue
			intrinsic_flux = spectrum(wave_grid)
			if intrinsic_flux_cache is not None:
				intrinsic_flux_cache[model_key] = intrinsic_flux

		for av in av_grid:
			attenuation = 10 ** (-0.4 * av * ext_curve)
			reddened_flux = intrinsic_flux * attenuation
			reddened_flux = add_gaussian_emission_lines(reddened_flux, wave_grid, injected_lines)
			reddened_spec = SourceSpectrum(Empirical1D, points=wave_grid, lookup_table=reddened_flux)

			mags = {}
			failed = False
			for filt, bp in bandpasses.items():
				try:
					apparent_mag = Observation(reddened_spec, bp, force='extrap').effstim('vegamag', vegaspec=Vega).value
					mags[filt] = apparent_mag - distmod
				except SynphotError:
					n_skipped_observations += 1
					failed = True
					break
			completed_cases += 1
			if pbar is not None:
				pbar.update(1)
				if completed_cases % 250 == 0 or completed_cases == total_cases:
					elapsed = time.time() - t_start
					rate = completed_cases / elapsed if elapsed > 0 else np.nan
					remaining = (total_cases - completed_cases) / rate if rate > 0 else np.nan
					pbar.set_postfix(
						rows=len(rows),
						skipped_model=n_skipped_models,
						skipped_obs=n_skipped_observations,
						rate=f'{rate:0.2f}/s',
						eta_min=f'{remaining/60.0:0.1f}' if np.isfinite(remaining) else 'nan',
					)
			if failed:
				continue

			row = {
				'grid': model['grid'],
				'teff': model['teff'],
				'metallicity': model['metallicity'],
				'logg': model['logg'],
				'stellar_class': stellar_class_from_logg(model['logg']),
				'av': float(av),
				'mag_vega_abs_f182m': mags['F182M'],
				'mag_vega_abs_f187n': mags['F187N'],
				'mag_vega_abs_f405n': mags['F405N'],
				'mag_vega_abs_f410m': mags['F410M'],
				'distance_pc': float(distance_pc),
				'line_tag': 'none' if not injected_lines else str(injected_lines[0]['tag']),
				'line_label': 'none' if not injected_lines else str(injected_lines[0]['label']),
				'line_peak_jy': 0.0 if not injected_lines else float(injected_lines[0]['peak_jy']),
				'line_fwhm_kms': 0.0 if not injected_lines else float(injected_lines[0]['fwhm_kms']),
			}
			row['color_f405n_f410m'] = row['mag_vega_abs_f405n'] - row['mag_vega_abs_f410m']
			row['color_f182m_f187n'] = row['mag_vega_abs_f182m'] - row['mag_vega_abs_f187n']
			row['color_f187n_f182m'] = row['mag_vega_abs_f187n'] - row['mag_vega_abs_f182m']
			rows.append(row)

	if pbar is not None:
		pbar.close()

	return Table(rows=rows), n_skipped_models, n_skipped_observations


def summarize_by_av(phot_table):
	"""Summarize color spread versus Av and by stellar class."""
	av_vals = np.unique(phot_table['av'])
	summary_rows = []

	for av in av_vals:
		sel_av = phot_table['av'] == av
		all_x = phot_table['color_f405n_f410m'][sel_av]
		all_y = phot_table['color_f187n_f182m'][sel_av]

		sel_giant = sel_av & (phot_table['stellar_class'] == 'giant')
		giant_x = phot_table['color_f405n_f410m'][sel_giant]
		giant_y = phot_table['color_f187n_f182m'][sel_giant]

		summary_rows.append(
			{
				'av': float(av),
				'n_all': int(sel_av.sum()),
				'n_giants': int(sel_giant.sum()),
				'x_min_all': float(np.min(all_x)),
				'x_max_all': float(np.max(all_x)),
				'x_std_all': float(np.std(all_x)),
				'y_min_all': float(np.min(all_y)),
				'y_max_all': float(np.max(all_y)),
				'y_std_all': float(np.std(all_y)),
				'x_min_giants': float(np.min(giant_x)) if len(giant_x) else np.nan,
				'x_max_giants': float(np.max(giant_x)) if len(giant_x) else np.nan,
				'y_min_giants': float(np.min(giant_y)) if len(giant_y) else np.nan,
				'y_max_giants': float(np.max(giant_y)) if len(giant_y) else np.nan,
			}
		)

	return Table(rows=summary_rows)


def make_ccd_plot(phot_table, output_path):
	"""Create CCD showing close-color spread from stellar SED + CT06 extinction only."""
	x = np.array(phot_table['color_f405n_f410m'])
	y = np.array(phot_table['color_f187n_f182m'])
	av = np.array(phot_table['av'])
	is_giant = np.array(phot_table['stellar_class'] == 'giant')

	fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

	all_sc = axes[0].scatter(x, y, c=av, s=3, cmap='viridis', alpha=0.5, linewidths=0)
	axes[0].set_title('All model spectra')
	axes[0].set_xlabel('F405N - F410M')
	axes[0].set_ylabel('F187N - F182M')
	axes[0].grid(alpha=0.3)

	axes[1].scatter(x[~is_giant], y[~is_giant], c='0.75', s=2, alpha=0.35, linewidths=0, label='non-giants')
	giant_sc = axes[1].scatter(x[is_giant], y[is_giant], c=av[is_giant], s=5, cmap='magma', alpha=0.7, linewidths=0,
							   label='giants (logg<=2)')
	axes[1].set_title('Giant subset highlighted')
	axes[1].set_xlabel('F405N - F410M')
	axes[1].grid(alpha=0.3)
	axes[1].legend(loc='best', markerscale=2)

	cbar0 = fig.colorbar(all_sc, ax=axes[0], fraction=0.046, pad=0.04)
	cbar0.set_label('Av')
	cbar1 = fig.colorbar(giant_sc, ax=axes[1], fraction=0.046, pad=0.04)
	cbar1.set_label('Av (giants)')

	plt.tight_layout()
	fig.savefig(output_path, dpi=250, bbox_inches='tight')
	plt.close(fig)


def make_cmd_plot(phot_table, output_path, color_col, mag_col, xlabel, ylabel, title):
	"""Create a color-magnitude diagram for a given color and magnitude axis pair."""
	fig, ax = plt.subplots(figsize=(7, 6))
	for stellar_class, style in STELLAR_CLASS_STYLE.items():
		sel = np.array(phot_table['stellar_class']) == stellar_class
		if not np.any(sel):
			continue
		x = np.array(phot_table[color_col][sel], dtype=float)
		y = np.array(phot_table[mag_col][sel], dtype=float)
		ax.scatter(
			x,
			y,
			s=14,
			c=style['color'],
			marker=style['marker'],
			alpha=0.70,
			linewidths=0.25,
			edgecolors='black',
			label=style['label'],
		)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.set_title(title)
	ax.grid(alpha=0.3)
	ax.invert_yaxis()
	ax.legend(loc='best', framealpha=0.9, fontsize=8)
	plt.tight_layout()
	fig.savefig(output_path, dpi=250, bbox_inches='tight')
	plt.close(fig)


def make_cmd_line_experiment_plot(base_table, line_table, output_path, color_col, mag_col, xlabel, ylabel, title):
	"""Compare baseline CMD against Pa alpha and Br alpha line-injection CMD experiments.
	Also shows a red clump star's photometric trajectory across flux levels for each A_V.
	"""
	fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
	line_order = ('paa', 'bra')
	peak_values = sorted({float(val) for val in np.array(line_table['line_peak_jy'], dtype=float)})
	colors = plt.get_cmap('turbo')(np.linspace(0.12, 0.90, max(len(peak_values), 1)))

	base_x = np.array(base_table[color_col], dtype=float)
	base_y = np.array(base_table[mag_col], dtype=float)

	for ax, line_tag in zip(axes, line_order):
		line_label = LINE_DEFINITIONS[line_tag]['label']
		ax.scatter(base_x, base_y, s=9, c='0.70', marker='x', alpha=0.45, linewidths=0.4, label='baseline (no line)')

		for color, peak_jy in zip(colors, peak_values):
			sel = (np.array(line_table['line_tag']) == line_tag) & np.isclose(np.array(line_table['line_peak_jy'], dtype=float), peak_jy)
			x = np.array(line_table[color_col][sel], dtype=float)
			y = np.array(line_table[mag_col][sel], dtype=float)
			marker = LINE_MARKERS.get(float(peak_jy), 'o')
			ax.scatter(x, y, s=18, color=color, marker=marker, alpha=0.72, linewidths=0.25, edgecolors='black',
					   label=f'{line_label}, {peak_jy:g} Jy')

		# Add tracking line for a red clump star: show its path through flux values at each A_V
		red_clump_base = base_table[base_table['stellar_class'] == 'red_clump']
		if len(red_clump_base) > 0:
			rc_star = red_clump_base[0]
			rc_grid = rc_star['grid']
			rc_teff = rc_star['teff']
			rc_metallicity = rc_star['metallicity']
			rc_logg = rc_star['logg']

			av_vals = sorted(set(base_table['av']))
			for av_val in av_vals:
				rc_sel = (
					(line_table['grid'] == rc_grid) &
					np.isclose(line_table['teff'], rc_teff) &
					np.isclose(line_table['metallicity'], rc_metallicity) &
					np.isclose(line_table['logg'], rc_logg) &
					np.isclose(line_table['av'], av_val) &
					(line_table['line_tag'] == line_tag)
				)
				rc_rows = np.where(rc_sel)[0]
				if len(rc_rows) == 0:
					continue

				fluxes = np.array(line_table['line_peak_jy'][rc_rows], dtype=float)
				sort_idx = np.argsort(fluxes)
				rc_rows = rc_rows[sort_idx]

				rc_x = np.array(line_table[color_col][rc_rows], dtype=float)
				rc_y = np.array(line_table[mag_col][rc_rows], dtype=float)

				ax.plot(rc_x, rc_y, 'k-', linewidth=1.5, alpha=0.6, zorder=2)
				if len(rc_x) > 1:
					ax.annotate('', xy=(rc_x[-1], rc_y[-1]), xytext=(rc_x[-2], rc_y[-2]),
							   arrowprops=dict(arrowstyle='->', color='black', lw=1.2, alpha=0.6))

		ax.set_title(f'{line_label}, FWHM={LINEWIDTH_KMS:g} km/s')
		ax.set_xlabel(xlabel)
		ax.grid(alpha=0.3)
		ax.invert_yaxis()

	axes[0].set_ylabel(ylabel)
	axes[1].legend(loc='best', fontsize=8, framealpha=0.9)
	fig.suptitle(title)
	plt.tight_layout()
	fig.savefig(output_path, dpi=250, bbox_inches='tight')
	plt.close(fig)


def make_giant_spectra_plot(
	evolved_models,
	output_path,
	filter_a='F405N',
	filter_b='F410M',
	default_filter_facility='JWST/NIRCam',
	center_norm_um=4.10,
	title='Synthetic evolved-star spectra',
	teff_min=3000.0,
	teff_max=4000.0,
):
	"""Plot normalized synthetic evolved-star spectra over a selected filter range."""
	filter_a_id = resolve_svo_filter_id(filter_a, default_facility=default_filter_facility)
	filter_b_id = resolve_svo_filter_id(filter_b, default_facility=default_filter_facility)
	f405 = SvoFps.get_transmission_data(filter_a_id)
	f410 = SvoFps.get_transmission_data(filter_b_id)

	w405 = f405['Wavelength'].quantity.to(u.um)
	t405 = np.array(f405['Transmission'], dtype=float)
	w410 = f410['Wavelength'].quantity.to(u.um)
	t410 = np.array(f410['Transmission'], dtype=float)

	keep405 = t405 > 1e-3
	keep410 = t410 > 1e-3
	plot_min = min(w405[keep405].min().to_value(u.um), w410[keep410].min().to_value(u.um)) - 0.03
	plot_max = max(w405[keep405].max().to_value(u.um), w410[keep410].max().to_value(u.um)) + 0.03

	wave_um = np.linspace(plot_min, plot_max, 2400)
	wave = wave_um * u.um

	fig, ax = plt.subplots(figsize=(11, 6))
	cmap = plt.get_cmap('viridis')

	n_plotted = 0
	for model in evolved_models:
		try:
			spectrum = grid_to_spec(model['grid'], model['teff'], model['metallicity'], model['logg'])
		except ParameterOutOfBounds:
			continue

		flux = spectrum(wave)
		norm = np.interp(center_norm_um, wave_um, flux.value)
		if norm <= 0:
			continue

		norm_flux = flux.value / norm
		line_color = cmap((model['teff'] - teff_min) / (teff_max - teff_min + 1e-6))
		ax.plot(
			wave_um,
			norm_flux,
			color=line_color,
			alpha=0.6,
			linewidth=1.1,
			label=f"{model['grid']} T={int(model['teff'])}K logg={model['logg']:.1f} [M/H]={model['metallicity']:.1f}",
		)
		n_plotted += 1

	ax2 = ax.twinx()
	ax2.fill_between(w405.to_value(u.um), 0, t405, color='tab:red', alpha=0.18, label=f'{filter_a} throughput')
	ax2.fill_between(w410.to_value(u.um), 0, t410, color='tab:blue', alpha=0.18, label=f'{filter_b} throughput')
	ax2.set_ylim(0, 1.05 * max(np.nanmax(t405), np.nanmax(t410)))
	ax2.set_ylabel('Filter throughput')

	ax.set_xlabel('Wavelength (um)')
	ax.set_ylabel(f'Normalized flux density (F_lambda / F_lambda[{center_norm_um:0.2f} um])')
	ax.set_xlim(plot_min, plot_max)
	ax.grid(alpha=0.3)
	ax.set_title(f'{title} ({teff_min:.0f}-{teff_max:.0f} K, logg<=2 or 2.3<=logg<=2.8)')

	if n_plotted <= 20:
		ax.legend(loc='upper left', fontsize=7, ncol=1, framealpha=0.9)

	h1, l1 = ax.get_legend_handles_labels()
	h2, l2 = ax2.get_legend_handles_labels()
	if l2:
		ax2.legend(h1[-1:] + h2, l1[-1:] + l2, loc='upper right', framealpha=0.9)

	plt.tight_layout()
	fig.savefig(output_path, dpi=250, bbox_inches='tight')
	plt.close(fig)

	return n_plotted


def build_argparser():
	parser = argparse.ArgumentParser(description='Synphot CCD uncertainty experiments for JWST Brick filters.')
	parser.add_argument('--grids', nargs='+', default=['phoenix', 'ck04models', 'k93models'],
						help='stsynphot stellar grids to include.')
	parser.add_argument('--av-min', type=float, default=0.0)
	parser.add_argument('--av-max', type=float, default=50.0)
	parser.add_argument('--av-step', type=float, default=5.0)
	parser.add_argument('--max-models', type=int, default=None,
						help='Optional uniform downsample of available synthetic spectra.')
	parser.add_argument('--compute-subset',
						choices=['all', 'evolved_3000_4000', 'evolved_3000_4500', 'evolved_3000_4000_solarplus',
								 'solar_3000_5000_logg_lt3'],
						default='solar_3000_5000_logg_lt3',
						help='Subset of model grid to use for full photometry computation.')
	parser.add_argument('--wave-step-angstrom', type=float, default=10.0,
						help='Wavelength grid step for synphot integration.')
	parser.add_argument('--distance-pc', type=float, default=10.0,
						help='Distance in parsec used to convert apparent Vega magnitudes to absolute magnitudes.')
	parser.add_argument('--output-dir', default=f'{basepath}/analysis/synphot_experiments',
						help='Directory for ECSV summaries and CCD figure output.')
	parser.add_argument('--giant-plot-max-models', type=int, default=24,
						help='Maximum number of giant models to overplot in the spectrum figure.')
	parser.add_argument('--no-progress', action='store_true',
						help='Disable progress bar output.')
	parser.add_argument('--show-extrapolation-warnings', action='store_true',
						help='Show repeated synphot extrapolation warnings.')
	parser.add_argument('--cache-synphot', dest='cache_synphot', action='store_true',
					help='Use on-disk synphot caches for baseline and line experiments.')
	parser.add_argument('--no-cache-synphot', dest='cache_synphot', action='store_false',
					help='Disable on-disk synphot caches.')
	parser.add_argument('--recompute-synphot-cache', action='store_true',
					help='Force recomputation and overwrite on-disk synphot caches.')
	parser.set_defaults(cache_synphot=True)
	return parser


def main(args=None):
	parser = build_argparser()
	parsed = parser.parse_args(args=args)

	os.makedirs(parsed.output_dir, exist_ok=True)

	phot_path = os.path.join(parsed.output_dir, 'synphot_closecolor_grid.ecsv')
	summary_path = os.path.join(parsed.output_dir, 'synphot_closecolor_summary_by_av.ecsv')
	fig_path = os.path.join(parsed.output_dir, 'synphot_closecolor_ccd_f405n-f410m_vs_f187n-f182m.png')
	cmd_182_187_path = os.path.join(parsed.output_dir, 'synphot_cmd_f182m-f187n_vs_f187n_baseline.png')
	cmd_405_410_path = os.path.join(parsed.output_dir, 'synphot_cmd_f405n-f410m_vs_f410m_baseline.png')
	line_grid_path = os.path.join(parsed.output_dir, 'synphot_closecolor_grid_lineexperiments.ecsv')
	cmd_182_187_line_path = os.path.join(parsed.output_dir, 'synphot_cmd_f182m-f187n_vs_f187n_lineexperiments.png')
	cmd_405_410_line_path = os.path.join(parsed.output_dir, 'synphot_cmd_f405n-f410m_vs_f410m_lineexperiments.png')
	giant_spec_path = os.path.join(parsed.output_dir, 'synphot_giant_spectra_3000-4000K_f405n-f410m.png')
	giant_spec_182_187_path = os.path.join(parsed.output_dir, 'synphot_giant_spectra_3000-4000K_f182m-f187n.png')
	giant_spec_356_444_path = os.path.join(parsed.output_dir, 'synphot_giant_spectra_3000-4000K_f356w-f444w_solarplus.png')
	giant_spec_vvv_ukidss_path = os.path.join(parsed.output_dir, 'synphot_giant_spectra_3000-4000K_vvvks-ukidssk.png')
	giant_spec_spitzer_i1_i2_path = os.path.join(parsed.output_dir, 'synphot_giant_spectra_3000-4000K_spitzeri1-spitzeri2.png')

	if not parsed.show_extrapolation_warnings:
		warnings.filterwarnings('ignore', message='Source spectrum will be extrapolated.*')

	bandpasses, wave_min_um, wave_max_um = build_bandpasses(filters=FILTERS)
	model_rows_all = collect_model_rows(parsed.grids, max_models=None)
	model_rows = filter_compute_models(model_rows_all, subset=parsed.compute_subset)
	if parsed.max_models is not None and parsed.max_models < len(model_rows):
		keep_idx = np.linspace(0, len(model_rows) - 1, parsed.max_models).astype(int)
		model_rows = [model_rows[ii] for ii in keep_idx]
	av_grid = np.arange(parsed.av_min, parsed.av_max + parsed.av_step / 2.0, parsed.av_step)

	evolved_models = collect_evolved_model_rows(
		grids=parsed.grids,
		teff_min=3000.0,
		teff_max=5000.0,
		giant_logg_max=2.0,
		red_clump_logg_min=2.3,
		red_clump_logg_max=2.8,
		max_models=parsed.giant_plot_max_models,
	)

	giant_models_solarplus = select_by_metallicity(evolved_models, min_metallicity=0.0)

	n_giant_plotted_182_187 = make_giant_spectra_plot(
		evolved_models=giant_models_solarplus,
		output_path=giant_spec_182_187_path,
		filter_a='F187N',
		filter_b='F182M',
		center_norm_um=1.82,
		title='Synthetic giant+red-clump spectra near F187N/F182M',
		teff_min=3000.0,
		teff_max=5000.0,
	)

	n_giant_plotted_356_444 = make_giant_spectra_plot(
		evolved_models=giant_models_solarplus,
		output_path=giant_spec_356_444_path,
		filter_a='F356W',
		filter_b='F444W',
		center_norm_um=4.00,
		title='Synthetic giant+red-clump spectra near F356W/F444W, [M/H]>=0',
		teff_min=3000.0,
		teff_max=5000.0,
	)

	n_giant_plotted = make_giant_spectra_plot(
		evolved_models=giant_models_solarplus,
		output_path=giant_spec_path,
		filter_a='F405N',
		filter_b='F410M',
		center_norm_um=4.10,
		title='Synthetic giant+red-clump spectra near F405N/F410M',
		teff_min=3000.0,
		teff_max=5000.0,
	)

	n_giant_plotted_vvv_ukidss = make_giant_spectra_plot(
		evolved_models=giant_models_solarplus,
		output_path=giant_spec_vvv_ukidss_path,
		filter_a='Paranal/VISTA.Ks',
		filter_b='UKIRT/WFCAM.K',
		center_norm_um=2.15,
		title='Synthetic giant+red-clump spectra near VVV Ks / UKIDSS K',
		teff_min=3000.0,
		teff_max=5000.0,
	)

	n_giant_plotted_spitzer_i1_i2 = make_giant_spectra_plot(
		evolved_models=giant_models_solarplus,
		output_path=giant_spec_spitzer_i1_i2_path,
		filter_a='Spitzer/IRAC.I1',
		filter_b='Spitzer/IRAC.I2',
		center_norm_um=4.05,
		title='Synthetic giant+red-clump spectra near Spitzer I1 / I2',
		teff_min=3000.0,
		teff_max=5000.0,
	)

	n_giant_plotted_356_444 = make_giant_spectra_plot(
		evolved_models=giant_models_solarplus,
		output_path=giant_spec_356_444_path,
		filter_a='F356W',
		filter_b='F444W',
		center_norm_um=4.00,
		title='Synthetic giant+red-clump spectra near F356W/F444W, [M/H]>=0',
		teff_min=3000.0,
		teff_max=5000.0,
	)

	intrinsic_flux_cache = {}
	cache_dir = os.path.join(parsed.output_dir, 'cache')
	if parsed.cache_synphot:
		os.makedirs(cache_dir, exist_ok=True)

	baseline_cache_key = make_synphot_cache_key(
		model_rows=model_rows,
		av_grid=av_grid,
		wave_step_angstrom=parsed.wave_step_angstrom,
		distance_pc=parsed.distance_pc,
		injected_lines=None,
	)
	baseline_cache_path = os.path.join(cache_dir, f'synphot_baseline_{baseline_cache_key}.ecsv')
	baseline_from_cache = False

	if parsed.cache_synphot and os.path.exists(baseline_cache_path) and not parsed.recompute_synphot_cache:
		phot_table = Table.read(baseline_cache_path)
		n_skipped_models = 0
		n_skipped_observations = 0
		baseline_from_cache = True
	else:
		phot_table, n_skipped_models, n_skipped_observations = compute_synphot_grid(
			model_rows=model_rows,
			bandpasses=bandpasses,
			av_grid=av_grid,
			wave_min_um=wave_min_um,
			wave_max_um=wave_max_um,
			wave_step_angstrom=parsed.wave_step_angstrom,
			distance_pc=parsed.distance_pc,
			intrinsic_flux_cache=intrinsic_flux_cache,
			show_progress=not parsed.no_progress,
		)
		if parsed.cache_synphot:
			phot_table.write(baseline_cache_path, overwrite=True)

	summary_table = summarize_by_av(phot_table)
	phot_table.write(phot_path, overwrite=True)
	summary_table.write(summary_path, overwrite=True)
	make_ccd_plot(phot_table, fig_path)
	make_cmd_plot(
		phot_table=phot_table,
		output_path=cmd_182_187_path,
		color_col='color_f182m_f187n',
		mag_col='mag_vega_abs_f187n',
		xlabel='F182M - F187N',
		ylabel='F187N (Vega abs mag)',
		title='CMD baseline: F182M-F187N vs F187N (Vega abs)',
	)
	make_cmd_plot(
		phot_table=phot_table,
		output_path=cmd_405_410_path,
		color_col='color_f405n_f410m',
		mag_col='mag_vega_abs_f410m',
		xlabel='F405N - F410M',
		ylabel='F410M (Vega abs mag)',
		title='CMD baseline: F405N-F410M vs F410M (Vega abs)',
	)

	line_cache_key = make_synphot_cache_key(
		model_rows=model_rows,
		av_grid=av_grid,
		wave_step_angstrom=parsed.wave_step_angstrom,
		distance_pc=parsed.distance_pc,
		injected_lines=[
			{'tag': tag, 'center_um': info['center_um'], 'peak_jy': peak, 'fwhm_kms': LINEWIDTH_KMS}
			for tag, info in LINE_DEFINITIONS.items()
			for peak in LINE_PEAK_JY_GRID
		],
	)
	line_cache_path = os.path.join(cache_dir, f'synphot_linegrid_{line_cache_key}.ecsv')
	line_from_cache = False

	if parsed.cache_synphot and os.path.exists(line_cache_path) and not parsed.recompute_synphot_cache:
		line_table_all = Table.read(line_cache_path)
		line_from_cache = True
	else:
		line_tables = []
		for line_tag, line_info in LINE_DEFINITIONS.items():
			for peak_jy in LINE_PEAK_JY_GRID:
				injected_lines = [
					{
						'tag': line_tag,
						'label': line_info['label'],
						'center_um': line_info['center_um'],
						'peak_jy': peak_jy,
						'fwhm_kms': LINEWIDTH_KMS,
					}
				]
				line_table, _nskip_model_line, _nskip_obs_line = compute_synphot_grid(
					model_rows=model_rows,
					bandpasses=bandpasses,
					av_grid=av_grid,
					wave_min_um=wave_min_um,
					wave_max_um=wave_max_um,
					wave_step_angstrom=parsed.wave_step_angstrom,
					injected_lines=injected_lines,
					distance_pc=parsed.distance_pc,
					intrinsic_flux_cache=intrinsic_flux_cache,
					show_progress=not parsed.no_progress,
				)
				line_tables.append(line_table)

		line_table_all = vstack(line_tables)
		if parsed.cache_synphot:
			line_table_all.write(line_cache_path, overwrite=True)

	line_table_all.write(line_grid_path, overwrite=True)

	vega_check = vega_like_sanity_check(
		bandpasses=bandpasses,
		wave_min_um=wave_min_um,
		wave_max_um=wave_max_um,
		grids=parsed.grids,
		wave_step_angstrom=parsed.wave_step_angstrom,
		distance_pc=10.0,
		teff=10000.0,
		logg=4.0,
		metallicity=0.0,
		radius_rsun=2.5,
	)

	make_cmd_line_experiment_plot(
		base_table=phot_table,
		line_table=line_table_all,
		output_path=cmd_182_187_line_path,
		color_col='color_f182m_f187n',
		mag_col='mag_vega_abs_f187n',
		xlabel='F182M - F187N',
		ylabel='F187N (Vega abs mag)',
		title='CMD line experiment: F182M-F187N vs F187N (Vega abs)',
	)
	make_cmd_line_experiment_plot(
		base_table=phot_table,
		line_table=line_table_all,
		output_path=cmd_405_410_line_path,
		color_col='color_f405n_f410m',
		mag_col='mag_vega_abs_f410m',
		xlabel='F405N - F410M',
		ylabel='F410M (Vega abs mag)',
		title='CMD line experiment: F405N-F410M vs F410M (Vega abs)',
	)

	x = np.array(phot_table['color_f405n_f410m'])
	y = np.array(phot_table['color_f187n_f182m'])
	giants = phot_table[phot_table['stellar_class'] == 'giant']

	print(f'Model spectra available total: {len(model_rows_all)}')
	print(f'Model subset: {parsed.compute_subset}')
	print(f'Model spectra used: {len(model_rows)}')
	print(f'Baseline synphot cache used: {baseline_from_cache}')
	print(f'Line-grid synphot cache used: {line_from_cache}')
	print(f'Magnitude system: Vega')
	print(f'Absolute magnitude calibration distance (pc): {parsed.distance_pc}')
	print(f'Skipped model entries (invalid catalog rows): {n_skipped_models}')
	print(f'Skipped model/Av cases (invalid synthetic observations): {n_skipped_observations}')
	print(f'Av samples: {len(av_grid)} spanning [{parsed.av_min}, {parsed.av_max}]')
	print(f'Total synthetic photometry rows: {len(phot_table)}')
	print(f'All-model F405N-F410M range: [{x.min():0.4f}, {x.max():0.4f}]')
	print(f'All-model F187N-F182M range: [{y.min():0.4f}, {y.max():0.4f}]')
	print(f'Giant rows: {len(giants)}')
	print(f'Evolved spectral models requested/plotted: {len(evolved_models)}/{n_giant_plotted}')
	print(f'Non-JWST K-band models requested/plotted: {len(giant_models_solarplus)}/{n_giant_plotted_vvv_ukidss}')
	print(f'Spitzer I1/I2 models requested/plotted: {len(giant_models_solarplus)}/{n_giant_plotted_spitzer_i1_i2}')
	print(f'Giant solar+ models requested/plotted: {len(giant_models_solarplus)}/{n_giant_plotted_356_444}')
	print('Vega-like sanity check (Teff=10000K, logg=4.0, [M/H]=0, R=2.5 Rsun, d=10pc):')
	print(f"  Grid: {vega_check['grid']}")
	for filt in FILTERS:
		print(f"  M_vega({filt}) = {vega_check['mags_abs_vega'][filt]:0.4f}")
	print(f"  max |M_vega - 0| = {vega_check['max_abs_mag_delta_from_zero']:0.4f}")
	if len(giants):
		gx = np.array(giants['color_f405n_f410m'])
		gy = np.array(giants['color_f187n_f182m'])
		print(f'Giant F405N-F410M range: [{gx.min():0.4f}, {gx.max():0.4f}]')
		print(f'Giant F187N-F182M range: [{gy.min():0.4f}, {gy.max():0.4f}]')

	print(f'Wrote: {phot_path}')
	print(f'Wrote: {summary_path}')
	print(f'Wrote: {fig_path}')
	print(f'Wrote: {cmd_182_187_path}')
	print(f'Wrote: {cmd_405_410_path}')
	print(f'Wrote: {line_grid_path}')
	print(f'Wrote: {cmd_182_187_line_path}')
	print(f'Wrote: {cmd_405_410_line_path}')
	print(f'Wrote: {giant_spec_path}')
	print(f'Wrote: {giant_spec_356_444_path}')
	print(f'Wrote: {giant_spec_vvv_ukidss_path}')
	print(f'Wrote: {giant_spec_spitzer_i1_i2_path}')


if __name__ == '__main__':
	main()