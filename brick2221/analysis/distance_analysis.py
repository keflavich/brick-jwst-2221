"""
Distance analysis for red clump (RC) stars.

This work is hacked "together" with AI agents with unknown version (vscode autoselect on March 17 2026, the week after they blocked direct access to Claude for students).  

Assumptions requested:
- RC absolute magnitude (K-like): M_K = -1.55 +/- 0.15 [from Girardi 2016 table 1, roughly approximated w/Claude]
- Change to -1.59+/-0.04 following Nogueras-Lara+ 2021 Distances to GC clouds
- Extinction model: G21_MWAvg to 8 kpc, then CT06_MWGC beyond 8 kpc

Workflow:
1. Load Brick downselected table and Cloud C table.
2. Build an RC selection function from an M_K prior across a distance grid.
3. Infer per-source distance and extinction from F182M/F212N color + mixed extinction law.
4. Deredden selected stars and make separate + overplotted distance histograms.
"""

import functools
import os
import sys
import numpy as np
import pylab as pl
from astropy.table import Table
from astropy import units as u
from astropy.modeling.models import BlackBody
from astroquery.svo_fps import SvoFps

from dust_extinction.averages import CT06_MWGC, G21_MWAvg

from analysis_setup import basepath
from selections import make_downselected_table_20251211
from plot_tools import cmd


# M_K from Nogueras-Lara+ 2021 (Vega, 2MASS Ks system)
RC_MK_VEGA = -1.59
RC_MK_SIGMA = 0.04
RC_INTRINSIC_COLOR_F182M_F212N = 0.0
GC_BOUNDARY_KPC = 8.0
CLOUDC_JWSTPLOTS_PATH = '/home/savannahgramze/orange_link/adamginsburg/jwst/cloudc/lactea-filament/lactea-filament'

# SVO FPS filter IDs for the M_K → M_F212N(AB) conversion
FILTER_ID_F212N = 'JWST/NIRCam.F212N'
FILTER_ID_KS = '2MASS/2MASS.Ks'
# Effective temperature of an RC K giant and of Vega (used for the color correction)
RC_TEFF_K = 4500.0    # K
VEGA_TEFF_K = 9550.0  # K


# Empirical RC CMD anchors requested by user
AK_F212N_AT_8KPC = 2.0
RC_CMD_ANCHOR_COLOR_1 = 1.0
RC_CMD_ANCHOR_F182M_1 = 17.0
RC_CMD_ANCHOR_COLOR_2 = 0.6
RC_CMD_ANCHOR_F182M_2 = 15.3

# Selection tolerances around the anchored RC CMD ridge
RC_COLOR_TOL_MAG = 0.15
RC_F182M_TOL_MAG = 0.35


@functools.lru_cache(maxsize=None)
def _compute_mk_to_mf212n_ab_offset():
	"""
	Compute the offset M_F212N(AB, JWST) − M_K(Vega, 2MASS Ks) for an RC K giant.

	Method:
	  1. Fetch filter transmission profiles for JWST/NIRCam.F212N and 2MASS/2MASS.Ks
	     from SVO FPS (used to get accurate pivot wavelengths and Vega zeropoints).
	  2. Integrate a blackbody SED at RC_TEFF_K through each filter profile
	     (photon-counting weighting: f_nu_eff = ∫ T·f_ν·λ dλ / ∫ T·λ dλ).
	  3. Divide by the same integral for a Vega-temperature blackbody to get the
	     Vega-system color F212N − Ks for a K giant.
	  4. Add the AB−Vega offset for F212N: −2.5·log₁₀(ZP_Vega_Jy / 3631 Jy).

	Returns
	-------
	float
	    Offset (mag) to add to M_K(Vega) to obtain M_F212N(AB, JWST).
	"""
	jfilts = SvoFps.get_filter_list('JWST')
	jfilts.add_index('filterID')
	tmass = SvoFps.get_filter_list('2MASS')
	tmass.add_index('filterID')

	f212_row = jfilts.loc[FILTER_ID_F212N]
	ks_row   = tmass.loc[FILTER_ID_KS]

	zp_f212n_jy    = float(f212_row['ZeroPoint'])          # Vega ZP in Jy
	pivot_f212n_um = float(f212_row['WavelengthPivot']) * 1e-4  # Å → µm
	pivot_ks_um    = float(ks_row['WavelengthPivot'])   * 1e-4

	trans_f212n = SvoFps.get_transmission_data(FILTER_ID_F212N)
	trans_ks    = SvoFps.get_transmission_data(FILTER_ID_KS)

	bb_star = BlackBody(temperature=RC_TEFF_K  * u.K)
	bb_vega = BlackBody(temperature=VEGA_TEFF_K * u.K)

	def _mean_fnu(trans_table, bb):
		"""Photon-counting mean B_ν: ∫ T·B_ν·λ dλ / ∫ T·λ dλ  (λ in Å).
		Units of B_ν cancel when computing Vega-normalised colour ratios.
		"""
		lam_aa = np.array(trans_table['Wavelength'], dtype=float)
		T      = np.array(trans_table['Transmission'], dtype=float)
		bnu    = bb(lam_aa * u.AA).value  # B_ν in erg/cm²/s/Hz/sr; units cancel in ratios
		return np.trapezoid(T * bnu * lam_aa, lam_aa) / np.trapezoid(T * lam_aa, lam_aa)

	fnu_star_f212n = _mean_fnu(trans_f212n, bb_star)
	fnu_star_ks    = _mean_fnu(trans_ks,    bb_star)
	fnu_vega_f212n = _mean_fnu(trans_f212n, bb_vega)
	fnu_vega_ks    = _mean_fnu(trans_ks,    bb_vega)

	# Vega-system color F212N − Ks for a K giant (relative to a Vega-temperature BB)
	color_f212n_minus_ks_vega = -2.5 * np.log10(
		(fnu_star_f212n / fnu_vega_f212n) / (fnu_star_ks / fnu_vega_ks)
	)

	# AB − Vega offset for F212N using the SVO Vega zeropoint flux
	ab_minus_vega_f212n = -2.5 * np.log10(zp_f212n_jy / 3631.0)

	offset = color_f212n_minus_ks_vega + ab_minus_vega_f212n

	print("SVO FPS M_K(Vega, 2MASS Ks) → M_F212N(AB, JWST) conversion:")
	print(f"  Pivot F212N (JWST/NIRCam): {pivot_f212n_um:.4f} µm")
	print(f"  Pivot Ks    (2MASS):        {pivot_ks_um:.4f} µm")
	print(f"  Vega ZP F212N:              {zp_f212n_jy:.2f} Jy")
	print(f"  F212N − Ks (Vega, BB {RC_TEFF_K:.0f} K):  {color_f212n_minus_ks_vega:+.4f} mag")
	print(f"  AB − Vega offset (F212N):   {ab_minus_vega_f212n:+.4f} mag")
	print(f"  Total offset:               {offset:+.4f} mag")
	print(f"  → M_F212N(AB) = {RC_MK_VEGA} + ({offset:+.4f}) = {RC_MK_VEGA + offset:.4f}")

	return float(offset)


def _filled_float(column):
	if hasattr(column, 'filled'):
		return np.array(column.filled(np.nan), dtype=float)
	return np.array(column, dtype=float)


def _distance_modulus(distance_kpc):
	return 5 * np.log10(distance_kpc * 1000.0) - 5


def _path_lengths(distance_kpc, gc_boundary_kpc=GC_BOUNDARY_KPC):
	d_mw = np.minimum(distance_kpc, gc_boundary_kpc)
	d_gc = np.maximum(distance_kpc - gc_boundary_kpc, 0.0)
	return d_mw, d_gc


def _mixed_extinction_terms(distance_grid_kpc):
	g21 = G21_MWAvg()
	ct06 = CT06_MWGC()

	k182_g21 = g21(1.82 * u.um)
	k212_g21 = g21(2.12 * u.um)
	k182_ct06 = ct06(1.82 * u.um)
	k212_ct06 = ct06(2.12 * u.um)

	d_mw, d_gc = _path_lengths(distance_grid_kpc)

	color_denom = d_mw * (k182_g21 - k212_g21) + d_gc * (k182_ct06 - k212_ct06)
	a212_factor = d_mw * k212_g21 + d_gc * k212_ct06
	a182_factor = d_mw * k182_g21 + d_gc * k182_ct06

	return color_denom, a182_factor, a212_factor


def _ak_f212n_profile(distance_kpc):
	"""Monotonic extinction profile with A_K(F212N)=2 at 8 kpc and 0 at 0 kpc."""
	distance_kpc = np.asarray(distance_kpc, dtype=float)
	return AK_F212N_AT_8KPC * np.maximum(distance_kpc, 0.0) / GC_BOUNDARY_KPC


@functools.lru_cache(maxsize=None)
def _calibrated_rc_track_parameters():
	"""
	Calibrate an empirical RC CMD ridge constrained by:
	- A_K(F212N)=2 at 8 kpc and linear-to-zero toward d=0
	- Ridge passes through (F182M-F212N, F182M) = (1.0, 17.0) and (0.6, 15.3)

	We use a monotonic color law: color(d) = color_8 * (d/8)^p.
	"""
	color1 = RC_CMD_ANCHOR_COLOR_1
	m182_1 = RC_CMD_ANCHOR_F182M_1
	color2 = RC_CMD_ANCHOR_COLOR_2
	m182_2 = RC_CMD_ANCHOR_F182M_2

	# At d=8 kpc: A212=2, observed color=color1, so A182=A212+color1
	a212_8 = _ak_f212n_profile(np.array([GC_BOUNDARY_KPC]))[0]
	a182_8 = a212_8 + color1
	m182_rc = m182_1 - _distance_modulus(GC_BOUNDARY_KPC) - a182_8

	# Fit p so the model also goes through anchor #2 in CMD
	p_grid = np.linspace(0.3, 4.0, 20000)
	d2_grid = GC_BOUNDARY_KPC * (color2 / color1) ** (1.0 / p_grid)
	a212_d2 = _ak_f212n_profile(d2_grid)
	a182_d2 = a212_d2 + color2
	m182_d2 = m182_rc + _distance_modulus(d2_grid) + a182_d2
	best_p = float(p_grid[np.argmin(np.abs(m182_d2 - m182_2))])

	print("Calibrated empirical RC CMD ridge:")
	print(f"  A_K(F212N) at 8 kpc: {AK_F212N_AT_8KPC:.2f}")
	print(f"  Color law: (F182M-F212N)(d) = {color1:.3f} * (d/8)^p,  p={best_p:.4f}")
	print(f"  RC M_F182M(AB) from anchor: {m182_rc:.4f}")

	return best_p, float(m182_rc)


def _empirical_rc_track(distance_grid_kpc, intrinsic_color=RC_INTRINSIC_COLOR_F182M_F212N):
	"""Return expected RC observables along distance from empirical anchored model."""
	p_color, m182_rc = _calibrated_rc_track_parameters()

	distance_grid_kpc = np.asarray(distance_grid_kpc, dtype=float)
	a212 = _ak_f212n_profile(distance_grid_kpc)
	color_obs = intrinsic_color + RC_CMD_ANCHOR_COLOR_1 * (distance_grid_kpc / GC_BOUNDARY_KPC) ** p_color
	a182 = a212 + (color_obs - intrinsic_color)
	m182_obs = m182_rc + _distance_modulus(distance_grid_kpc) + a182
	m212_obs = m182_obs - color_obs
	m212_abs = m182_rc - intrinsic_color

	return color_obs, m182_obs, m212_obs, a182, a212, m212_abs


def rc_selection_and_distances(
	basetable,
	distance_grid_kpc=np.linspace(0.5, 16.5, 551),
	rc_mk_vega=RC_MK_VEGA,
	rc_mk_sigma=RC_MK_SIGMA,
	intrinsic_color=RC_INTRINSIC_COLOR_F182M_F212N,
):
	mag182 = _filled_float(basetable['mag_ab_f182m'])
	mag212 = _filled_float(basetable['mag_ab_f212n'])
	emag212 = _filled_float(basetable['emag_ab_f212n'])
	emag182 = _filled_float(basetable['emag_ab_f182m']) if 'emag_ab_f182m' in basetable.colnames else np.zeros(len(basetable), dtype=float)
	emag182[~np.isfinite(emag182)] = 0.0

	valid = np.isfinite(mag182) & np.isfinite(mag212) & np.isfinite(emag212)
	observed_color = mag182 - mag212

	track_color, track_m182, _, track_a182, track_a212, track_m212_abs = _empirical_rc_track(
		distance_grid_kpc,
		intrinsic_color=intrinsic_color,
	)

	# Fit each star to nearest point on empirical RC ridge in CMD space
	color_resid_2d = observed_color[:, None] - track_color[None, :]
	m182_resid_2d = mag182[:, None] - track_m182[None, :]
	metric_2d = color_resid_2d**2 + m182_resid_2d**2

	best_index = np.zeros(len(basetable), dtype=int)
	best_distance_kpc = np.full(len(basetable), np.nan, dtype=float)
	best_color_residual = np.full(len(basetable), np.nan, dtype=float)
	best_m182_residual = np.full(len(basetable), np.nan, dtype=float)
	a212 = np.full(len(basetable), np.nan, dtype=float)
	a182 = np.full(len(basetable), np.nan, dtype=float)

	if np.any(valid):
		metric_valid = np.where(np.isfinite(metric_2d[valid, :]), metric_2d[valid, :], np.inf)
		idx_valid = np.argmin(metric_valid, axis=1)
		rows_valid = np.where(valid)[0]
		best_index[rows_valid] = idx_valid
		best_distance_kpc[rows_valid] = distance_grid_kpc[idx_valid]
		best_color_residual[rows_valid] = color_resid_2d[rows_valid, idx_valid]
		best_m182_residual[rows_valid] = m182_resid_2d[rows_valid, idx_valid]
		a212[rows_valid] = track_a212[idx_valid]
		a182[rows_valid] = track_a182[idx_valid]

	best_mk = mag212 - _distance_modulus(best_distance_kpc) - a212
	best_color0 = observed_color - (a182 - a212)

	# Bin-based RC selection around anchored CMD ridge
	interior_distance = (best_distance_kpc > distance_grid_kpc[0]) & (best_distance_kpc < distance_grid_kpc[-1])
	color_fit_good = np.abs(best_color_residual) < RC_COLOR_TOL_MAG
	mag_fit_good = np.abs(best_m182_residual) < RC_F182M_TOL_MAG

	rc_selection = (
		valid
		& np.isfinite(best_distance_kpc)
		& np.isfinite(best_mk)
		& color_fit_good
		& mag_fit_good
		& interior_distance
	)

	results = Table()
	results['distance_kpc'] = best_distance_kpc
	results['color_residual_mag'] = best_color_residual
	results['f182m_residual_mag'] = best_m182_residual
	results['mk_dered_f212n'] = best_mk
	results['ak_f212n'] = a212
	results['a182'] = a182
	results['f182m_minus_f212n_dered'] = best_color0
	results['mag_ab_f212n_dered'] = mag212 - a212
	results['mag_ab_f182m_dered'] = mag182 - a182
	results['rc_selected'] = rc_selection

	return results



def load_cloudc_catalog():
	try:
		import jwst_plots
	except ModuleNotFoundError:
		sys.path.append(CLOUDC_JWSTPLOTS_PATH)
		import jwst_plots

	cat = jwst_plots.make_cat_use().catalog
	return cat


def selected_distances(rc_table):
	selected = rc_table['rc_selected']
	distances = np.array(rc_table['distance_kpc'][selected], dtype=float)
	distances = distances[np.isfinite(distances)]
	return distances, selected


def _dist_to_mag(d_kpc):
	"""Convert distance (kpc) to dereddened F212N AB apparent magnitude for an RC star."""
	rc_mf212n_ab = RC_MK_VEGA + _compute_mk_to_mf212n_ab_offset()
	return rc_mf212n_ab + 5.0 * np.log10(np.asarray(d_kpc) * 1000.0) - 5.0


def _mag_to_dist(mag):
	"""Inverse of _dist_to_mag: dereddened F212N AB apparent magnitude → distance in kpc."""
	rc_mf212n_ab = RC_MK_VEGA + _compute_mk_to_mf212n_ab_offset()
	return 10.0 ** ((np.asarray(mag) - rc_mf212n_ab + 5.0) / 5.0) / 1000.0


def _add_mag_secondary_axis(ax):
	"""Attach a top x-axis showing dereddened RC F212N magnitude."""
	secax = ax.secondary_xaxis('top', functions=(_dist_to_mag, _mag_to_dist))
	secax.set_xlabel('Dereddened F212N magnitude (RC star)')
	# Put ticks at round magnitudes that fall inside the distance range shown
	d_lim = ax.get_xlim()
	mag_lo = _dist_to_mag(d_lim[0])
	mag_hi = _dist_to_mag(d_lim[1])
	mag_ticks = np.arange(np.ceil(mag_lo), np.floor(mag_hi) + 1, 1.0)
	secax.set_ticks(mag_ticks)
	return secax


def plot_histogram(distances, title, outpath, color='C0'):
	fig = pl.figure(figsize=(8, 5.5))
	ax = fig.gca()
	ax.hist(distances, bins=np.linspace(0.5, 16.5, 111), histtype='stepfilled', alpha=0.7, color=color)
	ax.set_xlabel('Distance (kpc)')
	ax.set_ylabel('N (RC-selected stars)')
	ax.set_title(title)
	ax.grid(alpha=0.3)
	ax.set_xlim(0.5, 16.5)
	_add_mag_secondary_axis(ax)
	fig.tight_layout()
	fig.savefig(outpath, dpi=200)
	print(f"Wrote histogram to {outpath}")


def plot_overplot(distances_brick, distances_cloudc, outpath):
	fig = pl.figure(figsize=(8, 5.5))
	ax = fig.gca()
	bins = np.linspace(0.5, 16.5, 111)  # 5 grid steps per bin, aligned to grid origin — no Moiré
	ax.hist(distances_brick, bins=bins, histtype='step', linewidth=2, label='Brick', color='C0')
	ax.hist(distances_cloudc, bins=bins, histtype='step', linewidth=2, label='Cloud C', color='C1')
	ax.set_xlabel('Distance (kpc)')
	ax.set_ylabel('N (RC-selected stars)')
	ax.set_title('RC distance histogram: Brick vs Cloud C')
	ax.grid(alpha=0.3)
	ax.legend(loc='best')
	ax.set_xlim(0.5, 16.5)
	_add_mag_secondary_axis(ax)
	fig.tight_layout()
	fig.savefig(outpath, dpi=200)
	print(f"Wrote Brick+Cloud C overplot to {outpath}")


def plot_overplot_normalized(distances_brick, distances_cloudc, outpath):
	"""Overplot Brick and Cloud C normalized to unit area for shape comparison."""
	fig = pl.figure(figsize=(8, 5.5))
	ax = fig.gca()
	bins = np.linspace(0.5, 16.5, 111)  # 5 grid steps per bin, aligned to grid origin — no Moiré
	ax.hist(distances_brick, bins=bins, histtype='step', linewidth=2,
	        label=f'Brick (N={len(distances_brick):,})', color='C0', density=True)
	ax.hist(distances_cloudc, bins=bins, histtype='step', linewidth=2,
	        label=f'Cloud C (N={len(distances_cloudc):,})', color='C1', density=True)
	ax.set_xlabel('Distance (kpc)')
	ax.set_ylabel('Probability density (kpc$^{-1}$)')
	ax.set_title('RC distance histogram: Brick vs Cloud C (normalized)')
	ax.grid(alpha=0.3)
	ax.legend(loc='best')
	ax.set_xlim(0.5, 16.5)
	_add_mag_secondary_axis(ax)
	fig.tight_layout()
	fig.savefig(outpath, dpi=200)
	print(f"Wrote normalized Brick+Cloud C overplot to {outpath}")


def plot_cmd_with_rc_selection(basetable, rc_table, name, outpath):
	"""Plot F182M vs (F182M-F212N) color-magnitude diagram with RC selection overlay.
	
	Parameters
	----------
	basetable : Table
	    Table with mag_ab_f182m, mag_ab_f212n, emag_ab_f182m columns
	rc_table : Table
	    Results table from rc_selection_and_distances() with rc_selected column
	name : str
	    Sample name (e.g., 'Brick', 'Cloud C') for title
	outpath : str
	    Output file path for the figure
	"""
	fig = pl.figure(figsize=(9, 6))
	ax = fig.gca()

	# Build selection mask: all valid stars and RC-selected stars
	mag182 = np.array(basetable['mag_ab_f182m'], dtype=float)
	mag212 = np.array(basetable['mag_ab_f212n'], dtype=float)
	valid = np.isfinite(mag182) & np.isfinite(mag212)
	rc_sel = np.array(rc_table['rc_selected'], dtype=bool)

	# Use plot_tools.cmd for consistent CMD behavior (hexbin + extinction vector)
	cmd(
		ax=ax,
		basetable=basetable,
		f1='f182m',
		f2='f212n',
		include=valid,
		sel=rc_sel,
		axlims=(0, 2.5, 21, 14.5),
		ext=CT06_MWGC(),
		extvec_scale=30,
		head_width=0.08,
		markersize=5,
		alpha=0.4,
		alpha_sel=0.8,
		color='gray',
		selcolor='red',
		rasterized=True,
		hexbin=True,
		n_hexbin_bins=90,
		hexbin_cmap='Greys',
		sel_hexbin_cmap='Reds',
		extvec_start=(1.2, 15.0),
	)

	ax.set_title(f'RC selection in F182M-(F182M-F212N) CMD (hexbin): {name}')
	#ax.grid(alpha=0.3)
	fig.tight_layout()
	fig.savefig(outpath, dpi=200)
	print(f"Wrote {name} CMD plot to {outpath}")


def _cmd_valid_xy(basetable, f182m_max=None):
	"""Return finite CMD coordinates: x=(F182M-F212N), y=F182M."""
	mag182 = np.array(basetable['mag_ab_f182m'], dtype=float)
	mag212 = np.array(basetable['mag_ab_f212n'], dtype=float)
	valid = np.isfinite(mag182) & np.isfinite(mag212)
	if f182m_max is not None:
		valid &= (mag182 <= f182m_max)
	x = (mag182 - mag212)[valid]
	y = mag182[valid]
	return x, y


def plot_cmd_difference_normalized_hexbin(basetable_brick, basetable_cloudc, outpath,
	                                      axlims=(0, 2.5, 19.7, 14.9), gridsize=90, f182m_max=19.7):
	"""Plot normalized CMD density difference (Brick - Cloud C) in hexbin space.

	Bins where both datasets have <3 stars are masked as noise-dominated.
	"""
	xb, yb = _cmd_valid_xy(basetable_brick, f182m_max=f182m_max)
	xc, yc = _cmd_valid_xy(basetable_cloudc, f182m_max=f182m_max)

	if len(xb) == 0 or len(xc) == 0:
		raise ValueError("Cannot make CMD difference plot: one sample has no finite CMD points")

	extent = (axlims[0], axlims[1], min(axlims[2], axlims[3]), max(axlims[2], axlims[3]))

	# Main normalized difference map using one shared hex grid
	x_all = np.concatenate([xb, xc])
	y_all = np.concatenate([yb, yc])
	weights = np.concatenate([
		np.full(len(xb), 1.0 / len(xb), dtype=float),
		np.full(len(xc), -1.0 / len(xc), dtype=float),
	])

	fig = pl.figure(figsize=(9, 6))
	ax = fig.gca()
	hb_diff = ax.hexbin(
		x_all,
		y_all,
		C=weights,
		reduce_C_function=np.sum,
		gridsize=gridsize,
		extent=extent,
		mincnt=1,
	)

	# Per-sample counts on the identical hex grid (for noise masking)
	fig_tmp1 = pl.figure(figsize=(1, 1))
	ax_tmp1 = fig_tmp1.gca()
	hb_b = ax_tmp1.hexbin(xb, yb, gridsize=gridsize, extent=extent, mincnt=1)
	b_offsets = hb_b.get_offsets()
	b_counts = hb_b.get_array()
	pl.close(fig_tmp1)

	fig_tmp2 = pl.figure(figsize=(1, 1))
	ax_tmp2 = fig_tmp2.gca()
	hb_c = ax_tmp2.hexbin(xc, yc, gridsize=gridsize, extent=extent, mincnt=1)
	c_offsets = hb_c.get_offsets()
	c_counts = hb_c.get_array()
	pl.close(fig_tmp2)

	# Match bins by center coordinates; centers are identical for shared gridsize/extent
	def _keyify(offsets):
		return [f"{xv:.7f},{yv:.7f}" for xv, yv in offsets]

	b_map = dict(zip(_keyify(b_offsets), b_counts))
	c_map = dict(zip(_keyify(c_offsets), c_counts))

	offsets = hb_diff.get_offsets()
	keys = _keyify(offsets)
	b_in_bin = np.array([b_map.get(key, 0.0) for key in keys], dtype=float)
	c_in_bin = np.array([c_map.get(key, 0.0) for key in keys], dtype=float)
	noise_mask = (b_in_bin < 3) & (c_in_bin < 3)

	diff_vals = np.array(hb_diff.get_array(), dtype=float)
	diff_vals_masked = np.ma.array(diff_vals, mask=noise_mask)

	vlim = np.nanmax(np.abs(diff_vals_masked))
	if not np.isfinite(vlim) or vlim == 0:
		vlim = 1e-4

	hb_diff.set_array(diff_vals_masked)
	hb_diff.set_cmap('RdBu_r')
	hb_diff.set_clim(-vlim, vlim)

	cbar = fig.colorbar(hb_diff, ax=ax)
	cbar.set_label('Normalized density difference (Brick - Cloud C)')

	ax.set_xlim(axlims[0], axlims[1])
	ax.set_ylim(axlims[2], axlims[3])
	ax.set_xlabel('F182M - F212N')
	ax.set_ylabel('F182M')
	ax.set_title('CMD normalized hexbin difference: Brick - Cloud C')
	ax.grid(alpha=0.2)

	fig.tight_layout()
	fig.savefig(outpath, dpi=200)
	print(f"Wrote normalized CMD difference hexbin plot to {outpath}")


def _rc_distance_reddening(rc_table):
	"""Return finite (distance_kpc, ak_f212n) for RC-selected stars."""
	rc_sel = np.array(rc_table['rc_selected'], dtype=bool)
	dist = np.array(rc_table['distance_kpc'], dtype=float)
	ak = np.array(rc_table['ak_f212n'], dtype=float)
	valid = rc_sel & np.isfinite(dist) & np.isfinite(ak)
	return dist[valid], ak[valid]


def plot_distance_vs_reddening(distances, reddening, title, outpath, color='C0'):
	"""Scatter plot of RC distance vs reddening."""
	fig = pl.figure(figsize=(8, 5.5))
	ax = fig.gca()
	ax.scatter(distances, reddening, s=8, alpha=0.25, c=color, linewidths=0, rasterized=True)
	ax.set_xlabel('Distance (kpc)')
	ax.set_ylabel('A_K (F212N)')
	ax.set_title(title)
	ax.grid(alpha=0.3)
	ax.set_xlim(0.5, 16.5)
	if len(reddening) > 0:
		ymin = np.nanmin(reddening)
		ymax = np.nanmax(reddening)
		pad = 0.05 * max(ymax - ymin, 0.1)
		ax.set_ylim(ymin - pad, ymax + pad)
	fig.tight_layout()
	fig.savefig(outpath, dpi=200)
	print(f"Wrote distance-vs-reddening plot to {outpath}")


def plot_distance_vs_reddening_overplot(dist_brick, ak_brick, dist_cloudc, ak_cloudc, outpath):
	"""Overplot RC distance vs reddening for Brick and Cloud C."""
	fig = pl.figure(figsize=(8, 5.5))
	ax = fig.gca()
	ax.scatter(dist_brick, ak_brick, s=8, alpha=0.20, c='C0', linewidths=0, rasterized=True,
	          label=f'Brick (N={len(dist_brick):,})')
	ax.scatter(dist_cloudc, ak_cloudc, s=8, alpha=0.20, c='C1', linewidths=0, rasterized=True,
	          label=f'Cloud C (N={len(dist_cloudc):,})')
	ax.set_xlabel('Distance (kpc)')
	ax.set_ylabel('A_K (F212N)')
	ax.set_title('RC distance vs reddening: Brick and Cloud C')
	ax.grid(alpha=0.3)
	ax.legend(loc='best')
	ax.set_xlim(0.5, 16.5)
	all_ak = np.concatenate([ak_brick, ak_cloudc]) if (len(ak_brick) + len(ak_cloudc)) > 0 else np.array([0.0])
	ymin = np.nanmin(all_ak)
	ymax = np.nanmax(all_ak)
	pad = 0.05 * max(ymax - ymin, 0.1)
	ax.set_ylim(ymin - pad, ymax + pad)
	fig.tight_layout()
	fig.savefig(outpath, dpi=200)
	print(f"Wrote overplotted distance-vs-reddening plot to {outpath}")


# ─────────────────────────────────────────────────────────────────────────────
# RC Peak Analysis  (after Nogueras-Lara+ 2021, 2021A&A...647L...6N)
#
# Project CMD stars along the extinction vector via:
#   m_dered = F182M - alpha * (F182M - F212N)
# where alpha = A_F182M / (A_F182M - A_F212N) = k_182 / (k_182 - k_212).
# When alpha is correct, all RC stars collapse to the same m_dered regardless
# of individual extinction → narrowest possible Gaussian peak.
# We iterate CMD-window selection to converge on the minimum-width solution.
# ─────────────────────────────────────────────────────────────────────────────


def _rc_peak_extinction_slopes():
	"""Return CMD slopes alpha for G21 and CT06 extinction laws.

	alpha = A_F182M / (A_F182M - A_F212N) = k_182 / (k_182 - k_212),
	i.e. dF182M / d(F182M-F212N) along the reddening vector.
	"""
	g21_ext = G21_MWAvg()
	ct06_ext = CT06_MWGC()
	k182_g21  = float(g21_ext(1.82 * u.um))
	k212_g21  = float(g21_ext(2.12 * u.um))
	k182_ct06 = float(ct06_ext(1.82 * u.um))
	k212_ct06 = float(ct06_ext(2.12 * u.um))
	alpha_g21  = k182_g21  / (k182_g21  - k212_g21)
	alpha_ct06 = k182_ct06 / (k182_ct06 - k212_ct06)
	return dict(alpha_g21=alpha_g21, k182_g21=k182_g21, k212_g21=k212_g21,
	            alpha_ct06=alpha_ct06, k182_ct06=k182_ct06, k212_ct06=k212_ct06)


def _rc_peak_gauss_fit(data, n_sigma_range=5):
	"""Fit a Gaussian to binned 1-D data via scipy.optimize.curve_fit.

	Returns (peak, sigma, amplitude).
	"""
	from scipy.optimize import curve_fit

	med = np.median(data)
	mad = max(np.median(np.abs(data - med)) * 1.4826, 1e-4)
	bins = np.linspace(med - n_sigma_range * mad, med + n_sigma_range * mad, 80)
	counts, edges = np.histogram(data, bins=bins)
	centers = 0.5 * (edges[:-1] + edges[1:])

	def gauss(x, amp, mu, sig):
		return amp * np.exp(-0.5 * ((x - mu) / sig) ** 2)

	p0 = [float(counts.max()), med, mad]
	bounds = ([0, med - 3 * mad, 1e-4], [float(counts.sum()) * 2, med + 3 * mad, 5 * mad])
	popt, _ = curve_fit(gauss, centers, counts.astype(float), p0=p0, bounds=bounds, maxfev=4000)
	return float(popt[1]), abs(float(popt[2])), float(popt[0])


def rc_peak_fit_slope(basetable, name,
	                  alpha_min=1.5, alpha_max=12.0, n_alpha=800,
	                  n_iter=8, sigma_clip=2.5,
	                  color_range=(0.3, 1.5), mag_range=(14.5, 19.7),
	                  sigma_clip_max_mag=0.80,
	                  seed_from_rc_selection=True):
	"""Iteratively fit the CMD extinction slope to minimise dereddened RC width.

	Parameters
	----------
	basetable : Table
	name : str
	alpha_min, alpha_max, n_alpha : grid-search bounds and density for slope
	n_iter : maximum selection iterations
	sigma_clip : keep stars within this many MAD-sigmas of the RC peak each iter
	color_range, mag_range : initial CMD window for RC candidates
	seed_from_rc_selection : if True, seed the iterative selection from the
	    existing RC strip determined by rc_selection_and_distances() so that
	    the distance spread does not dominate the width metric.

	Returns
	-------
	dict  — see inline keys for documentation
	"""
	from scipy.optimize import minimize_scalar

	slopes = _rc_peak_extinction_slopes()

	mag182 = _filled_float(basetable['mag_ab_f182m'])
	mag212 = _filled_float(basetable['mag_ab_f212n'])
	valid = np.isfinite(mag182) & np.isfinite(mag212)
	color = mag182 - mag212
	mag   = mag182

	in_window = (
		valid
		& (color >= color_range[0]) & (color <= color_range[1])
		& (mag   >= mag_range[0])   & (mag   <= mag_range[1])
	)

	alpha_grid = np.linspace(alpha_min, alpha_max, n_alpha)

	# Seed from the existing geometrically-selected RC strip when possible,
	# intersected with the CMD window.  This constrains the distance spread
	# so that the width-vs-alpha curve reflects the extinction-law slope
	# rather than the line-of-sight depth of the field.
	if seed_from_rc_selection:
		rc_tbl = rc_selection_and_distances(basetable)
		rc_sel_mask = np.asarray(rc_tbl['rc_selected'], dtype=bool)
		sel = in_window & rc_sel_mask
		if sel.sum() < 50:
			print(f"[rc_peak/{name}] RC-seeded window has only {sel.sum()} stars; "
			      f"falling back to full CMD window")
			sel = in_window.copy()
	else:
		sel = in_window.copy()

	alpha_prev_stored = np.nan
	for iteration in range(n_iter):
		color_sel = color[sel]
		mag_sel   = mag[sel]
		n_sel = int(np.sum(sel))
		if n_sel < 20:
			print(f"[rc_peak/{name}] iter {iteration}: {n_sel} stars — stopping early")
			break

		# Width metric at each candidate slope: normalised MAD of m_dered
		widths = np.empty(n_alpha)
		peaks  = np.empty(n_alpha)
		for ii, alpha in enumerate(alpha_grid):
			m_d = mag_sel - alpha * color_sel
			med = np.median(m_d)
			mad = np.median(np.abs(m_d - med)) * 1.4826
			peaks[ii]  = med
			widths[ii] = mad if mad > 0 else 1e9

		best_idx   = int(np.argmin(widths))
		best_alpha = float(alpha_grid[best_idx])
		best_sigma = float(widths[best_idx])
		best_peak  = float(peaks[best_idx])

		m_dered_all = mag - best_alpha * color
		# Clip radius is sigma_clip * current MAD, but never exceed sigma_clip_max_mag
		clip_rad = min(sigma_clip * best_sigma, sigma_clip_max_mag)
		new_sel = in_window & (np.abs(m_dered_all - best_peak) < clip_rad)

		converged = (iteration > 0) and (abs(best_alpha - alpha_prev_stored) < 0.02)
		sel = new_sel
		alpha_prev_stored = best_alpha
		print(f"[rc_peak/{name}] iter {iteration}: alpha={best_alpha:.3f}, "
		      f"sigma={best_sigma:.4f}, N={n_sel}"
		      + (" (converged)" if converged else ""))
		if converged:
			break

	# Precise minimum via bounded scalar optimisation on final selection
	color_sel = color[sel]
	mag_sel   = mag[sel]

	def _width(a):
		m_d = mag_sel - a * color_sel
		mad = np.median(np.abs(m_d - np.median(m_d))) * 1.4826
		return mad if mad > 0 else 1e9

	opt = minimize_scalar(_width, bounds=(alpha_min, alpha_max), method='bounded')
	best_alpha = float(opt.x)

	# Final width curve over the entire grid at the converged selection
	widths_final = np.array([_width(a) for a in alpha_grid])
	best_sigma = float(widths_final[np.argmin(widths_final)])

	m_dered_sel = mag_sel - best_alpha * color_sel

	# Gaussian fit for precise peak and width
	peak_fit, sigma_fit, _ = _rc_peak_gauss_fit(m_dered_sel)

	# Uncertainty: curvature of the width(alpha) parabola near the minimum
	near = np.abs(alpha_grid - best_alpha) < 1.0
	if near.sum() >= 5:
		pcoeff = np.polyfit(alpha_grid[near], widths_final[near], 2)
		curvature = pcoeff[0]
		delta_w = best_sigma / np.sqrt(max(len(color_sel), 1))
		alpha_err = float(np.sqrt(delta_w / curvature)) if curvature > 0 else 0.05
	else:
		alpha_err = 0.05

	print(f"[rc_peak/{name}] FINAL: alpha={best_alpha:.3f} ± {alpha_err:.3f}, "
	      f"peak={peak_fit:.3f}, sigma={sigma_fit:.4f}, N={int(np.sum(sel))}")

	return dict(
		name=name,
		best_alpha=best_alpha,
		alpha_err=alpha_err,
		peak_m_dered=peak_fit,
		sigma_m_dered=sigma_fit,
		n_selected=int(np.sum(sel)),
		color_sel=color_sel,
		mag_sel=mag_sel,
		m_dered_sel=m_dered_sel,
		alpha_grid=alpha_grid,
		widths=widths_final,
		sel_mask_full=sel,
		color_range=color_range,
		mag_range=mag_range,
		**slopes,
	)


def plot_rc_peak_cmd(basetable, rc_peak_result, name, outpath,
	                 axlims=(0, 2.5, 21, 14.5)):
	"""Hexbin CMD with best-fit, G21, and CT06 slope lines and uncertainty band."""
	mag182 = _filled_float(basetable['mag_ab_f182m'])
	mag212 = _filled_float(basetable['mag_ab_f212n'])
	valid  = np.isfinite(mag182) & np.isfinite(mag212)
	color  = (mag182 - mag212)[valid]
	mag    = mag182[valid]
	extent = (axlims[0], axlims[1], min(axlims[2], axlims[3]), max(axlims[2], axlims[3]))

	r          = rc_peak_result
	best_alpha = r['best_alpha']
	alpha_err  = r['alpha_err']
	alpha_g21  = r['alpha_g21']
	alpha_ct06 = r['alpha_ct06']
	peak       = r['peak_m_dered']

	# Anchor all slope lines at color0=1.0 and the RC peak
	color0 = 1.0
	color_line = np.array([axlims[0], axlims[1]])

	def _mag_line(alpha):
		"""Observed F182M for an RC star of dereddened magnitude `peak` as a function of color."""
		return peak + alpha * (color_line - color0) + alpha * color0

	fig, ax = pl.subplots(figsize=(9, 7))
	ax.hexbin(color, mag, mincnt=1, gridsize=90, extent=extent,
	          cmap='Greys', bins='log', rasterized=True)

	# RC selection used by the slope fit
	sel = r['sel_mask_full']
	ax.hexbin((mag182 - mag212)[sel], mag182[sel], mincnt=1, gridsize=60,
	          extent=extent, cmap='Reds', alpha=0.75, rasterized=True)

	# Best-fit slope with uncertainty band
	ml           = _mag_line(best_alpha)
	ml_lo        = _mag_line(best_alpha - alpha_err)
	ml_hi        = _mag_line(best_alpha + alpha_err)
	ax.plot(color_line, ml, 'y-', linewidth=2.5, zorder=6,
	        label=rf'Best fit  $\alpha={best_alpha:.2f}\pm{alpha_err:.2f}$')
	ax.fill_between(color_line, ml_lo, ml_hi, color='yellow', alpha=0.25, zorder=5)

	ax.plot(color_line, _mag_line(alpha_g21),  'b--', linewidth=1.8, zorder=6,
	        label=rf'G21   $\alpha={alpha_g21:.2f}$')
	ax.plot(color_line, _mag_line(alpha_ct06), 'c--', linewidth=1.8, zorder=6,
	        label=rf'CT06  $\alpha={alpha_ct06:.2f}$')

	ax.set_xlim(axlims[0], axlims[1])
	ax.set_ylim(axlims[2], axlims[3])
	ax.set_xlabel('F182M \u2212 F212N')
	ax.set_ylabel('F182M')
	ax.set_title(f'RC peak slope fit (rc_peak): {name}')
	ax.legend(loc='lower right', fontsize=9)
	ax.grid(alpha=0.2)
	fig.tight_layout()
	fig.savefig(outpath, dpi=200)
	print(f"Wrote rc_peak CMD to {outpath}")


def plot_rc_peak_histogram(rc_peak_result, name, outpath):
	"""Histogram of m_dered at the best-fit slope with Gaussian overlay."""
	r         = rc_peak_result
	m_d       = r['m_dered_sel']
	peak      = r['peak_m_dered']
	sigma     = r['sigma_m_dered']
	alpha     = r['best_alpha']
	alpha_err = r['alpha_err']
	n         = r['n_selected']

	mad  = max(np.median(np.abs(m_d - np.median(m_d))) * 1.4826, 1e-4)
	bins = np.linspace(peak - 6 * mad, peak + 6 * mad, 80)

	fig, ax = pl.subplots(figsize=(8, 5))
	counts, edges, _ = ax.hist(m_d, bins=bins, histtype='stepfilled',
	                            color='steelblue', alpha=0.7,
	                            label=f'Data  N={n:,}')
	x_fine = np.linspace(bins[0], bins[-1], 600)
	ax.plot(x_fine,
	        counts.max() * np.exp(-0.5 * ((x_fine - peak) / sigma) ** 2),
	        'r-', linewidth=2,
	        label=rf'Gaussian  $\mu={peak:.3f}$, $\sigma={sigma:.4f}$')
	ax.axvline(peak, color='r', linestyle='--', alpha=0.6)
	ax.set_xlabel(rf'$m_\mathrm{{dered}}$ = F182M $-$ {alpha:.2f}$\times$(F182M$-$F212N)')
	ax.set_ylabel('N')
	ax.set_title(rf'RC peak histogram ($\alpha={alpha:.2f}\pm{alpha_err:.2f}$, rc_peak): {name}')
	ax.legend(loc='best')
	ax.grid(alpha=0.3)
	fig.tight_layout()
	fig.savefig(outpath, dpi=200)
	print(f"Wrote rc_peak histogram to {outpath}")


def plot_rc_peak_width_vs_alpha(rc_peak_result, name, outpath):
	"""Width of the dereddened RC distribution vs slope -- shows clean minimum."""
	r          = rc_peak_result
	alpha_grid = r['alpha_grid']
	widths     = r['widths']
	best_alpha = r['best_alpha']
	alpha_err  = r['alpha_err']
	alpha_g21  = r['alpha_g21']
	alpha_ct06 = r['alpha_ct06']

	fig, ax = pl.subplots(figsize=(8, 5))
	ax.plot(alpha_grid, widths, 'k-', linewidth=1.5, label='RC width (normalised MAD)')
	ax.axvline(best_alpha, color='y', linewidth=2.5,
	           label=rf'Best fit  $\alpha={best_alpha:.2f}\pm{alpha_err:.2f}$')
	ax.axvspan(best_alpha - alpha_err, best_alpha + alpha_err,
	           color='yellow', alpha=0.25)
	ax.axvline(alpha_g21,  color='b', linestyle='--', linewidth=1.8,
	           label=rf'G21   $\alpha={alpha_g21:.2f}$')
	ax.axvline(alpha_ct06, color='c', linestyle='--', linewidth=1.8,
	           label=rf'CT06  $\alpha={alpha_ct06:.2f}$')
	ax.set_xlabel(r'CMD slope  $\alpha$ = d(F182M)/d(F182M$-$F212N)')
	ax.set_ylabel(r'RC width  (normalised MAD, mag)')
	ax.set_title(f'Width vs slope (rc_peak): {name}')
	ax.legend(loc='best')
	ax.grid(alpha=0.3)
	fig.tight_layout()
	fig.savefig(outpath, dpi=200)
	print(f"Wrote rc_peak width-vs-alpha to {outpath}")


def plot_rc_peak_vs_color(rc_peak_result, name, outpath, n_bins=10):
	"""RC peak position vs colour (extinction proxy).

	If alpha is correct the peak should be flat across all colour bins,
	confirming the extinction law slopes is correct.
	"""
	r          = rc_peak_result
	color_sel  = r['color_sel']
	m_dered    = r['m_dered_sel']
	best_alpha = r['best_alpha']
	alpha_err  = r['alpha_err']
	peak       = r['peak_m_dered']
	sigma      = r['sigma_m_dered']

	edges = np.linspace(color_sel.min(), color_sel.max(), n_bins + 1)
	centers = 0.5 * (edges[:-1] + edges[1:])
	bin_peaks = np.full(n_bins, np.nan)
	bin_errs  = np.full(n_bins, np.nan)

	for ii in range(n_bins):
		mask = (color_sel >= edges[ii]) & (color_sel < edges[ii + 1])
		if mask.sum() < 10:
			continue
		md = m_dered[mask]
		med = np.median(md)
		mad = np.median(np.abs(md - med)) * 1.4826
		bin_peaks[ii] = med
		bin_errs[ii]  = mad / np.sqrt(mask.sum())

	finite = np.isfinite(bin_peaks)
	fig, ax = pl.subplots(figsize=(8, 5))
	ax.errorbar(centers[finite], bin_peaks[finite], yerr=bin_errs[finite],
	            fmt='o', color='steelblue', capsize=4,
	            label=rf'RC peak per bin  ($\alpha={best_alpha:.2f}\pm{alpha_err:.2f}$)')
	ax.axhline(peak, color='r', linestyle='--', linewidth=1.5,
	           label=f'Global peak = {peak:.3f}')
	ax.fill_between([color_sel.min(), color_sel.max()],
	                peak - sigma, peak + sigma,
	                color='r', alpha=0.15, label=f'$\\pm\\sigma={sigma:.4f}$')
	ax.set_xlabel('F182M \u2212 F212N (observed colour, proxy for extinction)')
	ax.set_ylabel(rf'$m_\mathrm{{dered}}$ = F182M $-$ {best_alpha:.2f}$\times$(colour)')
	ax.set_title(f'RC peak vs extinction — flatness test (rc_peak): {name}')
	ax.legend(loc='best', fontsize=9)
	ax.grid(alpha=0.3)
	fig.tight_layout()
	fig.savefig(outpath, dpi=200)
	print(f"Wrote rc_peak vs-colour flatness plot to {outpath}")
def main():
	os.makedirs(f'{basepath}/distance', exist_ok=True)

	brick_table = make_downselected_table_20251211()
	brick_rc_table = rc_selection_and_distances(brick_table)
	brick_distances, brick_selected = selected_distances(brick_rc_table)

	cloudc_table = load_cloudc_catalog()
	cloudc_rc_table = rc_selection_and_distances(cloudc_table)
	cloudc_distances, cloudc_selected = selected_distances(cloudc_rc_table)

	print(f"Brick: total stars considered: {len(brick_rc_table)}")
	print(f"Brick: RC-selected stars: {brick_selected.sum()}")
	if len(brick_distances) > 0:
		print(f"Brick: distance median (kpc): {np.nanmedian(brick_distances):0.2f}")
		print(f"Brick: distance std (kpc): {np.nanstd(brick_distances):0.2f}")

	print(f"Cloud C: total stars considered: {len(cloudc_rc_table)}")
	print(f"Cloud C: RC-selected stars: {cloudc_selected.sum()}")
	if len(cloudc_distances) > 0:
		print(f"Cloud C: distance median (kpc): {np.nanmedian(cloudc_distances):0.2f}")
		print(f"Cloud C: distance std (kpc): {np.nanstd(cloudc_distances):0.2f}")

	plot_histogram(
		brick_distances,
		title='RC distance histogram (Brick; G21 to 8 kpc + CT06 in GC)',
		outpath=f'{basepath}/distance/red_clump_distance_histogram.png',
		color='C0',
	)

	plot_histogram(
		cloudc_distances,
		title='RC distance histogram (Cloud C; G21 to 8 kpc + CT06 in GC)',
		outpath=f'{basepath}/distance/red_clump_distance_histogram_cloudc.png',
		color='C1',
	)

	plot_overplot(
		distances_brick=brick_distances,
		distances_cloudc=cloudc_distances,
		outpath=f'{basepath}/distance/red_clump_distance_histogram_brick_cloudc_overplot.png',
	)

	plot_overplot_normalized(
		distances_brick=brick_distances,
		distances_cloudc=cloudc_distances,
		outpath=f'{basepath}/distance/red_clump_distance_histogram_brick_cloudc_normalized.png',
	)

	plot_cmd_with_rc_selection(
		basetable=brick_table,
		rc_table=brick_rc_table,
		name='Brick',
		outpath=f'{basepath}/distance/red_clump_cmd_brick.png',
	)

	plot_cmd_with_rc_selection(
		basetable=cloudc_table,
		rc_table=cloudc_rc_table,
		name='Cloud C',
		outpath=f'{basepath}/distance/red_clump_cmd_cloudc.png',
	)

	plot_cmd_difference_normalized_hexbin(
		basetable_brick=brick_table,
		basetable_cloudc=cloudc_table,
		outpath=f'{basepath}/distance/red_clump_cmd_brick_minus_cloudc_normalized_hexbin.png',
	)

	brick_ak_dist, brick_ak = _rc_distance_reddening(brick_rc_table)
	cloudc_ak_dist, cloudc_ak = _rc_distance_reddening(cloudc_rc_table)

	plot_distance_vs_reddening(
		distances=brick_ak_dist,
		reddening=brick_ak,
		title='RC distance vs reddening (Brick)',
		outpath=f'{basepath}/distance/red_clump_distance_vs_reddening_brick.png',
		color='C0',
	)

	plot_distance_vs_reddening(
		distances=cloudc_ak_dist,
		reddening=cloudc_ak,
		title='RC distance vs reddening (Cloud C)',
		outpath=f'{basepath}/distance/red_clump_distance_vs_reddening_cloudc.png',
		color='C1',
	)

	plot_distance_vs_reddening_overplot(
		dist_brick=brick_ak_dist,
		ak_brick=brick_ak,
		dist_cloudc=cloudc_ak_dist,
		ak_cloudc=cloudc_ak,
		outpath=f'{basepath}/distance/red_clump_distance_vs_reddening_brick_cloudc_overplot.png',
	)

	brick_table_outpath = f'{basepath}/distance/red_clump_distance_table.ecsv'
	brick_rc_table.write(brick_table_outpath, overwrite=True)
	print(f"Wrote Brick RC distance table to {brick_table_outpath}")

	cloudc_table_outpath = f'{basepath}/distance/red_clump_distance_table_cloudc.ecsv'
	cloudc_rc_table.write(cloudc_table_outpath, overwrite=True)
	print(f"Wrote Cloud C RC distance table to {cloudc_table_outpath}")

	return {
		'brick': brick_rc_table,
		'cloudc': cloudc_rc_table,
	}


def main_rc_peak():
	"""Run rc_peak analysis for Brick and Cloud C."""
	os.makedirs(f'{basepath}/distance', exist_ok=True)

	brick_table  = make_downselected_table_20251211()
	cloudc_table = load_cloudc_catalog()

	brick_rc_peak  = rc_peak_fit_slope(brick_table,  'Brick')
	cloudc_rc_peak = rc_peak_fit_slope(cloudc_table, 'Cloud C')

	for rc_peak_result, basetable, tag in [
		(brick_rc_peak,  brick_table,  'brick'),
		(cloudc_rc_peak, cloudc_table, 'cloudc'),
	]:
		nm = rc_peak_result['name']
		plot_rc_peak_cmd(
			basetable, rc_peak_result, nm,
			outpath=f'{basepath}/distance/rc_peak_cmd_{tag}.png',
		)
		plot_rc_peak_histogram(
			rc_peak_result, nm,
			outpath=f'{basepath}/distance/rc_peak_histogram_{tag}.png',
		)
		plot_rc_peak_width_vs_alpha(
			rc_peak_result, nm,
			outpath=f'{basepath}/distance/rc_peak_width_vs_alpha_{tag}.png',
		)
		plot_rc_peak_vs_color(
			rc_peak_result, nm,
			outpath=f'{basepath}/distance/rc_peak_vs_color_{tag}.png',
		)

	return {'brick': brick_rc_peak, 'cloudc': cloudc_rc_peak}


if __name__ == '__main__':
	main()
	main_rc_peak()