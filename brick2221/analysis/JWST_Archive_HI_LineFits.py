#!/usr/bin/env python

"""Fit Pa-alpha and Br-alpha line regions in JWST archive spectra with pyspeckit."""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pyspeckit
from astropy import units as u
from astropy.table import Table
from astropy.units import UnitsWarning
from tqdm.auto import tqdm


DEFAULT_TABLE = Path(
    "/orange/adamginsburg/jwst/spectra/mastDownload/JWST/"
    "jwst_archive_spectra_as_fluxes.ecsv"
)
DEFAULT_OUTTABLE = Path(
    "/orange/adamginsburg/jwst/spectra/mastDownload/JWST/"
    "jwst_archive_hydrogen_linefits.ecsv"
)


LINE_CONFIG = {
    "paa": {
        "name": "PaA",
        "center_um": 1.8751,
        "fit_min_um": 1.84,
        "fit_max_um": 1.92,
        "cont_windows_um": [1.80, 1.86, 1.89, 1.95],
        "sigma_guess_um": 0.003,
    },
    "bra": {
        "name": "BrA",
        "center_um": 4.0523,
        "fit_min_um": 4.00,
        "fit_max_um": 4.11,
        "cont_windows_um": [3.95, 4.02, 4.08, 4.15],
        "sigma_guess_um": 0.004,
    },
}


def load_spectrum(spectrum_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UnitsWarning)
        spectable = Table.read(spectrum_path, hdu=1)

    wavelength = np.array(spectable["WAVELENGTH"], dtype=float)
    flux = np.array(spectable["FLUX"], dtype=float)
    ok = np.isfinite(wavelength) & np.isfinite(flux)
    return wavelength[ok], flux[ok]


def fit_single_line(wavelength_um: np.ndarray, flux_jy: np.ndarray, config: dict) -> dict:
    if wavelength_um.size == 0 or flux_jy.size == 0:
        return {
            "line_center_fit_um": np.nan,
            "line_amplitude_jy": np.nan,
            "line_sigma_um": np.nan,
            "line_integrated_flux_jy_um": np.nan,
            "continuum_at_center_jy": np.nan,
            "coverage_ok": False,
            "fit_ok": False,
        }

    fit_min = config["fit_min_um"]
    fit_max = config["fit_max_um"]
    has_coverage = (np.nanmin(wavelength_um) <= fit_min) and (np.nanmax(wavelength_um) >= fit_max)
    if not has_coverage:
        return {
            "line_center_fit_um": np.nan,
            "line_amplitude_jy": np.nan,
            "line_sigma_um": np.nan,
            "line_integrated_flux_jy_um": np.nan,
            "continuum_at_center_jy": np.nan,
            "coverage_ok": False,
            "fit_ok": False,
        }

    spec = pyspeckit.Spectrum(xarr=wavelength_um * u.um, data=flux_jy * u.Jy)
    spec.baseline(
        include=config["cont_windows_um"],
        order=1,
        subtract=False,
        reset_selection=True,
        interactive=False,
    )

    baseline_model = np.array(spec.baseline.basespec, dtype=float)
    continuum_level = float(np.interp(config["center_um"], wavelength_um, baseline_model))

    line_only_flux = flux_jy - baseline_model
    line_spec = pyspeckit.Spectrum(xarr=wavelength_um * u.um, data=line_only_flux * u.Jy)

    in_fit = (wavelength_um >= fit_min) & (wavelength_um <= fit_max)
    if in_fit.sum() < 5:
        return {
            "line_center_fit_um": np.nan,
            "line_amplitude_jy": np.nan,
            "line_sigma_um": np.nan,
            "line_integrated_flux_jy_um": np.nan,
            "continuum_at_center_jy": continuum_level,
            "coverage_ok": True,
            "fit_ok": False,
        }

    if not np.any(np.isfinite(line_only_flux[in_fit])):
        return {
            "line_center_fit_um": np.nan,
            "line_amplitude_jy": np.nan,
            "line_sigma_um": np.nan,
            "line_integrated_flux_jy_um": np.nan,
            "continuum_at_center_jy": continuum_level,
            "coverage_ok": True,
            "fit_ok": False,
        }

    if np.nanstd(line_only_flux[in_fit]) == 0:
        return {
            "line_center_fit_um": np.nan,
            "line_amplitude_jy": np.nan,
            "line_sigma_um": np.nan,
            "line_integrated_flux_jy_um": np.nan,
            "continuum_at_center_jy": continuum_level,
            "coverage_ok": True,
            "fit_ok": False,
        }

    peak_guess = float(np.nanmax(line_only_flux[in_fit]))
    if not np.isfinite(peak_guess) or peak_guess <= 0:
        return {
            "line_center_fit_um": np.nan,
            "line_amplitude_jy": np.nan,
            "line_sigma_um": np.nan,
            "line_integrated_flux_jy_um": np.nan,
            "continuum_at_center_jy": continuum_level,
            "coverage_ok": True,
            "fit_ok": False,
        }

    line_spec.specfit(
        fittype="gaussian",
        guesses=[peak_guess, config["center_um"], config["sigma_guess_um"]],
        xmin=fit_min,
        xmax=fit_max,
        annotate=False,
    )

    amp_jy = float(line_spec.specfit.parinfo[0].value)
    center_um = float(line_spec.specfit.parinfo[1].value)
    sigma_um = float(line_spec.specfit.parinfo[2].value)
    integrated_jy_um = float(amp_jy * np.abs(sigma_um) * np.sqrt(2.0 * np.pi))

    return {
        "line_center_fit_um": center_um,
        "line_amplitude_jy": amp_jy,
        "line_sigma_um": sigma_um,
        "line_integrated_flux_jy_um": integrated_jy_um,
        "continuum_at_center_jy": continuum_level,
        "coverage_ok": True,
        "fit_ok": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit Pa-alpha and Br-alpha in JWST archive spectra and save a summary table."
    )
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE, help="Input archive ECSV table")
    parser.add_argument(
        "--outtable",
        type=Path,
        default=DEFAULT_OUTTABLE,
        help="Output ECSV table with fitted line properties",
    )
    parser.add_argument(
        "--max-spectra",
        type=int,
        default=None,
        help="Optional limit on number of spectra for quick testing",
    )
    args = parser.parse_args()

    table = Table.read(args.table)
    if args.max_spectra is not None:
        table = table[: args.max_spectra]

    rows = []
    for row in tqdm(table, desc="Fitting hydrogen lines"):
        spectrum_path = Path(str(row["Filename"]))
        wavelength_um, flux_jy = load_spectrum(spectrum_path)

        result_row = {
            "Filename": str(row["Filename"]),
            "Target": str(row["Target"]) if "Target" in table.colnames else "",
            "Object": str(row["Object"]) if "Object" in table.colnames else "",
        }

        for tag, config in LINE_CONFIG.items():
            fit = fit_single_line(wavelength_um=wavelength_um, flux_jy=flux_jy, config=config)
            for key, value in fit.items():
                result_row[f"{tag}_{key}"] = value

        rows.append(result_row)

    out = Table(rows=rows)
    args.outtable.parent.mkdir(parents=True, exist_ok=True)
    out.write(args.outtable, overwrite=True)
    print(f"Wrote {len(out)} fitted spectra to {args.outtable}")


if __name__ == "__main__":
    main()
