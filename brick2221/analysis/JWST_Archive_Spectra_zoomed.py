#!/usr/bin/env python

"""Plot zoomed spectra for archive sources selected by NIRCam colors/magnitudes."""

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from astropy.units import UnitsWarning
from astroquery.svo_fps import SvoFps


DEFAULT_TABLE = Path(
    "/orange/adamginsburg/jwst/spectra/mastDownload/JWST/"
    "jwst_archive_spectra_as_fluxes.ecsv"
)
DEFAULT_SPECTRUM_DIR = Path("/orange/adamginsburg/jwst/spectra/mastDownload/JWST")


def get_mag_column(table: Table, filt: str) -> str:
    candidates = [f"JWST/NIRCam.{filt}", f"JWST/NIRCam2025.{filt}"]
    for col in candidates:
        if col in table.colnames:
            return col
    raise KeyError(f"Could not find a magnitude column for {filt}. Tried {candidates}")


def get_filter_profile(filter_name: str):
    return SvoFps.get_transmission_data(f"JWST/NIRCam.{filter_name}")


def save_zoomed_spectrum(
    spectrum_path: Path,
    output_path: Path,
    label: str,
    bands: list[tuple[Table, str, str]],
    xmin: float = 1.5,
    xmax: float = 2.1,
) -> bool:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UnitsWarning)
        spectrum = Table.read(spectrum_path, hdu=1)
    wavelength = np.array(spectrum["WAVELENGTH"], dtype=float)
    flux = np.array(spectrum["FLUX"], dtype=float)

    finite = np.isfinite(wavelength) & np.isfinite(flux)
    wavelength = wavelength[finite]
    flux = flux[finite]

    sel = (wavelength >= xmin) & (wavelength <= xmax)
    if sel.sum() == 0:
        return False
    wavelength, flux = wavelength[sel], flux[sel]

    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    ax.plot(wavelength, flux, color="black", linewidth=0.8, label="Spectrum")
    ax.set_xlim(xmin, xmax)

    y_min = np.nanmin(flux)
    y_max = np.nanmax(flux)
    if np.isfinite(y_min) and np.isfinite(y_max) and y_max > y_min:
        y_span = y_max - y_min
        band_scale = y_min
        band_height = 0.60 * y_span
    else:
        band_scale = 0.0
        band_height = 1.0

    for band, color, name in bands:
        band_wavelength = np.array(band["Wavelength"], dtype=float) / 1e4
        band_transmission = np.array(band["Transmission"], dtype=float)
        ax.plot(
            band_wavelength,
            band_scale + band_transmission * band_height,
            color=color,
            linewidth=2.0,
            alpha=0.85,
            label=f"{name} transmission",
        )

    ax.set_xlabel(r"Wavelength [$\mu$m]")
    ax.set_ylabel(r"Flux [$\mathrm{Jy}$]")
    ax.set_title(label.split("/")[-1])
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create zoomed spectra for archive sources selected by NIRCam photometry."
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=DEFAULT_TABLE,
        help="Path to jwst_archive_spectra_as_fluxes.ecsv",
    )
    parser.add_argument(
        "--spectrum-dir",
        type=Path,
        default=DEFAULT_SPECTRUM_DIR,
        help="Directory containing the x1d.fits files referenced by the table",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Directory for output plots. Defaults to <table parent>/zoomed_spectra",
    )
    args = parser.parse_args()

    table = Table.read(args.table)
    f187_col = get_mag_column(table, "F187N")
    f182_col = get_mag_column(table, "F182M")
    f405_col = get_mag_column(table, "F405N")
    f410_col = get_mag_column(table, "F410M")

    f187 = np.array(table[f187_col], dtype=float)
    f182_minus_f187 = np.array(table[f182_col], dtype=float) - f187
    f405 = np.array(table[f405_col], dtype=float)
    f405_minus_410 = np.array(table[f405_col], dtype=float) - np.array(table[f410_col], dtype=float)
    selection = (f187 < 15) | ((f182_minus_f187 < -0.5) & (f187 < 20)) | ((f405_minus_410 < -0.5) & (f405 < 20))

    selected = table[selection]
    outdir = args.outdir if args.outdir is not None else args.table.parent / "zoomed_spectra"
    print(f"Selected {len(selected)} spectra for zoomed plotting")

    band_182 = get_filter_profile("F182M")
    band_187 = get_filter_profile("F187N")
    band_405 = get_filter_profile("F405N")
    band_410 = get_filter_profile("F410M")

    paa_bands = [(band_182, "tab:blue", "F182M"), (band_187, "tab:orange", "F187N")]
    bra_bands = [(band_405, "tab:green", "F405N"), (band_410, "tab:red", "F410M")]

    paa_saved = 0
    bra_saved = 0

    for row in selected:
        spectrum_path = Path(str(row["Filename"]))
        if not spectrum_path.is_absolute():
            spectrum_path = args.spectrum_dir / spectrum_path

        spectrum_name = str(row["Object"]) if "Object" in table.colnames else str(row["Target"])
        paa_label = (
            f"{spectrum_name} | {row['Target']} | {row['Filename']}\n"
            f"F187N={row[f187_col]:0.2f}, F182M-F187N={row[f182_col] - row[f187_col]:0.2f}"
        )
        bra_label = (
            f"{spectrum_name} | {row['Target']} | {row['Filename']}\n"
            f"F405N={row[f405_col]:0.2f}, F405N-F410M={row[f405_col] - row[f410_col]:0.2f}"
        )
        output_path_paa = outdir / f"{Path(str(row['Filename'])).stem}_zoomed_1p5_to_2p1um.png"
        if save_zoomed_spectrum(
            spectrum_path,
            output_path_paa,
            paa_label,
            bands=paa_bands,
            xmin=1.5,
            xmax=2.1,
        ):
            paa_saved += 1
            print(f"PaA: f182m-f187n={row[f182_col] - row[f187_col]:0.2f} | Saved {output_path_paa}")

        output_path_bra = outdir / f"{Path(str(row['Filename'])).stem}_zoomed_3p9_to_4p2um.png"
        if save_zoomed_spectrum(
            spectrum_path,
            output_path_bra,
            bra_label,
            bands=bra_bands,
            xmin=3.8,
            xmax=4.4,
        ):
            bra_saved += 1
            print(f"BrA: f405n-f410m={row[f405_col] - row[f410_col]:0.2f} | Saved {output_path_bra}")
    print(f"Saved PaA zooms: {paa_saved} / {len(selected)}")
    print(f"Saved BrA zooms: {bra_saved} / {len(selected)}")


if __name__ == "__main__":
    main()
