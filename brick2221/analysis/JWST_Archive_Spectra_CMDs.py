#!/usr/bin/env python

"""Create color-magnitude diagrams from JWST archive-derived photometry."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table


DEFAULT_TABLE = Path(
    "/orange/adamginsburg/jwst/spectra/mastDownload/JWST/"
    "jwst_archive_spectra_as_fluxes.ecsv"
)


def get_mag_column(table: Table, filt: str) -> str:
    """Return the preferred magnitude column name for a filter."""
    candidates = [f"JWST/NIRCam.{filt}", f"JWST/NIRCam2025.{filt}"]
    for col in candidates:
        if col in table.colnames:
            return col
    raise KeyError(f"Could not find a magnitude column for {filt}. Tried {candidates}")


def finite_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def make_cmd(
    table: Table,
    mag_filter: str,
    color_filter_1: str,
    color_filter_2: str,
    output_path: Path,
) -> None:
    mag_col = get_mag_column(table, mag_filter)
    c1_col = get_mag_column(table, color_filter_1)
    c2_col = get_mag_column(table, color_filter_2)

    mag = np.array(table[mag_col], dtype=float)
    color = np.array(table[c1_col], dtype=float) - np.array(table[c2_col], dtype=float)
    color, mag = finite_pair(color, mag)

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    ax.scatter(color, mag, s=8, alpha=0.5, linewidths=0, rasterized=True)

    ax.set_xlabel(f"{color_filter_1} - {color_filter_2}")
    ax.set_ylabel(mag_filter)
    ax.set_title(f"CMD: {mag_filter} vs {color_filter_1} - {color_filter_2}")

    # Conventional CMD orientation: brighter sources at the top.
    ax.invert_yaxis()
    ax.grid(alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Make requested JWST archive color-magnitude diagrams."
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=DEFAULT_TABLE,
        help="Path to jwst_archive_spectra_as_fluxes.ecsv",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("."),
        help="Directory for output figures",
    )
    args = parser.parse_args()

    table = Table.read(args.table)

    print("182-187 CMD")
    make_cmd(
        table=table,
        mag_filter="F187N",
        color_filter_1="F182M",
        color_filter_2="F187N",
        output_path=args.outdir / "NIRSpec_cmd_F187N_vs_F182M_minus_F187N.png",
    )
    print("405-410 CMD")
    make_cmd(
        table=table,
        mag_filter="F405N",
        color_filter_1="F405N",
        color_filter_2="F410M",
        output_path=args.outdir / "NIRSpec_cmd_F405N_vs_F405N_minus_F410M.png",
    )


if __name__ == "__main__":
    main()
