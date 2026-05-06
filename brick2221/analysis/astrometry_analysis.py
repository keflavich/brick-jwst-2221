#!/usr/bin/env python
"""Crossmatch final merged JWST catalogs against VVV and Gaia references.

This script measures absolute astrometric offsets for each merged catalog produced
by merge_catalogs.py, treating VVV and Gaia as the absolute reference frames.
It writes per-catalog match tables, summary tables, and xmatch-style residual plots.
"""

from __future__ import annotations

import argparse
import datetime
import glob as globmod
import os
from pathlib import Path
import re
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy import stats
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack

try:
    from brick2221.analysis.make_reference_from_pipeline_catalogs import (
        compute_query_footprint,
        fetch_gaia_catalog,
        fetch_vvv_catalog,
        resolve_default_basepath,
    )
except ImportError:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.append(str(repo_root))
    from brick2221.analysis.make_reference_from_pipeline_catalogs import (
        compute_query_footprint,
        fetch_gaia_catalog,
        fetch_vvv_catalog,
        resolve_default_basepath,
    )


DEFAULT_CATALOG_GLOBS = (
    "basic_merged_photometry_tables_merged.fits",
    "basic_merged_photometry_tables_merged.ecsv",
)

REFERENCE_SPECS = (
    {
        "name": "vvv",
        "label": "VVV",
        "mag_columns": ("Ks_refmag", "Ksmag3", "Ksmag", "Ks3mag"),
    },
    {
        "name": "gaia",
        "label": "Gaia DR3",
        "mag_columns": ("phot_g_mean_mag", "Gmag", "Gmag3", "Gmag2"),
    },
)

K_BAND_EQUIVALENT_FILTERS = ("f212n", "f210m", "f200w")
VVV_KS_ERROR_COLUMNS = ("e_Ks2ap3", "e_Ks1ap3", "e_Ks2ap1", "e_Ks1ap1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        default="brick",
        help="Target name used to resolve /orange/adamginsburg/jwst/{target}.",
    )
    parser.add_argument(
        "--basepath",
        default=None,
        help="Base data directory. Defaults to /orange/adamginsburg/jwst/{target} (or /NB for sgrb2).",
    )
    parser.add_argument(
        "--catalog-glob",
        action="append",
        default=None,
        help="Catalog glob to search under {basepath}/catalogs. May be repeated.",
    )
    parser.add_argument(
        "--catalog",
        action="append",
        default=None,
        help="Explicit catalog file path or glob to analyze. May be repeated.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory for plots and tables. Defaults to {basepath}/astrometry_analysis.",
    )
    parser.add_argument(
        "--max-sep-arcsec",
        type=float,
        default=0.5,
        help="Maximum separation for a JWST/reference match in arcsec.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Force refreshing cached VVV/Gaia query results.",
    )
    parser.add_argument(
        "--vvv-catalog",
        default="II/376/vvv4",
        help="Vizier catalog identifier for VVV reference queries.",
    )
    parser.add_argument(
        "--gaia-catalog",
        default="I/355/gaiadr3",
        help="Vizier catalog identifier for Gaia DR3 reference queries.",
    )
    parser.add_argument(
        "--max-catalogs",
        type=int,
        default=0,
        help="Optional cap on the number of catalogs to process (0 = no cap).",
    )
    parser.add_argument(
        "--vvv-photometric-nsigma",
        type=float,
        default=3.0,
        help=(
            "For VVV matches, require K-band-equivalent JWST and VVV magnitudes to agree "
            "within this many sigma of the less precise measurement (default: 3)."
        ),
    )
    return parser.parse_args()


def sanitize_label(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


def default_basepath(target: str) -> Path:
    return resolve_default_basepath(target)


def normalize_catalog_inputs(basepath: Path, args: argparse.Namespace) -> list[Path]:
    explicit = args.catalog or []
    if explicit:
        discovered: list[Path] = []
        for pattern in explicit:
            expanded = globmod.glob(pattern)
            if expanded:
                discovered.extend(Path(path) for path in expanded)
            else:
                path = Path(pattern)
                if path.exists():
                    discovered.append(path)
        return sorted(set(discovered))

    globs = args.catalog_glob or list(DEFAULT_CATALOG_GLOBS)
    discovered: list[Path] = []
    for pattern in globs:
        candidate = pattern if os.path.isabs(pattern) else str(basepath / "catalogs" / pattern)
        discovered.extend(Path(path) for path in globmod.glob(candidate))
    discovered = [path for path in discovered if path.is_file()]
    discovered = [path for path in discovered if "astrometry_analysis" not in path.name]
    if discovered:
        basic_fits = [path for path in discovered if path.name == "basic_merged_photometry_tables_merged.fits"]
        if basic_fits:
            return basic_fits
        basic_ecsv = [path for path in discovered if path.name == "basic_merged_photometry_tables_merged.ecsv"]
        if basic_ecsv:
            return basic_ecsv
    return sorted(set(discovered))


def find_skycoord_column(tbl: Table) -> str:
    for name in ("skycoord_ref", "skycoord", "skycoord_avg"):
        if name in tbl.colnames:
            return name
    skycoord_columns = [name for name in tbl.colnames if name.startswith("skycoord_")]
    if skycoord_columns:
        return skycoord_columns[0]
    raise ValueError("Could not find a skycoord column in the merged catalog.")


def get_catalog_coordinates(tbl: Table) -> SkyCoord:
    return tbl[find_skycoord_column(tbl)]


def get_reference_label(tbl: Table) -> str:
    if "skycoord_ref_filtername" in tbl.colnames:
        values = np.asarray(tbl["skycoord_ref_filtername"]).astype(str)
        if len(values) > 0:
            unique, counts = np.unique(values, return_counts=True)
            return unique[int(np.argmax(counts))]
    if "astrometric_reference_wavelength" in tbl.meta:
        return str(tbl.meta["astrometric_reference_wavelength"])
    return "reference"


def pick_mag_column(tbl: Table, candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in tbl.colnames:
            return name
    return None


def to_float_array(values) -> np.ndarray:
    return np.asarray(np.ma.filled(values, np.nan), dtype=float)


def pick_catalog_kband_columns(catalog: Table) -> tuple[str | None, str | None]:
    for filt in K_BAND_EQUIVALENT_FILTERS:
        mag_col = f"mag_ab_{filt}"
        err_col = f"emag_ab_{filt}"
        if mag_col in catalog.colnames and err_col in catalog.colnames:
            return mag_col, err_col
    return None, None


def build_vvv_sigma(reference: Table, reference_indices: np.ndarray) -> np.ndarray:
    sigma_candidates: list[np.ndarray] = []
    for colname in VVV_KS_ERROR_COLUMNS:
        if colname in reference.colnames:
            sigma_candidates.append(to_float_array(reference[colname][reference_indices]))

    if not sigma_candidates:
        return np.full(len(reference_indices), np.nan, dtype=float)

    sigma_stack = np.vstack(sigma_candidates)
    return np.nanmax(sigma_stack, axis=0)


def apply_vvv_photometric_filter(
    catalog: Table,
    reference: Table,
    reference_mag_column: str,
    match: dict[str, np.ndarray],
    nsigma: float,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    info: dict[str, object] = {
        "applied": False,
        "jwst_mag_column": "",
        "jwst_err_column": "",
        "n_before": int(len(match["catalog_indices"])),
        "n_after": int(len(match["catalog_indices"])),
        "median_mag_offset": np.nan,
    }

    jwst_mag_column, jwst_err_column = pick_catalog_kband_columns(catalog)
    if jwst_mag_column is None or jwst_err_column is None:
        return match, info
    if reference_mag_column not in reference.colnames:
        return match, info

    catalog_indices = match["catalog_indices"]
    reference_indices = match["reference_indices"]

    jwst_mag = to_float_array(catalog[jwst_mag_column][catalog_indices])
    jwst_sigma = to_float_array(catalog[jwst_err_column][catalog_indices])
    vvv_mag = to_float_array(reference[reference_mag_column][reference_indices])
    vvv_sigma = build_vvv_sigma(reference, reference_indices)

    finite = (
        np.isfinite(jwst_mag)
        & np.isfinite(jwst_sigma)
        & np.isfinite(vvv_mag)
        & np.isfinite(vvv_sigma)
        & (jwst_sigma > 0)
        & (vvv_sigma > 0)
    )
    if not np.any(finite):
        return match, info

    delta_mag = jwst_mag - vvv_mag
    median_delta = float(np.nanmedian(delta_mag[finite]))
    sigma_threshold = nsigma * np.maximum(jwst_sigma, vvv_sigma)
    phot_ok = finite & (np.abs(delta_mag - median_delta) <= sigma_threshold)

    filtered = {
        key: value[phot_ok]
        for key, value in match.items()
    }

    info.update(
        {
            "applied": True,
            "jwst_mag_column": jwst_mag_column,
            "jwst_err_column": jwst_err_column,
            "n_after": int(np.count_nonzero(phot_ok)),
            "median_mag_offset": median_delta,
        }
    )
    return filtered, info


def build_reference_cache_paths(outdir: Path, catalog_path: Path, ref_name: str) -> Path:
    cache_dir = outdir / "reference_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{sanitize_label(catalog_path.stem)}_{ref_name}.fits"


def load_reference_catalog(
    name: str,
    catalog_path: Path,
    center: SkyCoord,
    width: u.Quantity,
    height: u.Quantity,
    outdir: Path,
    refresh_cache: bool,
    vvv_catalog: str,
    gaia_catalog: str,
) -> tuple[Table, Path, str]:
    cache_path = build_reference_cache_paths(outdir, catalog_path, name)
    if name == "vvv":
        table = fetch_vvv_catalog(
            center=center,
            width=width,
            height=height,
            vvv_catalog=vvv_catalog,
            cache_path=cache_path,
            refresh_cache=refresh_cache,
        )
        mag_column = pick_mag_column(table, REFERENCE_SPECS[0]["mag_columns"])
        return table, cache_path, mag_column or "Ks_refmag"
    if name == "gaia":
        table = fetch_gaia_catalog(
            center=center,
            width=width,
            height=height,
            gaia_catalog=gaia_catalog,
            cache_path=cache_path,
            refresh_cache=refresh_cache,
        )
        mag_column = pick_mag_column(table, REFERENCE_SPECS[1]["mag_columns"])
        return table, cache_path, mag_column or "phot_g_mean_mag"
    raise ValueError(f"Unsupported reference catalog name: {name}")


def match_catalog_to_reference(
    catalog_coords: SkyCoord,
    reference_coords: SkyCoord,
    max_sep: u.Quantity,
) -> dict[str, np.ndarray]:
    catalog_idx, sep, _ = catalog_coords.match_to_catalog_sky(reference_coords, nthneighbor=1)
    reverse_idx, _, _ = reference_coords.match_to_catalog_sky(catalog_coords, nthneighbor=1)
    mutual = reverse_idx[catalog_idx] == np.arange(len(catalog_idx))
    keep = mutual & np.isfinite(sep) & (sep <= max_sep)

    match_indices = np.where(keep)[0]
    reference_indices = catalog_idx[keep]
    matched_catalog = catalog_coords[keep]
    matched_reference = reference_coords[reference_indices].transform_to(matched_catalog.frame)
    dra, ddec = matched_reference.spherical_offsets_to(matched_catalog)
    separation = matched_catalog.separation(matched_reference)

    return {
        "catalog_indices": match_indices,
        "reference_indices": reference_indices,
        "dra": dra.to(u.mas).value,
        "ddec": ddec.to(u.mas).value,
        "sep": separation.to(u.mas).value,
    }


def summarize_offsets(dra_mas: np.ndarray, ddec_mas: np.ndarray, sep_mas: np.ndarray) -> dict[str, float]:
    if len(dra_mas) == 0:
        return {
            "n_matches": 0,
            "median_dra_mas": np.nan,
            "median_ddec_mas": np.nan,
            "median_sep_mas": np.nan,
            "mad_dra_mas": np.nan,
            "mad_ddec_mas": np.nan,
            "mad_sep_mas": np.nan,
            "sem_dra_mas": np.nan,
            "sem_ddec_mas": np.nan,
            "sem_sep_mas": np.nan,
            "vector_median_offset_mas": np.nan,
            "vector_sem_mas": np.nan,
        }

    n_matches = len(dra_mas)
    med_dra = float(np.nanmedian(dra_mas))
    med_ddec = float(np.nanmedian(ddec_mas))
    med_sep = float(np.nanmedian(sep_mas))
    mad_dra = float(stats.mad_std(dra_mas, ignore_nan=True))
    mad_ddec = float(stats.mad_std(ddec_mas, ignore_nan=True))
    mad_sep = float(stats.mad_std(sep_mas, ignore_nan=True))
    sem_dra = mad_dra / np.sqrt(n_matches)
    sem_ddec = mad_ddec / np.sqrt(n_matches)
    sem_sep = mad_sep / np.sqrt(n_matches)
    vector_median = float(np.hypot(med_dra, med_ddec))
    vector_sem = float(np.hypot(sem_dra, sem_ddec))

    return {
        "n_matches": int(n_matches),
        "median_dra_mas": med_dra,
        "median_ddec_mas": med_ddec,
        "median_sep_mas": med_sep,
        "mad_dra_mas": mad_dra,
        "mad_ddec_mas": mad_ddec,
        "mad_sep_mas": mad_sep,
        "sem_dra_mas": sem_dra,
        "sem_ddec_mas": sem_ddec,
        "sem_sep_mas": sem_sep,
        "vector_median_offset_mas": vector_median,
        "vector_sem_mas": vector_sem,
    }


def make_match_table(
    catalog: Table,
    reference: Table,
    catalog_idx: np.ndarray,
    reference_idx: np.ndarray,
    dra_mas: np.ndarray,
    ddec_mas: np.ndarray,
    sep_mas: np.ndarray,
    ref_mag_column: str | None,
) -> Table:
    catalog_coords = get_catalog_coordinates(catalog)
    matched = Table()
    matched["catalog_index"] = np.asarray(catalog_idx, dtype=int)
    matched["reference_index"] = np.asarray(reference_idx, dtype=int)
    matched["dra_mas"] = np.asarray(dra_mas, dtype=float)
    matched["ddec_mas"] = np.asarray(ddec_mas, dtype=float)
    matched["sep_mas"] = np.asarray(sep_mas, dtype=float)
    matched["catalog_ra_deg"] = catalog_coords.ra.deg[catalog_idx]
    matched["catalog_dec_deg"] = catalog_coords.dec.deg[catalog_idx]
    matched["reference_ra_deg"] = reference["skycoord"].ra.deg[reference_idx]
    matched["reference_dec_deg"] = reference["skycoord"].dec.deg[reference_idx]
    if ref_mag_column is not None and ref_mag_column in reference.colnames:
        ref_mag = reference[ref_mag_column]
        matched["reference_mag"] = to_float_array(ref_mag[reference_idx])
    return matched


def plot_offsets(
    catalog_label: str,
    reference_label: str,
    catalog_ra_deg: np.ndarray,
    catalog_dec_deg: np.ndarray,
    dra_mas: np.ndarray,
    ddec_mas: np.ndarray,
    sep_mas: np.ndarray,
    ref_mag: np.ndarray | None,
    outdir: Path,
    summary: dict[str, float],
) -> tuple[Path, Path, Path, Path]:
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    stem = sanitize_label(f"{catalog_label}__{reference_label}")
    offset_path = plot_dir / f"{stem}_offsets.png"
    sep_path = plot_dir / f"{stem}_separation.png"
    mag_path = plot_dir / f"{stem}_sep_vs_mag.png"
    quiver_path = plot_dir / f"{stem}_quiver.png"

    if len(dra_mas) == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No mutual matches found", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.savefig(offset_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No mutual matches found", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.savefig(sep_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No mutual matches found", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.savefig(mag_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No mutual matches found", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.savefig(quiver_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return offset_path, sep_path, mag_path, quiver_path

    lim = float(np.nanpercentile(np.abs(np.concatenate([dra_mas, ddec_mas])), 99.5))
    lim = max(lim * 1.1, 5.0)

    fig, ax = plt.subplots(figsize=(7.8, 6.8))
    if len(dra_mas) > 5000:
        ax.hexbin(dra_mas, ddec_mas, gridsize=120, mincnt=1, cmap="inferno", norm=mpl.colors.LogNorm())
    else:
        ax.scatter(dra_mas, ddec_mas, s=2, alpha=0.15, color="k", rasterized=True)
    ax.axhline(0, color="0.5", lw=1)
    ax.axvline(0, color="0.5", lw=1)
    ax.arrow(
        0,
        0,
        summary["median_dra_mas"],
        summary["median_ddec_mas"],
        color="cyan",
        width=0,
        head_width=max(lim * 0.02, 1.0),
        length_includes_head=True,
        zorder=5,
    )
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("dRA cos(dec) [mas]")
    ax.set_ylabel("dDec [mas]")
    ax.set_title(
        f"{catalog_label} vs {reference_label}\n"
        f"N={summary['n_matches']}  med=({summary['median_dra_mas']:.2f}, {summary['median_ddec_mas']:.2f}) mas",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(offset_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    bins = np.linspace(0, max(float(np.nanmax(sep_mas)) * 1.05, 1.0), 80)
    ax.hist(sep_mas, bins=bins, histtype="stepfilled", color="0.2", alpha=0.75)
    ax.axvline(summary["median_sep_mas"], color="tab:cyan", lw=2, label="median")
    ax.set_xlabel("Separation [mas]")
    ax.set_ylabel("N")
    ax.set_title(
        f"{catalog_label} vs {reference_label}\n"
        f"median={summary['median_sep_mas']:.2f} mas  MAD={summary['mad_sep_mas']:.2f} mas",
        fontsize=11,
    )
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(sep_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    if ref_mag is None or len(ref_mag) != len(sep_mas):
        ax.text(0.5, 0.5, "No magnitude column available for this reference", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        good = np.isfinite(ref_mag) & np.isfinite(sep_mas)
        if np.count_nonzero(good) == 0:
            ax.text(0.5, 0.5, "No finite magnitude/separation values", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        else:
            if np.count_nonzero(good) > 5000:
                ax.hexbin(ref_mag[good], sep_mas[good], gridsize=90, mincnt=1, cmap="viridis", norm=mpl.colors.LogNorm())
            else:
                ax.scatter(ref_mag[good], sep_mas[good], s=2, alpha=0.15, color="k", rasterized=True)
            ax.set_xlabel("Reference magnitude")
            ax.set_ylabel("Separation [mas]")
            ax.set_title(f"{catalog_label} vs {reference_label}\nSeparation vs reference magnitude", fontsize=11)
    fig.tight_layout()
    fig.savefig(mag_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Quiver residual map in tangent-plane coordinates (arcsec relative to field center).
    center_ra = np.nanmedian(catalog_ra_deg)
    center_dec = np.nanmedian(catalog_dec_deg)
    x_arcsec = (catalog_ra_deg - center_ra) * np.cos(np.deg2rad(center_dec)) * 3600.0
    y_arcsec = (catalog_dec_deg - center_dec) * 3600.0
    u_arcsec = dra_mas / 1000.0
    v_arcsec = ddec_mas / 1000.0
    # Residual vectors are often a few mas and visually tiny on arcsec-scale axes.
    # Apply a display-only boost so the directional pattern is visible.
    vector_display_boost = 100.0
    u_disp = u_arcsec * vector_display_boost
    v_disp = v_arcsec * vector_display_boost

    fig, ax = plt.subplots(figsize=(8.0, 6.8))
    if len(x_arcsec) > 3000:
        stride = int(np.ceil(len(x_arcsec) / 3000.0))
        keep = np.arange(len(x_arcsec)) % stride == 0
    else:
        keep = np.ones(len(x_arcsec), dtype=bool)

    q = ax.quiver(
        x_arcsec[keep],
        y_arcsec[keep],
        u_disp[keep],
        v_disp[keep],
        sep_mas[keep],
        cmap="plasma",
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.0035,
        headwidth=5.5,
        headlength=7.5,
        headaxislength=6.5,
        alpha=0.85,
    )
    cbar = fig.colorbar(q, ax=ax, pad=0.02)
    cbar.set_label("Residual magnitude [mas]")
    ax.set_xlabel("Relative RA position [arcsec]")
    ax.set_ylabel("Relative Dec position [arcsec]")
    ax.set_title(f"{catalog_label} vs {reference_label}\nResidual vector field")
    ax.set_aspect("equal", adjustable="box")
    med_vec_true = max(summary["vector_median_offset_mas"] / 1000.0, 1e-3)
    med_vec_display = med_vec_true * vector_display_boost
    ax.quiverkey(
        q,
        X=0.88,
        Y=1.02,
        U=med_vec_display,
        label=f"{summary['vector_median_offset_mas']:.2f} mas (x{vector_display_boost:.0f} display)",
        labelpos="E",
    )
    fig.tight_layout()
    fig.savefig(quiver_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return offset_path, sep_path, mag_path, quiver_path


def analyze_catalog(
    catalog_path: Path,
    outdir: Path,
    refresh_cache: bool,
    vvv_catalog: str,
    gaia_catalog: str,
    max_sep: u.Quantity,
    vvv_photometric_nsigma: float,
) -> list[dict[str, object]]:
    catalog = Table.read(catalog_path)
    catalog_coords = get_catalog_coordinates(catalog)
    reference_label = get_reference_label(catalog)
    center, width, height = compute_query_footprint(catalog_coords)

    catalog_rows: list[dict[str, object]] = []
    catalog_label = catalog_path.stem
    catalog_cache_label = sanitize_label(catalog_label)

    for spec in REFERENCE_SPECS:
        reference, cache_path, mag_column = load_reference_catalog(
            name=spec["name"],
            catalog_path=catalog_path,
            center=center,
            width=width,
            height=height,
            outdir=outdir,
            refresh_cache=refresh_cache,
            vvv_catalog=vvv_catalog,
            gaia_catalog=gaia_catalog,
        )
        reference_coords = reference["skycoord"]
        match = match_catalog_to_reference(catalog_coords, reference_coords, max_sep=max_sep)
        vvv_filter_info = {
            "applied": False,
            "jwst_mag_column": "",
            "jwst_err_column": "",
            "n_before": int(len(match["catalog_indices"])),
            "n_after": int(len(match["catalog_indices"])),
            "median_mag_offset": np.nan,
        }
        if spec["name"] == "vvv":
            match, vvv_filter_info = apply_vvv_photometric_filter(
                catalog=catalog,
                reference=reference,
                reference_mag_column=mag_column,
                match=match,
                nsigma=vvv_photometric_nsigma,
            )
        catalog_match_coords = catalog_coords[match["catalog_indices"]]
        summary = summarize_offsets(match["dra"], match["ddec"], match["sep"])
        matched_tbl = make_match_table(
            catalog=catalog,
            reference=reference,
            catalog_idx=match["catalog_indices"],
            reference_idx=match["reference_indices"],
            dra_mas=match["dra"],
            ddec_mas=match["ddec"],
            sep_mas=match["sep"],
            ref_mag_column=mag_column,
        )

        pair_label = f"{catalog_cache_label}__{spec['name']}"
        matched_dir = outdir / "matches"
        matched_dir.mkdir(parents=True, exist_ok=True)
        matched_tbl.write(matched_dir / f"{sanitize_label(pair_label)}_matches.ecsv", overwrite=True)

        ref_mag = None
        if mag_column is not None and mag_column in reference.colnames:
            ref_mag = np.asarray(np.ma.filled(reference[mag_column][match["reference_indices"]], np.nan), dtype=float)

        offset_path, sep_path, mag_path, quiver_path = plot_offsets(
            catalog_label=catalog_label,
            reference_label=spec["label"],
            catalog_ra_deg=catalog_match_coords.ra.deg,
            catalog_dec_deg=catalog_match_coords.dec.deg,
            dra_mas=match["dra"],
            ddec_mas=match["ddec"],
            sep_mas=match["sep"],
            ref_mag=ref_mag,
            outdir=outdir,
            summary=summary,
        )

        row = {
            "catalog_file": str(catalog_path),
            "catalog_label": catalog_label,
            "catalog_rows": int(len(catalog)),
            "catalog_columns": int(len(catalog.colnames)),
            "reference_name": spec["label"],
            "reference_cache": str(cache_path),
            "reference_rows": int(len(reference)),
            "reference_mag_column": mag_column or "",
            "vvv_photometric_filter_applied": bool(vvv_filter_info["applied"]),
            "vvv_photometric_jwst_mag_column": str(vvv_filter_info["jwst_mag_column"]),
            "vvv_photometric_jwst_err_column": str(vvv_filter_info["jwst_err_column"]),
            "vvv_photometric_n_before": int(vvv_filter_info["n_before"]),
            "vvv_photometric_n_after": int(vvv_filter_info["n_after"]),
            "vvv_photometric_median_mag_offset": float(vvv_filter_info["median_mag_offset"])
            if np.isfinite(vvv_filter_info["median_mag_offset"])
            else np.nan,
            "n_matches": summary["n_matches"],
            "median_dra_mas": summary["median_dra_mas"],
            "median_ddec_mas": summary["median_ddec_mas"],
            "median_sep_mas": summary["median_sep_mas"],
            "mad_dra_mas": summary["mad_dra_mas"],
            "mad_ddec_mas": summary["mad_ddec_mas"],
            "mad_sep_mas": summary["mad_sep_mas"],
            "sem_dra_mas": summary["sem_dra_mas"],
            "sem_ddec_mas": summary["sem_ddec_mas"],
            "sem_sep_mas": summary["sem_sep_mas"],
            "vector_median_offset_mas": summary["vector_median_offset_mas"],
            "vector_sem_mas": summary["vector_sem_mas"],
            "footprint_ra_deg": float(center.ra.to_value(u.deg)),
            "footprint_dec_deg": float(center.dec.to_value(u.deg)),
            "footprint_width_arcmin": float(width.to_value(u.arcmin)),
            "footprint_height_arcmin": float(height.to_value(u.arcmin)),
            "offset_plot": str(offset_path),
            "separation_plot": str(sep_path),
            "mag_plot": str(mag_path),
            "quiver_plot": str(quiver_path),
            "observation_reference_label": reference_label,
        }
        catalog_rows.append(row)

    return catalog_rows


def write_markdown_report(summary: Table, outdir: Path) -> Path:
    report_path = outdir / "astrometry_analysis_report.md"
    lines = [
        "# Astrometry Analysis Report",
        "",
        f"Generated: {datetime.datetime.now().isoformat()}",
        "",
        "| catalog | reference | matches | med_dra_mas | med_ddec_mas | med_sep_mas | sem_dra_mas | sem_ddec_mas | sem_sep_mas |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            "| "
            f"{Path(str(row['catalog_file'])).name} | {row['reference_name']} | {int(row['n_matches'])} | "
            f"{row['median_dra_mas']:.3f} | {row['median_ddec_mas']:.3f} | {row['median_sep_mas']:.3f} | "
            f"{row['sem_dra_mas']:.3f} | {row['sem_ddec_mas']:.3f} | {row['sem_sep_mas']:.3f} |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    basepath = Path(args.basepath) if args.basepath else default_basepath(args.target)
    outdir = Path(args.outdir) if args.outdir else basepath / "astrometry_analysis"
    outdir.mkdir(parents=True, exist_ok=True)

    catalogs = normalize_catalog_inputs(basepath, args)
    if args.max_catalogs > 0:
        catalogs = catalogs[: args.max_catalogs]

    if not catalogs:
        raise FileNotFoundError(
            f"No final merged catalogs found under {basepath}. "
            "Pass --catalog or --catalog-glob to specify inputs explicitly."
        )

    summary_rows: list[dict[str, object]] = []
    for catalog_path in catalogs:
        summary_rows.extend(
            analyze_catalog(
                catalog_path=catalog_path,
                outdir=outdir,
                refresh_cache=args.refresh_cache,
                vvv_catalog=args.vvv_catalog,
                gaia_catalog=args.gaia_catalog,
                max_sep=args.max_sep_arcsec * u.arcsec,
                vvv_photometric_nsigma=args.vvv_photometric_nsigma,
            )
        )

    summary = Table(rows=[tuple(row.values()) for row in summary_rows], names=list(summary_rows[0].keys()))
    summary.write(outdir / "astrometry_analysis_summary.ecsv", overwrite=True)
    summary.write(outdir / "astrometry_analysis_summary.fits", overwrite=True)
    report_path = write_markdown_report(summary, outdir)

    print(f"Wrote summary table to {outdir / 'astrometry_analysis_summary.ecsv'}")
    print(f"Wrote summary table to {outdir / 'astrometry_analysis_summary.fits'}")
    print(f"Wrote markdown report to {report_path}")
    print(f"Wrote plots and matched tables under {outdir}")


if __name__ == "__main__":
    main()
