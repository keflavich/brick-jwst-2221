#!/usr/bin/env python
"""Build astrometric reference tables from JWST pipeline catalogs.

Default behavior is configured for Sickle (target=sickle, proposal_id=3958, field=007),
but this script can be used for other targets/proposals/fields (e.g. brick-2221,
brick-1182, sgrb2-5365) by changing CLI options.

Outputs:
- {basepath}/catalogs/pipeline_based_nircam-{filter}_reference_astrometric_catalog.ecsv
- {basepath}/catalogs/pipeline_based_nircam-{filter}_reference_astrometric_catalog.fits
"""

from __future__ import annotations

import argparse
import datetime
from glob import glob
import os
from pathlib import Path
import sys

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy import stats
from astropy.table import Table, vstack
import matplotlib.pyplot as plt
from astroquery.vizier import Vizier


try:
    from measure_offsets import measure_offsets
except ImportError:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.append(str(repo_root / "offsets"))
    from measure_offsets import measure_offsets


DEFAULT_GNS_CATALOG_PATH = Path("/orange/adamginsburg/jwst/brick/catalogs/GALACTICNUCLEUS_2021_merged.fits")
DEFAULT_GNS_CATALOG = "J/A+A/653/A133/central"
BOOTSTRAPPED_REFCAT_FILENAMES = {
    "vvv": "nircam_bootstrapped_to_vvv_refcat",
    "gns": "nircam_bootstrapped_to_gns_refcat",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build reference table from JWST pipeline source catalogs."
    )
    parser.add_argument(
        "--target",
        default="sickle",
        help="Target/region name used in /orange/adamginsburg/jwst/{target} (default: sickle).",
    )
    parser.add_argument(
        "--proposal-id",
        default="3958",
        help="Proposal ID label to store in metadata (default: 3958).",
    )
    parser.add_argument(
        "--field",
        default="007",
        help="Field ID label to store in metadata (default: 007).",
    )
    parser.add_argument(
        "--basepath",
        default=None,
        help="Root data directory. Defaults to /orange/adamginsburg/jwst/{target}.",
    )
    parser.add_argument(
        "--filter",
        default="F210M",
        help="Filter to use as the source for reference catalogs (default: F210M).",
    )
    parser.add_argument(
        "--min-flux",
        type=float,
        default=0.0,
        help="Minimum flux threshold applied after catalog merge.",
    )
    parser.add_argument(
        "--max-sources",
        type=int,
        default=50000,
        help="Maximum number of sources to keep (brightest by flux).",
    )
    parser.add_argument(
        "--generate-catalogs",
        action="store_true",
        help="Generate *_cat.ecsv catalogs from *_i2d.fits with SourceCatalogStep if missing.",
    )
    parser.add_argument(
        "--crds-server-url",
        default="https://jwst-crds.stsci.edu",
        help="CRDS server URL used when generating catalogs.",
    )
    parser.add_argument(
        "--crds-path",
        default=None,
        help="CRDS cache path to use when generating catalogs. Defaults to {basepath}/crds.",
    )
    parser.add_argument(
        "--vvv-catalog",
        default="II/376/vvv4",
        help="Vizier catalog identifier for VVV data.",
    )
    parser.add_argument(
        "--gaia-catalog",
        default="I/355/gaiadr3",
        help="Vizier catalog identifier for Gaia data (default: Gaia DR3).",
    )
    parser.add_argument(
        "--gns-catalog",
        default=DEFAULT_GNS_CATALOG,
        help=(
            "GALACTICNUCLEUS catalog source. Use a local FITS/ECSV path if available, "
            "or a Vizier catalog identifier if you need to download it."
        ),
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Force refreshing cached VVV/Gaia catalogs instead of reusing saved files.",
    )
    parser.add_argument(
        "--vvv-max-sep-arcsec",
        type=float,
        default=0.4,
        help="Maximum separation for JWST/VVV matching in arcsec.",
    )
    parser.add_argument(
        "--photometric-nsigma",
        type=float,
        default=3.0,
        help="Sigma threshold for flux-vs-Ks agreement filtering.",
    )
    parser.add_argument(
        "--offset-threshold-arcsec",
        type=float,
        default=0.01,
        help="Convergence threshold for iterative astrometric correction.",
    )
    return parser.parse_args()


def resolve_default_basepath(target: str) -> Path:
    target_root = Path(f"/orange/adamginsburg/jwst/{target}")
    if target.lower() == "sgrb2":
        nb_root = target_root / "NB"
        if nb_root.exists():
            return nb_root
    return target_root


def _catalog_patterns(pipeline_dir: Path) -> list[str]:
    return [
        str(pipeline_dir / "*_cat.ecsv"),
        str(pipeline_dir / "*_cat.fits"),
    ]


def find_catalogs(pipeline_dir: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in _catalog_patterns(pipeline_dir):
        files.extend(Path(x) for x in glob(pattern))
    unique = sorted(set(files))
    return unique


def generate_catalogs_from_i2d(pipeline_dir: Path) -> list[Path]:
    from jwst.source_catalog import SourceCatalogStep

    i2d_files = sorted(Path(x) for x in glob(str(pipeline_dir / "*_i2d.fits")))
    generated: list[Path] = []

    for i2d_file in i2d_files:
        stem = i2d_file.name.replace(".fits", "")
        expected = pipeline_dir / f"{stem}_cat.ecsv"
        if expected.exists():
            generated.append(expected)
            continue

        SourceCatalogStep.call(
            str(i2d_file),
            output_dir=str(pipeline_dir),
            save_results=True,
        )

        if expected.exists():
            generated.append(expected)

    return generated


def _extract_skycoord(tbl: Table) -> SkyCoord:
    if "sky_centroid" in tbl.colnames:
        return tbl["sky_centroid"]
    if "skycoord" in tbl.colnames:
        return tbl["skycoord"]
    if "RA" in tbl.colnames and "DEC" in tbl.colnames:
        return SkyCoord(tbl["RA"], tbl["DEC"], frame="fk5")
    if "ra" in tbl.colnames and "dec" in tbl.colnames:
        return SkyCoord(tbl["ra"], tbl["dec"], frame="fk5")
    raise ValueError("Could not find sky coordinates in catalog table")


def _extract_flux(tbl: Table) -> np.ndarray:
    preferred = (
        "aper_total_flux",
        "isophotal_flux",
        "segment_flux",
        "flux",
        "source_sum",
    )
    for name in preferred:
        if name in tbl.colnames:
            return np.asarray(tbl[name])
    raise ValueError("Could not find a supported flux column in catalog table")


def has_supported_sky_columns(tbl: Table) -> bool:
    if "sky_centroid" in tbl.colnames:
        return True
    if "skycoord" in tbl.colnames:
        return True
    if "RA" in tbl.colnames and "DEC" in tbl.colnames:
        return True
    if "ra" in tbl.colnames and "dec" in tbl.colnames:
        return True
    return False


def has_supported_flux_column(tbl: Table) -> bool:
    preferred = (
        "aper_total_flux",
        "isophotal_flux",
        "segment_flux",
        "flux",
        "source_sum",
    )
    return any(name in tbl.colnames for name in preferred)


def has_supported_catalog_schema(tbl: Table) -> bool:
    return has_supported_sky_columns(tbl) and has_supported_flux_column(tbl)


def load_supported_catalogs(catalog_files: list[Path]) -> list[Table]:
    tables: list[Table] = []
    skipped: list[Path] = []

    for path in catalog_files:
        tbl = Table.read(path)
        if has_supported_catalog_schema(tbl):
            tables.append(read_and_normalize(path))
        else:
            skipped.append(path)

    if not tables:
        raise ValueError(
            "No usable source catalogs were found. "
            "Catalogs must include sky coordinates (sky_centroid/skycoord/RA,DEC) and flux columns."
        )

    if skipped:
        print(f"Skipped {len(skipped)} catalogs without supported schema")

    return tables


def read_and_normalize(catalog_file: Path) -> Table:
    tbl = Table.read(catalog_file)
    sky = _extract_skycoord(tbl)
    flux = _extract_flux(tbl)

    good = np.isfinite(sky.ra) & np.isfinite(sky.dec) & np.isfinite(flux) & (flux > 0)

    out = Table()
    out["skycoord"] = sky[good]
    out["flux"] = flux[good]
    out["RA"] = out["skycoord"].ra
    out["DEC"] = out["skycoord"].dec
    out.meta["INPUT_FILE"] = str(catalog_file)
    return out


def compute_query_footprint(
    skycoord: SkyCoord,
    min_width: u.Quantity = 2 * u.arcmin,
    max_width: u.Quantity = 1.5 * u.deg,
) -> tuple[SkyCoord, u.Quantity, u.Quantity]:
    padding = 1 * u.arcmin
    ra = skycoord.ra.to(u.deg)
    dec = skycoord.dec.to(u.deg)
    center = SkyCoord(np.nanmedian(ra), np.nanmedian(dec), frame="fk5")

    width = (np.nanmax(ra) - np.nanmin(ra)).to(u.deg)
    height = (np.nanmax(dec) - np.nanmin(dec)).to(u.deg)
    width = np.clip(
        (width + 2 * padding).to_value(u.deg),
        min_width.to_value(u.deg),
        max_width.to_value(u.deg),
    ) * u.deg
    height = np.clip(
        (height + 2 * padding).to_value(u.deg),
        min_width.to_value(u.deg),
        max_width.to_value(u.deg),
    ) * u.deg
    return center, width, height


def query_vizier_with_cache(
    center: SkyCoord,
    width: u.Quantity,
    height: u.Quantity,
    catalog: str,
    cache_path: Path,
    refresh_cache: bool,
) -> Table:
    legacy_cache_path = cache_path.with_suffix(".ecsv")
    if cache_path.exists() and not refresh_cache:
        return Table.read(cache_path)
    if legacy_cache_path.exists() and not refresh_cache:
        table = Table.read(legacy_cache_path)
        # Migrate legacy ECSV cache to the requested cache format/path.
        table.write(cache_path, overwrite=True)
        return table

    Vizier.ROW_LIMIT = -1
    result = Vizier.query_region(center, width=width, height=height, catalog=[catalog])
    if len(result) == 0:
        raise ValueError(f"No sources found in {catalog} for center={center}.")

    table = result[0]
    table.meta["QUERY_CATALOG"] = catalog
    table.meta["QUERY_RA_DEG"] = float(center.ra.to_value(u.deg))
    table.meta["QUERY_DEC_DEG"] = float(center.dec.to_value(u.deg))
    table.meta["QUERY_WIDTH_DEG"] = float(width.to_value(u.deg))
    table.meta["QUERY_HEIGHT_DEG"] = float(height.to_value(u.deg))
    table.write(cache_path, overwrite=True)
    return table


def fetch_vvv_catalog(
    center: SkyCoord,
    width: u.Quantity,
    height: u.Quantity,
    vvv_catalog: str,
    cache_path: Path,
    refresh_cache: bool,
) -> Table:
    vvv = query_vizier_with_cache(
        center=center,
        width=width,
        height=height,
        catalog=vvv_catalog,
        cache_path=cache_path,
        refresh_cache=refresh_cache,
    )
    required = ("RAJ2000", "DEJ2000")
    for col in required:
        if col not in vvv.colnames:
            raise ValueError(f"VVV catalog {vvv_catalog} is missing required column {col}.")

    ks_candidates = [
        "Ksmag3",
        "Ksmag",
        "Ks3mag",
        "Ks2ap3",
        "Ks1ap3",
        "Ks2ap1",
        "Ks1ap1",
    ]
    available = [name for name in ks_candidates if name in vvv.colnames]
    if not available:
        raise ValueError(
            f"VVV catalog {vvv_catalog} is missing all supported Ks columns. "
            f"Tried {ks_candidates}; available columns include {vvv.colnames}."
        )

    ks_stack = np.vstack([np.asarray(vvv[name], dtype=float) for name in available])
    has_any_ks = np.isfinite(ks_stack).any(axis=0)
    ks_refmag = np.full(ks_stack.shape[1], np.nan, dtype=float)
    if np.any(has_any_ks):
        ks_refmag[has_any_ks] = np.nanmedian(ks_stack[:, has_any_ks], axis=0)
    vvv["Ks_refmag"] = ks_refmag

    finite = np.isfinite(vvv["RAJ2000"]) & np.isfinite(vvv["DEJ2000"]) & np.isfinite(vvv["Ks_refmag"])
    vvv = vvv[finite]
    vvv["skycoord"] = SkyCoord(vvv["RAJ2000"], vvv["DEJ2000"], frame="fk5")
    vvv.write(cache_path, overwrite=True)
    return vvv


def fetch_gaia_catalog(
    center: SkyCoord,
    width: u.Quantity,
    height: u.Quantity,
    gaia_catalog: str,
    cache_path: Path,
    refresh_cache: bool,
) -> Table:
    gaia = query_vizier_with_cache(
        center=center,
        width=width,
        height=height,
        catalog=gaia_catalog,
        cache_path=cache_path,
        refresh_cache=refresh_cache,
    )

    if "RA_ICRS" in gaia.colnames and "DE_ICRS" in gaia.colnames:
        finite = np.isfinite(gaia["RA_ICRS"]) & np.isfinite(gaia["DE_ICRS"])
        gaia = gaia[finite]
        gaia["skycoord"] = SkyCoord(gaia["RA_ICRS"], gaia["DE_ICRS"], frame="icrs")
    elif "RAJ2000" in gaia.colnames and "DEJ2000" in gaia.colnames:
        finite = np.isfinite(gaia["RAJ2000"]) & np.isfinite(gaia["DEJ2000"])
        gaia = gaia[finite]
        gaia["skycoord"] = SkyCoord(gaia["RAJ2000"], gaia["DEJ2000"], frame="icrs")
    else:
        raise ValueError(
            f"Gaia catalog {gaia_catalog} is missing RA/Dec columns. "
            "Expected RA_ICRS/DE_ICRS or RAJ2000/DEJ2000."
        )

    return gaia


def _normalize_reference_ks_magnitude(reference: Table, candidates: tuple[str, ...]) -> tuple[Table, str]:
    available = [name for name in candidates if name in reference.colnames]
    if not available:
        raise ValueError(
            f"Reference catalog is missing all supported Ks columns. Tried {candidates}; "
            f"available columns include {reference.colnames}."
        )

    ks_stack = np.vstack([np.asarray(reference[name], dtype=float) for name in available])
    has_any_ks = np.isfinite(ks_stack).any(axis=0)
    ks_refmag = np.full(ks_stack.shape[1], np.nan, dtype=float)
    if np.any(has_any_ks):
        ks_refmag[has_any_ks] = np.nanmedian(ks_stack[:, has_any_ks], axis=0)

    normalized = reference.copy()
    normalized["Ks_refmag"] = ks_refmag
    finite = np.isfinite(normalized["RAJ2000"]) & np.isfinite(normalized["DEJ2000"]) & np.isfinite(normalized["Ks_refmag"])
    normalized = normalized[finite]
    normalized["skycoord"] = SkyCoord(normalized["RAJ2000"], normalized["DEJ2000"], frame="fk5")
    return normalized, "Ks_refmag"


def fetch_gns_catalog(
    center: SkyCoord,
    width: u.Quantity,
    height: u.Quantity,
    gns_catalog: str,
    cache_path: Path,
    refresh_cache: bool,
) -> Table:
    source_path = Path(gns_catalog).expanduser()
    source_label = str(source_path if source_path.exists() else gns_catalog)
    if cache_path.exists() and not refresh_cache:
        cached = Table.read(cache_path)
        cached_source = str(cached.meta.get("SOURCE_CATALOG", ""))
        if cached_source == source_label and "skycoord" in cached.colnames and "Ks_refmag" in cached.colnames:
            return cached
        if cached_source == source_label:
            cached = _normalize_reference_ks_magnitude(
                cached,
                candidates=("Ksmag", "Ks", "Ks_mag", "Ksmag3", "Ks3mag", "Ks2ap3", "Ks1ap3", "Ks2ap1", "Ks1ap1"),
            )[0]
            cached.meta["SOURCE_CATALOG"] = source_label
            cached.write(cache_path, overwrite=True)
            return cached

    if source_path.exists():
        gns = Table.read(source_path)
    else:
        gns = query_vizier_with_cache(
            center=center,
            width=width,
            height=height,
            catalog=gns_catalog,
            cache_path=cache_path,
            refresh_cache=refresh_cache,
        )

    if "RAJ2000" not in gns.colnames or "DEJ2000" not in gns.colnames:
        raise ValueError(
            f"GALACTICNUCLEUS catalog {source_label} is missing RAJ2000/DEJ2000 coordinates."
        )

    if source_path.exists() and not cache_path.exists():
        pass

    normalized, _ = _normalize_reference_ks_magnitude(
        gns,
        candidates=("Ksmag", "Ks", "Ks_mag", "Ksmag3", "Ks3mag", "Ks2ap3", "Ks1ap3", "Ks2ap1", "Ks1ap1"),
    )
    normalized.meta["SOURCE_CATALOG"] = source_label
    normalized.write(cache_path, overwrite=True)
    return normalized


def initial_spatial_photometric_match(
    catalog_table: Table,
    reference_table: Table,
    reference_mag_column: str,
    reference_label: str,
    max_sep: u.Quantity,
    nsigma: float,
) -> tuple[np.ndarray, dict[str, float]]:
    jwst_coord = catalog_table["skycoord"]
    reference_coord = reference_table["skycoord"]

    idx, sep, _ = jwst_coord.match_to_catalog_sky(reference_coord, nthneighbor=1)
    reverse_idx, _, _ = reference_coord.match_to_catalog_sky(jwst_coord, nthneighbor=1)
    mutual = reverse_idx[idx] == np.arange(len(idx))
    spatial = sep < max_sep
    keep = mutual & spatial

    if keep.sum() < 10:
        raise ValueError(f"Only {keep.sum()} JWST/{reference_label} spatial matches were found; need >=10.")

    jwst_flux = np.asarray(catalog_table["flux"][keep])
    reference_mag = np.asarray(reference_table[reference_mag_column][idx[keep]])
    x = np.log10(jwst_flux)

    design = np.vstack([np.ones_like(x), x]).T
    coeff, _, _, _ = np.linalg.lstsq(design, reference_mag, rcond=None)
    model = design @ coeff
    resid = reference_mag - model

    med = np.nanmedian(resid)
    mad = stats.mad_std(resid, ignore_nan=True)
    if not np.isfinite(mad) or mad == 0:
        good = np.isfinite(resid)
    else:
        good = np.abs(resid - med) < (nsigma * mad)

    final_keep = np.zeros(len(catalog_table), dtype=bool)
    keep_indices = np.where(keep)[0]
    final_keep[keep_indices[good]] = True

    return final_keep, {
        "initial_matches": float(np.count_nonzero(keep)),
        "selected_matches": float(np.count_nonzero(final_keep)),
        "photometric_intercept": float(coeff[0]),
        "photometric_slope": float(coeff[1]),
        "photometric_residual_median": float(med),
        "photometric_residual_mad": float(mad),
    }


def match_catalog_to_reference(
    catalog_coords: SkyCoord,
    reference_coords: SkyCoord,
    max_sep: u.Quantity,
    mutual: bool = True,
) -> dict[str, np.ndarray]:
    catalog_idx, sep, _ = catalog_coords.match_to_catalog_sky(reference_coords, nthneighbor=1)
    keep = np.isfinite(sep) & (sep <= max_sep)
    if mutual:
        reverse_idx, _, _ = reference_coords.match_to_catalog_sky(catalog_coords, nthneighbor=1)
        keep &= reverse_idx[catalog_idx] == np.arange(len(catalog_idx))

    match_indices = np.where(keep)[0]
    reference_indices = catalog_idx[keep]
    matched_catalog = catalog_coords[keep]
    matched_reference = reference_coords[reference_indices].transform_to(matched_catalog.frame)
    dra = (matched_catalog.ra - matched_reference.ra).to(u.mas)
    ddec = (matched_catalog.dec - matched_reference.dec).to(u.mas)
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
            "mean_dra_arcsec": np.nan,
            "mean_ddec_arcsec": np.nan,
            "median_dra_arcsec": np.nan,
            "median_ddec_arcsec": np.nan,
            "median_sep_mas": np.nan,
            "mad_dra_arcsec": np.nan,
            "mad_ddec_arcsec": np.nan,
            "mad_sep_mas": np.nan,
            "vector_mean_offset_arcsec": np.nan,
            "vector_median_offset_arcsec": np.nan,
        }

    dra_arcsec = np.asarray(dra_mas, dtype=float) / 1000.0
    ddec_arcsec = np.asarray(ddec_mas, dtype=float) / 1000.0
    sep_mas = np.asarray(sep_mas, dtype=float)

    mean_dra = float(np.nanmean(dra_arcsec))
    mean_ddec = float(np.nanmean(ddec_arcsec))
    median_dra = float(np.nanmedian(dra_arcsec))
    median_ddec = float(np.nanmedian(ddec_arcsec))
    median_sep = float(np.nanmedian(sep_mas))
    mad_dra = float(stats.mad_std(dra_arcsec, ignore_nan=True))
    mad_ddec = float(stats.mad_std(ddec_arcsec, ignore_nan=True))
    mad_sep = float(stats.mad_std(sep_mas, ignore_nan=True))

    return {
        "n_matches": int(len(dra_arcsec)),
        "mean_dra_arcsec": mean_dra,
        "mean_ddec_arcsec": mean_ddec,
        "median_dra_arcsec": median_dra,
        "median_ddec_arcsec": median_ddec,
        "median_sep_mas": median_sep,
        "mad_dra_arcsec": mad_dra,
        "mad_ddec_arcsec": mad_ddec,
        "mad_sep_mas": mad_sep,
        "vector_mean_offset_arcsec": float(np.hypot(mean_dra, mean_ddec)),
        "vector_median_offset_arcsec": float(np.hypot(median_dra, median_ddec)),
    }


def plot_offset_distribution(
    dra_mas: np.ndarray,
    ddec_mas: np.ndarray,
    outpath: Path,
    title: str,
    summary: dict[str, float],
) -> Path:
    outpath.parent.mkdir(parents=True, exist_ok=True)

    dra_arcsec = np.asarray(dra_mas, dtype=float) / 1000.0
    ddec_arcsec = np.asarray(ddec_mas, dtype=float) / 1000.0
    radial_arcsec = np.sqrt(dra_arcsec**2 + ddec_arcsec**2)

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    ax = axes[0, 0]
    ax.scatter(dra_arcsec, ddec_arcsec, s=4, alpha=0.25, color="0.2")
    ax.axhline(0, color="tab:red", lw=1)
    ax.axvline(0, color="tab:red", lw=1)
    ax.axvline(summary["mean_dra_arcsec"], color="tab:blue", lw=1, ls="--")
    ax.axhline(summary["mean_ddec_arcsec"], color="tab:blue", lw=1, ls="--")
    ax.set_xlabel("dRA [arcsec]")
    ax.set_ylabel("dDec [arcsec]")
    ax.set_title("Offset cloud")

    ax = axes[0, 1]
    ax.hist(dra_arcsec, bins=80, histtype="stepfilled", color="tab:blue", alpha=0.7)
    ax.axvline(summary["mean_dra_arcsec"], color="k", lw=2, label="mean")
    ax.axvline(summary["median_dra_arcsec"], color="tab:red", lw=2, ls="--", label="median")
    ax.set_xlabel("dRA [arcsec]")
    ax.set_ylabel("N")
    ax.set_title("dRA distribution")
    ax.legend(loc="best")

    ax = axes[1, 0]
    ax.hist(ddec_arcsec, bins=80, histtype="stepfilled", color="tab:green", alpha=0.7)
    ax.axvline(summary["mean_ddec_arcsec"], color="k", lw=2, label="mean")
    ax.axvline(summary["median_ddec_arcsec"], color="tab:red", lw=2, ls="--", label="median")
    ax.set_xlabel("dDec [arcsec]")
    ax.set_ylabel("N")
    ax.set_title("dDec distribution")
    ax.legend(loc="best")

    ax = axes[1, 1]
    ax.hist(radial_arcsec, bins=80, histtype="stepfilled", color="0.3", alpha=0.7)
    ax.set_xlabel("Radial offset [arcsec]")
    ax.set_ylabel("N")
    ax.set_title("Radial offset distribution")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return outpath


def bootstrap_reference_catalog(
    merged: Table,
    reference: Table,
    reference_name: str,
    reference_mag_column: str,
    max_sep: u.Quantity,
    photometric_nsigma: float,
    threshold: u.Quantity,
    output_dir: Path,
) -> tuple[Table, dict[str, object]]:
    selection, phot_info = initial_spatial_photometric_match(
        merged,
        reference,
        reference_mag_column=reference_mag_column,
        reference_label=reference_name,
        max_sep=max_sep,
        nsigma=photometric_nsigma,
    )

    corrected = merged.copy()
    total_dra = 0.0 * u.arcsec
    total_ddec = 0.0 * u.arcsec
    iteration = 0
    max_iterations = 6

    while True:
        matched = match_catalog_to_reference(
            corrected["skycoord"][selection],
            reference["skycoord"],
            max_sep=max_sep,
            mutual=False,
        )
        if len(matched["dra"]) < 10:
            raise ValueError(
                f"Need at least 10 matches to bootstrap {reference_name}, got {len(matched['dra'])}."
            )

        dra_arcsec = np.asarray(matched["dra"], dtype=float) / 1000.0
        ddec_arcsec = np.asarray(matched["ddec"], dtype=float) / 1000.0
        finite = np.isfinite(dra_arcsec) & np.isfinite(ddec_arcsec)
        if np.count_nonzero(finite) < 10:
            raise ValueError(f"Too few finite offsets to bootstrap {reference_name}.")

        dra_clip = stats.sigma_clip(dra_arcsec[finite], sigma=4.0, maxiters=5)
        ddec_clip = stats.sigma_clip(ddec_arcsec[finite], sigma=4.0, maxiters=5)
        good = ~(dra_clip.mask | ddec_clip.mask)
        if np.count_nonzero(good) < 10:
            raise ValueError(f"Too few clipped offsets to bootstrap {reference_name}.")

        mean_dra = np.nanmean(dra_arcsec[finite][good]) * u.arcsec
        mean_ddec = np.nanmean(ddec_arcsec[finite][good]) * u.arcsec
        total_dra -= mean_dra
        total_ddec -= mean_ddec

        corrected["skycoord"] = SkyCoord(
            corrected["skycoord"].ra - mean_dra,
            corrected["skycoord"].dec - mean_ddec,
            frame=corrected["skycoord"].frame,
        )
        corrected["RA"] = corrected["skycoord"].ra
        corrected["DEC"] = corrected["skycoord"].dec

        iteration += 1
        if (np.abs(mean_dra) <= threshold and np.abs(mean_ddec) <= threshold) or iteration >= max_iterations:
            break

    final_match = match_catalog_to_reference(
        corrected["skycoord"][selection],
        reference["skycoord"],
        max_sep=max_sep,
        mutual=True,
    )
    summary = summarize_offsets(final_match["dra"], final_match["ddec"], final_match["sep"])
    plot_path = plot_offset_distribution(
        final_match["dra"],
        final_match["ddec"],
        output_dir / f"{reference_name}_offset_distribution.png",
        title=f"{reference_name.upper()} bootstrap offsets",
        summary=summary,
    )

    corrected.meta["VERSION"] = datetime.datetime.now().isoformat()
    corrected.meta["REFERENCE_NAME"] = reference_name.upper()
    corrected.meta["REFERENCE_MAG_COLUMN"] = reference_mag_column
    corrected.meta["N_REFERENCE_MATCHES"] = int(summary["n_matches"])
    corrected.meta["PHOTOMETRIC_MATCHES"] = int(phot_info["selected_matches"])
    corrected.meta["PHOTOMETRIC_INTERCEPT"] = float(phot_info["photometric_intercept"])
    corrected.meta["PHOTOMETRIC_SLOPE"] = float(phot_info["photometric_slope"])
    corrected.meta["PHOTOMETRIC_RESIDUAL_MEDIAN"] = float(phot_info["photometric_residual_median"])
    corrected.meta["PHOTOMETRIC_RESIDUAL_MAD"] = float(phot_info["photometric_residual_mad"])
    corrected.meta["MEAN_DRA_AS"] = float(summary["mean_dra_arcsec"])
    corrected.meta["MEAN_DDEC_AS"] = float(summary["mean_ddec_arcsec"])
    corrected.meta["MEDIAN_DRA_AS"] = float(summary["median_dra_arcsec"])
    corrected.meta["MEDIAN_DDEC_AS"] = float(summary["median_ddec_arcsec"])
    corrected.meta["TOTAL_DRA_AS"] = float(total_dra.to_value(u.arcsec))
    corrected.meta["TOTAL_DDEC_AS"] = float(total_ddec.to_value(u.arcsec))
    corrected.meta["BOOTSTRAP_ITER"] = int(iteration)
    corrected.meta["BOOTSTRAP_PLOT"] = str(plot_path)

    bootstrap_summary = {
        "reference_name": reference_name,
        "plot_path": str(plot_path),
        "mean_dra_arcsec": float(summary["mean_dra_arcsec"]),
        "mean_ddec_arcsec": float(summary["mean_ddec_arcsec"]),
        "median_dra_arcsec": float(summary["median_dra_arcsec"]),
        "median_ddec_arcsec": float(summary["median_ddec_arcsec"]),
        "median_sep_mas": float(summary["median_sep_mas"]),
        "n_matches": int(summary["n_matches"]),
        "photometric_matches": int(phot_info["selected_matches"]),
        "photometric_intercept": float(phot_info["photometric_intercept"]),
        "photometric_slope": float(phot_info["photometric_slope"]),
        "photometric_residual_median": float(phot_info["photometric_residual_median"]),
        "photometric_residual_mad": float(phot_info["photometric_residual_mad"]),
        "total_dra_arcsec": float(total_dra.to_value(u.arcsec)),
        "total_ddec_arcsec": float(total_ddec.to_value(u.arcsec)),
        "iterations": int(iteration),
    }
    return corrected, bootstrap_summary


def refine_with_vvv(
    ref_table: Table,
    vvv_table: Table,
    sel: np.ndarray,
    max_sep: u.Quantity,
    threshold: u.Quantity,
) -> tuple[Table, dict[str, object]]:
    if sel.sum() < 10:
        raise ValueError(f"Need >=10 VVV-vetted matches for iterative refinement, got {sel.sum()}.")

    refflux = 10 ** (-0.4 * np.asarray(vvv_table["Ks_refmag"]))
    skyflux = np.asarray(ref_table["flux"])

    (
        total_dra,
        total_ddec,
        med_dra,
        med_ddec,
        std_dra,
        std_ddec,
        keep,
        skykeep,
        reject,
        iteration,
    ) = measure_offsets(
        reference_coordinates=vvv_table["skycoord"],
        skycrds_cat=ref_table["skycoord"],
        refflux=refflux,
        skyflux=skyflux,
        max_offset=max_sep,
        threshold=threshold,
        sel=sel,
        verbose=True,
        ratio_match=True,
    )

    if not isinstance(total_dra, u.Quantity) or not isinstance(total_ddec, u.Quantity):
        raise ValueError("Iterative VVV refinement failed to converge to a valid astrometric shift.")

    updated = ref_table.copy()
    updated["skycoord"] = SkyCoord(
        updated["skycoord"].ra + total_dra,
        updated["skycoord"].dec + total_ddec,
        frame=updated["skycoord"].frame,
    )
    updated["RA"] = updated["skycoord"].ra
    updated["DEC"] = updated["skycoord"].dec

    meta = {
        "dra_arcsec": total_dra.to_value(u.arcsec),
        "ddec_arcsec": total_ddec.to_value(u.arcsec),
        "med_dra_arcsec": med_dra.to_value(u.arcsec),
        "med_ddec_arcsec": med_ddec.to_value(u.arcsec),
        "std_dra_arcsec": std_dra.to_value(u.arcsec),
        "std_ddec_arcsec": std_ddec.to_value(u.arcsec),
        "iteration": int(iteration),
        "n_keep": int(np.count_nonzero(keep)),
        "n_skykeep": int(np.count_nonzero(skykeep)),
        "n_reject": int(np.count_nonzero(reject)),
    }
    return updated, meta


def main() -> None:
    args = parse_args()

    basepath = Path(args.basepath) if args.basepath else resolve_default_basepath(args.target)
    pipeline_dir = basepath / args.filter.upper() / "pipeline"
    output_dir = basepath / "catalogs"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    crds_path = Path(args.crds_path) if args.crds_path else (basepath / "crds")
    os.environ["CRDS_SERVER_URL"] = args.crds_server_url
    os.environ["CRDS_PATH"] = str(crds_path)
    os.environ.pop("CRDS_CONFIG_URI", None)

    catalog_files = find_catalogs(pipeline_dir)

    if not catalog_files and args.generate_catalogs:
        generate_catalogs_from_i2d(pipeline_dir)
        catalog_files = find_catalogs(pipeline_dir)

    if not catalog_files:
        raise FileNotFoundError(
            f"No pipeline source catalogs found in {pipeline_dir}. "
            "Run with --generate-catalogs or run source_catalog in pipeline first."
        )

    tables = load_supported_catalogs(catalog_files)
    merged = vstack(tables, metadata_conflicts="silent")

    min_flux = float(args.min_flux)
    keep = merged["flux"] > min_flux
    merged = merged[keep]

    merged.sort("flux", reverse=True)
    max_sources = int(args.max_sources)
    if max_sources > 0 and len(merged) > max_sources:
        merged = merged[:max_sources]

    merged.meta["VERSION"] = datetime.datetime.now().isoformat()
    merged.meta["PARENT_FILTER"] = args.filter.upper()
    merged.meta["N_INPUT_CATALOGS"] = len(catalog_files)
    merged.meta["SCRIPT"] = Path(__file__).name
    merged.meta["TARGET"] = args.target
    merged.meta["PROPOSAL_ID"] = str(args.proposal_id)
    merged.meta["FIELD"] = str(args.field)

    center, width, height = compute_query_footprint(merged["skycoord"])
    cache_stem = f"{args.target}_p{args.proposal_id}_f{args.field}_{args.filter.lower()}"
    vvv_cache_path = cache_dir / f"vvv_{cache_stem}.fits"
    gaia_cache_path = cache_dir / f"gaia_{cache_stem}.fits"

    vvv = fetch_vvv_catalog(
        center=center,
        width=width,
        height=height,
        vvv_catalog=args.vvv_catalog,
        cache_path=vvv_cache_path,
        refresh_cache=args.refresh_cache,
    )
    gaia = fetch_gaia_catalog(
        center=center,
        width=width,
        height=height,
        gaia_catalog=args.gaia_catalog,
        cache_path=gaia_cache_path,
        refresh_cache=args.refresh_cache,
    )
    gns_cache_path = cache_dir / f"gns_{cache_stem}.fits"
    gns = fetch_gns_catalog(
        center=center,
        width=width,
        height=height,
        gns_catalog=args.gns_catalog,
        cache_path=gns_cache_path,
        refresh_cache=args.refresh_cache,
    )

    bootstrap_specs = [
        {
            "name": "vvv",
            "reference": vvv,
            "reference_mag_column": "Ks_refmag",
            "cache_path": vvv_cache_path,
            "input_label": args.vvv_catalog,
        },
        {
            "name": "gns",
            "reference": gns,
            "reference_mag_column": "Ks_refmag",
            "cache_path": gns_cache_path,
            "input_label": args.gns_catalog,
        },
    ]

    bootstrap_rows: list[dict[str, object]] = []
    for spec in bootstrap_specs:
        corrected, summary = bootstrap_reference_catalog(
            merged=merged,
            reference=spec["reference"],
            reference_name=spec["name"],
            reference_mag_column=spec["reference_mag_column"],
            max_sep=args.vvv_max_sep_arcsec * u.arcsec,
            photometric_nsigma=args.photometric_nsigma,
            threshold=args.offset_threshold_arcsec * u.arcsec,
            output_dir=output_dir,
        )

        outfile_stem = BOOTSTRAPPED_REFCAT_FILENAMES[spec["name"]]
        out_ecsv = output_dir / f"{outfile_stem}.ecsv"
        out_fits = output_dir / f"{outfile_stem}.fits"
        corrected.meta["SOURCE_F210M_CATALOG"] = str(catalog_files[0])
        corrected.meta["VVV_CATALOG"] = args.vvv_catalog
        corrected.meta["GNS_CATALOG"] = args.gns_catalog
        corrected.meta["GAIA_CATALOG"] = args.gaia_catalog
        corrected.meta["VVV_CACHE"] = str(vvv_cache_path)
        corrected.meta["GNS_CACHE"] = str(gns_cache_path)
        corrected.meta["GAIA_CACHE"] = str(gaia_cache_path)
        corrected.meta["QUERY_RA_DEG"] = float(center.ra.to_value(u.deg))
        corrected.meta["QUERY_DEC_DEG"] = float(center.dec.to_value(u.deg))
        corrected.meta["QUERY_W_DEG"] = float(width.to_value(u.deg))
        corrected.meta["QUERY_H_DEG"] = float(height.to_value(u.deg))
        corrected.meta["REFERENCE_NAME"] = f"{args.filter.upper()} Reference Astrometric Catalog ({spec['name'].upper()})"
        corrected.write(out_ecsv, overwrite=True)
        corrected.write(out_fits, overwrite=True)

        bootstrap_rows.append(
            {
                "reference": spec["name"],
                "input_catalog": spec["input_label"],
                "output_ecsv": str(out_ecsv),
                "output_fits": str(out_fits),
                "n_matches": int(summary["n_matches"]),
                "mean_dra_arcsec": float(summary["mean_dra_arcsec"]),
                "mean_ddec_arcsec": float(summary["mean_ddec_arcsec"]),
                "median_dra_arcsec": float(summary["median_dra_arcsec"]),
                "median_ddec_arcsec": float(summary["median_ddec_arcsec"]),
                "median_sep_mas": float(summary["median_sep_mas"]),
                "plot_path": summary["plot_path"],
                "photometric_matches": int(summary["photometric_matches"]),
                "photometric_intercept": float(summary["photometric_intercept"]),
                "photometric_slope": float(summary["photometric_slope"]),
                "total_dra_arcsec": float(summary["total_dra_arcsec"]),
                "total_ddec_arcsec": float(summary["total_ddec_arcsec"]),
                "iterations": int(summary["iterations"]),
            }
        )

        print(
            f"Bootstrapped {len(corrected)} rows to {out_fits} using {spec['name'].upper()}: "
            f"mean offset=({summary['mean_dra_arcsec']:.6f}, {summary['mean_ddec_arcsec']:.6f}) arcsec, "
            f"N={summary['n_matches']}, photometric matches={summary['photometric_matches']}"
        )

    summary_table = Table(rows=bootstrap_rows)
    summary_ecsv = output_dir / "nircam_bootstrap_offset_summary.ecsv"
    summary_fits = output_dir / "nircam_bootstrap_offset_summary.fits"
    summary_table.write(summary_ecsv, overwrite=True)
    summary_table.write(summary_fits, overwrite=True)

    report_path = output_dir / "nircam_bootstrap_offset_report.md"
    report_lines = [
        "# NIRCam bootstrap reference summary",
        "",
        f"Generated: {datetime.datetime.now().isoformat()}",
        "",
        "| reference | matches | mean_dra_arcsec | mean_ddec_arcsec | median_dra_arcsec | median_ddec_arcsec | median_sep_mas | total_dra_arcsec | total_ddec_arcsec |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in bootstrap_rows:
        report_lines.append(
            "| "
            f"{row['reference']} | {row['n_matches']} | {row['mean_dra_arcsec']:.6f} | {row['mean_ddec_arcsec']:.6f} | "
            f"{row['median_dra_arcsec']:.6f} | {row['median_ddec_arcsec']:.6f} | {row['median_sep_mas']:.3f} | "
            f"{row['total_dra_arcsec']:.6f} | {row['total_ddec_arcsec']:.6f} |"
        )
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"Wrote bootstrap summary to {summary_ecsv}")
    print(f"Wrote bootstrap summary to {summary_fits}")
    print(f"Wrote bootstrap report to {report_path}")
    print(f"Cached VVV catalog at {vvv_cache_path} with {len(vvv)} rows")
    print(f"Cached GNS catalog at {gns_cache_path} with {len(gns)} rows")
    print(f"Cached Gaia catalog at {gaia_cache_path} with {len(gaia)} rows")


if __name__ == "__main__":
    main()
