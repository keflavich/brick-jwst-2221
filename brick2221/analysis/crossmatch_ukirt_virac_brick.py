#!/usr/bin/env python
"""
Crossmatch UKIRT microlensing survey sources with VIRAC in a 15x15 arcmin
region centered on the Brick, then generate average photometry products and
basic color-color / color-magnitude diagrams.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pyvo
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.table import Table
from astroquery.vizier import Vizier
from dust_extinction.averages import CT06_MWGC


UKIRT_TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP"
UKIRT_TABLE = "ukirttimeseries"
DEFAULT_VIRAC_CATALOG = "II/387/virac2"
DEFAULT_SPITZER_SSTGC_CATALOG = "/orange/adamginsburg/spitzer/cmz_catalog_II_295_SSTGC_Ramirez2008.fits"
LAMBDA_H_UM = 1.65
LAMBDA_K_UM = 2.20
LAMBDA_KS_UM = 2.149

# Brick center from regions_/nircam_brick_fov.reg in this project.
DEFAULT_BRICK_RA_DEG = 266.534963671
DEFAULT_BRICK_DEC_DEG = -28.710074995


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crossmatch UKIRT with VIRAC and make H-K and K-Ks diagrams toward the Brick."
    )
    parser.add_argument("--ra", type=float, default=DEFAULT_BRICK_RA_DEG, help="Center RA in deg")
    parser.add_argument("--dec", type=float, default=DEFAULT_BRICK_DEC_DEG, help="Center Dec in deg")
    parser.add_argument(
        "--box-size-arcmin",
        type=float,
        default=15.0,
        help="Square region size in arcmin (width=height)",
    )
    parser.add_argument(
        "--match-radius-arcsec",
        type=float,
        default=0.3,
        help="Maximum match separation in arcsec",
    )
    parser.add_argument(
        "--virac-catalog",
        type=str,
        default=DEFAULT_VIRAC_CATALOG,
        help="Vizier VIRAC catalog ID (e.g. II/387/virac2 or II/364/virac)",
    )
    parser.add_argument(
        "--min-obs-year",
        type=int,
        default=2017,
        help="Minimum UKIRT obs_year to include (2017+ includes K-band years)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/orange/adamginsburg/jwst/brick/ukirt_virac_brick_crossmatch"),
        help="Directory for output tables and plots",
    )
    parser.add_argument(
        "--max-mag-err",
        type=float,
        default=0.05,
        help="Maximum allowed magnitude uncertainty for plotted sources",
    )
    parser.add_argument(
        "--spitzer-catalog",
        type=Path,
        default=Path(DEFAULT_SPITZER_SSTGC_CATALOG),
        help="Path to local Spitzer SSTGC source catalog",
    )
    parser.add_argument(
        "--spitzer-match-radius-arcsec",
        type=float,
        default=1.0,
        help="Maximum match separation for Spitzer x UKIRT/VIRAC crossmatch",
    )
    return parser.parse_args()


def rectangular_region_mask(coords: SkyCoord, center: SkyCoord, box_size_arcmin: float) -> np.ndarray:
    half_height_deg = (0.5 * box_size_arcmin * u.arcmin).to(u.deg).value
    half_width_deg = half_height_deg / np.cos(np.deg2rad(center.dec.deg))
    return (
        (coords.ra.deg >= center.ra.deg - half_width_deg)
        & (coords.ra.deg <= center.ra.deg + half_width_deg)
        & (coords.dec.deg >= center.dec.deg - half_height_deg)
        & (coords.dec.deg <= center.dec.deg + half_height_deg)
    )


def query_spitzer_sstgc_local(catalog_path: Path) -> Table:
    if not catalog_path.exists():
        raise FileNotFoundError(f"Spitzer catalog not found: {catalog_path}")
    print(f"Loading Spitzer SSTGC catalog from {catalog_path}")
    spitzer = Table.read(catalog_path)
    if len(spitzer) == 0:
        raise ValueError("Spitzer SSTGC catalog is empty.")
    return spitzer


def make_spitzer_cmds(spitzer: Table, center: SkyCoord, box_size_arcmin: float, output_dir: Path) -> None:
    i1 = np.asarray(spitzer["_3.6mag"], dtype=float)
    i2 = np.asarray(spitzer["_4.5mag"], dtype=float)
    spitzer_coords = SkyCoord(spitzer["coordinates"])

    finite = np.isfinite(i1) & np.isfinite(i2)
    color_i1_i2 = i1 - i2

    # Whole GC CMD from all finite SSTGC entries.
    fig, ax = plt.subplots(figsize=(7, 6))
    hb = ax.hexbin(
        color_i1_i2[finite],
        i2[finite],
        gridsize=180,
        mincnt=1,
        bins="log",
        cmap="viridis",
        linewidths=0,
    )
    fig.colorbar(hb, ax=ax, label="log10(N)")
    ax.set_xlabel("Spitzer I1 - I2")
    ax.set_ylabel("Spitzer I2")
    ax.set_title("Spitzer CMD: Whole GC")
    ax.set_xlim(-1.0, 3.0)
    ax.set_ylim(15.5, 7.0)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "spitzer_cmd_i1_minus_i2_whole_gc.png", dpi=220)
    plt.close(fig)

    # Brick-region CMD using same rectangular region as UKIRT/VIRAC.
    brick_mask = rectangular_region_mask(spitzer_coords, center, box_size_arcmin)
    sel = finite & brick_mask
    if np.count_nonzero(sel) == 0:
        raise ValueError("No Spitzer I1/I2 sources found in the requested Brick region.")

    fig, ax = plt.subplots(figsize=(7, 6))
    hb = ax.hexbin(
        color_i1_i2[sel],
        i2[sel],
        gridsize=160,
        mincnt=1,
        bins="log",
        cmap="viridis",
        linewidths=0,
    )
    fig.colorbar(hb, ax=ax, label="log10(N)")
    ax.set_xlabel("Spitzer I1 - I2")
    ax.set_ylabel("Spitzer I2")
    ax.set_title("Spitzer CMD: Brick Region")
    ax.set_xlim(-1.0, 3.0)
    ax.set_ylim(15.5, 7.0)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "spitzer_cmd_i1_minus_i2_brick_region.png", dpi=220)
    plt.close(fig)


def query_ukirt_box(
    center: SkyCoord,
    box_size_arcmin: float,
    min_obs_year: int,
    cache_dir: Path,
) -> Table:
    half_height_deg = (0.5 * box_size_arcmin * u.arcmin).to(u.deg).value
    half_width_deg = half_height_deg / np.cos(np.deg2rad(center.dec.deg))

    ra_min = center.ra.deg - half_width_deg
    ra_max = center.ra.deg + half_width_deg
    dec_min = center.dec.deg - half_height_deg
    dec_max = center.dec.deg + half_height_deg

    adql = f"""
    SELECT
      sourceid,
      AVG(ra) AS ra,
      AVG(dec) AS dec,
      AVG(h_mag) AS h_mag,
      AVG(k_mag) AS k_mag,
            SQRT(AVG(h_mag*h_mag) - AVG(h_mag)*AVG(h_mag)) AS h_mag_err,
            SQRT(AVG(k_mag*k_mag) - AVG(k_mag)*AVG(k_mag)) AS k_mag_err,
      COUNT(*) AS n_rows,
      MIN(obs_year) AS min_obs_year,
      MAX(obs_year) AS max_obs_year
    FROM {UKIRT_TABLE}
    WHERE ra BETWEEN {ra_min:.8f} AND {ra_max:.8f}
      AND dec BETWEEN {dec_min:.8f} AND {dec_max:.8f}
      AND obs_year >= {int(min_obs_year)}
      AND h_mag IS NOT NULL
      AND k_mag IS NOT NULL
    GROUP BY sourceid
    """

    cache_dir.mkdir(parents=True, exist_ok=True)
    adql_for_hash = "\n".join(line.strip() for line in adql.splitlines() if line.strip())
    adql_hash = hashlib.sha256(adql_for_hash.encode("utf-8")).hexdigest()[:16]
    cache_file = cache_dir / f"ukirt_adql_{adql_hash}.ecsv"

    if cache_file.exists():
        print(f"Loading UKIRT query result from cache: {cache_file}")
        ukirt = Table.read(cache_file)
        if len(ukirt) == 0:
            raise ValueError(f"Cached UKIRT table is empty: {cache_file}")
        print(f"Loaded {len(ukirt)} UKIRT sources from cache.")
        return ukirt

    print("Querying UKIRT TAP service...")
    service = pyvo.dal.TAPService(UKIRT_TAP_URL)
    result = service.search(adql)
    ukirt = result.to_table()
    print(f"Completed UKIRT query with {len(ukirt)} sources returned.")

    if len(ukirt) == 0:
        raise ValueError("UKIRT query returned zero rows for the requested region.")

    ukirt.write(cache_file, overwrite=True)
    print(f"Cached UKIRT query result to {cache_file}")

    return ukirt


def query_virac_box(center: SkyCoord, box_size_arcmin: float, catalog: str) -> Table:
    print(f"Querying vizier for {catalog} sources...")
    viz = Vizier(columns=["*"], row_limit=-1)
    result = viz.query_region(
        center,
        width=box_size_arcmin * u.arcmin,
        height=box_size_arcmin * u.arcmin,
        catalog=catalog,
    )
    print(f"Completed query with {len(result)} tables returned (first table has {len(result[0])} rows).")
    if len(result) == 0:
        raise ValueError(f"VIRAC query returned zero rows for catalog {catalog}.")
    return result[0]


def infer_virac_columns(virac: Table) -> tuple[str, str, str, str, str, str]:
    if "RAJ2000" in virac.colnames and "DEJ2000" in virac.colnames:
        ra_col = "RAJ2000"
        dec_col = "DEJ2000"
        h_err_col = "e_Hmag"
        ks_err_col = "e_Ksmag"
    elif "RA_ICRS" in virac.colnames and "DE_ICRS" in virac.colnames:
        ra_col = "RA_ICRS"
        dec_col = "DE_ICRS"
        h_err_col = "Hell"
        ks_err_col = "KsEll"
    else:
        raise ValueError("Could not find RA/Dec columns in VIRAC table.")

    if "Ksmag" not in virac.colnames:
        raise ValueError("Could not find Ksmag in VIRAC table.")
    if "Hmag" not in virac.colnames:
        raise ValueError("Could not find Hmag in VIRAC table.")

    if h_err_col not in virac.colnames or ks_err_col not in virac.colnames:
        raise ValueError("Could not find VIRAC uncertainty columns for H/Ks.")

    return ra_col, dec_col, "Hmag", "Ksmag", h_err_col, ks_err_col


def crossmatch_catalogs(ukirt: Table, virac: Table, match_radius_arcsec: float) -> Table:
    ra_col, dec_col, virac_h_col, virac_ks_col, virac_h_err_col, virac_ks_err_col = infer_virac_columns(virac)

    ukirt_ra = np.asarray(ukirt["ra"], dtype=float)
    ukirt_dec = np.asarray(ukirt["dec"], dtype=float)
    virac_ra = np.asarray(virac[ra_col], dtype=float)
    virac_dec = np.asarray(virac[dec_col], dtype=float)

    ukirt_coords = SkyCoord(ukirt_ra, ukirt_dec, unit="deg", frame="icrs")
    virac_coords = SkyCoord(virac_ra, virac_dec, unit="deg", frame="icrs")

    idx, sep2d, _ = match_coordinates_sky(ukirt_coords, virac_coords)
    keep = sep2d <= (match_radius_arcsec * u.arcsec)

    matched = Table()
    matched["sourceid"] = ukirt["sourceid"][keep]
    matched["min_obs_year"] = ukirt["min_obs_year"][keep]
    matched["max_obs_year"] = ukirt["max_obs_year"][keep]
    matched["n_ukirt_rows"] = ukirt["n_rows"][keep]
    matched["ra"] = ukirt["ra"][keep]
    matched["dec"] = ukirt["dec"][keep]
    matched["sep_arcsec"] = sep2d[keep].to(u.arcsec).value

    matched["ukirt_h_mag"] = np.asarray(ukirt["h_mag"][keep], dtype=float)
    matched["ukirt_k_mag"] = np.asarray(ukirt["k_mag"][keep], dtype=float)
    matched["ukirt_h_mag_err"] = np.asarray(ukirt["h_mag_err"][keep], dtype=float)
    matched["ukirt_k_mag_err"] = np.asarray(ukirt["k_mag_err"][keep], dtype=float)
    matched["virac_h_mag"] = np.asarray(virac[virac_h_col][idx[keep]], dtype=float)
    matched["virac_ks_mag"] = np.asarray(virac[virac_ks_col][idx[keep]], dtype=float)
    matched["virac_h_mag_err"] = np.asarray(virac[virac_h_err_col][idx[keep]], dtype=float)
    matched["virac_ks_mag_err"] = np.asarray(virac[virac_ks_err_col][idx[keep]], dtype=float)

    matched["H_minus_K"] = matched["ukirt_h_mag"] - matched["ukirt_k_mag"]
    matched["K_minus_Ks"] = matched["ukirt_k_mag"] - matched["virac_ks_mag"]
    matched["Ks_minus_K"] = matched["virac_ks_mag"] - matched["ukirt_k_mag"]

    finite = (
        np.isfinite(matched["H_minus_K"])
        & np.isfinite(matched["Ks_minus_K"])
        & np.isfinite(matched["ukirt_k_mag"])
    )
    return matched[finite]


def make_brick_spitzer_ukirt_virac_ccd(
    matched: Table,
    spitzer: Table,
    center: SkyCoord,
    box_size_arcmin: float,
    max_mag_err: float,
    spitzer_match_radius_arcsec: float,
    output_dir: Path,
) -> None:
    # Apply the same UKIRT/VIRAC quality cuts used for the existing diagrams.
    ukv_good = (
        np.isfinite(matched["ukirt_h_mag_err"])
        & np.isfinite(matched["ukirt_k_mag_err"])
        & np.isfinite(matched["virac_ks_mag_err"])
        & (matched["ukirt_h_mag_err"] < max_mag_err)
        & (matched["ukirt_k_mag_err"] < max_mag_err)
        & (matched["virac_ks_mag_err"] < max_mag_err)
    )

    ukv_ra = np.asarray(matched["ra"], dtype=float)
    ukv_dec = np.asarray(matched["dec"], dtype=float)
    ukv_coords = SkyCoord(ukv_ra, ukv_dec, unit="deg", frame="icrs")
    ukv_brick = rectangular_region_mask(ukv_coords, center, box_size_arcmin)
    ukv_sel = ukv_good & ukv_brick
    if np.count_nonzero(ukv_sel) == 0:
        raise ValueError("No UKIRT/VIRAC sources remain for Brick crossmatch after quality cuts.")

    i1 = np.asarray(spitzer["_3.6mag"], dtype=float)
    i2 = np.asarray(spitzer["_4.5mag"], dtype=float)
    spitzer_coords = SkyCoord(spitzer["coordinates"])
    spitzer_good = np.isfinite(i1) & np.isfinite(i2)
    spitzer_brick = rectangular_region_mask(spitzer_coords, center, box_size_arcmin)
    spitzer_sel = spitzer_good & spitzer_brick
    if np.count_nonzero(spitzer_sel) == 0:
        raise ValueError("No Spitzer I1/I2 sources remain in Brick region.")

    ukv_coords_sel = ukv_coords[ukv_sel]
    spitzer_coords_sel = spitzer_coords[spitzer_sel]
    spitzer_i1_i2_sel = i1[spitzer_sel] - i2[spitzer_sel]

    idx, sep2d, _ = match_coordinates_sky(ukv_coords_sel, spitzer_coords_sel)
    keep = sep2d <= (spitzer_match_radius_arcsec * u.arcsec)
    if np.count_nonzero(keep) == 0:
        raise ValueError("No UKIRT/VIRAC-Spitzer matches found in Brick region.")

    k_minus_ks = np.asarray(matched["K_minus_Ks"][ukv_sel], dtype=float)[keep]
    i1_minus_i2 = np.asarray(spitzer_i1_i2_sel[idx], dtype=float)[keep]

    finite = np.isfinite(k_minus_ks) & np.isfinite(i1_minus_i2)
    k_minus_ks = k_minus_ks[finite]
    i1_minus_i2 = i1_minus_i2[finite]
    if k_minus_ks.size == 0:
        raise ValueError("No finite K-Ks vs I1-I2 pairs after matching.")

    fig, ax = plt.subplots(figsize=(7, 6))
    hb = ax.hexbin(
        k_minus_ks,
        i1_minus_i2,
        gridsize=140,
        mincnt=1,
        bins="log",
        cmap="viridis",
        linewidths=0,
    )
    fig.colorbar(hb, ax=ax, label="log10(N)")
    ax.set_xlabel("UKIRT K - VIRAC Ks")
    ax.set_ylabel("Spitzer I1 - I2")
    ax.set_title("Brick CCD: K-Ks vs I1-I2")
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-1.0, 3.0)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "brick_ccd_k_minus_ks_vs_i1_minus_i2.png", dpi=220)
    plt.close(fig)


def summarize_matches(
    matched: Table,
    match_radius_arcsec: float,
    box_size_arcmin: float,
    center: SkyCoord,
) -> Table:
    summary = Table()
    summary["n_matched"] = [len(matched)]
    summary["ra_center_deg"] = [center.ra.deg]
    summary["dec_center_deg"] = [center.dec.deg]
    summary["box_size_arcmin"] = [box_size_arcmin]
    summary["match_radius_arcsec"] = [match_radius_arcsec]

    summary["mean_ukirt_h_mag"] = [np.nanmean(matched["ukirt_h_mag"])]
    summary["mean_ukirt_k_mag"] = [np.nanmean(matched["ukirt_k_mag"])]
    summary["mean_virac_ks_mag"] = [np.nanmean(matched["virac_ks_mag"])]
    summary["mean_H_minus_K"] = [np.nanmean(matched["H_minus_K"])]
    summary["mean_Ks_minus_K"] = [np.nanmean(matched["Ks_minus_K"])]

    summary["std_ukirt_h_mag"] = [np.nanstd(matched["ukirt_h_mag"])]
    summary["std_ukirt_k_mag"] = [np.nanstd(matched["ukirt_k_mag"])]
    summary["std_virac_ks_mag"] = [np.nanstd(matched["virac_ks_mag"])]
    summary["std_H_minus_K"] = [np.nanstd(matched["H_minus_K"])]
    summary["std_Ks_minus_K"] = [np.nanstd(matched["Ks_minus_K"])]

    return summary


def make_plots(matched: Table, output_dir: Path, max_mag_err: float) -> None:
    plot_mask = (
        np.isfinite(matched["ukirt_h_mag_err"])
        & np.isfinite(matched["ukirt_k_mag_err"])
        & np.isfinite(matched["virac_ks_mag_err"])
        & (matched["ukirt_h_mag_err"] < max_mag_err)
        & (matched["ukirt_k_mag_err"] < max_mag_err)
        & (matched["virac_ks_mag_err"] < max_mag_err)
    )
    plotted = matched[plot_mask]

    if len(plotted) == 0:
        raise ValueError("No sources remain after applying plot uncertainty cuts.")

    fig, ax = plt.subplots(figsize=(7, 6))
    sel = (plotted["H_minus_K"] > -0.5) & (plotted["H_minus_K"] < 4) & (plotted["Ks_minus_K"] > -0.5) & (plotted["Ks_minus_K"] < 0.5)
    hb = ax.hexbin(
        plotted["H_minus_K"][sel],
        plotted["Ks_minus_K"][sel],
        gridsize=180,
        mincnt=1,
        bins="log",
        cmap="viridis",
        linewidths=0,
    )
    fig.colorbar(hb, ax=ax, label="log10(N)")
    ax.set_xlabel("UKIRT H - K")
    ax.set_ylabel("VIRAC Ks - UKIRT K")
    ax.set_title("Brick: H-K vs Ks-K")
    ax.set_xlim(-0.5, 4)
    ax.set_ylim(-0.5, 0.5)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "brick_ccd_hk_kks.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    sel = (plotted["Ks_minus_K"] > -0.5) & (plotted["Ks_minus_K"] < 0.5) & (plotted["ukirt_k_mag"] > 11) & (plotted["ukirt_k_mag"] < 15)
    hb = ax.hexbin(
        plotted["Ks_minus_K"][sel],
        plotted["ukirt_k_mag"][sel],
        gridsize=180,
        mincnt=1,
        bins="log",
        cmap="viridis",
        linewidths=0,
    )
    fig.colorbar(hb, ax=ax, label="log10(N)")
    ax.set_xlabel("VIRAC Ks - UKIRT K")
    ax.set_ylabel("UKIRT K")
    ax.set_title("Brick: CMD using Ks-K color")
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(15, 11)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "brick_cmd_kks_k.png", dpi=220)
    plt.close(fig)

    # Deredden K and K-Ks using E(H-K) with the CT06_MWGC extinction law.
    ext = CT06_MWGC()
    ah_av = float(ext(LAMBDA_H_UM * u.micron))
    ak_av = float(ext(LAMBDA_K_UM * u.micron))
    aks_av = float(ext(LAMBDA_KS_UM * u.micron))

    ehk_obs = np.asarray(plotted["H_minus_K"], dtype=float)
    av_est = ehk_obs / (ah_av - ak_av)
    ak = av_est * ak_av
    aks = av_est * aks_av

    k_obs = np.asarray(plotted["ukirt_k_mag"], dtype=float)
    ks_obs = np.asarray(plotted["virac_ks_mag"], dtype=float)

    k0 = k_obs - ak
    k_minus_ks0 = (k_obs - ks_obs) - (ak - aks)

    sel = (
        np.isfinite(k0)
        & np.isfinite(k_minus_ks0)
        & (k_minus_ks0 > -1.0)
        & (k_minus_ks0 < 1.0)
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    hb = ax.hexbin(
        k_minus_ks0[sel],
        k0[sel],
        gridsize=180,
        mincnt=1,
        bins="log",
        cmap="viridis",
        linewidths=0,
    )
    fig.colorbar(hb, ax=ax, label="log10(N)")
    ax.set_xlabel("(UKIRT K - VIRAC Ks) dereddened")
    ax.set_ylabel("UKIRT K dereddened")
    ax.set_title("Brick: Dereddened CMD (CT06_MWGC from H-K)")
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(15, 8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "brick_cmd_k_minus_ks_dereddened_ct06.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    ukirt_cache_dir = output_dir / "ukirt_adql_cache"

    center = SkyCoord(args.ra * u.deg, args.dec * u.deg, frame="fk5")

    ukirt = query_ukirt_box(center, args.box_size_arcmin, args.min_obs_year, ukirt_cache_dir)
    virac = query_virac_box(center, args.box_size_arcmin, args.virac_catalog)
    spitzer = query_spitzer_sstgc_local(args.spitzer_catalog)
    print("Crossmatching catalogs...")
    matched = crossmatch_catalogs(ukirt, virac, args.match_radius_arcsec)

    if len(matched) == 0:
        raise ValueError("No crossmatches found. Try a larger match radius.")

    print("Summarizing")
    summary = summarize_matches(matched, args.match_radius_arcsec, args.box_size_arcmin, center)

    ukirt.write(output_dir / "ukirt_sources.ecsv", overwrite=True)
    virac.write(output_dir / "virac_sources.ecsv", overwrite=True)
    matched.write(output_dir / "ukirt_virac_matched.ecsv", overwrite=True)
    summary.write(output_dir / "ukirt_virac_summary.ecsv", overwrite=True)

    print("Making plots...")
    make_spitzer_cmds(spitzer, center, args.box_size_arcmin, output_dir)
    make_plots(matched, output_dir, args.max_mag_err)
    make_brick_spitzer_ukirt_virac_ccd(
        matched,
        spitzer,
        center,
        args.box_size_arcmin,
        args.max_mag_err,
        args.spitzer_match_radius_arcsec,
        output_dir,
    )

    print(f"Wrote outputs to {output_dir}")
    print(f"UKIRT sources: {len(ukirt)}")
    print(f"VIRAC sources: {len(virac)}")
    print(f"Matched sources: {len(matched)}")
    print("Average photometry summary:")
    for name in summary.colnames:
        print(f"  {name}: {summary[name][0]}")


if __name__ == "__main__":
    main()
