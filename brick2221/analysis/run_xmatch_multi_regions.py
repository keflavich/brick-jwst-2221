#!/usr/bin/env python
"""Run plot_tools.xmatch_plot for Brick, Cloud C, Sgr B2, and W51 catalogs."""

import argparse
import gc
import importlib.util
import os
import pathlib
import resource
import sys
import time

import matplotlib.pyplot as pl
import numpy as np
from astropy import units as u
from astropy.table import Table

import plot_tools

BRICK_CATALOG = "/orange/adamginsburg/jwst/brick/catalogs/crowdsource_nsky0_merged_photometry_tables_merged.fits"
CLOUDC_PATH = "/home/savannahgramze/orange_link/adamginsburg/jwst/cloudc/lactea-filament/lactea-filament"
SGRB2_CATALOG = "/orange/adamginsburg/jwst/sgrb2/NB/catalogs/crowdsource_nsky0_merged_photometry_tables_merged.fits"
W51_CATALOG_CANDIDATES = (
    "/orange/adamginsburg/jwst/w51/catalogs/final_nircam_miri_indivexp_merged_dao_refined_after_sat.fits",
    "/orange/adamginsburg/jwst/w51/catalogs/final_nircam_indivexp_merged_dao_refined_after_sat.fits",
)


def current_rss_mb():
    status_path = "/proc/self/status"
    if not os.path.exists(status_path):
        return np.nan

    with open(status_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    return int(parts[1]) / 1024.0
    return np.nan


def max_rss_mb():
    # Linux ru_maxrss is reported in KiB.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def log_stage(message, t0=None):
    elapsed = ""
    if t0 is not None:
        elapsed = f" elapsed={time.time() - t0:0.1f}s"
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        f" rss={current_rss_mb():0.1f} MB maxrss={max_rss_mb():0.1f} MB{elapsed}",
        flush=True,
    )


def filter_selection_mask(cat):
    filters_with_detections = ["f480m", "f410m", "f405n", "f360m"]
    return np.logical_and.reduce([~np.isnan(cat[f"mag_ab_{band}"]) for band in filters_with_detections])


def choose_existing_path(paths):
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def load_brickcat(path=BRICK_CATALOG):
    if not os.path.exists(path):
        return None
    return Table.read(path)


def load_cloudccat():
    if not os.path.isdir(CLOUDC_PATH):
        return None
    if CLOUDC_PATH not in sys.path:
        sys.path.append(CLOUDC_PATH)

    if importlib.util.find_spec("jwst_plots") is None:
        return None

    jwst_plots = importlib.import_module("jwst_plots")

    return jwst_plots.make_cat_use().catalog


def load_sgrb2cat(path=SGRB2_CATALOG):
    if not os.path.exists(path):
        return None
    sgrb2cat = Table.read(path)
    mask = filter_selection_mask(sgrb2cat)
    return sgrb2cat[mask]


def load_w51cat():
    path = choose_existing_path(W51_CATALOG_CANDIDATES)
    if path is None:
        return None
    return Table.read(path)


def infer_filternames(cat, ref_filter="f405n"):
    sep_filters = [name[4:] for name in cat.colnames if name.startswith("sep_")]
    skycoord_filters = [name[9:] for name in cat.colnames if name.startswith("skycoord_")]
    flux_filters = [name[5:] for name in cat.colnames if name.startswith("flux_")]

    valid = []
    for filt in sep_filters:
        if filt in skycoord_filters and filt in flux_filters:
            valid.append(filt)

    valid = sorted(set(valid))
    if ref_filter in valid:
        return ref_filter, valid

    if len(valid) == 0:
        return None, []

    return valid[0], valid


def qfit_column_name(cat, filt):
    for colname in (f"qfit_{filt}", f"qf_{filt}"):
        if colname in cat.colnames:
            return colname
    return None


def apply_qfit_limit(cat, filternames, qfit_limit):
    selected = np.ones(len(cat), dtype=bool)
    used_columns = []

    for filt in filternames:
        colname = qfit_column_name(cat, filt)
        if colname is None:
            print(f"No qfit column found for {filt}; skipping qfit cut for that filter", flush=True)
            continue

        qfit = np.ma.array(cat[colname], copy=False)
        qfit_value = qfit.filled(np.nan)
        good = (~np.ma.getmaskarray(qfit)) & np.isfinite(qfit_value) & (qfit_value < qfit_limit)
        excluded = (~good).sum()
        print(
            f"qfit cut for {filt} using {colname}: limit<{qfit_limit:0.3f}, "
            f"excluded={excluded} of {len(cat)}",
            flush=True,
        )
        selected &= good
        used_columns.append(colname)

    print(
        f"Combined qfit selection keeps {selected.sum()} of {len(cat)} rows "
        f"across {len(used_columns)} qfit columns",
        flush=True,
    )
    return cat[selected]


def slim_catalog_for_xmatch(cat, ref_filter, filternames):
    needed = [f"skycoord_{ref_filter}"]
    for filt in filternames:
        needed.extend((f"flux_{filt}", f"skycoord_{filt}", f"sep_{filt}"))

    needed_unique = [col for col in dict.fromkeys(needed) if col in cat.colnames]
    missing = [col for col in needed if col not in cat.colnames]
    if missing:
        print(f"Missing expected xmatch columns: {missing}", flush=True)

    return cat[needed_unique]


def stats_table_from_dict(statsd):
    rows = []
    for filt, stats in sorted(statsd.items()):
        rows.append(
            (
                filt,
                stats["med"].to(u.arcsec).value,
                stats["mad"].to(u.arcsec).value,
                stats["std"].to(u.arcsec).value,
                stats["med_thr"].to(u.arcsec).value,
                stats["mad_thr"].to(u.arcsec).value,
                stats["std_thr"].to(u.arcsec).value,
            )
        )

    return Table(
        rows=rows,
        names=(
            "filter",
            "med_arcsec",
            "mad_arcsec",
            "std_arcsec",
            "med_thr_arcsec",
            "mad_thr_arcsec",
            "std_thr_arcsec",
        ),
    )


def run_target(name, cat, outdir, maxsep=0.13 * u.arcsec, alpha=0.01, qfit_limit=0.06):
    t0 = time.time()
    log_stage(f"[{name}] starting run_target")
    ref_filter, filternames = infer_filternames(cat, ref_filter="f405n")
    if ref_filter is None or len(filternames) < 2:
        print(f"Skipping {name}: could not infer enough xmatch-compatible filters")
        return

    print(
        f"Running xmatch_plot for {name} with ref_filter={ref_filter} "
        f"and filters={filternames}",
        flush=True,
    )
    print(
        f"[{name}] catalog size: nrows={len(cat)} ncols={len(cat.colnames)}",
        flush=True,
    )
    log_stage(f"[{name}] before apply_qfit_limit", t0=t0)
    cat = apply_qfit_limit(cat, filternames=filternames, qfit_limit=qfit_limit)
    print(
        f"[{name}] after qfit selection: nrows={len(cat)} ncols={len(cat.colnames)}",
        flush=True,
    )
    log_stage(f"[{name}] after apply_qfit_limit", t0=t0)
    log_stage(f"[{name}] before slim_catalog_for_xmatch", t0=t0)
    cat = slim_catalog_for_xmatch(cat, ref_filter=ref_filter, filternames=filternames)
    print(
        f"[{name}] slimmed catalog size: nrows={len(cat)} ncols={len(cat.colnames)}",
        flush=True,
    )
    log_stage(f"[{name}] after slim_catalog_for_xmatch", t0=t0)

    pl.close("all")
    log_stage(f"[{name}] before xmatch_plot", t0=t0)

    statsd = plot_tools.xmatch_plot(
        cat,
        ref_filter=ref_filter,
        filternames=filternames,
        maxsep=maxsep,
        alpha=alpha,
        regs=None,
    )
    log_stage(f"[{name}] after xmatch_plot", t0=t0)

    fig1 = pl.figure(1)
    fig2 = pl.figure(2)
    log_stage(f"[{name}] before saving figures", t0=t0)
    fig1.savefig(outdir / f"{name}_xmatch_offsets.png", bbox_inches="tight", dpi=200)
    fig2.savefig(outdir / f"{name}_xmatch_hist.png", bbox_inches="tight", dpi=200)
    log_stage(f"[{name}] after saving figures", t0=t0)

    stat_tbl = stats_table_from_dict(statsd)
    stat_tbl.write(outdir / f"{name}_xmatch_stats.ecsv", overwrite=True)
    log_stage(f"[{name}] wrote stats table", t0=t0)

    # Encourage prompt memory release between targets.
    del statsd
    pl.close("all")
    log_stage(f"[{name}] completed run_target", t0=t0)


def main():
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        default="xmatch_multi_regions",
        help="Directory for output plots and stats tables",
    )
    parser.add_argument(
        "--maxsep-arcsec",
        type=float,
        default=0.13,
        help="Maximum separation in arcsec passed to xmatch_plot",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="Scatter alpha passed to xmatch_plot",
    )
    parser.add_argument(
        "--qfit-limit",
        type=float,
        default=0.06,
        help="Keep only rows with qfit < this value for every plotted filter",
    )
    args = parser.parse_args()
    log_stage("Parsed CLI arguments", t0=t0)

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    log_stage(f"Output directory ready at {outdir}", t0=t0)

    target_loaders = (
        ("brick", load_brickcat),
        ("cloudc", load_cloudccat),
        ("sgrb2", load_sgrb2cat),
        ("w51", load_w51cat),
    )

    maxsep = args.maxsep_arcsec * u.arcsec

    for name, loader in target_loaders:
        log_stage(f"[{name}] loading catalog", t0=t0)
        cat = loader()
        log_stage(f"[{name}] finished load attempt", t0=t0)
        if cat is None:
            print(f"Skipping {name}: catalog not found or not importable")
            continue

        print(f"[{name}] loaded nrows={len(cat)} ncols={len(cat.colnames)}", flush=True)
        run_target(
            name=name,
            cat=cat,
            outdir=outdir,
            maxsep=maxsep,
            alpha=args.alpha,
            qfit_limit=args.qfit_limit,
        )

        del cat
        gc.collect()
        log_stage(f"[{name}] after explicit cleanup", t0=t0)

    log_stage("All targets complete", t0=t0)


if __name__ == "__main__":
    main()
