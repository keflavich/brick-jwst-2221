#!/usr/bin/env python
"""Benchmark: parallel-chunked PSFPhotometry vs single-pass.

Picks one sgrc F115W destreak exposure, runs DAOStarFinder to get init
positions, then times:
  (1) single-process PSFPhotometry over all sources
  (2) multiprocessing.Pool with N workers, each fitting a chunk

The chunking respects SourceGrouper output so neighboring sources stay
together (otherwise overlapping PSFs would double-count flux).

Usage:
    python bench_psf_parallel.py \
        --image /orange/adamginsburg/jwst/sgrc/F115W/pipeline/<file>.fits \
        --psf /orange/adamginsburg/jwst/sgrc/psfs/nircam_nrca1_f115w_fovp101_samp2_npsf16.fits \
        --chunk-size 100 --n-workers 8 [--max-sources 2000]
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats import sigma_clipped_stats
from astropy.table import Table, vstack

from photutils.background import LocalBackground
from photutils.detection import DAOStarFinder
from photutils.psf import (
    GriddedPSFModel,
    IterativePSFPhotometry,
    PSFPhotometry,
    SourceGrouper,
)


def _std_for_finder(image: np.ndarray, mask) -> float:
    _, _, std = sigma_clipped_stats(image, sigma=3.0, mask=mask)
    return float(std)


def load_image(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    with fits.open(path) as hdul:
        sci = hdul["SCI"].data.astype(np.float32)
        try:
            err = hdul["ERR"].data.astype(np.float32)
        except KeyError:
            err = None
        try:
            dq = hdul["DQ"].data
        except KeyError:
            dq = None
        meta = dict(hdul["SCI"].header)
    return sci, err, dq, meta


def load_psf(path: Path) -> GriddedPSFModel:
    return GriddedPSFModel.read(path)


def find_sources(image: np.ndarray, fwhm_pix: float, mask: np.ndarray | None,
                 nsigma: float = 5.0) -> Table:
    mean, median, std = sigma_clipped_stats(image, sigma=3.0, mask=mask)
    finder = DAOStarFinder(threshold=nsigma * std, fwhm=fwhm_pix,
                           roundlo=-1.0, roundhi=1.0)
    sources = finder(image - median, mask=mask)
    if sources is None or len(sources) == 0:
        raise SystemExit("No sources found.")
    init = Table()
    init["x_init"] = sources["xcentroid"]
    init["y_init"] = sources["ycentroid"]
    init["flux_init"] = sources["flux"]
    return init


def assign_groups(init: Table, min_separation: float) -> np.ndarray:
    grouper = SourceGrouper(min_separation=min_separation)
    return grouper(init["x_init"], init["y_init"])


def chunk_by_group(init: Table, group_id: np.ndarray, target: int) -> list[np.ndarray]:
    """Return list of row-index arrays. Each chunk has ~target sources and
    never splits a group across chunks."""
    init = init.copy()
    init["_group"] = group_id
    init["_rownum"] = np.arange(len(init))

    by_group: dict[int, list[int]] = {}
    for r in init:
        by_group.setdefault(int(r["_group"]), []).append(int(r["_rownum"]))
    groups = sorted(by_group.values(), key=lambda g: -len(g))

    chunks: list[list[int]] = []
    current: list[int] = []
    for grp in groups:
        if current and len(current) + len(grp) > target:
            chunks.append(current)
            current = []
        current.extend(grp)
    if current:
        chunks.append(current)
    return [np.asarray(c, dtype=np.int64) for c in chunks]


# Module-level state set by _worker_init so each forked process can read it.
_W_IMAGE: np.ndarray | None = None
_W_ERR: np.ndarray | None = None
_W_MASK: np.ndarray | None = None
_W_PSF: GriddedPSFModel | None = None
_W_FIT_SHAPE: tuple[int, int] = (5, 5)
_W_APERTURE: int = 4


def _worker_init(image, err, mask, psf, fit_shape, aperture_radius):
    global _W_IMAGE, _W_ERR, _W_MASK, _W_PSF, _W_FIT_SHAPE, _W_APERTURE
    _W_IMAGE = image
    _W_ERR = err
    _W_MASK = mask
    _W_PSF = psf
    _W_FIT_SHAPE = fit_shape
    _W_APERTURE = aperture_radius


def _worker_fit(args: tuple[int, Table]) -> tuple[int, Table]:
    chunk_idx, chunk_init = args
    photom = PSFPhotometry(
        psf_model=_W_PSF,
        fit_shape=_W_FIT_SHAPE,
        fitter=LevMarLSQFitter(),
        finder=None,
        aperture_radius=_W_APERTURE,
        progress_bar=False,
    )
    tbl = photom(_W_IMAGE, error=_W_ERR, mask=_W_MASK, init_params=chunk_init)
    return chunk_idx, tbl


def run_single(image, err, mask, psf, init, fit_shape, aperture_radius):
    photom = PSFPhotometry(
        psf_model=psf,
        fit_shape=fit_shape,
        fitter=LevMarLSQFitter(),
        finder=None,
        aperture_radius=aperture_radius,
        progress_bar=False,
    )
    t0 = time.perf_counter()
    tbl = photom(image, error=err, mask=mask, init_params=init)
    dt = time.perf_counter() - t0
    return tbl, dt


def run_parallel(image, err, mask, psf, init, chunks, n_workers,
                 fit_shape, aperture_radius):
    payload = [(i, init[chunk]) for i, chunk in enumerate(chunks)]
    ctx = mp.get_context("fork")
    t0 = time.perf_counter()
    with ctx.Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(image, err, mask, psf, fit_shape, aperture_radius),
    ) as pool:
        results = pool.map(_worker_fit, payload)
    dt = time.perf_counter() - t0
    results.sort(key=lambda r: r[0])
    tbl = vstack([r[1] for r in results])
    return tbl, dt


def compare(single: Table, parallel: Table) -> None:
    if len(single) != len(parallel):
        print(f"DIFF: row counts differ single={len(single)} parallel={len(parallel)}")
        return
    s = single.copy()
    p = parallel.copy()
    # Match by (x_init, y_init) — they should be unique within init table.
    s.sort(["y_init", "x_init"])
    p.sort(["y_init", "x_init"])
    for col in ("x_fit", "y_fit", "flux_fit"):
        if col not in s.colnames or col not in p.colnames:
            continue
        d = np.asarray(s[col]) - np.asarray(p[col])
        d = d[np.isfinite(d)]
        if len(d) == 0:
            print(f"{col}: no finite values")
            continue
        print(f"{col}: max|delta|={np.max(np.abs(d)):.3e} "
              f"median|delta|={np.median(np.abs(d)):.3e} n={len(d)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--psf", required=True, type=Path)
    ap.add_argument("--chunk-size", type=int, default=100)
    ap.add_argument("--n-workers", type=int, default=8)
    ap.add_argument("--fwhm-pix", type=float, default=1.5,
                    help="DAOStarFinder FWHM (px). F115W nrca1 ~1.5.")
    ap.add_argument("--min-separation", type=float, default=4.0,
                    help="SourceGrouper min_separation (px).")
    ap.add_argument("--max-sources", type=int, default=0,
                    help="Truncate init table for quick experiments (0 = all).")
    ap.add_argument("--fit-shape", type=int, default=5)
    ap.add_argument("--aperture", type=int, default=4)
    ap.add_argument("--nsigma", type=float, default=5.0,
                    help="DAOStarFinder detection threshold (nsigma * std).")
    ap.add_argument("--iterative", action="store_true",
                    help="Also benchmark IterativePSFPhotometry (serial) "
                         "vs the parallel reimplementation in the pipeline.")
    ap.add_argument("--iterative-maxiters", type=int, default=5)
    ap.add_argument("--localbkg-inner", type=int, default=10)
    ap.add_argument("--localbkg-outer", type=int, default=20)
    args = ap.parse_args()

    print(f"Loading image  {args.image}")
    sci, err, dq, _ = load_image(args.image)
    print(f"  shape={sci.shape} dtype={sci.dtype}")

    print(f"Loading PSF    {args.psf}")
    psf = load_psf(args.psf)

    mask = None
    if dq is not None:
        mask = dq != 0  # any DQ bit set
        print(f"  masked pixels = {mask.sum()} ({100*mask.mean():.2f}%)")

    print(f"Finding sources (DAOStarFinder fwhm={args.fwhm_pix} px nsigma={args.nsigma})")
    init = find_sources(sci, args.fwhm_pix, mask=mask, nsigma=args.nsigma)
    print(f"  n_sources = {len(init)}")
    if args.max_sources and len(init) > args.max_sources:
        init = init[: args.max_sources]
        print(f"  truncated to {len(init)} (--max-sources)")

    print(f"Grouping (SourceGrouper min_separation={args.min_separation} px)")
    group_id = assign_groups(init, args.min_separation)
    n_groups = len(np.unique(group_id))
    print(f"  n_groups = {n_groups}")

    chunks = chunk_by_group(init, group_id, args.chunk_size)
    print(f"Chunking (target {args.chunk_size}/chunk): {len(chunks)} chunks "
          f"sizes min={min(len(c) for c in chunks)} max={max(len(c) for c in chunks)}")

    print("--- single-process run ---")
    single_tbl, single_dt = run_single(sci, err, mask, psf, init,
                                       (args.fit_shape, args.fit_shape),
                                       args.aperture)
    print(f"  single: {single_dt:.2f} s  n_fit={len(single_tbl)}")

    print(f"--- parallel run (n_workers={args.n_workers}) ---")
    par_tbl, par_dt = run_parallel(sci, err, mask, psf, init, chunks,
                                   args.n_workers,
                                   (args.fit_shape, args.fit_shape),
                                   args.aperture)
    print(f"  parallel: {par_dt:.2f} s  n_fit={len(par_tbl)}")

    speedup = single_dt / par_dt if par_dt > 0 else float("inf")
    print(f"speedup = {speedup:.2f}x  (theoretical max = {args.n_workers}x)")

    print("--- agreement (single vs parallel) ---")
    compare(single_tbl, par_tbl)

    if args.iterative:
        # Iterative-mode test: run photutils IterativePSFPhotometry serially,
        # then run our parallel reimplementation (the one wired into the
        # pipeline as _parallel_iterative_psfphotometry).  Compare result
        # tables.  Note the parallel reimplementation reproduces mode='new'
        # only; small differences from photutils' full IterativePSFPhotometry
        # are expected (different finder invocation timing, no shared
        # _fit_models state), but the source set and fluxes should be close.
        print("=" * 60)
        print("ITERATIVE TEST")
        print("=" * 60)
        from photutils.psf import IterativePSFPhotometry  # noqa: F401
        import sys as _sys
        _sys.path.insert(0, "/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline/jwst_gc_pipeline/photometry")
        # Import the pipeline helper without running its huge __main__.
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "ccl",
            "/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline/jwst_gc_pipeline/photometry/crowdsource_catalogs_long.py",
        )
        # Skip the heavy import side-effects: load only the parallel-helpers
        # by extracting the top of the module.  But the file is ~3800 lines
        # and side-effecting; cleaner to import the parallel helpers by
        # ast or copy the small functions inline.  For simplicity we just
        # exec the module top-level imports + the helper functions.
        # Simpler still: replicate the helper here for the test.

        from astropy.modeling.fitting import LevMarLSQFitter as _LM
        photom_kwargs = dict(
            psf_model=psf,
            fitter=_LM(),
            fit_shape=(args.fit_shape, args.fit_shape),
            aperture_radius=args.aperture,
            progress_bar=False,
            localbkg_estimator=LocalBackground(args.localbkg_inner,
                                                args.localbkg_outer),
        )
        print(f"--- IterativePSFPhotometry serial (maxiters={args.iterative_maxiters}) ---")
        iter_phot = IterativePSFPhotometry(
            finder=DAOStarFinder(threshold=args.nsigma *
                                 _std_for_finder(sci, mask),
                                 fwhm=args.fwhm_pix,
                                 roundlo=-1.0, roundhi=1.0),
            maxiters=args.iterative_maxiters,
            sub_shape=(15, 15),
            **photom_kwargs,
        )
        t0 = time.perf_counter()
        iter_serial = iter_phot(sci, mask=mask)
        dt_iter_serial = time.perf_counter() - t0
        print(f"  iter serial: {dt_iter_serial:.2f} s  n={len(iter_serial)}")

        # Use the SAME parallel iterative implementation that the pipeline
        # ships, so this validates the production code path rather than
        # an inlined copy.  The pipeline module has heavy import-time
        # side effects (webbpsf import + module-level prints), but the
        # parallel helpers themselves are pure.  Import only the helpers
        # via importlib to dodge the __main__ guard.
        print(f"--- parallel iterative (n_workers={args.n_workers}) ---")
        # Import the pipeline module under its real name so forked workers
        # can re-pickle the `_par_worker_fit` reference via normal module
        # lookup.  Importing here has heavy side effects (webbpsf, etc.)
        # but is necessary for fork-based workers to find the function.
        import sys as _sys
        _sys.path.insert(0, "/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline")
        from jwst_gc_pipeline.photometry import crowdsource_catalogs_long as _ccl
        finder = DAOStarFinder(
            threshold=args.nsigma * _std_for_finder(sci, mask),
            fwhm=args.fwhm_pix, roundlo=-1.0, roundhi=1.0,
        )
        t0 = time.perf_counter()
        iter_par = _ccl._parallel_iterative_psfphotometry(
            sci,
            photometry_kwargs=photom_kwargs,
            finder=finder,
            init_params=None,
            error=None,
            mask=mask,
            maxiters=args.iterative_maxiters,
            sub_shape=(15, 15),
            psf_model=psf,
            n_workers=args.n_workers,
            chunk_size=args.chunk_size,
            group_min_separation=args.min_separation,
        )
        dt_iter_par = time.perf_counter() - t0
        print(f"  iter parallel: {dt_iter_par:.2f} s  n={len(iter_par)}")
        print(f"iter speedup = {dt_iter_serial / dt_iter_par:.2f}x")

        # Cross-match: find each parallel source's nearest serial source,
        # report fraction within 0.5 px and median flux ratio.
        from astropy.coordinates import SkyCoord  # not really sky; reuse 2D match
        import astropy.units as u  # noqa
        if len(iter_serial) > 0 and len(iter_par) > 0:
            xs = np.asarray(iter_serial['x_fit'])
            ys = np.asarray(iter_serial['y_fit'])
            xp = np.asarray(iter_par['x_fit'])
            yp = np.asarray(iter_par['y_fit'])
            # naive nearest neighbor
            from scipy.spatial import cKDTree
            tree = cKDTree(np.column_stack([xs, ys]))
            d, idx = tree.query(np.column_stack([xp, yp]), k=1)
            within = d < 0.5
            print(f"--- iter agreement: {within.sum()}/{len(d)} parallel "
                  f"sources within 0.5 px of a serial source "
                  f"(median d = {np.median(d):.3f} px)")
            if within.any():
                fs = np.asarray(iter_serial['flux_fit'])[idx[within]]
                fp = np.asarray(iter_par['flux_fit'])[within]
                ratio = fp / fs
                print(f"  flux ratio (par/ser): "
                      f"median={np.median(ratio):.4f} "
                      f"p16={np.percentile(ratio, 16):.4f} "
                      f"p84={np.percentile(ratio, 84):.4f}")


if __name__ == "__main__":
    # Prevent over-subscription: each worker fork inherits numpy/MKL thread pools.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()
