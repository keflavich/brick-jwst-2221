#!/usr/bin/env python
"""Convert per-frame satstar catalogs (pixel coords) into per-obs sky
ECSVs that make_reference_from_pipeline_catalogs.py will ingest.

Writes one combined ECSV per (filter, obs) pair, suffix `_satstar_cat.ecsv`,
in the same pipeline/ directory as the source catalogs. The mkref glob
`*_cat.ecsv` then picks them up automatically alongside the i2d catalogs.
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS


OBS_RE = re.compile(r"jw\d{5}(\d{3})\d{3}_")


def find_satstar(pipeline_dir: Path, prefer_iter3: bool = True) -> list[Path]:
    """Per-frame satstar catalogs. Prefer iter3 (more complete + refined),
    fall back to regular per frame if no iter3 exists for that crf."""
    out: list[Path] = []
    crfs = sorted(pipeline_dir.glob("jw*_destreak_*_crf.fits"))
    for crf in crfs:
        base = crf.with_suffix("")
        iter3 = Path(f"{base}_iter3_satstar_catalog.fits")
        plain = Path(f"{base}_satstar_catalog.fits")
        if prefer_iter3 and iter3.exists():
            out.append(iter3)
        elif plain.exists():
            out.append(plain)
        elif iter3.exists():
            out.append(iter3)
    return out


def _crf_for_satstar(sat_path: Path) -> Path:
    name = sat_path.name
    name = name.replace("_iter3_satstar_catalog.fits", ".fits")
    name = name.replace("_satstar_catalog.fits", ".fits")
    return sat_path.with_name(name)


def satstar_to_table(sat_path: Path) -> Table | None:
    """Read satstar catalog + matching crf WCS, return sky-position table."""
    crf_path = _crf_for_satstar(sat_path)
    if not crf_path.exists():
        return None
    try:
        with fits.open(crf_path) as h:
            wcs = WCS(h["SCI"].header)
    except Exception:
        return None
    t = Table.read(sat_path)
    if "x_fit" not in t.colnames or "y_fit" not in t.colnames:
        return None
    x = np.asarray(t["x_fit"], dtype=float)
    y = np.asarray(t["y_fit"], dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() == 0:
        return None
    sky = wcs.pixel_to_world(x[ok], y[ok])
    flux_fit = np.asarray(t["flux_fit"], dtype=float)[ok] if "flux_fit" in t.colnames else np.full(ok.sum(), np.nan)
    flux_err = np.asarray(t["flux_err"], dtype=float)[ok] if "flux_err" in t.colnames else np.full(ok.sum(), np.nan)
    qfit = np.asarray(t["qfit"], dtype=float)[ok] if "qfit" in t.colnames else np.full(ok.sum(), np.nan)
    # mkref schema gating requires sky_centroid or RA,DEC AND one of:
    # aper_total_flux / isophotal_flux / segment_flux / flux / source_sum.
    # Provide both `flux` (passes gating) and `aper_total_flux` (used by
    # read_and_normalize for actual brightness ordering).
    # Write `sky_centroid` as a SkyCoord column so it matches the regular
    # _cat.ecsv schema exactly (ICRS frame). Plain RA/DEC cols would be
    # reloaded as FK5 by mkref's _extract_skycoord, causing a vstack frame
    # mismatch.
    out = Table({
        "sky_centroid": sky,
        "flux": flux_fit,
        "aper_total_flux": flux_fit,
        "aper_total_flux_err": flux_err,
        "flux_fit": flux_fit,
        "flux_err": flux_err,
        "qfit": qfit,
    })
    out["aper_total_flux"].unit = u.Jy
    out["aper_total_flux_err"].unit = u.Jy
    out["flux"].unit = u.Jy
    out.meta["frame_filename"] = str(crf_path.name)
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pipeline-dir", required=True,
                   help="Path to <target>/<FILTER>/pipeline directory.")
    p.add_argument("--out-prefix", default=None,
                   help="Output ECSV path prefix (default: <pipeline_dir>/satstar_for_mkref)")
    p.add_argument("--group-by-obs", action="store_true",
                   help="Write one ECSV per obs# (parsed from filename), else one combined.")
    return p.parse_args()


def main():
    args = parse_args()
    pdir = Path(args.pipeline_dir)
    out_prefix = Path(args.out_prefix) if args.out_prefix else (pdir / "satstar_for_mkref")
    paths = find_satstar(pdir)
    print(f"Found {len(paths)} satstar catalogs in {pdir}")

    groups: dict[str, list[Table]] = {}
    for p in paths:
        m = OBS_RE.search(p.name)
        obs = m.group(1) if m else "all"
        tbl = satstar_to_table(p)
        if tbl is None:
            continue
        groups.setdefault(obs if args.group_by_obs else "all", []).append(tbl)

    from astropy.table import vstack
    for key, tables in groups.items():
        big = vstack(tables, join_type="outer")
        if args.group_by_obs:
            out_path = pdir / f"satstar_obs{key}_satstar_cat.ecsv"
        else:
            out_path = pdir / "satstar_combined_satstar_cat.ecsv"
        big.write(out_path, overwrite=True, format="ascii.ecsv")
        print(f"  wrote {out_path}: {len(big)} satstar sources from {len(tables)} frames")


if __name__ == "__main__":
    main()
