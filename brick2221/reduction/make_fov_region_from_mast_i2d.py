#!/usr/bin/env python
"""Create a target FOV region from MAST i2d products.

This script writes regions_/nircam_<target>_fov.reg as a sky rectangle that
fully encloses the union of WCS footprints from matching NIRCam i2d files.

Typical usage:
    python make_fov_region_from_mast_i2d.py --target cloudef --proposal-id 2092 --field 005
"""

from __future__ import annotations

import argparse
from glob import glob
from pathlib import Path

import astropy.units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import regions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build target FOV region from MAST i2d footprints")
    parser.add_argument("--target", required=True, help="Target name (e.g. cloudef, sgrc, sgrb2)")
    parser.add_argument("--proposal-id", required=True, help="JWST proposal id (e.g. 2092)")
    parser.add_argument("--field", required=True, help="Field id without o-prefix (e.g. 005)")
    parser.add_argument(
        "--basepath",
        default=None,
        help="Target root path. Defaults to /orange/adamginsburg/jwst/{target}",
    )
    parser.add_argument(
        "--mast-root",
        default=None,
        help="MAST root path containing JWST directory. Defaults to {basepath}/mastDownload",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output region path. Defaults to {basepath}/regions_/nircam_{target}_fov.reg",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing output file")
    return parser.parse_args()


def default_basepath(target: str) -> Path:
    return Path(f"/orange/adamginsburg/jwst/{target}")


def default_mast_root(basepath: Path, target: str) -> Path:
    direct = basepath / "mastDownload"
    if direct.exists():
        return direct
    if target in ("arches", "quintuplet"):
        alt = Path("/orange/adamginsburg/jwst/arches_quintuplet/mastDownload")
        if alt.exists():
            return alt
    return direct


def find_i2d_files(mast_root: Path, proposal_id: str, field: str) -> list[Path]:
    patterns = [
        str(mast_root / "JWST" / f"jw0{proposal_id}-o{field}_t*_nircam*" / "*_i2d.fits"),
        str(mast_root / "JWST" / f"jw0{proposal_id}{field}*" / "*_i2d.fits"),
    ]

    files: set[Path] = set()
    for pattern in patterns:
        for filename in glob(pattern):
            path = Path(filename)
            if path.is_file():
                files.add(path)

    return sorted(files)


def footprint_from_file(path: Path) -> np.ndarray:
    with fits.open(path) as hdul:
        if len(hdul) < 2:
            raise ValueError(f"{path} does not have extension 1 with WCS")
        wcs = WCS(hdul[1].header)
        # calc_footprint returns Nx2 in world coordinates (deg)
        footprint = wcs.calc_footprint()
    if footprint.shape[1] != 2:
        raise ValueError(f"Unexpected footprint shape for {path}: {footprint.shape}")
    return footprint


def build_enclosing_rectangle(footprints: list[np.ndarray]) -> regions.RectangleSkyRegion:
    points = np.vstack(footprints)
    ra = Angle(points[:, 0] * u.deg).wrap_at(180 * u.deg)
    dec = Angle(points[:, 1] * u.deg)

    ra_min = np.nanmin(ra.deg)
    ra_max = np.nanmax(ra.deg)
    dec_min = np.nanmin(dec.deg)
    dec_max = np.nanmax(dec.deg)

    center_ra_wrapped = 0.5 * (ra_min + ra_max)
    center_ra = Angle(center_ra_wrapped * u.deg).wrap_at(360 * u.deg)
    center_dec = 0.5 * (dec_min + dec_max) * u.deg

    width = (ra_max - ra_min) * u.deg
    height = (dec_max - dec_min) * u.deg

    center = SkyCoord(ra=center_ra, dec=center_dec, frame="fk5")
    return regions.RectangleSkyRegion(center=center, width=width, height=height, angle=0 * u.deg)


def main() -> None:
    args = parse_args()

    basepath = Path(args.basepath) if args.basepath else default_basepath(args.target)
    mast_root = Path(args.mast_root) if args.mast_root else default_mast_root(basepath, args.target)

    output = Path(args.output) if args.output else (basepath / "regions_" / f"nircam_{args.target}_fov.reg")
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists() and not args.force:
        print(f"FOV region already exists, skipping: {output}")
        return

    i2d_files = find_i2d_files(mast_root, args.proposal_id, args.field)
    if len(i2d_files) == 0:
        raise FileNotFoundError(
            f"No matching NIRCam i2d files found under {mast_root} for proposal={args.proposal_id} field={args.field}"
        )

    footprints = [footprint_from_file(path) for path in i2d_files]
    rect = build_enclosing_rectangle(footprints)

    regions.Regions([rect]).write(output, format="ds9", overwrite=True)
    print(f"Wrote FOV region: {output}")
    print(f"Inputs used ({len(i2d_files)} files):")
    for path in i2d_files:
        print(f"  {path}")


if __name__ == "__main__":
    main()
