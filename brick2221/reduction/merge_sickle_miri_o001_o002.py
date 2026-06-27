#!/usr/bin/env python
"""Merge sickle MIRI o001 (POS1) + o002 (POS2) into a single per-filter mosaic
(2026-06-14).

o001 and o002 are adjacent pointings of the same field; both are now correctly
registered to the NIRCam/F480M frame (o001 was re-reduced with shift-only
tweakreg, fixing the F1130W 45-deg rotation and F1500W 6.9" shift).  This
drizzles the per-exposure _align.fits frames of BOTH observations together with
jwst ResampleStep, producing one combined i2d covering both tiles.

o003 is EXCLUDED: it is the Brick background pointing and is handled with the
Brick data, not Sickle.

Usage: python merge_sickle_miri_o001_o002.py [F1130W F1500W ...]
Writes /orange/adamginsburg/jwst/sickle/<FILT>/pipeline/jw03958-o001o002_t001_miri_<filt>_i2d.fits
"""
import os
import sys
import glob
import warnings
warnings.filterwarnings('ignore')
os.environ.setdefault("CRDS_PATH", "/orange/adamginsburg/jwst/brick/crds/")
os.environ.setdefault("CRDS_SERVER_URL", "https://jwst-crds.stsci.edu")

from jwst.resample import ResampleStep
from jwst.datamodels import ModelContainer
from jwst.datamodels import ImageModel

BASE = '/orange/adamginsburg/jwst/sickle'
FILTERS = sys.argv[1:] if len(sys.argv) > 1 else ['F1130W', 'F1500W']


def main():
    for filt in FILTERS:
        pdir = f'{BASE}/{filt}/pipeline'
        # o001 = jw03958001001_*, o002 = jw03958002001_*  (o003 EXCLUDED)
        frames = (sorted(glob.glob(f'{pdir}/jw03958001001_*_mirimage_align.fits'))
                  + sorted(glob.glob(f'{pdir}/jw03958002001_*_mirimage_align.fits')))
        if not frames:
            print(f'{filt}: no o001/o002 align frames found; SKIP', flush=True)
            continue
        print(f'{filt}: merging {len(frames)} frames '
              f'({sum("001001" in f for f in frames)} o001 + '
              f'{sum("002001" in f for f in frames)} o002)', flush=True)
        models = ModelContainer([ImageModel(f) for f in frames])
        out = f'{pdir}/jw03958-o001o002_t001_miri_{filt.lower()}_i2d.fits'
        res = ResampleStep.call(models, output_file=os.path.basename(out),
                                output_dir=pdir, save_results=True)
        print(f'{filt}: wrote {out}', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
