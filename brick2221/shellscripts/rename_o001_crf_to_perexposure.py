"""After re-reducing sickle MIRI o001 (PipelineMIRI), the corrected crf come out
PRODUCT-named (jw03958-o001_t001_miri_<filt>_N_o001_crf.fits) because the asn
product name is set before image3. But cataloging globs PER-EXPOSURE crf
(jw03958001001*mirimage*o001_crf.fits). Map new->per-exposure by EXPSTART (1:1),
archive the stale broken per-exposure crf, copy the corrected ones into place.

Usage: python rename_o001_crf_to_perexposure.py F770W [F1130W F1500W ...]
"""
import sys, glob, os, shutil
from astropy.io import fits

BASE = '/orange/adamginsburg/jwst/sickle'
ARCH = lambda f: os.path.join(os.path.dirname(f), '_broken_rotated_crf')

def expstart(fn):
    h = fits.getheader(fn)
    es = h.get('EXPSTART')
    if es is None and len(fits.open(fn)) > 1:
        es = fits.getheader(fn, 1).get('EXPSTART')
    return round(float(es), 6)

def run(filt):
    pdir = f'{BASE}/{filt}/pipeline'
    new = sorted(glob.glob(f'{pdir}/jw03958-o001_t001_miri_{filt.lower()}_?_o001_crf.fits'))
    old = sorted(glob.glob(f'{pdir}/jw03958001001_*_mirimage_o001_crf.fits'))
    if not new:
        print(f"[{filt}] NO new product-named crf found -- skip (reduction not done?)")
        return
    old_by_es = {expstart(f): f for f in old}
    arch = ARCH(pdir + '/x')
    os.makedirs(arch, exist_ok=True)
    print(f"[{filt}] {len(new)} new crf, {len(old)} stale per-exposure crf")
    for nf in new:
        es = expstart(nf)
        target = old_by_es.get(es)
        if target is None:
            print(f"  !! {os.path.basename(nf)} EXPSTART={es} has NO per-exposure match -- ABORT")
            return
        # archive stale, copy new into per-exposure name
        shutil.move(target, os.path.join(arch, os.path.basename(target)))
        shutil.copy(nf, target)
        print(f"  {os.path.basename(nf)} -> {os.path.basename(target)}  (stale archived)")
    print(f"[{filt}] done. stale crf in {arch}")

if __name__ == '__main__':
    for f in sys.argv[1:]:
        run(f)
