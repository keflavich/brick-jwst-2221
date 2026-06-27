"""
Build the full 1182+2221 UNION multiband catalog from the per-filter module-locked
(VIRAC2-tied) combined catalogs.

Replaces the stale `basic_merged_indivexp_..._ok2221or1182` merge (built on the old
per-detector-tweakreg positions -> within-1182 quiltwork + cross-program offset). All
10 per-filter inputs here are `<filt>_merged_indivexp_LOCKED_dao_basic.fits`, each one
module-locked (one rigid shift per visit, all detectors) and tied to VIRAC2 PM-propagated
to the observation epoch, so they share one absolute astrometric frame (to the ~12 mas
cross-program filter-distortion floor; F182M vs F200W is the worst case).

Method: union cross-match (same logic as merge_catalogs.merge_catalogs but on the already-
combined per-filter products, not per-frame). Start from the reference filter, then for each
other filter add its unmatched (mutual-NN, sep>radius) sources to the master coordinate list.
Then attach every filter's columns matched to the master coords, masked where sep>radius.

Output: catalogs/union_merged_indivexp_LOCKED_ok2221or1182_dao_basic.fits

Per-filter columns: flux_<f>, std_ra_mas_<f>, std_dec_mas_<f>, nframes_<f>, qfit_<f>,
is_saturated_<f>, sep_<f> (arcsec to master), ra_<f>, dec_<f>, det_<f> (bool: detected).
Master: skycoord (master position = ref-filter position where matched, else first filter that
added the source), ndet (number of filters detecting the source), ref_filter (which filter
contributed the master position).
"""
import sys
import numpy as np
from astropy.table import Table, MaskedColumn
from astropy.coordinates import SkyCoord
from astropy import units as u

CD = '/orange/adamginsburg/jwst/brick/catalogs'
# union order: deepest/broadest first so the master frame is anchored on F200W (1182 SW).
ORDER = ['f200w', 'f115w', 'f356w', 'f444w', 'f182m', 'f212n', 'f410m', 'f187n', 'f405n', 'f466n']
RADIUS = 0.10 * u.arcsec
PROP = {'f115w': '1182', 'f200w': '1182', 'f356w': '1182', 'f444w': '1182',
        'f182m': '2221', 'f187n': '2221', 'f212n': '2221',
        'f405n': '2221', 'f410m': '2221', 'f466n': '2221'}
CARRY = ['flux', 'std_ra_mas', 'std_dec_mas', 'nframes', 'qfit', 'is_saturated']


def load(filt):
    t = Table.read(f'{CD}/{filt}_merged_indivexp_LOCKED_dao_basic.fits')
    return t, SkyCoord(t['skycoord'])


def main():
    filts = [f for f in ORDER if __import__('os').path.exists(
        f'{CD}/{f}_merged_indivexp_LOCKED_dao_basic.fits')]
    missing = [f for f in ORDER if f not in filts]
    if missing:
        print(f"WARNING: missing LOCKED catalogs, skipping: {missing}", flush=True)

    tabs = {}
    scs = {}
    for f in filts:
        t, sc = load(f)
        tabs[f] = t
        scs[f] = sc
        print(f"loaded {f}: {len(t)} ({PROP[f]})", flush=True)

    # ---- build master coordinate list (union) ----
    ref = filts[0]
    master = scs[ref]
    origin = np.array([ref] * len(master), dtype='U6')
    print(f"\nmaster start: {ref} ({len(master)})", flush=True)
    for f in filts[1:]:
        sc = scs[f]
        idx, sep, _ = sc.match_to_catalog_sky(master)
        ridx, _, _ = master.match_to_catalog_sky(sc)
        mutual = ridx[idx] == np.arange(len(idx))
        add = (sep > RADIUS) | (~mutual)
        newsc = sc[add]
        master = SkyCoord([master, newsc])
        origin = np.concatenate([origin, np.array([f] * len(newsc), dtype='U6')])
        print(f"  + {f}: added {add.sum()} -> master {len(master)}", flush=True)

    out = Table()
    out['skycoord'] = master
    out['RA'] = master.ra.deg
    out['DEC'] = master.dec.deg
    out['ref_filter'] = origin
    ndet = np.zeros(len(master), int)

    # ---- attach each filter matched to master ----
    for f in filts:
        sc = scs[f]
        t = tabs[f]
        idx, sep, _ = master.match_to_catalog_sky(sc)
        ridx, _, _ = sc.match_to_catalog_sky(master)
        mutual = ridx[idx] == np.arange(len(idx))
        det = (sep < RADIUS) & mutual
        ndet += det.astype(int)
        out[f'det_{f}'] = det
        out[f'sep_{f}'] = sep.to(u.arcsec).value
        out[f'ra_{f}'] = MaskedColumn(sc.ra.deg[idx], mask=~det)
        out[f'dec_{f}'] = MaskedColumn(sc.dec.deg[idx], mask=~det)
        for c in CARRY:
            if c in t.colnames:
                out[f'{c}_{f}'] = MaskedColumn(np.asarray(t[c])[idx], mask=~det)
        print(f"attach {f}: {det.sum()} detections matched to master", flush=True)

    out['ndet'] = ndet
    out.meta['CONTENT'] = 'union 1182+2221 multiband, module-locked VIRAC2-tied per-filter inputs'
    out.meta['INPUTS'] = ','.join(filts)
    out.meta['RADIUS_AS'] = RADIUS.to(u.arcsec).value
    out.meta['REFFRAME'] = 'VIRAC2 PM-propagated to obs epoch (per-filter LOCKED); ~12mas cross-program floor'
    outpath = f'{CD}/union_merged_indivexp_LOCKED_ok2221or1182_dao_basic.fits'
    out.write(outpath, overwrite=True)
    print(f"\nwrote {outpath}: {len(out)} rows, {len(out.colnames)} cols", flush=True)
    for n in range(1, len(filts) + 1):
        print(f"  ndet>={n}: {(ndet >= n).sum()}", flush=True)


if __name__ == '__main__':
    main()
