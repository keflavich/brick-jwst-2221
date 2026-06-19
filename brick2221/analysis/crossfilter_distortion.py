"""
Cross-filter distortion correction: force every filter onto F200W's (module-locked) frame with a
SMOOTH 2D residual map, removing the filter-dependent CRDS distortion residual (~12 mas, e.g. F200W
uses distortion_0281, F182M uses 0185) that the old per-detector tweakreg used to mask.

Method (catalog-level, JWST-internal -> tight, no VVV/blend ambiguity, same physical stars):
  ref = F200W module-locked combined catalog (already VIRAC2-bulk-tied).
  For each other filter's module-locked catalog:
    1. mutual-NN match to F200W (0.1"), sigma-clip the offset cloud
    2. bin the (dRA,dDec) on a coarse sky grid (robust median per cell), Gaussian-smooth, interpolate
    3. subtract the smooth map from the filter's positions -> on F200W's frame
  Result: 115/200/182/... mutually consistent (no quiltwork, no cross-filter distortion residual);
  the common frame inherits F200W's VIRAC2 bulk tie for the absolute zero-point.

Input:  catalogs/<filt>_merged_indivexp_LOCKED_dao_basic.fits
Output: catalogs/<filt>_merged_indivexp_XFILT_dao_basic.fits   (F200W copied through unchanged)
"""
import sys
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.stats import mad_std
from scipy.stats import binned_statistic_2d
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
import warnings
warnings.filterwarnings('ignore')

CD = '/orange/adamginsburg/jwst/brick/catalogs'
REFFILT = 'f200w'
CELL_ARCSEC = 12.0           # grid cell for the smooth map
SMOOTH_CELLS = 1.5           # gaussian smoothing (cells)
RA0, DEC0 = 266.53, -28.70
COSD = np.cos(np.radians(DEC0))


def load(filt, locked=True):
    tag = '_LOCKED' if locked else ''
    t = Table.read(f'{CD}/{filt}_merged_indivexp{tag}_dao_basic.fits')
    return t, SkyCoord(t['skycoord'])


def bright_sc(t, sc, pct=70):
    """sparse bright subset (flux above pct percentile) -> clean cross-filter NN matches."""
    fl = np.asarray(t['flux'], float)
    m = np.isfinite(fl) & (fl > np.nanpercentile(fl, pct))
    return sc[m]


def smooth_map(ra, dec, val, redges, dedges):
    stat, _, _, _ = binned_statistic_2d(ra, dec, val, statistic='median', bins=[redges, dedges])
    # fill empty cells by nearest-finite via iterative gaussian of a filled copy
    filled = np.where(np.isfinite(stat), stat, 0.0)
    weight = np.isfinite(stat).astype(float)
    fs = gaussian_filter(filled, SMOOTH_CELLS); ws = gaussian_filter(weight, SMOOTH_CELLS)
    sm = fs / np.maximum(ws, 1e-6)
    rc = 0.5 * (redges[:-1] + redges[1:]); dc = 0.5 * (dedges[:-1] + dedges[1:])
    return RegularGridInterpolator((rc, dc), sm, bounds_error=False, fill_value=None)


def correct_filter(filt, ref_sc, ref_bright):
    t, sc = load(filt)
    if filt == REFFILT:
        t.write(f'{CD}/{filt}_merged_indivexp_XFILT_dao_basic.fits', overwrite=True)
        print(f"  {filt}: reference frame, copied unchanged ({len(t)})")
        return
    # build the smooth map from BRIGHT stars only, MUTUAL NN + ISOLATION (reject ambiguous pairs
    # where a 2nd ref star is comparably close) -> clean cross-filter offsets even at ~12 mas in
    # the crowded GC.
    sc_b = bright_sc(t, sc)
    idx, sep, _ = sc_b.match_to_catalog_sky(ref_bright)
    ridx, _, _ = ref_bright.match_to_catalog_sky(sc_b)
    mutual = ridx[idx] == np.arange(len(idx))
    _, sep2, _ = sc_b.match_to_catalog_sky(ref_bright, nthneighbor=2)
    isolated = sep2 > 0.25 * u.arcsec               # no 2nd ref within 250 mas -> unambiguous
    m = mutual & (sep < 0.12 * u.arcsec) & isolated
    a = sc_b[m]; b = ref_bright[idx[m]]
    dra = ((b.ra - a.ra) * np.cos(a.dec.radian)).to(u.mas).value     # ref - filt
    ddec = (b.dec - a.dec).to(u.mas).value
    # sigma-clip the matched cloud (drop crowding mismatches)
    keep = np.ones(len(dra), bool)
    for _ in range(3):
        mr, sr = np.median(dra[keep]), mad_std(dra[keep]); md, sd = np.median(ddec[keep]), mad_std(ddec[keep])
        keep = (np.abs(dra - mr) < 4 * sr) & (np.abs(ddec - md) < 4 * sd)
    ra = a.ra.deg[keep]; dec = a.dec.deg[keep]
    redges = np.arange(ra.min(), ra.max() + 1e-6, CELL_ARCSEC / 3600. / COSD)
    dedges = np.arange(dec.min(), dec.max() + 1e-6, CELL_ARCSEC / 3600.)
    fra = smooth_map(ra, dec, dra[keep], redges, dedges)
    fde = smooth_map(ra, dec, ddec[keep], redges, dedges)
    # apply smooth correction (add ref-filt offset) to ALL of this filter's sources
    allra = sc.ra.deg; alldec = sc.dec.deg
    pts = np.column_stack([np.clip(allra, redges[0], redges[-1]), np.clip(alldec, dedges[0], dedges[-1])])
    cra = fra(pts); cde = fde(pts)
    new_ra = allra + (cra / 3.6e6) / np.cos(np.radians(alldec))
    new_dec = alldec + cde / 3.6e6
    t['RA'] = new_ra; t['DEC'] = new_dec; t['skycoord'] = SkyCoord(new_ra * u.deg, new_dec * u.deg)
    t.meta['XFILT'] = f'cross-filter smooth-2D distortion correction onto {REFFILT} frame'
    t.write(f'{CD}/{filt}_merged_indivexp_XFILT_dao_basic.fits', overwrite=True)
    # verify
    sc2 = SkyCoord(new_ra * u.deg, new_dec * u.deg)
    idx2, sep2, _ = sc2.match_to_catalog_sky(ref_sc); m2 = sep2 < 0.08 * u.arcsec
    A = sc2[m2]; B = ref_sc[idx2[m2]]
    r = ((B.ra - A.ra) * np.cos(A.dec.radian)).to(u.mas).value; d = (B.dec - A.dec).to(u.mas).value
    rb = np.linspace(np.percentile(A.ra.deg, 1), np.percentile(A.ra.deg, 99), 45)
    db = np.linspace(np.percentile(A.dec.deg, 1), np.percentile(A.dec.deg, 99), 45)
    qr = np.nanstd(binned_statistic_2d(A.ra.deg, A.dec.deg, r, 'median', bins=[rb, db])[0])
    qd = np.nanstd(binned_statistic_2d(A.ra.deg, A.dec.deg, d, 'median', bins=[rb, db])[0])
    print(f"  {filt} -> F200W: corrected MAD=({mad_std(r):.2f},{mad_std(d):.2f}) QUILT=({qr:.2f},{qd:.2f}) mas (N={m2.sum()})")


def main():
    filts = sys.argv[1:] or ['f115w', 'f182m']
    tref, ref_sc = load(REFFILT)
    ref_bright = bright_sc(tref, ref_sc)
    print(f"ref = {REFFILT} LOCKED ({len(ref_sc)}; bright {len(ref_bright)})")
    correct_filter(REFFILT, ref_sc, ref_bright)
    for f in filts:
        if f == REFFILT:
            continue
        correct_filter(f, ref_sc, ref_bright)


if __name__ == '__main__':
    main()
