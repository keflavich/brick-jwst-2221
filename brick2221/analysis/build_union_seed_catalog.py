#!/usr/bin/env python
"""Build a cross-band union seed catalog for the iter3 photometry pass.

Reads every per-filter iter2 ``daoiterative`` cross-exposure-merged catalog
for the requested target, concatenates them with no quality pre-filter,
clusters detections that fall within a filter-aware radius, and emits a
single sky-unique seed catalog whose authoritative position for each
cluster comes from the **shortest-wavelength** detection available
(SW NIRCam centroids beat LW NIRCam centroids; smaller PSF -> better
position).

Per-frame ``_satstar_catalog.fits`` rows are folded in as well so the
union seed list reproduces the satstar handling end-to-end.

The output catalog is consumed by the iter3 entry point in
``crowdsource_catalogs_long.do_photometry_step`` via
``--seed-catalog=<this file>``.

Schema
------
- ``skycoord``                : authoritative SkyCoord (best-PSF detection)
- ``source_id_union``         : unique integer id per cluster
- ``seed_filter_origin``      : filter whose detection set the position
- ``is_saturated``            : True if any contributing row was a satstar
- ``n_filters``               : number of filters that contributed to the cluster
- ``flux_{filter}``           : flux_fit (or flux) from that filter, NaN if absent
- ``fluxerr_{filter}``        : flux_err from that filter, NaN if absent
- ``qfit_{filter}``           : qfit  from that filter, NaN if absent
- ``detected_{filter}``       : bool, True iff that filter contributed a row
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Column, Table, vstack
from astropy import units as u

# Per-target configuration.
# ``catalog_dir`` holds the per-filter daoiterative cross-exposure merges.
# ``catalog_pattern`` is a glob whose first ``%s`` is the lowercase filter name.
# ``satstar_dir`` holds per-frame ``*_satstar_catalog.fits`` (one per pipeline frame).
TARGETS = {
    'sickle': {
        'catalog_dir': '/orange/adamginsburg/jwst/sickle/catalogs',
        # The current sickle merge runs with --modules=merged so the
        # per-filter cross-exposure merge files are written as
        # ``<filt>_merged_indivexp_merged_daoiterative_iterative.fits``.
        # An older naming used ``_nrcb_`` (when the merge ran with
        # --modules=nrcb); keep both in mind when cleaning stale files.
        'catalog_pattern': '%s_merged_indivexp_merged_daoiterative_iterative.fits',
        'satstar_glob': '/orange/adamginsburg/jwst/sickle/*/pipeline/*destreak_o007_crf_iter2_satstar_catalog.fits',
        'output_path': '/orange/adamginsburg/jwst/sickle/catalogs/seed_union_iter3_sickle.fits',
        # Glob patterns for iter3 residual mosaics used by --residual-peaks.
        # Any peaks detected above the threshold that are not within
        # exclusion_radius of an existing catalog entry are injected as seeds.
        'residual_globs': [
            '/orange/adamginsburg/jwst/sickle/*/pipeline/*iter3_daophot_iterative_residual_i2d.fits',
        ],
    },
    'brick': {
        'catalog_dir': '/blue/adamginsburg/adamginsburg/jwst/brick/catalogs',
        'catalog_pattern': '%s_merged_indivexp_merged_daoiterative_iterative.fits',
        'satstar_glob': '/blue/adamginsburg/adamginsburg/jwst/brick/*/pipeline/*destreak*_iter2_satstar_catalog.fits',
        'output_path': '/blue/adamginsburg/adamginsburg/jwst/brick/catalogs/seed_union_iter3_brick.fits',
    },
    'cloudc': {
        'catalog_dir': '/blue/adamginsburg/adamginsburg/jwst/cloudc/catalogs',
        'catalog_pattern': '%s_merged_indivexp_merged_daoiterative_iterative.fits',
        'satstar_glob': '/blue/adamginsburg/adamginsburg/jwst/cloudc/*/pipeline/*destreak*_iter2_satstar_catalog.fits',
        'output_path': '/blue/adamginsburg/adamginsburg/jwst/cloudc/catalogs/seed_union_iter3_cloudc.fits',
    },
    'sgrb2': {
        'catalog_dir': '/orange/adamginsburg/jwst/sgrb2/catalogs',
        'catalog_pattern': '%s_merged_indivexp_merged_daoiterative_iterative.fits',
        # Two suffix types: destreak (SW narrow/medium) and align (F150W + LW)
        'satstar_glob': '/orange/adamginsburg/jwst/sgrb2/*/pipeline/*_iter2_satstar_catalog.fits',
        'output_path': '/orange/adamginsburg/jwst/sgrb2/catalogs/seed_union_iter3_sgrb2.fits',
    },
}


def load_fwhm_table():
    fwhm_path = (Path(__file__).resolve().parents[1]
                 / 'reduction' / 'fwhm_table.ecsv')
    return Table.read(fwhm_path)


def fwhm_arcsec_for(filtername, fwhm_table):
    row = fwhm_table[fwhm_table['Filter'] == filtername.upper()]
    if len(row) == 0:
        raise KeyError(f'filter {filtername} not found in fwhm_table.ecsv')
    return float(row['PSF FWHM (arcsec)'][0])


def discover_filters(target_cfg):
    pattern = target_cfg['catalog_pattern']
    cat_dir = target_cfg['catalog_dir']
    matches = sorted(glob.glob(os.path.join(cat_dir, pattern % '*')))
    filters = []
    for path in matches:
        base = os.path.basename(path)
        # Filter is the leading f###[xy] token before the first underscore.
        token = base.split('_', 1)[0]
        filters.append(token)
    return filters, matches


def load_per_filter(filtername, path):
    """Return a normalized table (per-row source) for one filter's iter2
    daoiterative cross-exposure merge."""
    t = Table.read(path)
    flo = filtername.lower()

    # Sky position. Some merged catalogs use ``skycoord`` (already a
    # SkyCoord column), others store ``skycoord_*`` separately.
    if 'skycoord' in t.colnames:
        sk = SkyCoord(t['skycoord'])
    elif f'skycoord_{flo}' in t.colnames:
        sk = SkyCoord(t[f'skycoord_{flo}'])
    elif 'skycoord_centroid' in t.colnames:
        sk = SkyCoord(t['skycoord_centroid'])
    else:
        raise KeyError(f"no recognized SkyCoord column in {path}; "
                       f"colnames: {t.colnames}")

    # Per-source flux / err / qfit. Naming differs slightly between
    # cross-exposure merges (``flux``/``flux_err``/``qfit``) and
    # per-frame outputs (``flux_fit``/``flux_err``).
    if 'flux' in t.colnames:
        flux = t['flux']
    elif 'flux_fit' in t.colnames:
        flux = t['flux_fit']
    else:
        flux = np.full(len(t), np.nan)
    flux_err = (t['flux_err'] if 'flux_err' in t.colnames
                else np.full(len(t), np.nan))
    qfit = (t['qfit'] if 'qfit' in t.colnames
            else np.full(len(t), np.nan))
    is_sat = (np.asarray(t['is_saturated'], dtype=bool)
              if 'is_saturated' in t.colnames
              else np.zeros(len(t), dtype=bool))

    out = Table()
    out['ra'] = sk.ra.deg
    out['dec'] = sk.dec.deg
    out['flux'] = np.asarray(flux, dtype=float)
    out['flux_err'] = np.asarray(flux_err, dtype=float)
    out['qfit'] = np.asarray(qfit, dtype=float)
    out['is_saturated'] = is_sat
    out['seed_filter_origin'] = filtername.upper()
    out.meta['filter'] = filtername.upper()
    return out


def load_satstars(satstar_glob):
    """Concatenate every per-frame satstar catalog into one table of sky
    positions tagged with their source filter (read from the FITS header
    if present, else inferred from the path)."""
    files = sorted(glob.glob(satstar_glob))
    rows = []
    for path in files:
        try:
            t = Table.read(path)
        except (OSError, ValueError):
            continue
        if len(t) == 0 or 'skycoord_fit' not in t.colnames:
            continue
        sk = SkyCoord(t['skycoord_fit'])
        # Filter from the parent directory: ``.../<FILTER>/pipeline/...``
        parts = path.split(os.sep)
        try:
            filt = parts[parts.index('pipeline') - 1]
        except ValueError:
            filt = 'UNKNOWN'
        sub = Table()
        sub['ra'] = sk.ra.deg
        sub['dec'] = sk.dec.deg
        sub['flux'] = (np.asarray(t['flux_fit'], dtype=float)
                       if 'flux_fit' in t.colnames
                       else np.full(len(t), np.nan))
        sub['flux_err'] = (np.asarray(t['flux_err'], dtype=float)
                           if 'flux_err' in t.colnames
                           else np.full(len(t), np.nan))
        sub['qfit'] = (np.asarray(t['qfit'], dtype=float)
                       if 'qfit' in t.colnames
                       else np.full(len(t), np.nan))
        sub['is_saturated'] = np.ones(len(t), dtype=bool)
        sub['seed_filter_origin'] = filt.upper()
        rows.append(sub)
    if not rows:
        return None
    return vstack(rows)


def cluster_friends(sk, link_radius_arcsec):
    """Friends-of-friends grouping: any two sources within
    ``link_radius_arcsec`` are members of the same cluster.

    Returns a length-N integer label array (``label[i]`` is the cluster id of
    source ``i``).  Uses a 3D ``scipy.spatial.cKDTree`` on unit vectors and
    :func:`scipy.sparse.csgraph.connected_components` so it scales to
    O(10^7) inputs without a Python-level union-find loop.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    n = len(sk)
    if n == 0:
        return np.zeros(0, dtype=np.int64)

    ra = np.asarray(sk.ra.rad, dtype=np.float64)
    dec = np.asarray(sk.dec.rad, dtype=np.float64)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    xyz = np.stack([x, y, z], axis=1)

    # Chord length on the unit sphere subtended by ``link_radius_arcsec``.
    radius_rad = (link_radius_arcsec * u.arcsec).to(u.rad).value
    chord = 2.0 * np.sin(radius_rad / 2.0)

    tree = cKDTree(xyz)
    pairs = tree.query_pairs(chord, output_type='ndarray')
    if len(pairs) == 0:
        return np.arange(n, dtype=np.int64)
    rows = pairs[:, 0]
    cols = pairs[:, 1]
    data = np.ones(len(rows), dtype=np.int8)
    graph = coo_matrix((data, (rows, cols)), shape=(n, n))
    _, labels = connected_components(graph, directed=False, return_labels=True)
    return labels.astype(np.int64)


# Filter wavelength order for "shortest-wavelength wins" tie-break on the
# authoritative position.  Lower index = shorter wavelength = smaller PSF.
FILTER_ORDER = [
    'F115W', 'F150W', 'F162M', 'F182M', 'F187N', 'F200W', 'F210M', 'F212N',
    'F250M', 'F277W', 'F300M', 'F322W2', 'F323N', 'F335M', 'F356W', 'F360M',
    'F405N', 'F410M', 'F430M', 'F444W', 'F460M', 'F466N', 'F470N', 'F480M',
]


def filter_priority(name):
    name = name.upper()
    return (FILTER_ORDER.index(name) if name in FILTER_ORDER else 999)


def build_union(target):
    cfg = TARGETS[target]
    fwhm_table = load_fwhm_table()
    filters, files = discover_filters(cfg)
    if not files:
        raise SystemExit(
            f'no per-filter daoiterative catalogs found for target {target!r}; '
            f'looked in {cfg["catalog_dir"]} for pattern {cfg["catalog_pattern"]!r}'
        )
    print(f'[{target}] found {len(files)} per-filter catalogs:')
    for f, p in zip(filters, files):
        print(f'    {f.upper():6s}  {p}')

    rows = []
    for f, p in zip(filters, files):
        sub = load_per_filter(f, p)
        print(f'    {f.upper()}: {len(sub):>7d} rows')
        rows.append(sub)
    sat = load_satstars(cfg['satstar_glob'])
    if sat is not None:
        print(f'    satstars: {len(sat):>7d} rows from '
              f'{cfg["satstar_glob"]}')
        rows.append(sat)
    raw = vstack(rows)
    print(f'[{target}] total raw rows = {len(raw)}')

    # Linking radius: the largest FWHM among the contributing filters.
    # 2026-04-24: raised from FWHM/2 to FWHM.  Per-source astrometric
    # scatter on bright stars can be 0.1-0.2", which at FWHM/2 = 0.08"
    # fragmented genuine single-star clusters into 8+ seeds in some
    # dense regions; those seeds then evaded the post-fit dedup (which
    # only catches pairs within 1.5*FWHM in pixel space) and drove
    # deep-negative residuals.  Using the full FWHM brings the FoF
    # link scale in line with the typical per-source scatter at the
    # cost of occasionally merging close pairs; such pairs should
    # reappear as residual peaks in iterative_psf_photometry's
    # DAOStarFinder pass.
    max_fwhm = max(fwhm_arcsec_for(f, fwhm_table) for f in filters) if filters else 0.16
    link_radius_arcsec = max_fwhm
    print(f'[{target}] FoF link radius = {link_radius_arcsec:.4f}" '
          f'(= max FWHM)')

    sk_all = SkyCoord(raw['ra'] * u.deg, raw['dec'] * u.deg)
    labels = cluster_friends(sk_all, link_radius_arcsec)
    n_clusters = labels.max() + 1
    print(f'[{target}] {len(raw)} rows -> {n_clusters} clusters '
          f'(reduction {len(raw) / max(1, n_clusters):.1f}x)')

    # Assemble per-cluster row.
    raw['label'] = labels
    raw['_priority'] = np.array([filter_priority(s) for s in raw['seed_filter_origin']])
    raw_sorted = raw.copy()
    raw_sorted.sort(['label', '_priority'])

    # Take first row per cluster (= shortest-wavelength filter contributing).
    _, first_idx = np.unique(raw_sorted['label'], return_index=True)
    head = raw_sorted[first_idx]
    head.sort('label')

    out = Table()
    out['source_id_union'] = np.arange(len(head), dtype=np.int64)
    # Store sky coords as plain ra/dec floats *and* as an object-dtype
    # ``skycoord_ref`` column holding SkyCoord instances. The latter is
    # what ``_resolve_seed_skycoords`` in crowdsource_catalogs_long.py
    # looks for; naming it 'skycoord_ref' (instead of 'skycoord') means
    # the augmentation routine builds a fresh object-dtype 'skycoord'
    # column at runtime, matching what the per-frame detection_table has
    # so vstack doesn't choke on a SkyCoord-vs-Column type mismatch.
    out['ra'] = head['ra']
    out['dec'] = head['dec']
    sk_head = SkyCoord(head['ra'] * u.deg, head['dec'] * u.deg)
    skycoord_ref = np.empty(len(head), dtype=object)
    for ii, c in enumerate(sk_head):
        skycoord_ref[ii] = c
    out['skycoord_ref'] = skycoord_ref
    out['seed_filter_origin'] = head['seed_filter_origin']

    # Aggregate per-cluster information vectorized: for each filter, find
    # the brightest row per cluster and scatter its flux/err/qfit into the
    # head-indexed column.  Each cluster has at most one row per filter
    # for the cross-exposure merges, but satstars can contribute multiple
    # rows per cluster, so the "max-flux wins" rule is enforced explicitly.
    filt_names = [f.upper() for f in filters]
    n = len(head)
    # Map raw cluster label -> head index.
    head_labels = np.asarray(head['label'], dtype=np.int64)
    label_to_head = np.full(int(head_labels.max()) + 1, -1, dtype=np.int64)
    label_to_head[head_labels] = np.arange(n, dtype=np.int64)

    raw_labels = np.asarray(raw['label'], dtype=np.int64)
    raw_filt = np.asarray(raw['seed_filter_origin'])
    raw_flux = np.asarray(raw['flux'], dtype=float)
    raw_err = np.asarray(raw['flux_err'], dtype=float)
    raw_qfit = np.asarray(raw['qfit'], dtype=float)
    raw_sat = np.asarray(raw['is_saturated'], dtype=bool)

    is_sat_arr = np.zeros(n, dtype=bool)
    head_idx_all = label_to_head[raw_labels]
    np.logical_or.at(is_sat_arr, head_idx_all[raw_sat], True)

    flux_cols = {}
    err_cols = {}
    qfit_cols = {}
    det_cols = {}
    for f in filt_names:
        mask = raw_filt == f
        if not np.any(mask):
            flux_cols[f] = np.full(n, np.nan)
            err_cols[f] = np.full(n, np.nan)
            qfit_cols[f] = np.full(n, np.nan)
            det_cols[f] = np.zeros(n, dtype=bool)
            continue
        f_head = head_idx_all[mask]
        f_flux = raw_flux[mask]
        f_err = raw_err[mask]
        f_qfit = raw_qfit[mask]
        # Stable sort by (head_idx ascending, flux descending) so the
        # first occurrence per head_idx is the brightest row.
        order = np.lexsort((-np.where(np.isfinite(f_flux), f_flux, -np.inf),
                            f_head))
        f_head_s = f_head[order]
        # Pick first occurrence per cluster.
        uniq_head, first = np.unique(f_head_s, return_index=True)
        sel = order[first]
        flux_arr = np.full(n, np.nan)
        err_arr = np.full(n, np.nan)
        qfit_arr = np.full(n, np.nan)
        det_arr = np.zeros(n, dtype=bool)
        flux_arr[uniq_head] = f_flux[sel]
        err_arr[uniq_head] = f_err[sel]
        qfit_arr[uniq_head] = f_qfit[sel]
        det_arr[uniq_head] = True
        flux_cols[f] = flux_arr
        err_cols[f] = err_arr
        qfit_cols[f] = qfit_arr
        det_cols[f] = det_arr

    n_filters_arr = np.sum([det_cols[f].astype(np.int32) for f in filt_names],
                           axis=0).astype(np.int32)

    out['is_saturated'] = is_sat_arr
    out['n_filters'] = n_filters_arr
    for f in filt_names:
        out[f'flux_{f.lower()}'] = flux_cols[f]
        out[f'fluxerr_{f.lower()}'] = err_cols[f]
        out[f'qfit_{f.lower()}'] = qfit_cols[f]
        out[f'detected_{f.lower()}'] = det_cols[f]

    out.meta['target'] = target
    out.meta['filters'] = filt_names
    out.meta['link_radius_arcsec'] = link_radius_arcsec
    out.meta['source'] = (
        'build_union_seed_catalog.py: union over per-filter iter2 '
        'daoiterative cross-exposure merges + per-frame satstar catalogs')
    return out, cfg


def detect_residual_peaks(residual_path, fwhm_pix, threshold_nsigma=10.0):
    """
    Detect bright point-source peaks in one iter3 residual mosaic using
    DAOStarFinder.

    Only peaks above ``threshold_nsigma × sigma_bkg`` are returned; with a
    high threshold (default 10) this suppresses PSF-model imperfections on
    well-subtracted stars and retains only sources that were genuinely missed
    or whose positions were too far from the seed for iter3's xy_bounds to
    reach.

    Parameters
    ----------
    residual_path : str
        Path to the iter3 residual i2d FITS file.
    fwhm_pix : float
        PSF FWHM in pixels for this filter/pixel-scale.
    threshold_nsigma : float
        Detection threshold in units of the sigma-clipped background std.

    Returns
    -------
    Table with columns ``ra``, ``dec``, ``peak_flux``, ``seed_filter_origin``
    or None if no peaks found.
    """
    import warnings
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.stats import sigma_clipped_stats
    from photutils.detection import DAOStarFinder

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        hdul = fits.open(residual_path, memmap=False)
    ext = 'SCI' if 'SCI' in [e.name for e in hdul] else 0
    data = hdul[ext].data.astype(np.float64)
    hdr  = hdul[ext].header
    hdul.close()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        wcs = WCS(hdr, naxis=2)

    finite = data[np.isfinite(data)]
    if finite.size < 100:
        return None
    _, bkg_med, bkg_std = sigma_clipped_stats(finite, sigma=3.0, maxiters=5)
    if bkg_std <= 0:
        return None

    daofind = DAOStarFinder(
        threshold=threshold_nsigma * bkg_std + bkg_med,
        fwhm=fwhm_pix,
        roundness_range=(-0.5, 0.5),
        sharpness_range=(0.2, 1.8),
    )
    sources = daofind(data - bkg_med)
    if sources is None or len(sources) == 0:
        return None

    xcol = 'x_centroid' if 'x_centroid' in sources.colnames else 'xcentroid'
    ycol = 'y_centroid' if 'y_centroid' in sources.colnames else 'ycentroid'
    px = np.array(sources[xcol])
    py = np.array(sources[ycol])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sc = wcs.pixel_to_world(px, py)

    out = Table()
    out['ra']       = sc.ra.deg
    out['dec']      = sc.dec.deg
    out['peak_flux'] = np.array(sources['peak'])
    return out


def append_residual_peak_seeds(out, residual_globs, fwhm_table,
                                threshold_nsigma=10.0,
                                exclusion_radius_arcsec=0.031):
    """
    Detect bright peaks in iter3 residual mosaics and append those not within
    ``exclusion_radius_arcsec`` of any existing catalog entry as new seeds.

    The exclusion radius defaults to the iter3 xy_bounds (±1 SW pixel =
    0.031").  Any peak further than that from its nearest catalog entry is a
    star iter3 could not have fitted — either because it was absent from the
    seed catalog or because its seed position was too far from the true center.

    Both failure modes produce the same observable: a bright peak in the
    residual with no well-positioned existing seed, so one detection pass
    handles both.

    Parameters
    ----------
    out : Table
        Union catalog to extend.
    residual_globs : list[str]
        Glob patterns for iter3 residual i2d mosaics (non-bgsub).
    fwhm_table : Table
        FWHM lookup table from fwhm_table.ecsv.
    threshold_nsigma : float
        Peaks must exceed this multiple of the local background sigma.
    exclusion_radius_arcsec : float
        Peaks within this radius of any existing catalog entry are dropped.

    Returns
    -------
    Table
        Catalog with residual-peak seeds appended.
    """
    all_peaks = []
    for pattern in residual_globs:
        for path in sorted(glob.glob(pattern)):
            if 'bgsub' in os.path.basename(path):
                continue
            # Infer filter from parent directory name
            parts = path.split(os.sep)
            try:
                filt = parts[parts.index('pipeline') - 1].upper()
            except ValueError:
                filt = 'UNKNOWN'
            try:
                fwhm_as = fwhm_arcsec_for(filt, fwhm_table)
            except KeyError:
                fwhm_as = 0.10  # fallback
            # Get pixel scale from filename convention or use a default
            # (SW: 0.031"/px for F187N/F210M; LW: 0.063"/px for the rest)
            is_sw = filt in ('F115W', 'F150W', 'F162M', 'F182M', 'F187N',
                              'F200W', 'F210M', 'F212N')
            pixscale = 0.031 if is_sw else 0.063
            fwhm_pix = fwhm_as / pixscale

            peaks = detect_residual_peaks(path, fwhm_pix=fwhm_pix,
                                          threshold_nsigma=threshold_nsigma)
            if peaks is None:
                print(f'  [residual-peaks] no peaks in {os.path.basename(path)}')
                continue
            peaks['seed_filter_origin'] = filt
            all_peaks.append(peaks)
            print(f'  [residual-peaks] {len(peaks):4d} peaks in '
                  f'{os.path.basename(path)} '
                  f'(σ={threshold_nsigma:.0f}, fwhm={fwhm_pix:.2f}px)')

    if not all_peaks:
        return out

    combined = vstack(all_peaks)
    peaks_sk = SkyCoord(combined['ra'] * u.deg, combined['dec'] * u.deg)

    # Cross-match peaks against existing catalog: drop anything already
    # covered within the exclusion radius.
    cat_sk = SkyCoord(out['ra'] * u.deg, out['dec'] * u.deg)
    _, sep, _ = peaks_sk.match_to_catalog_sky(cat_sk)
    new_mask = sep.arcsec > exclusion_radius_arcsec
    n_dup = (~new_mask).sum()
    if n_dup:
        print(f'  [residual-peaks] {n_dup} peaks within {exclusion_radius_arcsec}" '
              f'of existing entries — dropped as already covered')

    new_peaks = combined[new_mask]
    if len(new_peaks) == 0:
        print('  [residual-peaks] no new positions to add')
        return out

    # Deduplicate peaks against each other (multiple filters may detect the
    # same star) using the same FoF clustering as the main catalog build.
    new_sk = SkyCoord(new_peaks['ra'] * u.deg, new_peaks['dec'] * u.deg)
    labels = cluster_friends(new_sk, exclusion_radius_arcsec)
    _, first_idx = np.unique(labels, return_index=True)
    new_peaks = new_peaks[first_idx]
    new_sk = new_sk[first_idx]

    n_existing = len(out)
    n_new = len(new_peaks)
    filt_names = out.meta.get('filters', [])

    rows = Table()
    rows['source_id_union'] = np.arange(n_existing, n_existing + n_new, dtype=np.int64)
    rows['ra']  = np.array(new_peaks['ra'])
    rows['dec'] = np.array(new_peaks['dec'])
    skycoord_col = np.empty(n_new, dtype=object)
    for ii, c in enumerate(new_sk):
        skycoord_col[ii] = c
    rows['skycoord_ref']       = skycoord_col
    rows['seed_filter_origin'] = np.array(new_peaks['seed_filter_origin'])
    rows['is_saturated']       = np.zeros(n_new, dtype=bool)
    rows['n_filters']          = np.zeros(n_new, dtype=np.int32)
    for f in filt_names:
        fl = f.lower()
        rows[f'flux_{fl}']     = np.full(n_new, np.nan)
        rows[f'fluxerr_{fl}']  = np.full(n_new, np.nan)
        rows[f'qfit_{fl}']     = np.full(n_new, np.nan)
        rows[f'detected_{fl}'] = np.zeros(n_new, dtype=bool)

    combined_out = vstack([out, rows])
    print(f'  [residual-peaks] added {n_new} new seeds '
          f'({n_existing} → {len(combined_out)} total)')
    return combined_out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target', required=True, choices=sorted(TARGETS.keys()))
    p.add_argument('-o', '--output', default=None,
                   help='output FITS path (default: per-target convention)')
    p.add_argument('--dry-run', action='store_true',
                   help='compute clusters but do not write output')
    p.add_argument('--residual-peaks', action='store_true',
                   help='detect bright peaks in iter3 residual mosaics and inject '
                        'unmatched ones as additional seeds (general fix for missed '
                        'and position-biased sources)')
    p.add_argument('--residual-peaks-threshold', type=float, default=10.0,
                   metavar='NSIGMA',
                   help='detection threshold for residual peaks in units of the '
                        'sigma-clipped background std (default: 10)')
    p.add_argument('--residual-peaks-exclusion', type=float, default=0.031,
                   metavar='ARCSEC',
                   help='peaks within this radius of any existing catalog entry are '
                        'considered already-covered and dropped (default: 0.031" = '
                        'iter3 xy_bounds = ±1 SW pixel)')
    p.add_argument('--residual-glob', dest='residual_globs', nargs='+',
                   metavar='GLOB',
                   help='override the residual mosaic glob patterns for --residual-peaks')
    args = p.parse_args(argv)

    out, cfg = build_union(args.target)
    out_path = args.output or cfg['output_path']

    # Residual-peak injection: general fix for missed and position-biased sources.
    if args.residual_peaks:
        fwhm_table = load_fwhm_table()
        res_globs = args.residual_globs or cfg.get('residual_globs', [])
        if not res_globs:
            print('[warn] --residual-peaks requested but no residual_globs configured '
                  'for this target; use --residual-glob to specify patterns')
        else:
            out = append_residual_peak_seeds(
                out, res_globs, fwhm_table,
                threshold_nsigma=args.residual_peaks_threshold,
                exclusion_radius_arcsec=args.residual_peaks_exclusion,
            )

    print(f'\n[{args.target}] resolved {len(out)} unique seed positions')
    print(f'[{args.target}] saturated seeds: {int(out["is_saturated"].sum())}')
    for f in out.meta['filters']:
        n_det = int(out[f'detected_{f.lower()}'].sum())
        print(f'    {f}: detected in {n_det} clusters '
              f'({100 * n_det / len(out):.1f}%)')
    if args.dry_run:
        print('[dry-run] not writing output')
        return 0
    print(f'\nwriting {out_path}')
    out.write(out_path, overwrite=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
