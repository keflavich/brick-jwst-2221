"""Standalone SED generator for MIRI F1500W sources — satstar photometry primary."""
import sys
sys.stdout.reconfigure(line_buffering=True)

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from astropy.table import Table
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.visualization import simple_norm
import astropy.units as u
from reproject import reproject_interp
import warnings
import pandas as pd
from PIL import Image, PngImagePlugin
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
BRICK_BASE   = Path('/blue/adamginsburg/adamginsburg/jwst/brick')
SICKLE_BASE  = Path('/orange/adamginsburg/jwst/sickle')
BRICK_ORANGE = Path('/orange/adamginsburg/jwst/brick')
IMGS_DIR     = BRICK_BASE / 'images'

MIRI_CAT_FILE   = SICKLE_BASE / 'catalogs' / 'o003_miri_cmd_matched.fits'
NIRCAM_CAT_FILE = BRICK_ORANGE / 'catalogs' / 'basic_merged_indivexp_photometry_tables_merged_resbgsub_m8_dedup.fits'
SESHAT_CSV      = BRICK_BASE / 'seshat_results_f1500w_all.csv'
SATSTAR_CAT_DIR = BRICK_BASE / 'catalogs'
OUT_DIR = BRICK_BASE / 'sed_figures_f1500w_all'
OUT_DIR.mkdir(exist_ok=True)

IMAGE_FILES = {
    # 1182 NIRCam: use the current drizzled pipeline mosaics, NOT the stale
    # images/ copies -- those predate the astrometry corrections, so catalog
    # positions land ~0.1" off the stars in the cutouts.
    'f115w': BRICK_ORANGE / 'F115W' / 'pipeline' / 'jw01182-o004_t001_nircam_clear-f115w-merged_i2d.fits',
    'f200w': BRICK_ORANGE / 'F200W' / 'pipeline' / 'jw01182-o004_t001_nircam_clear-f200w-merged_i2d.fits',
    'f356w': BRICK_ORANGE / 'F356W' / 'pipeline' / 'jw01182-o004_t001_nircam_clear-f356w-merged_i2d.fits',
    'f444w': BRICK_ORANGE / 'F444W' / 'pipeline' / 'jw01182-o004_t001_nircam_clear-f444w-merged_i2d.fits',
    'f182m': BRICK_ORANGE / 'F182M' / 'pipeline' / 'jw02221-o001_t001_nircam_clear-f182m-merged_i2d.fits',
    'f187n': BRICK_ORANGE / 'F187N' / 'pipeline' / 'jw02221-o001_t001_nircam_clear-f187n-merged_i2d.fits',
    'f212n': BRICK_ORANGE / 'F212N' / 'pipeline' / 'jw02221-o001_t001_nircam_clear-f212n-merged_i2d.fits',
    'f405n': BRICK_ORANGE / 'F405N' / 'pipeline' / 'jw02221-o001_t001_nircam_clear-f405n-merged_i2d.fits',
    'f410m': BRICK_ORANGE / 'F410M' / 'pipeline' / 'jw02221-o001_t001_nircam_clear-f410m-merged_i2d.fits',
    'f466n': BRICK_ORANGE / 'F466N' / 'pipeline' / 'jw02221-o001_t001_nircam_clear-f466n-merged_i2d.fits',
    'f770w':  SICKLE_BASE / 'F770W/pipeline/jw03958-o003_t001_miri_f770w_i2d.fits',
    'f1130w': SICKLE_BASE / 'F1130W/pipeline/jw03958-o003_t001_miri_f1130w_i2d.fits',
    'f1500w': SICKLE_BASE / 'F1500W/pipeline/jw03958-o003_t001_miri_f1500w_i2d.fits',
    'f2550w': BRICK_ORANGE / 'F2550W/pipeline/jw02221-o002_t001_miri_f2550w_i2d.fits',
}

MIRI_VEGA_ZP_JY = {'f770w': 64.13, 'f1130w': 29.40, 'f1500w': 17.60, 'f2550w': 6.72}
WAVELENGTHS_UM = {
    'f115w': 1.154, 'f182m': 1.845, 'f187n': 1.874, 'f200w': 1.989, 'f212n': 2.121,
    'f356w': 3.563, 'f405n': 4.052, 'f410m': 4.082, 'f444w': 4.421, 'f466n': 4.654,
    'f770w': 7.639, 'f1130w': 11.309, 'f1500w': 15.065, 'f2550w': 25.363,
}
NIRCAM_FILTERS = ['f115w','f182m','f187n','f200w','f212n','f356w','f405n','f410m','f444w','f466n']
CUTOUT_FILTERS = [
    'f115w','f182m','f187n','f200w','f212n',
    'f356w','f405n','f410m','f444w','f466n',
    'f770w','f1130w','f1500w','f2550w',
]
N_COLS = 5
MIRI_FILTS = set(MIRI_VEGA_ZP_JY.keys())
PHOT_APER_ARCSEC = {
    'f115w': 0.15, 'f182m': 0.15, 'f187n': 0.15, 'f200w': 0.15, 'f212n': 0.15,
    'f356w': 0.25, 'f405n': 0.25, 'f410m': 0.25, 'f444w': 0.25, 'f466n': 0.25,
    'f770w': 0.50, 'f1130w': 0.75, 'f1500w': 1.00, 'f2550w': 1.50,
}
SATSTAR_FALLBACK = {
    'f182m': ('flux_jy_182m187', WAVELENGTHS_UM['f182m']),
    'f187n': ('flux_jy_187m182', WAVELENGTHS_UM['f187n']),
    'f405n': ('flux_jy_405m410', WAVELENGTHS_UM['f405n']),
    'f410m': ('flux_jy_410m405', WAVELENGTHS_UM['f410m']),
}
CUTOUT_SIZE = 5.0   # arcsec displayed
CUTOUT_PAD  = 8.0   # arcsec padded cutout
_SATSTAR_RADIUS_ARCSEC = 0.5
XMATCH_RADIUS = 0.5 * u.arcsec

# ── Load MIRI catalog ─────────────────────────────────────────────────────────
print('Loading MIRI catalog...')
miri_cat = Table.read(str(MIRI_CAT_FILE))
for col in list(miri_cat.colnames):
    if col != col.lower():
        miri_cat.rename_column(col, col.lower())

def _to_float(val):
    if val is None: return np.nan
    if hasattr(val, 'mask') and val.mask: return np.nan
    try:
        v = float(val)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError): return np.nan

mag_f15 = np.array([_to_float(v) for v in miri_cat['mag_f1500w']])
order   = np.argsort(mag_f15)
sources     = miri_cat[order]
source_mags = mag_f15[order]
print(f'F1500W sources: {len(sources)}')

source_coords = SkyCoord(
    np.array([_to_float(r['ra'])  for r in sources]) * u.deg,
    np.array([_to_float(r['dec']) for r in sources]) * u.deg,
)

# ── Load NIRCam catalog ───────────────────────────────────────────────────────
print('Loading NIRCam catalog...')
_RA_MIN, _RA_MAX   = 266.35, 266.75
_DEC_MIN, _DEC_MAX = -28.85, -28.55
with fits.open(str(NIRCAM_CAT_FILE), memmap=True) as _hdul:
    _raw_ra  = np.array(_hdul[1].data['skycoord_ref.ra'])
    _raw_dec = np.array(_hdul[1].data['skycoord_ref.dec'])
    _bmask   = ((_raw_ra >= _RA_MIN) & (_raw_ra <= _RA_MAX) &
                (_raw_dec >= _DEC_MIN) & (_raw_dec <= _DEC_MAX))
    nircam_cat = Table(_hdul[1].data[_bmask])
_NC_RA_COL, _NC_DEC_COL = 'skycoord_ref.ra', 'skycoord_ref.dec'
nircam_coords = SkyCoord(nircam_cat[_NC_RA_COL].astype(float) * u.deg,
                          nircam_cat[_NC_DEC_COL].astype(float) * u.deg)
print(f'NIRCam loaded: {len(nircam_cat):,} rows')

idx_nc, sep_nc, _ = match_coordinates_sky(source_coords, nircam_coords)
has_nircam = sep_nc < XMATCH_RADIUS
print(f'NIRCam matches: {has_nircam.sum()} / {len(sources)}')

# ── Load SESHAT results ───────────────────────────────────────────────────────
print('Loading SESHAT results...')
seshat_result = pd.read_csv(str(SESHAT_CSV))
seshat_classes = [c.replace('Prob ', '') for c in seshat_result.columns if c.startswith('Prob ')]
print(f'SESHAT: {len(seshat_result)} rows, classes={seshat_classes}')

# ── Load satstar catalogs ─────────────────────────────────────────────────────
print('Loading satstar catalogs...')
SATSTAR_CATS, SATSTAR_COORDS, SATSTAR_PIX_SR = {}, {}, {}
for _filt in NIRCAM_FILTERS + ['f2550w']:
    _path = SATSTAR_CAT_DIR / f'{_filt}_consolidated_satstar_catalog.fits'
    if not _path.exists():
        continue
    _cat = Table.read(str(_path))
    SATSTAR_CATS[_filt] = _cat
    _sc = _cat['skycoord_fit']
    SATSTAR_COORDS[_filt] = SkyCoord(_sc.ra.deg * u.deg, _sc.dec.deg * u.deg)
    _img = IMAGE_FILES.get(_filt)
    if _img is None or not _img.exists():
        continue
    _pixar = None
    with fits.open(str(_img)) as _hdul:
        for _ext in [('SCI', 1), 1, 0]:
            try:
                _pixar = _hdul[_ext].header.get('PIXAR_SR')
                if _pixar is not None: break
            except (KeyError, IndexError):
                pass
        if _pixar is None:
            for _ext in [1, 0]:
                try:
                    _ww = WCS(_hdul[_ext].header)
                    _pixar = float(_ww.proj_plane_pixel_area().to(u.sr).value)
                    break
                except (KeyError, IndexError, ValueError):
                    pass
    if _pixar is not None:
        SATSTAR_PIX_SR[_filt] = float(_pixar)
    print(f'  {_filt}: {len(_cat)} satstar, PIXAR_SR={SATSTAR_PIX_SR.get(_filt, float("nan")):.4e}')

# ── Load images ───────────────────────────────────────────────────────────────
print('Loading images...')
image_data = {}
for filt, path in IMAGE_FILES.items():
    if not path.exists():
        print(f'  {filt}: missing')
        continue
    try:
        with fits.open(str(path), memmap=True) as hdul:
            for ext in [('SCI', 1), 1, 0]:
                try:
                    arr = hdul[ext].data
                    wcs = WCS(hdul[ext].header)
                    if arr is not None:
                        image_data[filt] = (arr.astype(float), wcs)
                        print(f'  Loaded {filt}: {arr.shape}')
                        break
                except (KeyError, IndexError, ValueError):
                    pass
    except (OSError, fits.VerifyError) as e:
        print(f'  {filt}: {e}')

# ── Helpers ───────────────────────────────────────────────────────────────────
def vega_mag_to_jy(mag, filt):
    zp = MIRI_VEGA_ZP_JY.get(filt.lower())
    m = _to_float(mag)
    if zp is None or not np.isfinite(m): return np.nan
    return zp * 10.0 ** (-m / 2.5)

def _make_northup_wcs(coord, pixscale_deg, npix):
    w = WCS(naxis=2)
    w.wcs.crpix = [npix / 2 + 0.5, npix / 2 + 0.5]
    w.wcs.cdelt = [-pixscale_deg, pixscale_deg]
    w.wcs.crval = [float(coord.ra.deg), float(coord.dec.deg)]
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    return w

def get_cutout(coord, filt):
    if filt not in image_data:
        return None, None
    data, wcs = image_data[filt]
    try:
        pm = wcs.pixel_scale_matrix
        ps_deg = float(np.sqrt(abs(np.linalg.det(pm))))
        ps_arcsec = ps_deg * 3600.0
        pad_pix = int(CUTOUT_PAD / ps_arcsec)
        cut = Cutout2D(data, coord, pad_pix, wcs=wcs, mode='partial', fill_value=np.nan)
        if np.all(~np.isfinite(cut.data)):
            return None, None
        npix = max(5, int(round(CUTOUT_SIZE / ps_arcsec)))
        target_wcs = _make_northup_wcs(coord, ps_deg, npix)
        arr, footprint = reproject_interp(
            (cut.data, cut.wcs), target_wcs, shape_out=(npix, npix), order='bilinear')
        arr = np.array(arr, dtype=float)
        arr[footprint < 0.5] = np.nan
        if np.all(~np.isfinite(arr)) or np.nansum(np.abs(arr)) == 0:
            return None, None
        return arr, ps_arcsec
    except (ValueError,):
        return None, None

def measure_flux_image(coord, filt):
    arr, ps_arcsec = get_cutout(coord, filt)
    if arr is None: return np.nan
    ap_pix = PHOT_APER_ARCSEC.get(filt, 0.5) / ps_arcsec
    ny, nx = arr.shape
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    yy, xx = np.mgrid[:ny, :nx].astype(float)
    r = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    ap_vals = arr[(r <= ap_pix) & np.isfinite(arr)]
    bg_vals = arr[(r > ap_pix*2) & (r <= ap_pix*3) & np.isfinite(arr)]
    if len(ap_vals) < 3: return np.nan
    bg = float(np.nanmedian(bg_vals)) if len(bg_vals) >= 3 else 0.0
    net = float(np.sum(ap_vals - bg))
    ps_rad = ps_arcsec * np.pi / (180.0 * 3600.0)
    flux_jy = net * ps_rad**2 * 1e6
    return float(flux_jy) if np.isfinite(flux_jy) and flux_jy > 0 else np.nan

def get_satstar_flux(filt, coord):
    if filt not in SATSTAR_COORDS or filt not in SATSTAR_PIX_SR:
        return np.nan, np.nan
    idx, sep, _ = coord.match_to_catalog_sky(SATSTAR_COORDS[filt])
    if float(np.asarray(sep.arcsec).flat[0]) > _SATSTAR_RADIUS_ARCSEC:
        return np.nan, np.nan
    cat = SATSTAR_CATS[filt]
    row = cat[int(np.asarray(idx).flat[0])]
    pixar_sr = SATSTAR_PIX_SR[filt]
    flux_jy = float(row['flux_fit']) * pixar_sr * 1e6
    if not (np.isfinite(flux_jy) and flux_jy > 0): return np.nan, np.nan
    ec = ('flux_err' if 'flux_err' in cat.colnames
          else 'flux_unc' if 'flux_unc' in cat.colnames else None)
    eflux = float(row[ec]) * pixar_sr * 1e6 if ec is not None else np.nan
    return float(flux_jy), float(eflux)

def get_nircam_coord(i):
    if not has_nircam[i]: return None
    nc_row = nircam_cat[int(idx_nc[i])]
    try:
        return SkyCoord(float(nc_row[_NC_RA_COL]) * u.deg,
                        float(nc_row[_NC_DEC_COL]) * u.deg)
    except (KeyError, TypeError, ValueError):
        return None

def get_source_fluxes(i):
    row           = sources[i]
    miri_coord    = source_coords[i]
    nc_coord      = get_nircam_coord(i)
    fluxes        = {}
    nc_phot_coord = nc_coord if nc_coord is not None else miri_coord

    for filt in ['f770w','f1130w','f1500w']:
        col = f'mag_{filt}'
        if col not in miri_cat.colnames: continue
        f = vega_mag_to_jy(_to_float(row[col]), filt)
        if np.isfinite(f) and f > 0:
            fluxes[filt] = (WAVELENGTHS_UM[filt], f, np.nan, False)

    sat_f, sat_ef = get_satstar_flux('f2550w', miri_coord)
    if np.isfinite(sat_f):
        fluxes['f2550w'] = (WAVELENGTHS_UM['f2550w'], sat_f, sat_ef, False)
    else:
        for _col in ['mag_f2550w','mag_f2550w_forced']:
            if _col in miri_cat.colnames:
                f = vega_mag_to_jy(_to_float(row[_col]), 'f2550w')
                if np.isfinite(f) and f > 0:
                    fluxes['f2550w'] = (WAVELENGTHS_UM['f2550w'], f, np.nan, False)
                    break
        if 'f2550w' not in fluxes:
            f = measure_flux_image(miri_coord, 'f2550w')
            if np.isfinite(f):
                fluxes['f2550w'] = (WAVELENGTHS_UM['f2550w'], f, np.nan, True)

    nc_row = nircam_cat[int(idx_nc[i])] if has_nircam[i] else None
    for filt in NIRCAM_FILTERS:
        sat_f, sat_ef = get_satstar_flux(filt, miri_coord)
        if np.isfinite(sat_f):
            fluxes[filt] = (WAVELENGTHS_UM[filt], sat_f, sat_ef, False)
            continue
        if nc_row is not None:
            fcol = f'flux_jy_{filt}'
            ecol = f'eflux_jy_{filt}'
            f  = _to_float(nc_row[fcol]) if fcol in nircam_cat.colnames else np.nan
            ef = _to_float(nc_row[ecol]) if ecol in nircam_cat.colnames else np.nan
            if np.isfinite(f) and f > 0:
                fluxes[filt] = (WAVELENGTHS_UM[filt], f, ef, False)
                continue
            fb = SATSTAR_FALLBACK.get(filt)
            if fb is not None and fb[0] in nircam_cat.colnames:
                f2 = _to_float(nc_row[fb[0]])
                if np.isfinite(f2) and f2 > 0:
                    fluxes[filt] = (fb[1], f2, np.nan, False)
                    continue
        f = measure_flux_image(nc_phot_coord, filt)
        if np.isfinite(f):
            fluxes[filt] = (WAVELENGTHS_UM[filt], f, np.nan, True)

    return dict(sorted(fluxes.items(), key=lambda kv: kv[1][0]))

# ── Plot function (matches notebook format) ───────────────────────────────────
def plot_source_sed(i_src):
    miri_coord = source_coords[i_src]
    nc_coord   = get_nircam_coord(i_src)
    rank       = i_src + 1
    mag15      = source_mags[i_src]
    fluxes     = get_source_fluxes(i_src)

    n_cat = sum(1 for v in fluxes.values() if not v[3])
    n_img = sum(1 for v in fluxes.values() if     v[3])

    sr       = seshat_result.iloc[i_src]
    cls      = str(sr['Predicted_Class'])
    cls_prob = float(sr[f'Prob {cls}'])

    fig = plt.figure(figsize=(14, 11))
    fig.suptitle(
        f'Rank {rank}  |  RA={miri_coord.ra.deg:.5f}  Dec={miri_coord.dec.deg:.5f}  '
        f'|  F1500W={mag15:.3f} Vega  '
        f'|  SESHAT: {cls} ({cls_prob:.0%})  '
        f'|  {n_cat} cat + {n_img} img-phot',
        fontsize=10, y=1.00,
    )

    gs = gridspec.GridSpec(4, N_COLS,
                           height_ratios=[1, 1, 1, 1.4],
                           hspace=0.35, wspace=0.08,
                           top=0.96, bottom=0.06, left=0.04, right=0.98)

    for idx, filt in enumerate(CUTOUT_FILTERS):
        row_g = idx // N_COLS
        col_g = idx %  N_COLS
        ax    = fig.add_subplot(gs[row_g, col_g])

        coord = miri_coord if filt in MIRI_FILTS else (nc_coord if nc_coord is not None else miri_coord)
        arr, pix_scale = get_cutout(coord, filt)

        if arr is not None:
            finite = arr[np.isfinite(arr)]
            norm = simple_norm(finite, stretch='log', percent=99.5) if (finite.size > 0 and finite.max() > finite.min()) else None
            ax.imshow(arr, norm=norm, origin='lower', cmap='inferno')
            cx_s, cy_s = arr.shape[1] / 2, arr.shape[0] / 2
            ap_pix = PHOT_APER_ARCSEC.get(filt, 0.5) / pix_scale
            ax.add_patch(plt.Circle((cx_s, cy_s), ap_pix, color='cyan', fill=False, lw=0.5))
            _gap = ap_pix + 2; _arm = max(3, ap_pix * 0.4)
            ax.plot([cx_s - _gap - _arm, cx_s - _gap], [cy_s, cy_s], color='cyan', lw=0.5, alpha=0.7)
            ax.plot([cx_s + _gap, cx_s + _gap + _arm], [cy_s, cy_s], color='cyan', lw=0.5, alpha=0.7)
            ax.plot([cx_s, cx_s], [cy_s - _gap - _arm, cy_s - _gap], color='cyan', lw=0.5, alpha=0.7)
            ax.plot([cx_s, cx_s], [cy_s + _gap, cy_s + _gap + _arm], color='cyan', lw=0.5, alpha=0.7)
        else:
            ax.set_facecolor('#111')
            ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=7, color='gray')

        flux_entry = fluxes.get(filt)
        clr = ('orange' if flux_entry[3] else 'lime') if flux_entry is not None else 'white'
        ax.set_title(filt.upper(), fontsize=8, pad=2, color=clr)
        ax.set_xticks([]); ax.set_yticks([])

    ax_sed = fig.add_subplot(gs[3, :])

    if fluxes:
        cat_items = [(k, v) for k, v in fluxes.items() if not v[3]]
        img_items = [(k, v) for k, v in fluxes.items() if     v[3]]

        if cat_items:
            wc  = np.array([v[0] for _, v in cat_items])
            fc  = np.array([v[1] for _, v in cat_items])
            efc = np.array([v[2] for _, v in cat_items])
            ax_sed.plot(wc, fc, '-', color='steelblue', lw=0.8, alpha=0.4, zorder=2)
            ax_sed.errorbar(wc, fc, yerr=np.where(np.isfinite(efc), efc, 0),
                            fmt='o', color='steelblue', ms=5, capsize=3, lw=1, zorder=3,
                            label='catalog')
            for w, f, lbl in zip(wc, fc, [k for k, _ in cat_items]):
                ax_sed.annotate(lbl.upper(), (w, f),
                                textcoords='offset points', xytext=(0, 6),
                                ha='center', fontsize=7)

        if img_items:
            wi = np.array([v[0] for _, v in img_items])
            fi = np.array([v[1] for _, v in img_items])
            ax_sed.plot(wi, fi, '--', color='orange', lw=0.6, alpha=0.4, zorder=2)
            ax_sed.errorbar(wi, fi, fmt='^', color='orange', ms=5, lw=1, zorder=3,
                            label='image aperture (no ap-corr)')
            for w, f, lbl in zip(wi, fi, [k for k, _ in img_items]):
                ax_sed.annotate(lbl.upper(), (w, f),
                                textcoords='offset points', xytext=(0, 6),
                                ha='center', fontsize=7, color='orange')

        ax_sed.legend(fontsize=8, loc='upper left')

    ax_sed.set_xscale('log'); ax_sed.set_yscale('log')
    ax_sed.set_xlabel(r'Wavelength ($\mu$m)', fontsize=9)
    ax_sed.set_ylabel('Flux (Jy)', fontsize=9)
    ax_sed.grid(True, alpha=0.3)

    outname = OUT_DIR / f'sed_rank{rank:03d}_F1500W{mag15:.2f}.png'
    fig.savefig(str(outname), dpi=150, bbox_inches='tight')
    plt.close(fig)

    img  = Image.open(str(outname))
    meta = PngImagePlugin.PngInfo()
    meta.add_text('SESHAT_CLASS', cls)
    meta.add_text('SESHAT_PROB',  f'{cls_prob:.6f}')
    for c in seshat_classes:
        meta.add_text(f'SESHAT_PROB_{c}', f'{float(sr[f"Prob {c}"]):.6f}')
    img.save(str(outname), pnginfo=meta)

    return outname

# ── Main ──────────────────────────────────────────────────────────────────────
print(f'\nGenerating {len(sources)} SEDs → {OUT_DIR}')
for i in range(len(sources)):
    try:
        out = plot_source_sed(i)
    except (ValueError, RuntimeError, KeyError, OSError, TypeError) as e:
        print(f'  ERROR rank {i+1}: {e}')
        continue
    if (i + 1) % 25 == 0 or (i + 1) == len(sources):
        print(f'{i+1}/{len(sources)} done: {Path(str(out)).name}')

print(f'Done. Figures in {OUT_DIR}')
