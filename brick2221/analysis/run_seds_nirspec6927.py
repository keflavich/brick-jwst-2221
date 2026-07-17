"""SED generator for NIRSpec 6927 targets — format matches sed_figures_f1500w_all."""
import sys
sys.stdout.reconfigure(line_buffering=True)

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pandas as pd
from astropy.table import Table
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.visualization import simple_norm
import astropy.units as u
from reproject import reproject_interp
import warnings
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
BRICK_BASE   = Path('/blue/adamginsburg/adamginsburg/jwst/brick')
BRICK_ORANGE = Path('/orange/adamginsburg/jwst/brick')
SICKLE_BASE  = Path('/orange/adamginsburg/jwst/sickle')
IMGS_DIR     = BRICK_BASE / 'images'

TARGET_CSV  = BRICK_BASE / 'nirspec_6927' / 'all_pointings_sources_20260715.csv'
NIRCAM_CAT  = BRICK_ORANGE / 'catalogs' / 'basic_merged_indivexp_photometry_tables_merged_resbgsub_m8_dedup.fits'
MIRI_CAT    = SICKLE_BASE / 'catalogs' / 'o003_miri_cmd_matched.fits'
SATSTAR_DIR = BRICK_BASE / 'catalogs'
OUT_DIR     = BRICK_BASE / 'nirspec_6927' / 'sed_figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_FILES = {
    'f115w':  IMGS_DIR    / 'jw01182-o004_t001_nircam_clear-f115w-merged_i2d.fits',
    'f200w':  IMGS_DIR    / 'jw01182-o004_t001_nircam_clear-f200w-merged_i2d.fits',
    'f356w':  IMGS_DIR    / 'jw01182-o004_t001_nircam_clear-f356w-merged_i2d.fits',
    'f444w':  IMGS_DIR    / 'jw01182-o004_t001_nircam_clear-f444w-merged_i2d.fits',
    'f182m':  BRICK_ORANGE / 'F182M' / 'pipeline' / 'jw02221-o001_t001_nircam_clear-f182m-merged_i2d.fits',
    'f187n':  BRICK_ORANGE / 'F187N' / 'pipeline' / 'jw02221-o001_t001_nircam_clear-f187n-merged_i2d.fits',
    'f212n':  BRICK_ORANGE / 'F212N' / 'pipeline' / 'jw02221-o001_t001_nircam_clear-f212n-merged_i2d.fits',
    'f405n':  BRICK_ORANGE / 'F405N' / 'pipeline' / 'jw02221-o001_t001_nircam_clear-f405n-merged_i2d.fits',
    'f410m':  BRICK_ORANGE / 'F410M' / 'pipeline' / 'jw02221-o001_t001_nircam_clear-f410m-merged_i2d.fits',
    'f466n':  BRICK_ORANGE / 'F466N' / 'pipeline' / 'jw02221-o001_t001_nircam_clear-f466n-merged_i2d.fits',
    'f770w':  SICKLE_BASE / 'F770W'  / 'pipeline' / 'jw03958-o003_t001_miri_f770w_i2d.fits',
    'f1130w': SICKLE_BASE / 'F1130W' / 'pipeline' / 'jw03958-o003_t001_miri_f1130w_i2d.fits',
    'f1500w': SICKLE_BASE / 'F1500W' / 'pipeline' / 'jw03958-o003_t001_miri_f1500w_i2d.fits',
    'f2550w': BRICK_ORANGE / 'F2550W' / 'pipeline' / 'jw02221-o002_t001_miri_f2550w_i2d.fits',
}

# NIRCam Vega ZPs (SVO filter profile service, Jy)
NIRCAM_VEGA_ZP_JY = {
    'f115w': 1746.1179, 'f182m': 844.9433, 'f187n': 794.8482,
    'f200w': 757.6538,  'f212n': 674.8317, 'f356w': 271.3926,
    'f405n': 206.9694,  'f410m': 208.7505, 'f444w': 184.1022,
    'f466n': 157.7731,
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
_SATSTAR_RADIUS_ARCSEC = 0.5
_XMATCH_RADIUS = 0.5   # arcsec
CUTOUT_SIZE = 5.0      # arcsec displayed
CUTOUT_PAD  = 8.0      # arcsec padded

# ── Load data ─────────────────────────────────────────────────────────────────
print('Loading NIRSpec target list...')
targets = pd.read_csv(str(TARGET_CSV))
targets.columns = [c.strip('# ') for c in targets.columns]
print(f'  {len(targets)} targets, pointings={sorted(targets["Pointing"].unique())}')
source_coords = SkyCoord(targets['RA'].values * u.deg, targets['Dec'].values * u.deg)

print('Loading satstar catalogs...')
SATSTAR_CATS, SATSTAR_COORDS, SATSTAR_PIX_SR = {}, {}, {}
for _filt in NIRCAM_FILTERS + ['f2550w']:
    _path = SATSTAR_DIR / f'{_filt}_consolidated_satstar_catalog.fits'
    if not _path.exists(): continue
    _cat = Table.read(str(_path))
    SATSTAR_CATS[_filt] = _cat
    _sc = _cat['skycoord_fit']
    SATSTAR_COORDS[_filt] = SkyCoord(_sc.ra.deg * u.deg, _sc.dec.deg * u.deg)
    _img = IMAGE_FILES.get(_filt)
    if _img is None or not _img.exists(): continue
    _pixar = None
    with fits.open(str(_img)) as _hdul:
        for _ext in [('SCI', 1), 1, 0]:
            try:
                _pixar = _hdul[_ext].header.get('PIXAR_SR')
                if _pixar is not None: break
            except (KeyError, IndexError): pass
        if _pixar is None:
            for _ext in [1, 0]:
                try:
                    _ww = WCS(_hdul[_ext].header)
                    _pixar = float(_ww.proj_plane_pixel_area().to(u.sr).value)
                    break
                except (KeyError, IndexError, ValueError): pass
    if _pixar is not None:
        SATSTAR_PIX_SR[_filt] = float(_pixar)
    print(f'  {_filt}: {len(_cat)} satstar, PIXAR_SR={SATSTAR_PIX_SR.get(_filt, float("nan")):.4e}')

print('Loading NIRCam m8_dedup catalog...')
with fits.open(str(NIRCAM_CAT), memmap=True) as _h:
    _ra  = np.array(_h[1].data['skycoord_ref.ra'],  dtype=float)
    _dec = np.array(_h[1].data['skycoord_ref.dec'], dtype=float)
    _bm  = ((_ra >= 266.35) & (_ra <= 266.75) & (_dec >= -28.85) & (_dec <= -28.55))
    nircam_cat = Table(_h[1].data[_bm])
nircam_coords = SkyCoord(nircam_cat['skycoord_ref.ra'].astype(float) * u.deg,
                          nircam_cat['skycoord_ref.dec'].astype(float) * u.deg)
print(f'  {len(nircam_cat):,} rows in region')
_idx_nc, _sep_nc, _ = match_coordinates_sky(source_coords, nircam_coords)
has_nircam = _sep_nc.arcsec < _XMATCH_RADIUS
print(f'  {has_nircam.sum()} / {len(targets)} matched to nircam_cat')

print('Loading MIRI o003 catalog...')
miri_cat = Table.read(str(MIRI_CAT))
miri_coords = SkyCoord(miri_cat['ra'].astype(float) * u.deg,
                        miri_cat['dec'].astype(float) * u.deg)
_idx_miri, _sep_miri, _ = match_coordinates_sky(source_coords, miri_coords)
has_miri = _sep_miri.arcsec < _XMATCH_RADIUS
print(f'  {has_miri.sum()} / {len(targets)} matched to MIRI o003')

print('Loading images...')
image_data = {}
for filt, path in IMAGE_FILES.items():
    if not path.exists(): print(f'  {filt}: missing'); continue
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
                except (KeyError, IndexError, ValueError): pass
    except (OSError, fits.VerifyError) as e:
        print(f'  {filt}: {e}')

# ── Helpers ───────────────────────────────────────────────────────────────────
def _to_float(x):
    try:
        v = float(x)
        return v if np.isfinite(v) and v > -90 else np.nan
    except (TypeError, ValueError): return np.nan

def vega_to_jy(mag, filt):
    m = _to_float(mag)
    zp = NIRCAM_VEGA_ZP_JY.get(filt) or MIRI_VEGA_ZP_JY.get(filt)
    if zp is None or not np.isfinite(m): return np.nan
    return float(zp * 10.0 ** (-m / 2.5))

def _make_northup_wcs(coord, pixscale_deg, npix):
    w = WCS(naxis=2)
    w.wcs.crpix = [npix / 2 + 0.5, npix / 2 + 0.5]
    w.wcs.cdelt = [-pixscale_deg, pixscale_deg]
    w.wcs.crval = [float(coord.ra.deg), float(coord.dec.deg)]
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    return w

def get_cutout(coord, filt):
    if filt not in image_data: return None, None
    data, wcs = image_data[filt]
    try:
        pm = wcs.pixel_scale_matrix
        ps_deg = float(np.sqrt(abs(np.linalg.det(pm))))
        ps_arcsec = ps_deg * 3600.0
        pad_pix = int(CUTOUT_PAD / ps_arcsec)
        cut = Cutout2D(data, coord, pad_pix, wcs=wcs, mode='partial', fill_value=np.nan)
        if np.all(~np.isfinite(cut.data)): return None, None
        npix = max(5, int(round(CUTOUT_SIZE / ps_arcsec)))
        target_wcs = _make_northup_wcs(coord, ps_deg, npix)
        arr, footprint = reproject_interp(
            (cut.data, cut.wcs), target_wcs, shape_out=(npix, npix), order='bilinear')
        arr = np.array(arr, dtype=float)
        arr[footprint < 0.5] = np.nan
        if np.all(~np.isfinite(arr)) or np.nansum(np.abs(arr)) == 0: return None, None
        return arr, ps_arcsec
    except (ValueError,): return None, None

def measure_flux_image(coord, filt):
    arr, ps = get_cutout(coord, filt)
    if arr is None: return np.nan
    ap_pix = PHOT_APER_ARCSEC.get(filt, 0.5) / ps
    ny, nx = arr.shape
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    yy, xx = np.mgrid[:ny, :nx].astype(float)
    r = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    ap_vals = arr[(r <= ap_pix) & np.isfinite(arr)]
    bg_vals = arr[(r > ap_pix*2) & (r <= ap_pix*3) & np.isfinite(arr)]
    if len(ap_vals) < 3: return np.nan
    bg = float(np.nanmedian(bg_vals)) if len(bg_vals) >= 3 else 0.0
    net = float(np.sum(ap_vals - bg))
    ps_rad = ps * np.pi / (180.0 * 3600.0)
    flux_jy = net * ps_rad**2 * 1e6
    return float(flux_jy) if np.isfinite(flux_jy) and flux_jy > 0 else np.nan

def get_satstar_flux(filt, coord):
    if filt not in SATSTAR_COORDS or filt not in SATSTAR_PIX_SR: return np.nan, np.nan
    idx, sep, _ = coord.match_to_catalog_sky(SATSTAR_COORDS[filt])
    if float(np.asarray(sep.arcsec).flat[0]) > _SATSTAR_RADIUS_ARCSEC: return np.nan, np.nan
    cat = SATSTAR_CATS[filt]
    row = cat[int(np.asarray(idx).flat[0])]
    pixar_sr = SATSTAR_PIX_SR[filt]
    flux_jy = float(row['flux_fit']) * pixar_sr * 1e6
    if not (np.isfinite(flux_jy) and flux_jy > 0): return np.nan, np.nan
    ec = ('flux_err' if 'flux_err' in cat.colnames
          else 'flux_unc' if 'flux_unc' in cat.colnames else None)
    eflux = float(row[ec]) * pixar_sr * 1e6 if ec is not None else np.nan
    return float(flux_jy), float(eflux)

def get_source_fluxes(i):
    row    = targets.iloc[i]
    coord  = source_coords[i]
    fluxes = {}

    # NIRCam: satstar → m8_dedup → CSV Vega mag → image
    for filt in NIRCAM_FILTERS:
        sf, sef = get_satstar_flux(filt, coord)
        if np.isfinite(sf):
            fluxes[filt] = (WAVELENGTHS_UM[filt], sf, sef, False); continue
        if has_nircam[i]:
            nc_row = nircam_cat[int(_idx_nc[i])]
            fcol, ecol = f'flux_jy_{filt}', f'eflux_jy_{filt}'
            if fcol in nircam_cat.colnames:
                f = _to_float(nc_row[fcol])
                e = _to_float(nc_row[ecol]) if ecol in nircam_cat.colnames else np.nan
                if np.isfinite(f) and f > 0:
                    fluxes[filt] = (WAVELENGTHS_UM[filt], f, e, False); continue
        col = filt.upper()
        if col in targets.columns:
            f = vega_to_jy(row[col], filt)
            if np.isfinite(f):
                fluxes[filt] = (WAVELENGTHS_UM[filt], f, np.nan, False); continue
        f = measure_flux_image(coord, filt)
        if np.isfinite(f):
            fluxes[filt] = (WAVELENGTHS_UM[filt], f, np.nan, True)

    # MIRI F770W, F1130W, F1500W: MIRI catalog → image
    for filt in ['f770w','f1130w','f1500w']:
        col = f'mag_{filt.upper()}'
        if has_miri[i] and col in miri_cat.colnames:
            m = _to_float(miri_cat[int(_idx_miri[i])][col])
            f = vega_to_jy(m, filt)
            if np.isfinite(f):
                fluxes[filt] = (WAVELENGTHS_UM[filt], f, np.nan, False); continue
        f = measure_flux_image(coord, filt)
        if np.isfinite(f):
            fluxes[filt] = (WAVELENGTHS_UM[filt], f, np.nan, True)

    # F2550W: satstar → MIRI catalog → image
    sf, sef = get_satstar_flux('f2550w', coord)
    if np.isfinite(sf):
        fluxes['f2550w'] = (WAVELENGTHS_UM['f2550w'], sf, sef, False)
    else:
        done = False
        if has_miri[i]:
            for col25 in ['mag_F2550W','mag_F2550W_forced']:
                if col25 in miri_cat.colnames:
                    m = _to_float(miri_cat[int(_idx_miri[i])][col25])
                    f = vega_to_jy(m, 'f2550w')
                    if np.isfinite(f):
                        fluxes['f2550w'] = (WAVELENGTHS_UM['f2550w'], f, np.nan, False)
                        done = True; break
        if not done:
            f = measure_flux_image(coord, 'f2550w')
            if np.isfinite(f):
                fluxes['f2550w'] = (WAVELENGTHS_UM['f2550w'], f, np.nan, True)

    return dict(sorted(fluxes.items(), key=lambda kv: kv[1][0]))

# ── Plot function ─────────────────────────────────────────────────────────────
def plot_source_sed(i):
    row    = targets.iloc[i]
    coord  = source_coords[i]
    src_id = int(row['ID'])
    pointing = int(row['Pointing'])
    ref    = str(row.get('REFERENCE', '')).upper() == 'TRUE'
    fluxes = get_source_fluxes(i)

    n_cat = sum(1 for v in fluxes.values() if not v[3])
    n_img = sum(1 for v in fluxes.values() if     v[3])

    # Build NIRSpec mag string from available CSV columns
    mag_str = ''
    for col_label in [('NRS_CLEAR', 'K'), ('F200W', 'F200W'), ('F444W', 'F444W')]:
        col, lbl = col_label
        if col in row.index:
            m = _to_float(row[col])
            if np.isfinite(m):
                mag_str += f'  {lbl}={m:.2f}'

    fig = plt.figure(figsize=(14, 11))
    ref_flag = '  [REF]' if ref else ''
    fig.suptitle(
        f'ID={src_id}{ref_flag}  P{pointing}  |  '
        f'RA={coord.ra.deg:.5f}  Dec={coord.dec.deg:.5f}'
        f'{mag_str}  |  {n_cat} cat + {n_img} img-phot',
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

        arr, pix_scale = get_cutout(coord, filt)

        if arr is not None:
            finite = arr[np.isfinite(arr)]
            norm = simple_norm(finite, stretch='log', percent=99.5) if (finite.size > 0 and finite.max() > finite.min()) else None
            ax.imshow(arr, norm=norm, origin='lower', cmap='inferno')
            cx_s, cy_s = arr.shape[1] / 2, arr.shape[0] / 2
            ap_pix = PHOT_APER_ARCSEC.get(filt, 0.5) / pix_scale
            ax.add_patch(plt.Circle((cx_s, cy_s), ap_pix, color='cyan', fill=False, lw=0.8))
            ax.axhline(cy_s, color='cyan', lw=0.4, alpha=0.5)
            ax.axvline(cx_s, color='cyan', lw=0.4, alpha=0.5)
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
                ax_sed.annotate(lbl.upper(), (w, f), textcoords='offset points',
                                xytext=(0, 6), ha='center', fontsize=7)

        if img_items:
            wi = np.array([v[0] for _, v in img_items])
            fi = np.array([v[1] for _, v in img_items])
            ax_sed.plot(wi, fi, '--', color='orange', lw=0.6, alpha=0.4, zorder=2)
            ax_sed.errorbar(wi, fi, fmt='^', color='orange', ms=5, lw=1, zorder=3,
                            label='image aperture (no ap-corr)')
            for w, f, lbl in zip(wi, fi, [k for k, _ in img_items]):
                ax_sed.annotate(lbl.upper(), (w, f), textcoords='offset points',
                                xytext=(0, 6), ha='center', fontsize=7, color='orange')

        ax_sed.legend(fontsize=8, loc='upper left')

    ax_sed.set_xscale('log'); ax_sed.set_yscale('log')
    ax_sed.set_xlabel(r'Wavelength ($\mu$m)', fontsize=9)
    ax_sed.set_ylabel('Flux (Jy)', fontsize=9)
    ax_sed.grid(True, alpha=0.3)

    outname = OUT_DIR / f'sed_id{src_id:06d}_p{pointing}.png'
    fig.savefig(str(outname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    return outname

# ── Main ──────────────────────────────────────────────────────────────────────
print(f'\nGenerating {len(targets)} SEDs → {OUT_DIR}')
for i in range(len(targets)):
    try:
        out = plot_source_sed(i)
    except (ValueError, RuntimeError, KeyError, OSError, TypeError) as e:
        print(f'  ERROR ID {int(targets.iloc[i]["ID"])}: {e}')
        continue
    if (i + 1) % 25 == 0 or (i + 1) == len(targets):
        print(f'{i+1}/{len(targets)} done: {Path(str(out)).name}')

print('Done.')
