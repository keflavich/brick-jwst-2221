"""Cutout grid at each fake-bright location (and A/B) on the NEW joint F770W
product: data / model / residual columns.  Confirms the fake-bright gate left no
phantom star where the user marked one, and that A/B survive."""
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.visualization import simple_norm
import warnings; warnings.filterwarnings('ignore')

P = '/orange/adamginsburg/jwst/sickle/F770W/pipeline/'
BASE = 'jw03958-o001-002_t001_miri_clear-f770w-mirimage'
M6 = '_resbgsub_group_m6_daophot_basic_mergedcat'

def load(fn):
    with fits.open(fn) as h:
        ext = 'SCI' if 'SCI' in [x.name for x in h] else 0
        return h[ext].data.astype(float), wcs.WCS(h[ext].header)

DAT, WD = load(P + BASE + '_data_i2d.fits')
MOD, WM = load(P + BASE + M6 + '_model_i2d.fits')
RES, WR = load(P + BASE + M6 + '_residual_i2d.fits')

def read_reg(fn):
    out = []
    for line in open(fn):
        line = line.strip()
        if line.startswith('point('):
            ra, dec = line.split('(')[1].split(')')[0].split(',')[:2]
            out.append(SkyCoord(float(ra), float(dec), unit='deg'))
    return out

fakes = read_reg('/orange/adamginsburg/jwst/sickle/regions_/f770w_fake_bright_stars_20260622.reg')
pts = [(f'F{i+1}', c) for i, c in enumerate(fakes)]
pts += [('A', SkyCoord(266.57431, -28.80958, unit='deg')),
        ('B', SkyCoord(266.57297, -28.80342, unit='deg'))]

H = 18
n = len(pts)
fig, ax = plt.subplots(n, 3, figsize=(7.5, 2.4 * n))
for row, (nm, c) in enumerate(pts):
    for col, (d, w, t) in enumerate([(DAT, WD, 'data'), (MOD, WM, 'model'), (RES, WR, 'residual')]):
        x, y = w.world_to_pixel(c); x, y = int(round(float(x))), int(round(float(y)))
        sub = d[y-H:y+H, x-H:x+H]
        a = ax[row, col]
        if col == 1:  # model: fixed low-high to expose any phantom
            a.imshow(sub, origin='lower', cmap='viridis', vmin=0, vmax=2000)
        else:
            a.imshow(sub, origin='lower', cmap='gray',
                     norm=simple_norm(np.nan_to_num(sub), 'asinh', min_percent=10, max_percent=99.5))
        a.add_patch(plt.Circle((H, H), 5, fill=False, ec='red', lw=0.8))
        a.set_xticks([]); a.set_yticks([])
        if col == 0:
            a.set_ylabel(nm, fontsize=10, rotation=0, labelpad=14, va='center')
        if row == 0:
            a.set_title(t, fontsize=11)
plt.suptitle('F770W joint (new): fakes F1-F16 must show NO model star; A/B kept', fontsize=12)
plt.tight_layout()
out = P + 'DIAG_fake_cutouts_new.png'
plt.savefig(out, dpi=70, bbox_inches='tight'); plt.close()
print("saved", out)
