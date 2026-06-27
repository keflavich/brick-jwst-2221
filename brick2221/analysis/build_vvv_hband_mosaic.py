"""
Build a single VVV H-band reference mosaic over the whole GC JWST footprint.

VVV Ks deep imaging is unavailable from HiPerGator (CASU VSA / ROE unreachable; no Ks
HiPS exists). The CDS VVV DR4 H/Bulge HiPS (~0.2"/pix) covers the GC bulge and is the
reachable real-image substitute -- H-band shows the same bright stars as Ks (only the
most reddened sources differ), which is what we need to spot bright stars just outside
each JWST FOV.

At 0.25"/pix the GC footprint is ~1.4 Gpix, too large for one hips2fits request, so we
define ONE global TAN WCS and render it in tiles (each tile is a window into the global
frame -> pixel grids align exactly), assembling a single mosaic FITS.

Output: /orange/adamginsburg/jwst/vvv/vvv_dr4_Hband_GC_mosaic_0p25.fits
"""
import os
import numpy as np
from astropy.wcs import WCS
from astropy.io import fits
from astroquery.hips2fits import hips2fits
from astropy import units as u

HIPS = 'CDS/P/VISTA/VVV/DR4/H/Bulge'
OUTDIR = '/orange/adamginsburg/jwst/vvv'
OUT = f'{OUTDIR}/vvv_dr4_Hband_GC_mosaic_0p25.fits'

# Global frame: TAN centered on the GC, covering all GC JWST fields
# (Sgr C l~359.4, Sgr A* l~0.0, Arches/Quintuplet, Brick l~0.25, Sgr B2 l~0.67).
CRVAL = (266.45, -28.90)            # deg, center
PIXSCALE = 0.25 / 3600.0           # deg/pix
HALF_X_DEG = 0.62                  # angular half-width  (x)  -> ~RA 265.8..267.1
HALF_Y_DEG = 0.78                  # angular half-height (y)  -> Dec -29.68..-28.12
TILE = 2000                       # render tile size (pix); larger -> server read-timeout

hips2fits.timeout = 300           # CDS render+transfer can be slow for big tiles


def global_wcs():
    nx = int(np.ceil(2 * HALF_X_DEG / PIXSCALE))
    ny = int(np.ceil(2 * HALF_Y_DEG / PIXSCALE))
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crval = list(CRVAL)
    w.wcs.crpix = [nx / 2.0 + 0.5, ny / 2.0 + 0.5]
    w.wcs.cd = [[-PIXSCALE, 0.0], [0.0, PIXSCALE]]
    w.wcs.cunit = ['deg', 'deg']
    return w, nx, ny


def tile_wcs(gw, x0, y0, tnx, tny):
    """sub-WCS = global WCS with CRPIX shifted so (x0,y0) is the tile origin."""
    w = gw.deepcopy()
    w.wcs.crpix = [gw.wcs.crpix[0] - x0, gw.wcs.crpix[1] - y0]
    w.array_shape = (tny, tnx)
    return w


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    gw, nx, ny = global_wcs()
    print(f"global frame {nx} x {ny} = {nx*ny/1e6:.0f} Mpix ({nx*ny*4/1e9:.2f} GB float32)", flush=True)
    mosaic = np.full((ny, nx), np.nan, dtype='float32')

    ntx = int(np.ceil(nx / TILE)); nty = int(np.ceil(ny / TILE))
    print(f"{ntx} x {nty} = {ntx*nty} tiles", flush=True)
    k = 0
    for jy in range(nty):
        y0 = jy * TILE; tny = min(TILE, ny - y0)
        for ix in range(ntx):
            x0 = ix * TILE; tnx = min(TILE, nx - x0)
            k += 1
            tw = tile_wcs(gw, x0, y0, tnx, tny)
            for attempt in range(4):
                try:
                    hdu = hips2fits.query_with_wcs(hips=HIPS, wcs=tw, format='fits',
                                                   get_query_payload=False)
                    data = hdu[0].data if isinstance(hdu, fits.HDUList) else hdu.data
                    mosaic[y0:y0+tny, x0:x0+tnx] = np.asarray(data, dtype='float32')
                    nvalid = np.isfinite(data).sum()
                    print(f"  tile {k}/{ntx*nty} [{x0}:{x0+tnx},{y0}:{y0+tny}] ok, {nvalid/1e6:.1f}Mpix valid", flush=True)
                    break
                except Exception as e:
                    print(f"  tile {k} attempt {attempt} ERR {repr(e)[:160]}", flush=True)
                    if attempt == 3:
                        print(f"  tile {k} FAILED, leaving NaN", flush=True)

    hdr = gw.to_header()
    hdr['BUNIT'] = 'arbitrary'
    hdr['SURVEY'] = 'VVV DR4'
    hdr['BAND'] = 'H'
    hdr['HIPS'] = HIPS
    hdr['COMMENT'] = 'VVV DR4 H-band GC reference mosaic rendered from CDS HiPS via hips2fits'
    fits.PrimaryHDU(data=mosaic, header=hdr).writeto(OUT, overwrite=True)
    print(f"wrote {OUT} ({os.path.getsize(OUT)/1e9:.2f} GB)", flush=True)


if __name__ == '__main__':
    main()
