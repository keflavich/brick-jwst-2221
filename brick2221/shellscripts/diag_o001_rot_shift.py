"""Quantify the rigid transform (rotation theta about field center + shift) that
maps OUR o001 star catalog onto the MAST (truth) catalog. Scan theta, NN-match,
find peak. Also report the crf WCS to localize where the error entered."""
import numpy as np, glob, os
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy import units as u
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats

mast = '/orange/adamginsburg/jwst/sickle/mastDownload/JWST/jw03958-o001_t001_miri_f770w-brightsky/jw03958-o001_t001_miri_f770w-brightsky_i2d.fits'
ours = '/orange/adamginsburg/jwst/sickle/F770W/pipeline/jw03958-o001_t001_miri_clear-f770w-mirimage_data_i2d.fits'

def load(fn):
    h=fits.open(fn)
    for hdu in h:
        if hdu.data is not None and getattr(hdu.data,'ndim',0)==2 and 'CRVAL1' in hdu.header:
            return hdu.data, WCS(hdu.header)
    raise RuntimeError(fn)

def detect(data,w,n=120):
    mean,med,std=sigma_clipped_stats(data,sigma=3.0)
    s=DAOStarFinder(fwhm=3.0,threshold=20*std)(data-med)
    s.sort('flux'); s.reverse(); s=s[:n]
    return w.pixel_to_world(s['xcentroid'],s['ycentroid'])

dm,wm=load(mast); do,wo=load(ours)
sky_m=detect(dm,wm); sky_o=detect(do,wo)
# pivot = MAST field center
cen=wm.pixel_to_world(dm.shape[1]/2,dm.shape[0]/2)

def rotate_about(coords, pivot, theta_deg):
    # work in local tangent plane (arcsec) about pivot
    fr = pivot.skyoffset_frame()
    off = coords.transform_to(fr)
    lon = off.lon.to(u.arcsec).value
    lat = off.lat.to(u.arcsec).value
    th=np.radians(theta_deg)
    lon2 = lon*np.cos(th) - lat*np.sin(th)
    lat2 = lon*np.sin(th) + lat*np.cos(th)
    return SkyCoord(lon2*u.arcsec, lat2*u.arcsec, frame=fr).icrs

best=None
for theta in np.arange(-60,60.1,1.0):
    for dlon in [0]:  # rotation only first
        rot=rotate_about(sky_o,cen,theta)
        idx,sep,_=match_coordinates_sky(rot,sky_m)
        nmatch=np.sum(sep.arcsec<1.0)
        if best is None or nmatch>best[1]:
            best=(theta,nmatch,np.median(sep.arcsec))
print(f"ROTATION-ONLY scan (about MAST center): best theta={best[0]:.1f} deg, matches<1\"={best[1]}/{len(sky_o)}, median sep={best[2]:.2f}\"")

# now allow shift: for the best theta, also try a grid of shifts
from itertools import product
bestf=None
for theta in np.arange(best[0]-5,best[0]+5.1,1.0):
    rot=rotate_about(sky_o,cen,theta)
    fr=cen.skyoffset_frame()
    o=rot.transform_to(fr); lon=o.lon.to(u.arcsec).value; lat=o.lat.to(u.arcsec).value
    m=sky_m.transform_to(fr); mlon=m.lon.to(u.arcsec).value; mlat=m.lat.to(u.arcsec).value
    for dx in np.arange(-50,50.1,5):
        for dy in np.arange(-50,50.1,5):
            cc=SkyCoord((lon+dx)*u.arcsec,(lat+dy)*u.arcsec,frame=fr).icrs
            idx,sep,_=match_coordinates_sky(cc,sky_m)
            nm=np.sum(sep.arcsec<1.0)
            if bestf is None or nm>bestf[0]:
                bestf=(nm,theta,dx,dy,np.median(sep.arcsec))
print(f"ROT+SHIFT scan: matches<1\"={bestf[0]}/{len(sky_o)} at theta={bestf[1]:.1f} dx={bestf[2]:.0f}\" dy={bestf[3]:.0f}\" median={bestf[4]:.2f}\"")

# crf WCS check (input to resample)
print("\n--- crf (resample inputs) PA + center ---")
for fn in sorted(glob.glob('/orange/adamginsburg/jwst/sickle/F770W/pipeline/jw03958001001_02101_0000?_mirimage_o001_crf.fits'))[:2]:
    h=fits.open(fn)
    for hdu in h:
        if hdu.data is not None and getattr(hdu.data,'ndim',0)==2 and 'CRVAL1' in hdu.header:
            w=WCS(hdu.header); cd=w.pixel_scale_matrix
            pa=np.degrees(np.arctan2(cd[0,1],cd[1,1]))
            c=w.pixel_to_world(hdu.data.shape[1]/2,hdu.data.shape[0]/2)
            print(f"  {os.path.basename(fn)}: PA={pa:.3f} cen=({c.ra.deg:.5f},{c.dec.deg:.5f})")
            break
