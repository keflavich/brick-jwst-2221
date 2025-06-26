"""
Compare A_V to dust emission-derived column density
"""
from astropy.table import Table
import numpy as np
import astropy.units as u
import matplotlib.pyplot as pl
import mpl_plot_templates
from molmass import Formula
import re
from astropy.io import fits

from brick2221.analysis.analysis_setup import filternames
from brick2221.analysis.selections import load_table
from brick2221.analysis.analysis_setup import fh_merged as fh, ww410_merged as ww410, ww410_merged as ww
from brick2221.analysis.analysis_setup import basepath

from icemodels.core import composition_to_molweight

from dust_extinction.averages import CT06_MWGC, G21_MWAvg


from astropy.io import fits
import pylab as pl
from astropy.wcs import WCS
from astropy.table import Table
from dust_extinction.averages import CT06_MWGC, G21_MWAvg
from astropy import units as u

from brick2221.analysis.make_icecolumn_fig9 import compute_molecular_column, calc_av


from brick2221.analysis.analysis_setup import basepath
fh = fits.open(f'{basepath}/brandt_ice/brick.dust_column_density_cf.fits')
column = fits.getdata(f'{basepath}/brandt_ice/brick.dust_column_density_cf.fits')
cocol = np.load(f'{basepath}/brandt_ice/COIceMap_0.npy')
h2ocol = np.load(f'{basepath}/brandt_ice/H2OIceMap_0.npy')

basetable = tbl = Table.read(f'{basepath}/catalogs/basic_merged_indivexp_photometry_tables_merged_ok2221_20250324.fits')
measured_466m410 = basetable['mag_ab_f466n'] - basetable['mag_ab_f410m']

crds = tbl['skycoord_ref']

ww = WCS(fh[0].header).celestial
keep = ww.footprint_contains(crds)
pcrds = [x.astype(int) for x in ww.world_to_pixel(crds[keep])]
columns = column[pcrds[1], pcrds[0]]




avfilts = ['F182M', 'F212N']

ext = CT06_MWGC()
av = calc_av(avfilts=avfilts, basetable=basetable, ext=ext)

E_V_410_466 = (ext(4.10*u.um) - ext(4.66*u.um))

unextincted_466m410 = measured_466m410 + E_V_410_466 * av

dmag_tbl = Table.read(f'{basepath}/tables/H2O+CO_ice_absorption_tables.ecsv')
dmag_tbl.add_index('composition')

inferred_molecular_column = compute_molecular_column(unextincted_1m2=unextincted_466m410,
                                                     dmag_tbl=dmag_tbl.loc['H2O:CO:CO2 (5:1:0.5)'])

match = av[keep] > (columns/2.21e21 + 20)

pl.figure(1)
pl.scatter(columns, av[keep], s=1)
pl.scatter(columns[match], av[keep][match], s=1, c='r')
#pl.plot([2.21e21, 80*2.21e21], [1, 80], 'k--')
#pl.plot([21*2.21e21, (80+20)*2.21e21], [1, 80], 'k--')
pl.xscale('log');
pl.xlabel('Dust-Derived Column Density [cm$^{-2}$]')
pl.ylabel('A$_V$ [mag]')
ax1 = pl.gca()
ax2 = ax1.twiny()
ax2.plot([1, 80], [1, 80], 'k--', label='$A_V$ = N(H) * 2.21e21')
ax2.plot([1+20, 80+20], [1, 80], 'k:', label='$A_V$ = N(H) * 2.21e21 + 20')
ax2.set_xlim(np.array(ax1.get_xlim()) / 2.21e21)
ax2.set_xlabel('$A_V$ from emission')
pl.legend(loc='best');

cocols = cocol[pcrds[1], pcrds[0]]
pl.figure(2)
pl.scatter(cocols, inferred_molecular_column[keep], s=1)
pl.scatter(cocols[match], inferred_molecular_column[keep][match], s=1, c='r', label='A$_V$ > c NH + 20')
pl.plot(np.logspace(16, 20, 100), np.logspace(16, 20, 100), 'k--', label='1:1')
pl.plot(np.logspace(13, 20, 100), np.logspace(13, 20, 100)*10, 'k:', label='10:1')
pl.plot(np.logspace(13, 20, 100), np.logspace(13, 20, 100)*100, 'k-.', label='100:1')
pl.xscale('log');
pl.yscale('log');
pl.xlabel('Model CO Column [cm$^{-2}$]')
pl.ylabel('Inferred CO Column [cm$^{-2}$]')
pl.ylim(1e16,1e20)
pl.legend(loc='best');

pl.figure(3)
ax = pl.subplot(projection=ww.celestial)
ax.imshow(column, origin='lower', cmap='gray_r')
ax.scatter_coord(crds[keep][~match], s=1, alpha=0.5)
ax.scatter_coord(crds[keep][match], s=1, c='r', alpha=0.5)
