"""
We ran a trilegal model with the following parameters modified from the default:
 center = 0.25, 0
 field area = 0.01 deg^2
 JWST NIRCAm narrow-band filters
 limiting mag in 4th filter is m=26 (that filter is f323n)
 Chabrier lognormal + salpeter [instead of chabrier lognormal]
 local calibration of 0.15 mag/kpc [lower than usual; used instead of calibration at infinity]
 galactocentric radius = 8.1 kpc  [down from 8.7]
"""
from astropy.table import Table
import numpy as np
from astropy import units as u
import warnings
warnings.simplefilter('ignore')

from dust_extinction.averages import CT06_MWGC, G21_MWAvg
ext = G21_MWAvg()

tbl = Table.read('output997585460850.dat', format='ascii.csv', delimiter=' ')

percentiles = [0.1, 1, 5, 16, 50, 84, 95, 99, 99.9]
print(f"What are the surface temperatures of stars in the detectable range? {percentiles}")
dmod = tbl['m-M0']
distance_kpc = 10**((dmod + 5)/5) / 1000.

# 17 magnitudes of extinction at 8.1 kpc
dmag = (17/8.1)
f212n_ext = ext(2.12*u.um) * dmag * distance_kpc + tbl['F212N']
extinction = distance_kpc * dmag + tbl['Av']

sel = (f212n_ext > 14) & (f212n_ext < 20) & (distance_kpc > 7)
print(sel.sum(), sel.sum()/len(tbl), len(tbl))
print("temperature", np.array(np.percentile(tbl['logTe'][sel], percentiles)))
print("temperature", np.array(10**np.percentile(tbl['logTe'][sel], percentiles)))
print("initial mass", np.array(np.percentile(tbl['m_ini'][sel], percentiles)))
print("extinction", np.array(np.percentile(extinction[sel], percentiles)))
print("distance", np.array(np.percentile(distance_kpc[sel], percentiles)))
print("log age", np.array(np.percentile(tbl['logAge'][sel], percentiles)))

# Calculate percentile at which 4000K is reached
temperatures = 10**tbl['logTe'][sel]
percentile_4000K = (temperatures <= 4000).sum() / len(temperatures) * 100
print(f"Percentile at which 4000K is reached: {percentile_4000K:.2f}%")

