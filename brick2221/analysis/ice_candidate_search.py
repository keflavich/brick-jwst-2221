"""
# Search for ice absorption to explain everything at N~10^19

410 - 466 < -0.5
212 - 466 < -0.5
356 - 444 > 0.25
356 - 410 > 0.25
|410 - 444| < 0.1
"""
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as pl
from brick2221.analysis.analysis_setup import basepath
from brick2221.analysis.make_ccd_with_icemodels import plot_ccd_with_icemodels, carbon_abundance, oxygen_abundance

dmag_all = Table.read(f'{basepath}/tables/combined_ice_absorption_tables.ecsv')
dmag_all.add_index('mol_id')
dmag_all.add_index('composition')
dmag_all.add_index('temperature')
dmag_all.add_index('database')
dmag_all.add_index('column')

N19 = dmag_all.loc['column', 1e19]
ice_candidate_selection = (((N19['F410M'] - N19['F466N']) < -0.5) &
                           (( - N19['F466N']) < -0.5) &
                           ((N19['F356W'] - N19['F444W']) > 0.25) &
                           ((N19['F356W'] - N19['F410M']) > 0.25) &
                           (np.abs(N19['F410M'] - N19['F444W']) < 0.1))

# print(N19[ice_candidate_selection])


percent = 25
fig = pl.figure()

for ii, (color1, color2, lims) in enumerate(((['F182M', 'F212N'], ['F410M', 'F466N'], (0, 3, -1.5, 1.0)),
                                             (['F115W', 'F200W'], ['F356W', 'F444W'], (0, 20, -0.5, 1.5)),
                                             (['F356W', 'F410M'], ['F410M', 'F444W'], (-0.5, 2, -0.5, 0.5)),
                                             (['F182M', 'F212N'], ['F212N', 'F466N'], (0, 3, -0.1, 2.5)),
                                            )):
    ax = fig.add_subplot(2, 2, ii+1)
    a_color1, a_color2, c1, c2, sel, E_V_color1, E_V_color2, tb = plot_ccd_with_icemodels(color1, color2,
                                                                                          molcomps=N19[ice_candidate_selection]['composition'],
                                                                                          dmag_tbl=dmag_all.loc['composition', N19[ice_candidate_selection]['composition']],
                                                                                          abundance=(percent/100.)*carbon_abundance,
                                                                                          max_column=2e20)
    pl.title(f"{percent}% of CO in ice");
    pl.axis(lims);
pl.legend(loc='upper left', bbox_to_anchor=(1,1,0,0))