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
from brick2221.analysis.analysis_setup import basepath, molscomps
from brick2221.analysis.make_ccd_with_icemodels import plot_ccd_with_icemodels, carbon_abundance, oxygen_abundance

dmag_all = Table.read(f'{basepath}/tables/combined_ice_absorption_tables.ecsv')
dmag_all.add_index('mol_id')
dmag_all.add_index('composition')
dmag_all.add_index('temperature')
dmag_all.add_index('database')
dmag_all.add_index('column')

print('loaded dmag_all')
ucol = np.unique(dmag_all['column'])
c19 = ucol[np.argmin(np.abs(ucol - 1e19))]
N19 = dmag_all[np.abs(dmag_all['column'] - c19)/c19 < 1e-4]

c195 = ucol[np.argmin(np.abs(ucol - 10**19.5))]
N195 = dmag_all[np.abs(dmag_all['column'] - c195)/c195 < 1e-4]

ice_candidate_selection = (((N19['F410M'] - N19['F466N']) < -0.5) &
                           (( - N19['F466N']) < -0.5) &
                           ((N19['F405N'] - N19['F466N']) < -0.5) &
                           ((N19['F356W'] - N19['F444W']) > 0.25) &
                           ((N19['F356W'] - N19['F410M']) > 0.25) &
                           (np.abs(N19['F410M'] - N19['F444W']) < 0.1))

ice_candidate_selection_narrow = (((N19['F410M'] - N19['F466N']) < -0.5) &
                           (( - N19['F466N']) < -0.5) &
                           ((N19['F405N'] - N19['F466N']) < -0.5) &
                           (np.abs(N19['F405N'] - N19['F410M']) < 0.1))

ice_candidate_selection_narrow_195 = (((N195['F410M'] - N195['F466N']) < -0.5) &
                           (( - N19['F466N']) < -0.5) &
                           ((N19['F405N'] - N19['F466N']) < -0.5) &
                           (np.abs(N19['F405N'] - N19['F410M']) < 0.1))

print("These are only pure ice column densities - so the column threshold is 10^19 of total ice, not of CO ice")
print("Ice candidates:")
N19[ice_candidate_selection]['molecule', 'database', 'author', 'F405N', 'F466N'].pprint(max_width=200)
print("--------------------------------\n\n", flush=True)
print("Ice candidates no wideband:")
narrow_options = N19[ice_candidate_selection_narrow]
narrow_options.sort('F466N')
narrow_options['molecule', 'database', 'author', 'F405N', 'F466N'].pprint(max_width=200, max_lines=100)
print("--------------------------------\n\n", flush=True)
print("Ice candidates no wideband 19.5:")
narrow_options_195 = N195[ice_candidate_selection_narrow]
narrow_options_195.sort('F466N')
narrow_options_195['molecule', 'database', 'author', 'F405N', 'F466N'].pprint(max_width=200, max_lines=100)
print("--------------------------------\n\n", flush=True)


# toggle this
ice_candidate_selection = ice_candidate_selection_narrow


dmag_lt_195 = dmag_all[dmag_all['column'] < 10**19.5]

percent = 25
fig = pl.figure(figsize=(14, 14))

for ii, (color1, color2, lims) in enumerate(((['F182M', 'F212N'], ['F410M', 'F466N'], (0, 2.1, -1.5, 1.0)),
                                             (['F115W', 'F200W'], ['F356W', 'F444W'], (0, 12.5, -0.5, 1.5)),
                                             #(['F356W', 'F410M'], ['F410M', 'F444W'], (-0.5, 2, -0.5, 0.5)),
                                             (['F182M', 'F212N'], ['F405N', 'F410M'], (0, 2.1, -0.3, 0.1)),
                                             #(['F182M', 'F212N'], ['F212N', 'F466N'], (0, 2.1, -0.1, 2.5)),
                                             (['F200W', 'F356W'], ['F356W', 'F444W'], (0, 5, -0.5, 1.5)),
                                            )):
    print(color1, color2)
    ax = fig.add_subplot(2, 2, ii+1)
    a_color1, a_color2, c1, c2, sel, E_V_color1, E_V_color2, tb = plot_ccd_with_icemodels(color1, color2,
                                                                                          molcomps=list(zip(N19[ice_candidate_selection]['composition'],
                                                                                                            N19[ice_candidate_selection]['temperature'])),
                                                                                          dmag_tbl=dmag_lt_195.loc['composition', N19[ice_candidate_selection]['composition']],
                                                                                          abundance_wrt_h2=(percent/100.)*carbon_abundance,
                                                                                          max_column=2e20)
    pl.title(f"{percent}% of CO in ice");
    pl.axis(lims);
pl.legend(loc='upper left', bbox_to_anchor=(1,1,0,1.2))
pl.subplots_adjust(wspace=0.25)