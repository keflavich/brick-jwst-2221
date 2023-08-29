import numpy as np
from astropy import units as u
from astropy.io import ascii
from astropy.table import Table

import analysis_setup
import selections
import plot_tools

from analysis_setup import filternames, basepath
from analysis_setup import fh_merged_reproject as fh, ww410_merged_reproject as ww410, ww410_merged_reproject as ww
from selections import main, basetable_merged_reproject, basetable_merged_reproject  as basetable

print()
print("merged-reproject")
result = main(basetable_merged_reproject, ww=ww)
globals().update(result)
# we want to measure the RMS before excludign...
all_good = result['all_good_phot']
globals().update({key+"_mr": val for key, val in result.items()})

oksep = result['oksep']
print(f"There are {oksep.sum()} out of {len(oksep)} oksep, and {(oksep & all_good).sum()} all_good & oksep")

# because we're computing stats, we don't want to exclude the big offsets
stats = plot_tools.xmatch_plot(basetable, sel=all_good, axlims=[-0.2,0.2,-0.2,0.2], ref_filter='f405n', maxsep=1*u.arcsec)


sfilternames = sorted(filternames)
astrom_tbl = Table({'Filter Name': [fil.upper() for fil in sfilternames],
                    'RMS Offset': u.Quantity([stats[fil]['std'] if fil in stats else np.nan*u.arcsec for fil in sfilternames], u.arcsec),
                    #'mad': u.Quantity([stats[fil]['mad'] if fil in stats else np.nan*u.arcsec for fil in sfilternames], u.arcsec),
                    '90th percentile': [np.nanpercentile(np.array(basetable[all_good][f'mag_ab_{fil}']), 90)
                                              for fil in sfilternames],
                    r'\# of sources': [basetable[f'good_{fil}'].sum() for fil in sfilternames],
                   })
astrom_tbl['RMS Offset'].format = "%0.2g"
astrom_tbl['90th percentile'].format = "%0.1f"
astrom_tbl['90th percentile'].unit = u.ABmag
ld = ascii.latex.latexdicts['AA'].copy()
ld.update({'tabletype': 'table',
           'tablefoot': ("\par\nThe RMS offset reports the standard deviation of the source position difference between the specified filter and the reference filter, F405N.\n"
                         "The 90th percentile column reports the 90th percentile magnitude in the catalog to give a general sense of depth.\n"
                         r"% this table is produced by \texttt{make\_observations\_table.py}"
                        )
          })
astrom_tbl.write(f'{basepath}/paper_co/observations_table.tex',
                 caption=r'Observations \label{tab:observations}',
                 latexdict=ld,
                 formats={'RMS Offset': lambda x: "-" if np.isnan(x) else f"{x:0.03f}"[:5]},
                 overwrite=True)
