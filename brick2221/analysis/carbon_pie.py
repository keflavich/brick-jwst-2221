"""
Show pie charts of carbon fraction in dust/ism.

GPT research:
https://chatgpt.com/share/6936e2fc-6c98-8004-a454-78bcb5418235
"""
import pylab as pl


import socket
if socket.gethostname() in ('cyg', 'cyg2022.local'):
    rootpath = '/Users/adam/Dropbox/talks/talks/ice_assets/'
elif 'ufhpc' in socket.gethostname():
    rootpath = '/orange/adamginsburg/ice/colors_of_ices_overleaf/figures/'
else:
    raise ValueError(f"Unknown host {socket.gethostname()}")

pl.clf();

startangle = 80

pl.pie([0.5, 0.1666, 0.11111, 0.11111, ], explode=(0,0.1,0,0), labels=['PAHs & Dust', 'CO gas / H = 5$\\times10^{-5}$', 'CO$_2$ ice', 'Other'], labeldistance=0.4, startangle=startangle, rotatelabels=True, textprops={'fontsize': 8});

pl.title("Local Carbon:\n C/H$\sim10^{-3.5} = 3\\times10^{-4}$")

pl.savefig(f"{rootpath}/standard_carbon_distribution_exploded.png", bbox_inches='tight', dpi=300)
print(f"{rootpath}/standard_carbon_distribution_exploded.png")

pl.clf();

pl.pie([0.5, 0.1666, 0.11111, 0.11111], labels=['PAHs & Dust', 'CO gas', 'CO$_2$ ice',  'Other'], labeldistance=0.4, startangle=startangle, rotatelabels=True, textprops={'fontsize': 8});

pl.title("Local Carbon:\n C/H$\sim10^{-3.5} = 3\\times10^{-4}$")

pl.savefig(f"{rootpath}/standard_carbon_distribution.png", bbox_inches='tight', dpi=300)
print(f"{rootpath}/standard_carbon_distribution.png")


pl.clf();

pl.pie([0.5, 0.1666, 0.11111, 0.11111], explode=(0,0.1,0,0), labels=['PAHs & Dust', 'CO gas / H$_2$ = 1$\\times10^{-4}$', 'CO$_2$ ice', 'Other'], labeldistance=0.4, startangle=startangle, rotatelabels=True, textprops={'fontsize': 8});

pl.title("Local Carbon:\n C/H$\sim10^{-3.5} = 3\\times10^{-4}$")

pl.savefig(f"{rootpath}/standard_carbon_distribution_exploded_h2.png", bbox_inches='tight', dpi=300)

pl.clf();

pl.pie([0.5, 0.1666, 0.11111, 0.11111], explode=(0,0.1,0,0), labels=['PAHs & Dust', 'CO ice / H$_2$ = 1$\\times10^{-4}$', 'CO$_2$ ice','Other'], labeldistance=0.4, startangle=startangle, rotatelabels=True, textprops={'fontsize': 8});

pl.title("Local Carbon:\n C/H$\sim10^{-3.5} = 3\\times10^{-4}$")

pl.savefig(f"{rootpath}/standard_carbon_distribution_exploded_h2_ice.png", bbox_inches='tight', dpi=300)
print(f"{rootpath}/standard_carbon_distribution_exploded_h2_ice.png")



pl.clf();

pl.pie([0.35, 0.4166666, 0.0777777, 0.0777777, ], explode=(0,0.1,0,0), labels=['PAHs & Dust', 'CO ice / H$_2$ = 2.5$\\times10^{-4}$', 'CO$_2$ ice',  'Other'], labeldistance=0.4, startangle=113, rotatelabels=True, textprops={'fontsize': 8});

pl.title("Local Carbon:\n C/H$\sim10^{-3.5} = 3\\times10^{-4}$\nwith GC measured CO ice")

pl.savefig(f"{rootpath}/standard_carbon_distribution_exploded_h2_with_GC_CO_levels.png", bbox_inches='tight', dpi=300)
print(f"{rootpath}/standard_carbon_distribution_exploded_h2_with_GC_CO_levels.png")
pl.clf();





pl.clf();

pl.pie([0.5, 0.1666666, 0.1111111, 0.1111111], explode=(0,0.1,0,0), labels=['PAHs & Dust', 'CO ice / H$_2$ = 2.5$\\times10^{-4}$', 'CO$_2$ ice',  'Other'], labeldistance=0.4, startangle=80, rotatelabels=True, textprops={'fontsize': 8});

pl.title("Galactic Center Carbon:\n C/H$\sim10^{-3.12} = 7.5\\times10^{-4}$")

pl.savefig(f"{rootpath}/standard_carbon_distribution_exploded_h2_gc.png", bbox_inches='tight', dpi=300)
print(f"{rootpath}/standard_carbon_distribution_exploded_h2_gc.png")


pl.clf();

pl.pie([0.5, 0.1666666, 0.111111, 0.01, 0.111111, 0.111111],
       explode=(0,0,0,0,0,0),
       labels=['PAHs & Dust', 'CO ice', 'CO$_2$ ice', 'C/C+', 'Other ice', 'Other'],
       labeldistance=0.4,
       startangle=92,
       rotatelabels=True, textprops={'fontsize': 8});

pl.title("Galactic Center Carbon:\n C/H$\sim10^{-3.12} = 7.5\\times10^{-4}$")

pl.savefig(f"{rootpath}/frozen_carbon_distribution.png", bbox_inches='tight', dpi=300)
print(f"{rootpath}/frozen_carbon_distribution.png")


pl.clf()
pl.pie([0., 1, 0, 0, 0], explode=(0,0.01,0,0,0), labels=['', '', '', '', ''], labeldistance=0.4, startangle=170, rotatelabels=True, textprops={'fontsize': 8});
circ = pl.matplotlib.patches.Circle((-0.35, 0.6), 0.1, color='black')
pl.gca().add_patch(circ)
pl.savefig(f"{rootpath}/allco_pacman.png", bbox_inches='tight', dpi=300)
pl.clf()
pl.pie([.1, 1, 0, 0, 0], explode=(0,0.01,0,0,0), labels=['', '', '', '', ''], labeldistance=0.4, startangle=160, rotatelabels=True, textprops={'fontsize': 8});
circ = pl.matplotlib.patches.Circle((-0.35, 0.6), 0.1, color='black')
pl.gca().add_patch(circ)
pl.savefig(f"{rootpath}/allco_pacman_90pct.png", bbox_inches='tight', dpi=300)
pl.clf()
pl.pie([.05, 1, 0, 0, 0], explode=(0,0.01,0,0,0), labels=['', '', '', '', ''], labeldistance=0.4, startangle=165, rotatelabels=True, textprops={'fontsize': 8});
circ = pl.matplotlib.patches.Circle((-0.35, 0.6), 0.1, color='black')
pl.gca().add_patch(circ)
pl.savefig(f"{rootpath}/allco_pacman_95pct.png", bbox_inches='tight', dpi=300)
pl.clf()
pl.pie([.16777, .8333, 0, 0, 0], explode=(0,0.01,0,0,0), labels=['', '', '', '', ''], labeldistance=0.4, startangle=150, rotatelabels=True, textprops={'fontsize': 8});
circ = pl.matplotlib.patches.Circle((-0.35, 0.6), 0.1, color='black')
pl.gca().add_patch(circ)
pl.savefig(f"{rootpath}/allco_pacman_83pct.png", bbox_inches='tight', dpi=300)


pl.clf()
pl.pie([0.1, 0.8333333, 0.0222222, 0.0222222], explode=(0,0.1,0,0), labels=['PAHs & Dust', 'CO ice / H$_2$ = 5$\\times10^{-4}$', 'CO$_2$ ice', 'Other'], labeldistance=0.4, startangle=170, rotatelabels=True, textprops={'fontsize': 8});

pl.title("Local Carbon:\n C/H$\sim10^{-3.5} = 3\\times10^{-4}$\nwith GC measured CO ice")

pl.savefig(f"{rootpath}/standard_carbon_distribution_exploded_h2_with_higher_GC_CO_levels.png", bbox_inches='tight', dpi=300)
print(f"{rootpath}/standard_carbon_distribution_exploded_h2_with_higher_GC_CO_levels.png")

circ = pl.matplotlib.patches.Circle((-0.35, 0.6), 0.1, color='black')
pl.gca().add_patch(circ)
pl.savefig(f"{rootpath}/standard_carbon_distribution_exploded_h2_with_higher_GC_CO_levels_pacman.png", bbox_inches='tight', dpi=300)
