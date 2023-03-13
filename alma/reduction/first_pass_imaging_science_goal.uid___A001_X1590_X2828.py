
"""
from sys import path
path.append('/users/lvidela/AIV/science/analysis_scripts/')
import analysisUtils as aU
es = aU.stuffForScienceDataReduction()

# Obtained cellsize,  0.016260514167053215
# Expected cellSize,  0.03
# Obtained FoV,  17.898948195446465
# imsize,  596.6316065148822
#uid___A001_X1590_X282a.CloudC_sci.spw{spw}.cube.I.manual



#Mosaic, Continuum
#==================================================
tclean(vis=['uid___A002_Xf287d3_Xcd1e.ms','uid___A002_Xfbe192_X54c.ms','uid___A002_Xfbf8a1_Xfe1.ms'],
       imagename='uid___A001_X1590_X282a.CloudC_sci.spw{spw}.cube.I.manual',
       field='CloudC',
       datacolumn='corrected',
       spw=['25,27,29,31'],
       specmode='cube',
       threshold='0.001mJy',
       imsize=[6000,7000],
       cell=['0.03arcsec'],
       niter=10000,
       deconvolver='hogbom',
       phasecenter = 'J2000 17:46:19.157 -028.35.15.041',
       gridder='mosaic',
       weighting='briggs',
       robust=0.5,
       pbcor=True,
       pblimit=0.2,
       interactive=True)



#Create fits datacubes for science targets
exportfits(imagename='uid___A001_X1590_X282a.CloudC_sci.spw{spw}.cube.I.manual.image.pbcor', fitsimage='uid___A001_X1590_X282a.CloudC_sci.spw{spw}.cube.I.manual.image.pbcor.fits')
exportfits(imagename='uid___A001_X1590_X282a.CloudC_sci.spw{spw}.cube.I.manual.pb', fitsimage='uid___A001_X1590_X282a.CloudC_sci.spw{spw}.cube.I.manual.pb.fits')
exportfits(imagename='uid___A001_X1590_X282a.CloudC_sci.spw{spw}.cube.I.manual.mask', fitsimage='uid___A001_X1590_X282a.CloudC_sci.spw{spw}.cube.I.manual.mask.fits')


"""

for spw in (25,27,29,31):
    tclean(vis=['uid___A002_Xf287d3_Xcd1e.ms','uid___A002_Xfbe192_X54c.ms','uid___A002_Xfbf8a1_Xfe1.ms'],
           imagename=f'uid___A001_X1590_X282a.CloudC_sci.spw{spw}.cube.I.manual',
           field='CloudC',
           datacolumn='corrected',
           spw=[str(spw)],
           specmode='cube',
           threshold='0.1mJy',
           imsize=[6000,7000],
           cell=['0.03arcsec'],
           niter=1000,
           deconvolver='hogbom',
           phasecenter = 'J2000 17:46:19.157 -028.35.15.041',
           gridder='mosaic',
           weighting='briggs',
           robust=0.5,
           pbcor=True,
           pblimit=0.2,
           interactive=False)



    #Create fits datacubes for science targets
    exportfits(imagename=f'uid___A001_X1590_X282a.CloudC_sci.spw{spw}.cube.I.manual.image.pbcor', fitsimage=f'uid___A001_X1590_X282a.CloudC_sci.spw{spw}.cube.I.manual.image.pbcor.fits')
    exportfits(imagename=f'uid___A001_X1590_X282a.CloudC_sci.spw{spw}.cube.I.manual.pb', fitsimage=f'uid___A001_X1590_X282a.CloudC_sci.spw{spw}.cube.I.manual.pb.fits')
    exportfits(imagename=f'uid___A001_X1590_X282a.CloudC_sci.spw{spw}.cube.I.manual.mask', fitsimage=f'uid___A001_X1590_X282a.CloudC_sci.spw{spw}.cube.I.manual.mask.fits')

