# copied from scriptforimaging
"""
from sys import path
path.append('/users/lvidela/AIV/science/analysis_scripts/')
import analysisUtils as aU
es = aU.stuffForScienceDataReduction()

# Obtained cellsize,  0.020638347590721137
# Expected cellSize,  0.03
# Obtained FoV,  17.89895052525803
# imsize,  596.6316841752677

#Mosaic, Continuum
#==================================================
tclean(vis=['uid___A002_Xfb8480_X136b9.ms','uid___A002_Xfb8480_X1827.ms'],
       imagename='uid___A001_X1590_X281a.Brick_sci.spw29.mfs.I.manual',
       field='Brick',
       datacolumn='corrected',
       spw=['25,27,29,31'],
       specmode='mfs',
       threshold='0.001mJy',
       imsize=[7500,5000],
       cell=['0.03arcsec'],
       niter=10000,
       phasecenter = 'J2000 17:46:09.294 -028.45.04.424' ,
       deconvolver='hogbom',
       gridder='mosaic',
       weighting='briggs',
       robust=0.5,
       pbcor=True,
       pblimit=0.2,
       interactive=True)


##Create fits datacubes for science targets
exportfits(imagename='uid___A001_X1590_X281a.Brick_sci.spw29.mfs.I.manual.image.pbcor', fitsimage='uid___A001_X1590_X281a.Brick_sci.spw29.mfs.I.manual.image.pbcor.fits')
exportfits(imagename='uid___A001_X1590_X281a.Brick_sci.spw29.mfs.I.manual.pb', fitsimage='uid___A001_X1590_X281a.Brick_sci.spw29.mfs.I.manual.pb.fits')
exportfits(imagename='uid___A001_X1590_X281a.Brick_sci.spw29.mfs.I.manual.mask', fitsimage='uid___A001_X1590_X281a.Brick_sci.spw29.mfs.I.manual.mask.fits')

"""

for spw in (25,27,29,31):

    tclean(vis=['uid___A002_Xfb8480_X136b9.ms','uid___A002_Xfb8480_X1827.ms'],
           imagename=f'uid___A001_X1590_X281a.Brick_sci.spw{spw}.cube.I.manual',
           field='Brick',
           datacolumn='corrected',
           spw=[str(spw)],
           specmode='cube',
           threshold='0.1mJy',
           imsize=[7500,5000],
           cell=['0.03arcsec'],
           niter=1000,
           phasecenter = 'J2000 17:46:09.294 -028.45.04.424' ,
           deconvolver='hogbom',
           gridder='mosaic',
           weighting='briggs',
           robust=0.5,
           pbcor=True,
           pblimit=0.2,
           interactive=False)


    ##Create fits datacubes for science targets
    exportfits(imagename=f'uid___A001_X1590_X281a.Brick_sci.spw{spw}.cube.I.manual.image.pbcor', fitsimage=f'uid___A001_X1590_X281a.Brick_sci.spw{spw}.cube.I.manual.image.pbcor.fits')
    exportfits(imagename=f'uid___A001_X1590_X281a.Brick_sci.spw{spw}.cube.I.manual.pb', fitsimage=f'uid___A001_X1590_X281a.Brick_sci.spw{spw}.cube.I.manual.pb.fits')
    exportfits(imagename=f'uid___A001_X1590_X281a.Brick_sci.spw{spw}.cube.I.manual.mask', fitsimage=f'uid___A001_X1590_X281a.Brick_sci.spw{spw}.cube.I.manual.mask.fits')
