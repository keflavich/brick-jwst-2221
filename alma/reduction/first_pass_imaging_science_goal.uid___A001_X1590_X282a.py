# Use 2828 - that's the Sci Goal
# copied from scriptforimaging
"""

"""

for spw in (25,27,29,31):

    tclean(vis=['.ms','ms'],
           imagename=f'.Brick_sci.spw{spw}.cube.I.manual',
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
