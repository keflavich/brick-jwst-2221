import glob, os, shutil


# make sure we're here...
tmpdir = os.getenv('SLURM_TMPDIR')
os.chdir(f'{tmpdir}/cloudc_2828')
assert os.getcwd() == f'{tmpdir}/cloudc_2828'

workdir = '/orange/adamginsburg/jwst/brick/alma/2021.1.00363.S/science_goal.uid___A001_X1590_X2828/group.uid___A001_X1590_X2829/member.uid___A001_X1590_X282a/calibrated/working'
mses = ['uid___A002_Xf287d3_Xcd1e.ms','uid___A002_Xfbe192_X54c.ms','uid___A002_Xfbf8a1_Xfe1.ms']

for spw in (25,27,29,31):
    splitnames = [msname.replace(".ms", f'_{spw}.split') for msname in mses]
    for msname in mses:
        split(f'{workdir}/{msname}', msname.replace(".ms", f'_{spw}.split'),
              spw=[str(spw)],
              datacolumn='corrected', field='CloudC')
    tclean(vis=splitnames,
           imagename=f'uid___A001_X1590_X282a.CloudC_sci.spw{spw}.cube.I.manual',
           field='CloudC',
           #datacolumn='corrected',
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

    for fn in glob.glob(f'uid___A001_X1590_X282a.CloudC_sci.spw{spw}.cube.I.manual*'):
        shutil.move(fn, workdir)

    for fn in splitnames:
        shutil.rmtree(fn)
