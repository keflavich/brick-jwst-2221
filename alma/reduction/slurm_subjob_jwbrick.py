import glob, os, shutil
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u

import sys
# TODO: make ALMA-IMF pipeline installable
sys.path.append('/orange/adamginsburg/ALMA_IMF/reduction/reduction/')
import metadata_tools

# this script has to be run interactively, so casalog should be in the namespace
def logprint(string, origin='jwbrickjob', priority='INFO'):
    print(string)
    casalog.post(string, origin=origin, priority=priority)


# make sure we're here...
tmpdir = os.getenv('SLURM_TMPDIR')
fieldname = os.getenv('FIELDNAME') or 'cloudc_2828'

fulltmpdir = f'{tmpdir}/{fieldname}'
if not os.path.exists(fulltmpdir):
    os.mkdir(fulltmpdir)
os.chdir(fulltmpdir)
assert os.getcwd() == fulltmpdir

field = os.getenv('FIELD') or 'CloudC'
mous = os.getenv('MOUS') or 'uid___A001_X1590_X282a'

spw = os.getenv('SPW')
if spw is None:
    raise ValueError("Specify SPW")
logprint(f"SPW = {spw}")

nchan = int(os.getenv('NCHAN') or 16)
start = int(os.getenv('STARTCHAN') or 0)

workdir = os.getenv('WORK_DIR') or '/orange/adamginsburg/jwst/brick/alma/2021.1.00363.S/science_goal.uid___A001_X1590_X2828/group.uid___A001_X1590_X2829/member.uid___A001_X1590_X282a/calibrated/working'
mses = os.getenv('MSES').split() or ['uid___A002_Xf287d3_Xcd1e.ms','uid___A002_Xfbe192_X54c.ms','uid___A002_Xfbf8a1_Xfe1.ms']

logprint(f"Working in {tmpdir} based on files in {workdir}, using MSes ${mses}")
logprint(f"Variables are: startchan={start:04d}, nchan={nchan}, spw={spw}, mous={mous}, field={field}, fieldname={fieldname}, tmpdir={tmpdir}, fulltmpdir={fulltmpdir}")

if os.getenv('DOMERGE') and int(os.getenv('DOMERGE')) == 1:

    os.chdir(workdir)
    print(f"Merging in {os.getcwd()}")
    totalnchan = int(os.getenv('TOTALNCHAN'))
    for suffix in (".image", ".image.pbcor", ".model", ".mask", ".pb", ".psf", ".residual", ".weight", ".sumwt"):
        infiles = [f'{mous}.{field}_sci.spw{spw}.{ii:04d}+{nchan:03d}.cube.I.manual{suffix}' for ii in range(0, totalnchan+1, nchan)]
        print(f"Merging input files.  Infiles are:\n{infiles}")
        for fn in infiles:
            if not os.path.exists(fn):
                raise ValueError(f"File {fn} does not exist")
        print("Running the 'p' job")
        ia.imageconcat(outfile=f'{mous}.{field}.spw{spw}.merge.p{suffix}',
                       infiles=infiles, mode='p', relax=True)
else:
    if os.path.exists(os.path.join(workdir,
                                   f'{mous}.{field}_sci.spw{spw}.{start:04d}+{nchan:03d}.cube.I.manual.image')):
        results = glob.glob(f'{workdir}/{mous}.{field}_sci.spw{spw}.{start:04d}+{nchan:03d}.cube.I.manual*')
        logprint(f"Found completed results {results}.  Finishing without doing any more work.")
    else:
        splitnames = [msname.replace(".ms", f'_{spw}.split') for msname in mses]
        for msname in mses:
            # cannot split channels out because we don't know which ones we need yet
            split_kwargs = dict(vis=f'{workdir}/{msname}',
                                outputvis=msname.replace(".ms", f'_{spw}.split'),
                                spw=str(spw),
                                datacolumn='corrected',
                                field=field)
            logprint(f"split kwargs: {split_kwargs}")

            split(**split_kwargs)

        coosys, racen, deccen = metadata_tools.determine_phasecenter(splitnames[0], field, formatted=False)
        phasecenter = (racen, deccen)
        dra, ddec, pixscale = metadata_tools.determine_imsize(splitnames[0], field, phasecenter, pixfraction_of_fwhm=1/3.)
        imsize = [dra, ddec]
        cellsize = ['{0:0.2f}arcsec'.format(pixscale)] * 2
        phasecenter_formatted = metadata_tools.determine_phasecenter(splitnames[0], field, formatted=True)

        tclean_kwargs = dict(vis=splitnames,
                             imagename=f'{mous}.{field}_sci.spw{spw}.{start:04d}+{nchan:03d}.cube.I.manual',
               field=field,
               start=start,
               nchan=nchan,
               specmode='cube',
               threshold='6mJy',
               imsize=imsize,
               cell=cellsize,
               niter=10000,
               deconvolver='hogbom',
               phasecenter=phasecenter_formatted,
               # phasecenter='J2000 17:46:19.157 -028.35.15.041',
               gridder='mosaic',
               weighting='briggs',
               robust=0.5,
               pbcor=True,
               pblimit=0.2,
               interactive=False)
        logprint(f'Tclean kwargs: {tclean_kwargs}')

        tclean(**tclean_kwargs)



        #Create fits datacubes for science targets
        # (no reason to do this - we have to merge them)
        #exportfits(imagename=f'{mous}.{field}_sci.spw{spw}.{start:04d}+{nchan:03d}.cube.I.manual.image.pbcor', fitsimage=f'{mous}.{field}_sci.spw{spw}.{start:04d}+{nchan:03d}.cube.I.manual.image.pbcor.fits')
        #exportfits(imagename=f'{mous}.{field}_sci.spw{spw}.{start:04d}+{nchan:03d}.cube.I.manual.pb', fitsimage=f'{mous}.{field}_sci.spw{spw}.{start:04d}+{nchan:03d}.cube.I.manual.pb.fits')
        #exportfits(imagename=f'{mous}.{field}_sci.spw{spw}.{start:04d}+{nchan:03d}.cube.I.manual.mask', fitsimage=f'{mous}.{field}_sci.spw{spw}.{start:04d}+{nchan:03d}.cube.I.manual.mask.fits')

        for fn in glob.glob(f'{mous}.{field}_sci.spw{spw}.{start:04d}+{nchan:03d}.cube.I.manual*'):
            if os.path.realpath(fn) == os.path.join(workdir, fn):
                logprint(f"File {fn} is already in {workdir}")
            else:
                logprint(f"Moving {fn} to {workdir}")
                shutil.move(fn, workdir)

        # no need to do this
        #for fn in splitnames:
        #    logprint(f"Removing {fn}")
        #    shutil.rmtree(fn)
