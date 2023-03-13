
"""
# ALMA Data Reduction Script

# Created using $Id: almaqa2isg.py,v 2.2 2022/12/19 16:37:37 dpetry Exp $

thesteps = []
step_title = {0: 'Continuum image for target CloudC, spws [25, 27, 29, 31]',
              1: 'Export images to FITS format'}

if 'applyonly' not in globals(): applyonly = False
try:
  print('List of steps to be executed ...'+str(mysteps))
  thesteps = mysteps
except:
  print('global variable mysteps not set.')
if (thesteps==[]):
  thesteps = range(0,len(step_title))
  print('Executing all steps: ', thesteps)

# The Python variable 'mysteps' will control which steps
# are executed when you start the script using
#   execfile('scriptForCalibration.py')
# e.g. setting
#   mysteps = [2,3,4]
# before starting the script will make the script execute
# only steps 2, 3, and 4
# Setting mysteps = [] will make it execute all steps.


import os
import sys

thevis = ['uid___A002_Xfbf8a1_X246d.ms']

# Continuum image for target CloudC, spws [25, 27, 29, 31]
mystep = 0
if(mystep in thesteps):
  casalog.post('Step '+str(mystep)+' '+step_title[mystep],'INFO')
  print('\nStep '+str(mystep)+' '+step_title[mystep])


  os.system('rm -rf CloudC_sci.spw25_27_29_31.cont.I.manual*')
  tclean(vis = thevis,
         imagename = 'CloudC_sci.spw25_27_29_31.cont.I.manual',
         field = 'CloudC', # IDs from representative MS: '4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153'
         intent = 'OBSERVE_TARGET#ON_SOURCE',
         phasecenter = 3,
         stokes = 'I',
         spw = '25,27,29,31',
         outframe = 'LSRK',
         specmode = 'cont',
         nterms = 1,
         imsize = [5120, 6000],
         cell = '0.036arcsec',
         deconvolver = 'hogbom',
         niter = 700,
         weighting = 'briggs',
         robust = 0.5,
         mask = '',
         gridder = 'mosaic',
         pbcor = True,
         threshold = '0.1 mJy',
         restoringbeam = 'common',
         interactive = True
         )

  # NOTE: enter the continuum channel selection in the spw parameter!


# Export images to FITS format
mystep = 1
if(mystep in thesteps):
  casalog.post('Step '+str(mystep)+' '+step_title[mystep],'INFO')
  print('\nStep '+str(mystep)+' '+step_title[mystep])

  myimages = ['CloudC_sci.spw25_27_29_31.cont.I.manual']

  for myimagebase in myimages:
    exportfits(imagename = myimagebase+'.image.pbcor',
               fitsimage = myimagebase+'.pbcor.fits',
               overwrite = True
               )
    if os.path.exists(myimagebase+'.pb'):
      exportfits(imagename = myimagebase+'.pb',
                 fitsimage = myimagebase+'.pb.fits',
                 overwrite = True
                 )
    if os.path.exists(myimagebase+'.mask'):
      exportfits(imagename = myimagebase+'.mask',
                 fitsimage = myimagebase+'.mask.fits',
                 overwrite = True
                 )

"""

