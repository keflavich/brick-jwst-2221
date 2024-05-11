import os
from astropy import coordinates
from astropy import units as u
from radio_beam import Beam
from astropy.io import fits
import pylab as pl
from astropy import visualization
from astropy import wcs
import PIL

import matplotlib.colors as mcolors
import numpy as np
from astropy.visualization import simple_norm
import pyavm
from pyavm.exceptions import NoXMPPacketFound
import glob

from toasty import study, image as timage, pyramid, builder, merge
from wwt_data_formats import write_xml_doc, folder, place, imageset
from astropy.coordinates import SkyCoord

from astropy.wcs.utils import fit_wcs_from_points

from aces.visualization.toast_aces import fits_to_avmpng


def toast(imfn, targetdir=None):

    if targetdir is None:
        tdr = os.path.basename(imfn).replace(".png", "")
        targetdir = f'/orange/adamginsburg/web/public/jwst/brick_2221/toasts/{tdr}'

    try:
        img = PIL.Image.open(imfn)
        avm = pyavm.AVM.from_image(imfn)
        wcs = avm.to_wcs()
    except NoXMPPacketFound:
        return None

    tim = timage.Image.from_pil(img)
    data = np.array(img)

    if 'GAL' in wcs.wcs.ctype[0] or 'GLON' in wcs.wcs.ctype[0]:
        points = np.mgrid[0:data.shape[1]:1000, 0:data.shape[0]:1000]
        points = points.reshape(2, np.prod(points.shape[1:])).T
        wpoints = wcs.pixel_to_world(points[:, 0], points[:, 1])

        wcsfk5 = fit_wcs_from_points((points[:, 0], points[:, 1]), wpoints.fk5)
    else:
        wcsfk5 = wcs

    height, width, _ = data.shape

    bui = builder.Builder(pyramid.PyramidIO(targetdir))
    stud = bui.prepare_study_tiling(tim)
    if not os.path.exists(f'{targetdir}/0/0/0_0.png'):
    #if True:  # always redo
        bui.execute_study_tiling(tim, stud)
        merge.cascade_images(
            bui.pio, start=7, merger=merge.averaging_merger, cli_progress=True
        )
    assert os.path.exists(f'{targetdir}/0/0/0_0.png'), "Failed"
    url = targetdir.replace("/orange/adamginsburg/web/public/",
                            "https://data.rc.ufl.edu/pub/adamginsburg/")
    buisuf = bui.imgset.url
    bui.imgset.url = url + '/' + buisuf
    # not sure I have this right yet... there is a more sensible way to construct this URL
    bui.imgset.credits_url = bui.imgset.url.replace("pub", "secure").replace('/' + buisuf, '.fits')
    wcsfk5 = timage._flip_wcs_parity(wcsfk5, height - 1)
    bui.apply_wcs_info(wcsfk5, width=width, height=height)
    bui.imgset.thumbnail_url = bui.imgset.url.format(0, 0, 0, 0)
    bui.imgset.name = os.path.basename(targetdir)
    bui.imgset.set_position_from_wcs(wcsfk5.to_header(), width=width, height=height,
                                     place=bui.place, fov_factor=2.5)
    # maybe this is only for Brick?
    bui.imgset.bottoms_up = False
    #bui.imgset.rotation_deg = 90
    bui.imgset.offset_x = width/2
    bui.imgset.offset_y = height/2

    #ctr = SkyCoord(0.1189 * u.deg, -0.05505 * u.deg, frame='galactic').fk5
    ctr = wcs.pixel_to_world(tim.shape[1]/2, tim.shape[0]/2)
    bui.place.ra_hr = ctr.ra.hourangle
    bui.place.dec_deg = ctr.dec.deg
    bui.place.zoom_level = 6

    fldr = bui.create_wtml_folder()

    # write both the 'rel' and 'full' URL versions
    bui.write_index_rel_wtml()
    with open(os.path.join(bui.pio._base_dir, "index.wtml"), 'w') as fh:
        write_xml_doc(fldr.to_xml(), dest_stream=fh)
    print("Wrote ", os.path.join(bui.pio._base_dir, "index.wtml"))
    return os.path.join(bui.pio._base_dir, "index.wtml")


def make_all_indexes():
    indexes = []
    imlist = (glob.glob("/orange/adamginsburg/web/public/jwst/brick_2221/*png") +
              glob.glob("/orange/adamginsburg/web/public/jwst/brick_2221/cloudc/*png"))
    assert "/orange/adamginsburg/web/public/jwst/brick_2221/BrickJWST_212-187-182_RGB_unrotated.png" in imlist
    print(f"imlist is {imlist}")
    for imfn in imlist:
        if 'residual' in imfn:
            continue
        tdr = os.path.basename(imfn).replace(".png", "")
        print(imfn, tdr)
        try:
            ind = toast(imfn,
                        targetdir=f'/orange/adamginsburg/web/public/jwst/brick_2221/toasts/{tdr}')
        except Exception as ex:
            print(f'{imfn} failed')
            print(ex)
            #raise
            continue
        if ind is not None:
            indexes.append(ind)

    return indexes


def make_joint_index(indexes):
    fld = folder.Folder()
    fld.browseable = True
    fld.group = 'Explorer'
    fld.name = 'Brick + Cloud C'
    fld.searchable = True
    fld.thumbnail = 'https://data.rc.ufl.edu/pub/adamginsburg/toasts/brick_2221/BrickJWST_merged_longwave_narrowband_lighter/0/0/0_0.png'
    fld.type = 'SKY'
    fld.children = []

    acestens = folder.Folder.from_file("/orange/adamginsburg/web/public/ACES/MUSTANG_Feather/index_rel.wtml")
    acestens.children[0].name = 'ACES+TENS (toasty)'
    acestens.children[0].thumbnail = 'https://data.rc.ufl.edu/pub/adamginsburg/ACES/MUSTANG_Feather/0/0/0_0.png'

    ctr = SkyCoord(0.1189 * u.deg, -0.05505 * u.deg, frame='galactic').fk5
    acestens.children[0].ra_hr = ctr.ra.hourangle
    acestens.children[0].dec_deg = ctr.dec.deg
    acestens.children[0].zoom_level = 4

    fld.children.extend(acestens.children)

    for ind in indexes:
        newfld = folder.Folder.from_file(ind)
        name = ind.split(os.sep)[-2].replace("_mosaic", "")
        assert '/' not in name
        newfld.children[0].name = name
        for child in newfld.children:
            child.thumbnail = child.foreground_image_set.url.format("", "0", "0", "0")
        fld.children.extend(newfld.children)
        with open('/orange/adamginsburg/web/public/jwst/brick_2221/toasts/BrickToast.wtml', 'w') as fh:
            write_xml_doc(fld.to_xml(), dest_stream=fh)


def make_place_notoast(fn):
    imfn = fn

    url = fn.replace("/orange/adamginsburg/web/public/",
                     "https://data.rc.ufl.edu/pub/adamginsburg/")

    img = PIL.Image.open(imfn)
    img.thumbnail((100, 100))
    img.save(imfn.replace(".png", "_thumbnail.png"))
    img = PIL.Image.open(imfn)

    avm = pyavm.AVM.from_image(imfn)
    ww = avm.to_wcs()
    height, width, _ = np.array(img).shape

    pl = place.Place()
    pl.name = os.path.splitext(os.path.basename(imfn))[0]
    pl.thumbnail = f'{url.replace(".png", "_thumbnail.png")}'
    pl.constellation = 'SAGITTARIUS'
    pl.data_set_type = 'SKY'
    pl.opacity = 100
    pl.zoom_level = 2

    imgset = imageset.ImageSet()
    imgset.set_position_from_wcs(ww.to_header(), width=width, height=height, place=pl, fov_factor=2.5)
    imgset.url = url
    imgset.thumbnail_url = pl.thumbnail
    imgset.bottoms_up = False
    #imgset.rotation_deg = 90
    imgset.band_pass = "INFRARED"
    imgset.offset_x = width/2
    imgset.offset_y = height/2
    #observed_xml = imgset.to_xml()

    pl.ra_hr = imgset.center_x / 15
    pl.dec_deg = imgset.center_y

    pl.foreground_image_set = imgset
    pl.zoom_level = 1

    return pl

def make_joint_index_notoast():
    fld = folder.Folder()
    fld.browseable = True
    fld.group = 'Explorer'
    fld.name = 'Brick + Cloud C'
    fld.searchable = True
    fld.thumbnail = 'https://data.rc.ufl.edu/pub/adamginsburg/toasts/brick_2221/BrickJWST_merged_longwave_narrowband_lighter/0/0/0_0.png'
    fld.type = 'SKY'
    fld.children = []

    acestens = folder.Folder.from_file("/orange/adamginsburg/web/public/ACES/MUSTANG_Feather/index_rel.wtml")
    acestens.children[0].name = 'ACES+TENS (toasty)'
    acestens.children[0].thumbnail = 'https://data.rc.ufl.edu/pub/adamginsburg/ACES/MUSTANG_Feather/0/0/0_0.png'

    ctr = SkyCoord(0.1189 * u.deg, -0.05505 * u.deg, frame='galactic').fk5
    acestens.children[0].ra_hr = ctr.ra.hourangle
    acestens.children[0].dec_deg = ctr.dec.deg
    acestens.children[0].zoom_level = 4

    fld.children.extend(acestens.children)


    imlist = (glob.glob("/orange/adamginsburg/web/public/jwst/brick_2221/*png") +
              glob.glob("/orange/adamginsburg/web/public/jwst/brick_2221/cloudc/*png"))
    assert "/orange/adamginsburg/web/public/jwst/brick_2221/BrickJWST_212-187-182_RGB_unrotated.png" in imlist
    #print(f"imlist is {imlist}")
    for imfn in imlist:
        if 'thumbnail' in imfn:
            continue
        try:
            avm = pyavm.AVM.from_image(imfn)
            avm.to_wcs()
            print(imfn, 'success')
        except Exception as ex:
            print(imfn, ex)
            continue


        try:
            place = make_place_notoast(imfn)
            fld.children.append(place)
        except Exception as ex:
            print(ex)
            continue

    with open('/orange/adamginsburg/web/public/jwst/brick_2221/Brick_NoToast.wtml', 'w') as fh:
        write_xml_doc(fld.to_xml(), dest_stream=fh)



def main():
    print("Making no-toast index")
    make_joint_index_notoast()

    print("Making all indexes")
    indexes = make_all_indexes()
    print(indexes)

    print("Making joint indexes")
    make_joint_index(indexes)


if __name__ == "__main__":
    main()
    make_joint_index(indexes)


if __name__ == "__main__":
    main()
