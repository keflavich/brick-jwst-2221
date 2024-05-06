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
from wwt_data_formats import write_xml_doc, folder
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
    wcsfk5 = wcs

    tim = timage.Image.from_pil(img)
    data = np.array(img)

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
    bui.imgset.rotation_deg = 90

    #ctr = SkyCoord(0.1189 * u.deg, -0.05505 * u.deg, frame='galactic').fk5
    ctr = wcs.pixel_to_world(tim.shape[1]/2, tim.shape[0]/2)
    bui.place.ra_hr = ctr.ra.hourangle
    bui.place.dec_deg = ctr.dec.deg
    bui.place.zoom_level = 4

    fldr = bui.create_wtml_folder()

    # write both the 'rel' and 'full' URL versions
    bui.write_index_rel_wtml()
    with open(os.path.join(bui.pio._base_dir, "index.wtml"), 'w') as fh:
        write_xml_doc(fldr.to_xml(), dest_stream=fh)
    print("Wrote ", os.path.join(bui.pio._base_dir, "index.wtml"))
    return os.path.join(bui.pio._base_dir, "index.wtml")


def make_all_indexes():
    indexes = []
    for imfn in (glob.glob("/orange/adamginsburg/web/public/jwst/brick_2221/*png") +
                 glob.glob("/orange/adamginsburg/web/public/jwst/brick_2221/cloudc/*png")):
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
            raise
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


def main():
    print("Making all indexes")
    indexes = make_all_indexes()
    print(indexes)

    print("Making joint indexes")
    make_joint_index(indexes)


if __name__ == "__main__":
    main()
