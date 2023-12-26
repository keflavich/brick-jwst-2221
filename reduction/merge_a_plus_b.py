import warnings
from align_to_catalogs import merge_a_plus_b, realign_to_catalog
from astropy.table import Table
from astropy import units as u

if __name__ == "__main__":
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-f", "--filternames", dest="filternames",
                      default='F466N,F405N,F410M,F212N,F182M,F187N',
                      help="filter name list", metavar="filternames")
    (options, args) = parser.parse_args()

    filternames = options.filternames.split(",")
    print(options)

    field_to_reg_mapping = {'2221': {'001': 'brick', '002': 'cloudc'},
                            '1182': {'004': 'brick'}}

    fields = ('001', '002', '004')
    basepath = '/orange/adamginsburg/jwst/brick/'

    abs_refcat = f'{basepath}/catalogs/crowdsource_based_nircam-f405n_reference_astrometric_catalog.ecsv'
    reftbl = Table.read(abs_refcat)

    raoffset = 0*u.deg
    decoffset = 0*u.deg

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for field in fields:
            for filtername in filternames:
                wavelength = int(filtername[1:4])
                for destreak_suffix in ('_destreak', '', '_nodestreak', ):
                    # _nodestreak probably doesn't exist
                    for proposal_id in ('1182', '2221'):
                        if field in field_to_reg_mapping[proposal_id]:
                            print(f"Main Loop: {proposal_id} + {filtername} + {field}={field_to_reg_mapping[proposal_id][field]} + destreak={destreak_suffix}")

                            try:
                                for module in ('nrca', 'nrcb'):
                                    realigned_refcat_filename = f'{basepath}/{filtername.upper()}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_clear-{filtername.lower()}-{module}{destreak_suffix}_realigned-to-refcat.fits'
                                    realigned = realign_to_catalog(reftbl['skycoord'],
                                                                   filtername=filtername.lower(),
                                                                   basepath=basepath, module=module,
                                                                   fieldnumber=field,
                                                                   mag_limit=20,
                                                                   proposal_id=proposal_id,
                                                                   max_offset=(0.4 if wavelength > 250 else 0.2)*u.arcsec,
                                                                   imfile=realigned_refcat_filename,
                                                                   raoffset=raoffset,
                                                                   decoffset=decoffset)
                            except Exception as ex:
                                print(ex)

                            try:
                                merge_a_plus_b(filtername, basepath=basepath, fieldnumber=field, suffix=f'{destreak_suffix}_realigned-to-refcat',
                                                proposal_id=proposal_id)
                                print("DONE Merging already-combined nrca + nrcb modules"
                                      f": {proposal_id} + {filtername} + {field}={field_to_reg_mapping[proposal_id][field]} + destreak={destreak_suffix}")
                            except Exception as ex:
                                print(ex)

                            try:
                                module = 'merged-reproject'
                                realigned_refcat_filename = f'{basepath}/{filtername.upper()}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_clear-{filtername.lower()}-{module}{destreak_suffix}_realigned-to-refcat.fits'
                                realigned = realign_to_catalog(reftbl['skycoord'],
                                                               filtername=filtername.lower(),
                                                               basepath=basepath, module='merged-reproject',
                                                               fieldnumber=field,
                                                               mag_limit=20,
                                                               proposal_id=proposal_id,
                                                               max_offset=(0.4 if wavelength > 250 else 0.2)*u.arcsec,
                                                               imfile=realigned_refcat_filename,
                                                               raoffset=raoffset,
                                                               decoffset=decoffset)
                                print("Realigned merged-reproject; it should have had zero offset")
                            except Exception as ex:
                                print(ex)