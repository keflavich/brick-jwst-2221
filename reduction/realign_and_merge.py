"""
Run only the realign & merge a plus b steps
"""
import sys
sys.path.remove('')
import regions

from align_to_catalogs import realign_to_vvv, realign_to_catalog, merge_a_plus_b, retrieve_vvv, main as align_main

propid_filters = {'2221': ( 'f405n', 'f410m', 'f466n', 'f182m', 'f187n', 'f212n',),
                  '1182': ('f444w', 'f356w', 'f200w', 'f115w')
                   }

def main(field='001'):
    align_main()
    
    for propid in propid_filters:
        for filtername in propid_filters[propid]:
            print(f"Reprojecting filter {filtername}")
            out = merge_a_plus_b(filtername, fieldnumber=field, suffix='realigned-to-refcat', proposal_id=propid)
            print(f"Wrote {out}")
            out = merge_a_plus_b(filtername, fieldnumber=field, suffix='realigned-to-vvv', outsuffix='merged-reproject-vvv', proposal_id=propid)
            print(f"Wrote {out}")

if __name__ == '__main__':
    main()