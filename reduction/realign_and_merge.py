"""
Run only the realign & merge a plus b steps
"""
import sys
sys.path.remove('')
import regions

from align_to_catalogs import realign_to_vvv, realign_to_catalog, merge_a_plus_b, retrieve_vvv, main as align_main

def main(field='001'):
    align_main()
    
    for filtername in ( 'f405n', 'f410m', 'f466n', 'f182m', 'f187n', 'f212n',):
        print(f"Reprojecting filter {filtername}")
        out = merge_a_plus_b(filtername, fieldnumber=field, suffix='realigned-to-refcat')
        print(f"Wrote {out}")
        out = merge_a_plus_b(filtername, fieldnumber=field, suffix='realigned-to-vvv', outsuffix='merged-reproject-vvv')
        print(f"Wrote {out}")

if __name__ == '__main__':
    main()