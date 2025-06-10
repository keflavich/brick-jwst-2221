"""
run this in /orange/adamginsburg/jwst/spectra?
"""
from astroquery.mast import Observations
from astroquery.mast import Mast

# 1802: IPA
obstable = Observations.query_criteria(project='JWST', instrument_name='NIRSpec*', proposal_id=1802)

data_products_by_obs = Observations.get_product_list(obstable)

# only x1d and s3d here
products = Observations.filter_products(data_products_by_obs, productType=["SCIENCE",], extension='fits', calib_level=[3,4])

Observations.download_products(products)


# JOYS: Sources targeted within the GTO programs 1290 (PI: E. F. van Dishoeck), 1236 (PI: M.E. Ressler), 1257 (PI: T. P. Ray), 1186 (PI: T. P. Greene) and the GO program 1960 (PI: E. F. van Dishoeck)
# 1939: galcen
# 3222: IRAS16293 "Cask-strength clouds: high percentage of methanol and HDO ices"
# 2151: only MIRI MRS
for proposal_id in [1959, 2640, 1960, 1854, 3222, 3702, 4358, 5437, 5804, 5064, 6095, 6161, 1309, 1611, 3222, 5552, 1939, 1236, 1257, 1186, 1960, 1290, 5791, 5437]:
    obstable = Observations.query_criteria(project='JWST', instrument_name='NIRSpec*', proposal_id=proposal_id)
    if len(obstable) > 0:
        for ii in range(len(obstable)):
            try:
                data_products_by_obs = Observations.get_product_list(obstable[ii:ii+1])
                products = Observations.filter_products(data_products_by_obs, productType=["SCIENCE",], extension='fits', calib_level=[3,4])
                Observations.download_products(products)
            except Exception as e:
                print(f"Error downloading products for {obstable[ii]['obsid']}: {e}")
                continue
