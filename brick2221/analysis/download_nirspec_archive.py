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
for proposal_id in [1959, 1854, 3702, 4358, 5437, 5804, 5064, 6095, 6161, 1309, 1611, 3222, 5552, 1939, 1236, 1257, 1186, 1960, 1290, 5791, ]:
    obstable = Observations.query_criteria(project='JWST', instrument_name='NIRSpec*', proposal_id=proposal_id)
    if len(obstable) > 0:
        data_products_by_obs = Observations.get_product_list(obstable)
        products = Observations.filter_products(data_products_by_obs, productType=["SCIENCE",], extension='fits', calib_level=[3,4])
        Observations.download_products(products)
