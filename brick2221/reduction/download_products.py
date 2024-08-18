from astroquery.mast import Mast, Observations
from astropy import table
from tqdm import tqdm
import os


with open(os.path.expanduser('/home/adamginsburg/.mast_api_token'), 'r') as fh:
    api_token = fh.read().strip()
Mast.login(api_token.strip())
Observations.login(api_token)

basepath = '/orange/adamginsburg/jwst/brick/cache'
Observations.cache_location = basepath

obs_table = Observations.query_criteria(
    calib_level=3,
    proposal_id="2221",
    proposal_pi="Ginsburg*",
    )
# obs_table = Observations.query_criteria(
#     calib_level=3,
#     proposal_id="1182",
#     )
# for finding filters : obs_table[np.char.find(obs_table['obs_id'], filtername.lower()) >= 0]

#data_products_by_obs = Observations.get_product_list(obs_table[obs_table['calib_level'] == 3])

product_list = [Observations.get_product_list(obs) for obs in tqdm(obs_table)]
data_products_by_id = table.vstack(product_list)
#data_products_by_id = Observations.get_product_list(obs_table['obsid'])

data_products_by_id = data_products_by_id[data_products_by_id['calib_level']==3]

products = Observations.filter_products(data_products_by_id,
                                        productType=["SCIENCE", "PREVIEW"],
                                        extension="fits")

manifest = Observations.download_products(products, download_dir='/orange/adamginsburg/jwst/brick')


if False:
    products_asn = Observations.filter_products(data_products_by_id,
                                                extension="json")


    manifest = Observations.download_products(products_asn, download_dir='/orange/adamginsburg/jwst/brick')
