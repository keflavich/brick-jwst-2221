"""
Need isochrones.

From

 * https://waps.cfa.harvard.edu/MIST/
 * http://stev.oapd.inaf.it/cgi-bin/cmd
 * No: only very very old stars http://stellar.dartmouth.edu/models/isolf_new.html
 * no web interface http://www.astro.yale.edu/yapsi/
 * older models http://astro.df.unipi.it/stellar-models/index.php
"""

import requests
from bs4 import BeautifulSoup
from astropy.table import Table

def trynum(x):
    try:
        return int(x)
    except:
        try:
            return float(x)
        except:
            return x

def data_to_json(x):
    return {y.split(":")[0]: trynum(y.split(":")[1].strip()) for y in x.split("\n")}

# MIST
resp = requests.post('https://waps.cfa.harvard.edu/MIST/iso_form.php',
                     data=data_to_json("""version: 1.2
v_div_vcrit: vvcrit0.4
age_scale: log10
age_value:
age_range_low:
age_range_high:
age_range_delta:
age_list:
age_type: standard
FeH_value: 0
theory_output: basic
output_option: photometry
output: JWST
Av_value: 0"""))
resp.raise_for_status()
resp

url = 'https://waps.cfa.harvard.edu/MIST/' + BeautifulSoup(resp.text).find('a').attrs['href']
filename = os.path.basename(url)
url, filename

from tqdm.notebook import tqdm
with requests.get(url, stream=True) as stream:
    stream.raise_for_status()
    with open(f'{basepath}/isochrones/MIST_isochrone_package.zip', 'wb') as fh:
        for chunk in tqdm(stream.iter_content(8192)):
            fh.write(chunk)

import zipfile

ls -lh $basepath/isochrones/MIST_isochrone_package.zip

with zipfile.ZipFile(f'{basepath}/isochrones/MIST_isochrone_package.zip') as zf:
    print(zf.infolist())
    zf.extractall(f'{basepath}/isochrones')


### stev / padova

S = requests.Session()
S.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36'
rr1 = S.get('http://stev.oapd.inaf.it/cgi-bin/cmd')
rr1.raise_for_status()
padov = S.post('http://stev.oapd.inaf.it/cgi-bin/cmd_3.6',
                      data=data_to_json("""submit_form: Submit
cmd_version: 3.5
track_parsec: parsec_CAF09_v1.2S
track_colibri: parsec_CAF09_v1.2S_S_LMC_08_web
track_postagb: no
n_inTPC: 10
eta_reimers: 0.2
kind_interp: 1
kind_postagb: -1
photsys_file: YBC_tab_mag_odfnew/tab_mag_ubvrijhk.dat
photsys_version: YBCnewVega
dust_sourceM: dpmod60alox40
dust_sourceC: AMCSIC15
kind_mag: 2
kind_dust: 0
extinction_av: 0.0
extinction_coeff: constant
extinction_curve: cardelli
kind_LPV: 3
imf_file: tab_imf/imf_kroupa_orig.dat
isoc_agelow: 1.0e
isoc_ageupp: 1.0e10
isoc_dage: 0.0
isoc_isagelog: 1
isoc_lagelow: 5
isoc_lageupp: 10.13
isoc_dlage: 0.25
isoc_ismetlog: 0
isoc_zlow: 0.0152
isoc_zupp: 0.03
isoc_dz: 0.0
isoc_metlow: -2
isoc_metupp: 0.3
isoc_dmet: 0.0
output_kind: 0
output_evstage: 1
lf_maginf: -15
lf_magsup: 20
lf_deltamag: 0.5
sim_mtot: 1.0e4
.cgifields: dust_sourceM
.cgifields: photsys_version
.cgifields: track_postagb
.cgifields: extinction_coeff
.cgifields: dust_sourceC
.cgifields: track_colibri
.cgifields: isoc_isagelog
.cgifields: isoc_ismetlog
.cgifields: track_parsec
.cgifields: extinction_curve
.cgifields: output_kind
.cgifields: kind_LPV
.cgifields: output_gzip"""))


url = 'http://stev.oapd.inaf.it/cmd/' + BeautifulSoup(padov.text).find('a').attrs['href']
filename = os.path.basename(url)
url, filename


from tqdm.notebook import tqdm
with requests.get(url, stream=True) as stream:
    stream.raise_for_status()
    with open(f'{basepath}/isochrones/padova_isochrone_package.dat', 'wb') as fh:
        for chunk in tqdm(stream.iter_content(8192)):
            fh.write(chunk)


