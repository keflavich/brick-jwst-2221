from bs4 import BeautifulSoup
import requests
import keyring
from tqdm import tqdm

S = requests.Session()
resp = S.get('https://bulk.cv.nrao.edu/almadata/proprietary/2021.1.00363.S/X282a')

soup = BeautifulSoup(resp.text)

execution = soup.find('input', type='hidden').attrs['value']

password = keyring.get_password('https://asa.alma.cl/cas/login', 'keflavich')

resp_post = S.post('https://asa.alma.cl/cas/login?service=https%3a%2f%2fbulk.cv.nrao.edu%2falmadata%2fproprietary%2f2021.1.00363.S%2fX281a', data={'username': 'keflavich', 'password': password, 'execution': execution, '_eventId': 'submit', 'geolocation': None})

soup2 = BeautifulSoup(resp_post.text)

for file in soup2.findAll('a'):
    filename = file.text
    if 'gz' in filename:
        url = '/'.join([resp_post.url, filename])
        print(url)
        with S.get(url, stream=True) as stream:
            stream.raise_for_status()
            with open(filename, 'wb') as fh:
                for chunk in tqdm(stream.iter_content(1024)):
                    fh.write(chunk)
