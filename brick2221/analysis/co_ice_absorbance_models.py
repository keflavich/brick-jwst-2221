"""
Superceded by icemodels/absorbance_in_filters.py
"""
raise Exception("This file is superceded by icemodels/absorbance_in_filters.py")
import glob
import os
import time
import multiprocessing as mp
from functools import partial

import numpy as np
import astropy.units as u
from astropy.table import Table
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map
import molmass

import icemodels
from icemodels import absorbed_spectrum, absorbed_spectrum_Gaussians, convsum, fluxes_in_filters, load_molecule, load_molecule_ocdb, atmo_model, molecule_data
from icemodels.core import composition_to_molweight, retrieve_gerakines_co, optical_constants_cache_dir, read_lida_file, read_ocdb_file

from brick2221.analysis.analysis_setup import basepath

import unicodedata

from astroquery.svo_fps import SvoFps
jfilts = SvoFps.get_filter_list('JWST')
jfilts.add_index('filterID')


def unicode_to_ascii(text):
    return ''.join(
        c if unicodedata.category(c) != 'No' else str(ord(c) - 8320)
        if '₀' <= c <= '₉' else unicodedata.normalize('NFKD', c).encode('ascii', 'ignore').decode('ascii')
        for c in text
    )


def load_tables(cache):
    # load tables
    if 'cotbs' not in cache:
        cotbs, h2otbs, co2tbs = {}, {}, {}
        for molname, tbs in {'H2O': h2otbs, 'CO': cotbs, 'CO2': co2tbs}.items():
            for fn in tqdm(glob.glob(f"{optical_constants_cache_dir}/*_{molname}:*") +
                        glob.glob(f"{optical_constants_cache_dir}/*_{molname}_*") +
                        glob.glob(f"{optical_constants_cache_dir}/*_{molname} *")
                        ):
                try:
                    tb = read_ocdb_file(fn)
                    basename = os.path.basename(os.path.splitext(fn)[0])
                    spl = basename.split("_")
                    idnum = int(spl[0])
                    #temperature = int(float([x for x in spl if 'K' in x][0].strip('K')))
                    temperature = int(tb.meta['temperature'].strip('K'))
                    tbs[('ocdb', idnum, temperature)] = tb
                    #tbs['ocdb'][idnum][temperature] = tb
                except Exception as ex:
                    with open(fn, 'r') as fh:
                        if 'ocdb' in fh.read().lower():
                            #print(fn)
                            continue
                    tb = read_lida_file(fn)
                    temperature = int(tb.meta['temperature'])
                    tbs[('lida', int(tb.meta['index']), temperature)] = tb
                except Exception as ex:
                    #print(fn, spl)
                    continue


        # rename columns to k
        for mol, tbs in {'H2O': h2otbs, 'CO': cotbs, 'CO2': co2tbs}.items():
            for key, tb in tbs.items():
                if 'k₁' in tb.colnames:
                    tb['k'] = tb['k₁']
                else:
                    pass
                    #print(f'{key} -> {tb.meta} had no k₁')

    cotbs[('ocdb', 63, 25)] = retrieve_gerakines_co(resolution='low')
    cotbs[('ocdb', 64, 25)] = retrieve_gerakines_co(resolution='high')

    return cotbs, h2otbs, co2tbs


def tryfloat(x):
    try:
        return float(x)
    except ValueError:
        return np.nan


def read_table_file(fn):
    try:
        tb = read_lida_file(fn)
        tb.meta['database'] = 'lida'
    except Exception as ex:
        try:
            tb = read_ocdb_file(fn)
            tb.meta['database'] = 'ocdb'
            if 'index' not in tb.meta:
                tb.meta['index'] = int(os.path.basename(fn).split('_')[0])
        except Exception as ex:
            return None

    try:
        tb['k'] = tb['k'].astype(float)
    except ValueError:
        try:
            kk = [tryfloat(x) for x in tb['k']]
            keep = ~np.isnan(kk)
            tb = tb[keep]
            tb['k'] = tb['k'].astype(float)
        except Exception as ex:
            print(f"Error reading table {fn}: {ex}")
            return None

    return tb


def make_mymix_tables():
    # make up our own H2O + CO at 25K with 3:1 ratio
    # 3-1 comes from Brandt's models plus the McClure 2023 paper
    mymix_tables = {}

    water_mastrapa = read_ocdb_file(f'{optical_constants_cache_dir}/240_H2O_(1)_25K_Mastrapa.txt') # h2otbs[('ocdb', 242, 25)] 242 is 50K....
    co2_gerakines = read_ocdb_file(f'{optical_constants_cache_dir}/55_CO2_(1)_8K_Gerakines.txt') # co2tbs[('ocdb', 55, 8)]
    ethanol = read_lida_file(f'{optical_constants_cache_dir}/87_CH3CH2OH_1_30.0K.txt')
    methanol = read_lida_file(f'{optical_constants_cache_dir}/58_CH3OH_1_25.0K.txt')
    ocn = read_lida_file(f'{optical_constants_cache_dir}/158_OCN-_1_12.0K.txt')
    #nh3 = read_ocdb_file(f'{optical_constants_cache_dir}/65_NH3_(1)_100K_Gerakines.txt')

    # modify OCN to get rid of the non-OCN contributions
    ocn['k'][(ocn['Wavelength'] < 4.5*u.um) | (ocn['Wavelength'] > 4.75*u.um)] = 0

    co_gerakines = gerakines = retrieve_gerakines_co()
    moltbls = {'CO': co_gerakines, 'H2O': water_mastrapa, 'CO2': co2_gerakines, 'CH3OH': methanol, 'CH3CH2OH': ethanol, 'OCN': ocn}

    grid = co_gerakines['Wavelength']
    grid = np.linspace(2.5*u.um, 5.0*u.um, 20000)

    for ii, (mol, composition) in enumerate([
                                            ('COplusH2O', 'H2O:CO (0.5:1)'),
                                            ('COplusH2O', 'H2O:CO (1:1)'),
                                            ('COplusH2O', 'H2O:CO (3:1)'),
                                            ('COplusH2O', 'H2O:CO (5:1)'),
                                            ('COplusH2O', 'H2O:CO (7:1)'),
                                            ('COplusH2O', 'H2O:CO (10:1)'),
                                            ('COplusH2O', 'H2O:CO (15:1)'),
                                            ('COplusH2O', 'H2O:CO (20:1)'),
                                            ('COplusH2OplusCO2', 'H2O:CO:CO2 (1:1:1)'),
                                            ('COplusH2OplusCO2', 'H2O:CO:CO2 (1:1:0.1)'),
                                            ('COplusH2OplusCO2', 'H2O:CO:CO2 (3:1:0.1)'),
                                            ('COplusH2OplusCO2', 'H2O:CO:CO2 (5:1:0.5)'),
                                            ('COplusH2OplusCO2', 'H2O:CO:CO2 (3:1:0.5)'),
                                            ('COplusH2OplusCO2', 'H2O:CO:CO2 (5:1:0.5)'),
                                            ('COplusH2OplusCO2', 'H2O:CO:CO2 (3:1:1)'),
                                            ('COplusH2OplusCO2', 'H2O:CO:CO2 (5:1:1)'),
                                            ('COplusH2OplusCO2', 'H2O:CO:CO2 (5:1:2)'),
                                            ('COplusH2OplusCO2', 'H2O:CO:CO2 (10:1:2)'),
                                            ('COplusH2OplusCO2', 'H2O:CO:CO2 (10:1:1)'),
                                            ('COplusH2OplusCO2', 'H2O:CO:CO2 (10:1:0.5)'),
                                            ('COplusH2OplusCO2', 'H2O:CO:CO2 (10:1:10)'),
                                            ('COplusH2OplusCO2', 'H2O:CO:CO2 (1:1:10)'),
                                            ('COplusH2OplusCO2', 'H2O:CO:CO2 (3:1:0.8)'),
                                            ('CO', 'CO 1'),
                                            ('H2O', 'H2O 1'),
                                            ('CO2', 'CO2 1'),
                                            ('COplusH2OplusCO2plusCH3OH', 'H2O:CO:CO2:CH3OH (1:1:1:1)'),
                                            ('COplusH2OplusCO2plusCH3OH', 'H2O:CO:CO2:CH3OH (1:1:0.1:0.1)'),
                                            ('COplusH2OplusCO2plusCH3OH', 'H2O:CO:CO2:CH3OH (1:1:0.1:1)'),
                                            ('COplusH2OplusCO2plusCH3OHplusCH3CH2OH', 'H2O:CO:CO2:CH3OH:CH3CH2OH (1:1:0.1:1:0.1)'),
                                            ('COplusH2OplusCO2plusCH3OHplusCH3CH2OH', 'H2O:CO:CO2:CH3OH:CH3CH2OH (1:1:0.1:0.1:0.1)'),
                                            ('COplusH2OplusCO2plusCH3OHplusCH3CH2OH', 'H2O:CO:CO2:CH3OH:CH3CH2OH (0.1:1:0.1:1:0.1)'),
                                            ('COplusH2OplusCO2plusCH3OHplusCH3CH2OH', 'H2O:CO:CO2:CH3OH:CH3CH2OH (0.01:1:0.1:0.1:1)'),
                                            ('COplusH2OplusCO2plusCH3OHplusCH3CH2OH', 'H2O:CO:CO2:CH3OH:CH3CH2OH (0.01:0.1:0.1:0.1:1)'),
                                            ('COplusH2OplusCO2plusCH3OHplusCH3CH2OH', 'H2O:CO:CO2:CH3OH:CH3CH2OH (1:1:1:1:1)'),
                                            ('COplusOCN', 'CO:OCN (1:1)'),
                                            ('COplusH2OplusOCN', 'H2O:CO:OCN (1:1:1)'),
                                            ('COplusH2OplusOCN', 'H2O:CO:OCN (1:1:0.02)'),
                                            ('COplusH2OplusOCN', 'H2O:CO:OCN (2:1:0.1)'),
                                            ('COplusH2OplusOCN', 'H2O:CO:OCN (2:1:0.5)'),
                                            ]):
        molspl = composition.split(' ')[0].split(':')
        compspl = composition.split(' ')[1].strip('()').split(':')

        mults = {key: float(mul) for key, mul in zip(molspl, compspl, )}

        co_plus_co2_plus_water_k = np.sum([
            (mult * np.interp(grid,
                              moltbls[mol]['Wavelength'][np.argsort(moltbls[mol]['Wavelength'])],
                              moltbls[mol]['k'][np.argsort(moltbls[mol]['Wavelength'])])) for mol, mult in mults.items()
        ], axis=0) / np.sum(list(mults.values()))
        # co_plus_co2_plus_water_k = (co_mult * gerakines['k'] +
        #                             h2o_mult * np.interp(grid, water_mastrapa['Wavelength'][inds], water_mastrapa['k'][inds],) +
        #                             co2_mult * np.interp(grid, co2_gerakines['Wavelength'], co2_gerakines['k']) +
        #                             ch3oh_mult * np.interp(grid, methanol['Wavelength'], methanol['k']) +
        #                             ch3ch2oh_mult * np.interp(grid, ethanol['Wavelength'], ethanol['k'])
        #                             ) / (
        #                                 co_mult + h2o_mult + co2_mult + ch3oh_mult + ch3ch2oh_mult
        #                                 )

        tbl = Table({'Wavelength': grid, 'k': co_plus_co2_plus_water_k})
        tbl.meta['composition'] = composition
        tbl.meta['density'] = 1*u.g/u.cm**3 # everything is close to 1 g/cm^3.... so this is just a close-enough guess
        tbl.meta['temperature'] = 25*u.K # really 8-25 K depending on molecule
        tbl.meta['index'] = ii
        tbl.meta['molecule'] = mol
        tbl.meta['database'] = 'mymix'
        tbl.meta['author'] = 'Mastrapa 2024, Gerakines 2020, etc'


        mymix_tables[(mol, ii, 25)] = tbl
        tbl.write(f'{optical_constants_cache_dir}/mymixes/{composition.replace(" ","_")}.ecsv', overwrite=True)

    return mymix_tables

mymix_tables = make_mymix_tables()


xarr = np.linspace(2.5*u.um, 5.0*u.um, 10000)
phx4000 = atmo_model(4000, xarr=xarr)
#xarr = phx4000['nu'].quantity.to(u.um, u.spectral())
cols = np.geomspace(1e15, 1e21, 25)

def process_table(args):
    if len(args) == 9:
        mol, key, consts, xarr, phx4000, cols, filter_data, transdata, basepath = args
        molfn = None
        tb = consts
    else:
        molfn, xarr, phx4000, cols, filter_data, transdata, basepath = args
        consts = tb = read_table_file(molfn)
        if tb is None:
            return []


    mol = tb.meta['molecule']
    database = tb.meta['database']
    mol_id = tb.meta['index']
    temperature = tb.meta['temperature']

    dmag_rows = []

    if 'k' not in consts.colnames:
        return []

    # if the wavelength range doesn't match, it just extrapolates.
    if u.Quantity(consts['Wavelength'].min(), u.um) > 5.0*u.um:
        return []

    dmags410, dmags466, dmags444, dmags356, dmags405, dmags323, dmags277, dmags300, dmags250, dmags335, dmags360, dmags480 = [], [], [], [], [], [], [], [], [], [], [], []

    molecule = mol.lower()
    try:
        compstr = unicode_to_ascii(consts.meta['composition'].replace("^", "").replace('c-', '').replace("Helium", "He"))
        molwt = u.Quantity(composition_to_molweight(compstr), u.Da)
    except molmass.molmass.FormulaError:
        print(f"Molecule {mol} with composition {consts.meta['composition']} failed to convert to molwt")
        return []
    except ValueError:
        if molecule == 'HCOOH' or molecule.split()[0].lower() == 'hcooh':
            molwt = 46*u.Da
        elif molecule == 'H2CO' or molecule.split()[0].lower() == 'h2co':
            molwt = 30*u.Da
        else:
            print(f"Molecule {mol} with composition {consts.meta['composition']} failed to convert to molwt")
            return []

    cmd_x = ('JWST/NIRCam.F410M', # 0
             'JWST/NIRCam.F466N', # 1
             'JWST/NIRCam.F356W', # 2
             'JWST/NIRCam.F444W', # 3
             'JWST/NIRCam.F405N', # 4
             'JWST/NIRCam.F323N', # 5
             'JWST/NIRCam.F277W', # 6
             'JWST/NIRCam.F300M', # 7
             'JWST/NIRCam.F250M', # 8
             'JWST/NIRCam.F335M', # 9
             'JWST/NIRCam.F360M', # 10
             'JWST/NIRCam.F480M', # 11
             )
    flxd_ref = fluxes_in_filters(xarr, phx4000['fnu'].quantity, filterids=cmd_x, transdata=transdata)

    author = consts.meta['author'] if 'author' in consts.meta else ''

    print(f"Processing file {molfn}: {mol} with composition {consts.meta['composition']} and molwt {molwt}")

    for col in cols:
        try:
            spec = absorbed_spectrum(col*u.cm**-2, consts, molecular_weight=molwt,
                                    spectrum=phx4000['fnu'].quantity,
                                    xarr=xarr,
                                    )
        except Exception as ex:
            print(f"Error processing file {molfn}: {ex}")
            print(consts)
            raise
        flxd = fluxes_in_filters(xarr, spec, filterids=cmd_x, transdata=transdata)

        # Calculate magnitudes
        mags_x_star = tuple(-2.5*np.log10(flxd_ref[cmd] / u.Quantity(filter_data[cmd], u.Jy))
                           for cmd in cmd_x)
        mags_x = tuple(-2.5*np.log10(flxd[cmd] / u.Quantity(filter_data[cmd], u.Jy))
                       for cmd in cmd_x)

        dmags405.append(mags_x[4]-mags_x_star[4])
        dmags444.append(mags_x[3]-mags_x_star[3])
        dmags356.append(mags_x[2]-mags_x_star[2])
        dmags466.append(mags_x[1]-mags_x_star[1])
        dmags410.append(mags_x[0]-mags_x_star[0])
        dmags323.append(mags_x[5]-mags_x_star[5])
        dmags277.append(mags_x[6]-mags_x_star[6])
        dmags300.append(mags_x[7]-mags_x_star[7])
        dmags250.append(mags_x[8]-mags_x_star[8])
        dmags335.append(mags_x[9]-mags_x_star[9])
        dmags360.append(mags_x[10]-mags_x_star[10])
        dmags480.append(mags_x[11]-mags_x_star[11])

        dmag_rows.append({
            'molecule': molecule,
            'mol_id': mol_id,
            'molwt': molwt,
            'database': database,
            'author': author,
            'composition': consts.meta['composition'],
            'temperature': consts.meta['temperature'],
            'density': consts.meta['density'],
            'column': col,
            'F356W': dmags356[-1],
            'F410M': dmags410[-1],
            'F444W': dmags444[-1],
            'F466N': dmags466[-1],
            'F405N': dmags405[-1],
            'F323N': dmags323[-1],
            'F277W': dmags277[-1],
            'F300M': dmags300[-1],
            'F250M': dmags250[-1],
            'F335M': dmags335[-1],
            'F360M': dmags360[-1],
            'F480M': dmags480[-1],
        })

    return dmag_rows

if __name__ == '__main__':


    #cotbs, h2otbs, co2tbs = load_tables(cache=locals())

    #molecules = {'H2O': h2otbs, 'CO': cotbs, 'CO2': co2tbs, 'H2O+CO': mymix_tables}

    # Create a dictionary of filter zero points
    filter_ids = ['JWST/NIRCam.F410M', 'JWST/NIRCam.F466N', 'JWST/NIRCam.F356W',
                  'JWST/NIRCam.F444W', 'JWST/NIRCam.F405N', 'JWST/NIRCam.F300M',
                  'JWST/NIRCam.F335M', 'JWST/NIRCam.F360M', 'JWST/NIRCam.F212N',
                  'JWST/NIRCam.F430M', 'JWST/NIRCam.F460M', 'JWST/NIRCam.F480M',
                  'JWST/NIRCam.F323N', 'JWST/NIRCam.F277W', 'JWST/NIRCam.F300M',
                  'JWST/NIRCam.F250M', 'JWST/NIRCam.F335M', 'JWST/NIRCam.F360M',
                  'JWST/NIRCam.F182M',
                  ]
    filter_data = {fid: float(jfilts.loc[fid]['ZeroPoint']) for fid in filter_ids}
    transdata = {fid: SvoFps.get_transmission_data(fid) for fid in filter_ids}

    # Create list of all tables to process
    all_tables = []
    #for mol, tbs in molecules.items():
    #    for key, consts in tbs.items():
    #        all_tables.append((mol, key, consts, xarr, phx4000, cols, filter_data, transdata, basepath))

    for key, consts in mymix_tables.items():
        all_tables.append(('H2O+CO', key, consts, xarr, phx4000, cols, filter_data, transdata, basepath))
    for fn in glob.glob(f'{optical_constants_cache_dir}/*txt'):
        #all_tables.append((tb.meta['molecule'], (db, int(tb.meta['index']), tb.meta['temperature']), tb, xarr, phx4000, cols, filter_data, transdata, basepath))
        all_tables.append((fn, xarr, phx4000, cols, filter_data, transdata, basepath))

    # Process all tables in parallel
    results = process_map(process_table,
                          all_tables,
                          max_workers=mp.cpu_count(),
                          desc="Processing tables",
                          unit="table")

    # Combine results and write tables for each molecule
    mol_rows = []
    for result in results:
        if result:
            mol_rows.extend(result)

    dmag_tbl = Table(mol_rows)
    dmag_tbl.write(f'{basepath}/tables/combined_ice_absorption_tables.ecsv', overwrite=True)
    dmag_tbl.add_index('database')
    dmag_tbl.add_index('mol_id')