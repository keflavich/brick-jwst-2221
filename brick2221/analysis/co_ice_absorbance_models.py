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

import icemodels
from icemodels import absorbed_spectrum, absorbed_spectrum_Gaussians, convsum, fluxes_in_filters, load_molecule, load_molecule_ocdb, atmo_model, molecule_data
from icemodels.core import composition_to_molweight, retrieve_gerakines_co, optical_constants_cache_dir, read_lida_file, read_ocdb_file

from brick2221.analysis.analysis_setup import basepath

from astroquery.svo_fps import SvoFps
jfilts = SvoFps.get_filter_list('JWST')
jfilts.add_index('filterID')

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

    return cotbs, h2otbs, co2tbs

cotbs, h2otbs, co2tbs = load_tables(cache=locals())


def make_mymix_tables():
    # make up our own H2O + CO at 25K with 3:1 ratio
    # 3-1 comes from Brandt's models plus the McClure 2023 paper
    mymix_tables = {}

    water_mastrapa = h2otbs[('ocdb', 242, 25)]
    co2_gerakines = co2tbs[('ocdb', 55, 8)]

    co_gerakines = gerakines = retrieve_gerakines_co()

    cotbs[('ocdb', 63, 25)] = co_gerakines
    cotbs[('ocdb', 64, 25)] = co_gerakines

    grid = co_gerakines['Wavelength']

    for ii, (mol, composition) in enumerate([('COplusH2O', 'H2O:CO (3:1)'),
                                            ('COplusH2O', 'H2O:CO (10:1)'),
                                            ('COplusH2O', 'H2O:CO (20:1)'),
                                            ('COplusH2OplusCO2', 'H2O:CO:CO2 (3:1:0.1)'),
                                            ('COplusH2OplusCO2', 'H2O:CO:CO2 (5:1:0.5)'),
                                            ('COplusH2OplusCO2', 'H2O:CO:CO2 (3:1:0.5)'),
                                            ('COplusH2OplusCO2', 'H2O:CO:CO2 (5:1:0.5)'),
                                            ('COplusH2OplusCO2', 'H2O:CO:CO2 (3:1:1)'),
                                            ('COplusH2OplusCO2', 'H2O:CO:CO2 (5:1:1)'),
                                            ('COplusH2OplusCO2', 'H2O:CO:CO2 (5:1:2)'),
                                            ('COplusH2OplusCO2', 'H2O:CO:CO2 (10:1:2)'),
                                            ('CO', 'CO 1'),
                                            ]):
        compspl = composition.split(' ')[1].strip('()').split(':')
        if len(compspl) == 1:
            co_mult = 1
            h2o_mult = co2_mult = 0
        else:
            h2o_mult = float(compspl[0])
            co_mult = float(compspl[1])
            co2_mult = float(compspl[2] if len(compspl) > 2 else 0)

        inds = np.argsort(water_mastrapa['Wavelength'])
        co_plus_co2_plus_water_k = (co_mult * gerakines['k'] +
                                    h2o_mult * np.interp(grid, water_mastrapa['Wavelength'][inds], water_mastrapa['k'][inds],) +
                                    co2_mult * np.interp(grid, co2_gerakines['Wavelength'], co2_gerakines['k'])) / (
                                        co_mult + h2o_mult + co2_mult
                                        )

        tbl = Table({'Wavelength': grid, 'k': co_plus_co2_plus_water_k})
        tbl.meta['composition'] = composition
        tbl.meta['density'] = 1*u.g/u.cm**3
        tbl.meta['temperature'] = 25*u.K
        tbl.meta['index'] = ii
        mymix_tables[(mol, ii, 25)] = tbl

    return mymix_tables

mymix_tables = make_mymix_tables()


xarr = np.linspace(3.1*u.um, 4.9*u.um, 10000)
phx4000 = atmo_model(4000, xarr=xarr)
#xarr = phx4000['nu'].quantity.to(u.um, u.spectral())
cols = np.geomspace(1e15, 1e21, 25)

def process_table(args):
    mol, key, consts, xarr, phx4000, cols, filter_data, transdata, basepath = args
    dmag_rows = []

    if 'k' not in consts.colnames:
        return []

    # if the wavelength range doesn't match, it just extrapolates.
    if u.Quantity(consts['Wavelength'].min(), u.um) > 5.0*u.um:
        return []
    
    dmags410, dmags466, dmags444, dmags356, dmags405 = [], [], [], [], []

    molecule = mol.lower()
    try:
        molwt = u.Quantity(composition_to_molweight(consts.meta['composition'].replace("^", "")), u.Da)
    except ValueError:
        print(f"Molecule {mol} with composition {consts.meta['composition']} failed to convert to molwt")
        return []

    cmd_x = ('JWST/NIRCam.F410M',
             'JWST/NIRCam.F466N',
             'JWST/NIRCam.F356W',
             'JWST/NIRCam.F444W',
             'JWST/NIRCam.F405N',
             )
    flxd_ref = fluxes_in_filters(xarr, phx4000['fnu'].quantity, filterids=cmd_x, transdata=transdata)

    for col in cols:
        spec = absorbed_spectrum(col*u.cm**-2, consts, molecular_weight=molwt,
                                  spectrum=phx4000['fnu'].quantity,
                                  xarr=xarr,
                                 )
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

        try:
            database, mol_id, temperature = key
        except TypeError:
            mol_id = key
            database = 'ocdb'
            
        dmag_rows.append({
            'molecule': molecule,
            'mol_id': mol_id,
            'database': database,
            'composition': consts.meta['composition'],
            'temperature': consts.meta['temperature'],
            'density': consts.meta['density'],
            'column': col,
            'F356W': dmags356[-1],
            'F410M': dmags410[-1],
            'F444W': dmags444[-1],
            'F466N': dmags466[-1],
            'F405N': dmags405[-1]
        })

    return dmag_rows

if __name__ == '__main__':
    molecules = {'H2O': h2otbs, 'CO': cotbs, 'CO2': co2tbs, 'H2O+CO': mymix_tables}
    
    # Create a dictionary of filter zero points
    filter_ids = ['JWST/NIRCam.F410M', 'JWST/NIRCam.F466N', 'JWST/NIRCam.F356W', 
                 'JWST/NIRCam.F444W', 'JWST/NIRCam.F405N']
    filter_data = {fid: float(jfilts.loc[fid]['ZeroPoint']) for fid in filter_ids}
    transdata = {fid: SvoFps.get_transmission_data(fid) for fid in filter_ids}

    # Create list of all tables to process
    all_tables = []
    for mol, tbs in molecules.items():
        for key, consts in tbs.items():
            all_tables.append((mol, key, consts, xarr, phx4000, cols, filter_data, transdata, basepath))

    # Process all tables in parallel
    results = process_map(process_table, 
                        all_tables,
                        max_workers=mp.cpu_count(),
                        desc="Processing tables",
                        unit="table")

    # Combine results and write tables for each molecule
    for mol in molecules:
        mol_rows = []
        for result in results:
            if result and result[0].get('molecule') == mol.lower():
                mol_rows.extend(result)
                
        if mol_rows:
            dmag_tbl = Table(mol_rows)
            dmag_tbl.write(f'{basepath}/tables/{mol}_ice_absorption_tables.ecsv', overwrite=True)
            dmag_tbl.add_index('database')
            dmag_tbl.add_index('mol_id')