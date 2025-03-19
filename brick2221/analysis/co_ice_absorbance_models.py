import glob
import os
import time

import numpy as np
import astropy.units as u
from astropy.table import Table

import icemodels
from icemodels import absorbed_spectrum, absorbed_spectrum_Gaussians, convsum, fluxes_in_filters, load_molecule, load_molecule_ocdb, atmo_model, molecule_data
from icemodels.core import composition_to_molweight, retrieve_gerakines_co, optical_constants_cache_dir, read_lida_file, read_ocdb_file

# load tables
cotbs, h2otbs, co2tbs = {}, {}, {}
for molname, tbs in {'H2O': h2otbs, 'CO': cotbs, 'CO2': co2tbs}.items():
    for fn in glob.glob(f"{optical_constants_cache_dir}/*_{molname}:*") + glob.glob(f"{optical_constants_cache_dir}/*_{molname} *"):
        try:
            tb = read_ocdb_file(fn)
            basename = os.path.basename(os.path.splitext(fn)[0])
            spl = basename.split("_")
            idnum = int(spl[0])
            tbs[('ocdb', idnum)] = tb
        except Exception as ex:
            with open(fn, 'r') as fh:
                if 'ocdb' in fh.read().lower():
                    #print(fn)
                    continue
            tb = read_lida_file(fn)
            tbs[('lida', int(tb.meta['index']))] = tb
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


# make up our own H2O + CO at 25K with 3:1 ratio
# 3-1 comes from Brandt's models plus the McClure 2023 paper
mymix_tables = {}

water_mastrapa = h2otbs[('ocdb', 242)]
co2_gerakines = co2tbs[('ocdb', 55)] # TODO: mix this in at about 0.5 CO

co_gerakines = gerakines = retrieve_gerakines_co()
grid = co_gerakines['Wavelength']

for ii, (mol, composition) in enumerate([('COplusH2O', 'H2O:CO (3:1)'),
                                         ('COplusH2OplusCO2', 'H2O:CO:CO2 (3:1:0.1)'),
                                         ('COplusH2OplusCO2', 'H2O:CO:CO2 (5:1:0.5)'),
                                         ('COplusH2OplusCO2', 'H2O:CO:CO2 (3:1:0.5)'),
                                         ('COplusH2OplusCO2', 'H2O:CO:CO2 (5:1:0.5)'),
                                         ('COplusH2OplusCO2', 'H2O:CO:CO2 (3:1:1)'),
                                         ('COplusH2OplusCO2', 'H2O:CO:CO2 (5:1:1)'),
                                         ]):
    compspl = composition.split(' ')[1].strip('()').split(':')
    h2o_mult = float(compspl[0])
    co_mult = float(compspl[1])
    co2_mult = float(compspl[2] if len(compspl) > 2 else 0)

    inds = np.argsort(water_mastrapa['Wavelength'])
    co_plus_water_k = gerakines['k'] + np.interp(grid, water_mastrapa['Wavelength'][inds], h2o_mult*water_mastrapa['k'][inds],)
    co_plus_co2_plus_water_k = co_plus_water_k + co2_mult*np.interp(grid, co2_gerakines['Wavelength'], co2_gerakines['k'])

    tbl = Table({'Wavelength': grid, 'k': co_plus_co2_plus_water_k})
    tbl.meta['composition'] = composition
    tbl.meta['density'] = 1*u.g/u.cm**3
    mymix_tables[(mol, ii)] = tbl


xarr = np.linspace(3.1*u.um, 4.9*u.um, 10000)
phx4000 = atmo_model(4000, xarr=xarr)
#xarr = phx4000['nu'].quantity.to(u.um, u.spectral())
cols = np.geomspace(1e15, 1e21, 25)

for mol, tbs in {'H2O': h2otbs, 'CO': cotbs, 'CO2': co2tbs, 'H2O+CO': coh2otbs}.items():
    dmag_rows = []
    
    for key, consts in tbs.items():

        if 'k' not in consts.colnames:
            #print(f"Molecule {mol} with key {key} failed with no opacities")
            continue
        
        dmags410 = []
        dmags466 = []
        dmags444 = []
        dmags356 = []
        dmags405 = []
    
        molecule = mol.lower()
        molwt = u.Quantity(composition_to_molweight(consts.meta['composition']), u.Da)

        tt0 = time.time()
        #print(molecule, key, end=":  ")
        #print(f"  column,   mag410,  mag410*,  mag466n, mag466n*, dmag410, dmag466")
        for col in cols:
            t0 = time.time()
            spec = absorbed_spectrum(col*u.cm**-2, consts, molecular_weight=molwt,
                                      spectrum=phx4000['fnu'].quantity,
                                      xarr=xarr,
                                     )
            t1 = time.time()
            cmd_x = ('JWST/NIRCam.F410M', 'JWST/NIRCam.F466N', 'JWST/NIRCam.F356W', 'JWST/NIRCam.F444W',  'JWST/NIRCam.F405N')
            flxd_ref = fluxes_in_filters(xarr, phx4000['fnu'].quantity, filterids=cmd_x)
            t2 = time.time()
            flxd = fluxes_in_filters(xarr, spec, filterids=cmd_x)
            t3 = time.time()
            mags_x_star = (-2.5*np.log10(flxd_ref[cmd_x[0]] / u.Quantity(jfilts.loc[cmd_x[0]]['ZeroPoint'], u.Jy)),
                           -2.5*np.log10(flxd_ref[cmd_x[1]] / u.Quantity(jfilts.loc[cmd_x[1]]['ZeroPoint'], u.Jy)),
                           -2.5*np.log10(flxd_ref[cmd_x[2]] / u.Quantity(jfilts.loc[cmd_x[2]]['ZeroPoint'], u.Jy)),
                           -2.5*np.log10(flxd_ref[cmd_x[3]] / u.Quantity(jfilts.loc[cmd_x[3]]['ZeroPoint'], u.Jy)),
                           -2.5*np.log10(flxd_ref[cmd_x[4]] / u.Quantity(jfilts.loc[cmd_x[4]]['ZeroPoint'], u.Jy)),
                          )
            mags_x = (-2.5*np.log10(flxd[cmd_x[0]] / u.Quantity(jfilts.loc[cmd_x[0]]['ZeroPoint'], u.Jy)),
                      -2.5*np.log10(flxd[cmd_x[1]] / u.Quantity(jfilts.loc[cmd_x[1]]['ZeroPoint'], u.Jy)),
                      -2.5*np.log10(flxd[cmd_x[2]] / u.Quantity(jfilts.loc[cmd_x[2]]['ZeroPoint'], u.Jy)),
                      -2.5*np.log10(flxd[cmd_x[3]] / u.Quantity(jfilts.loc[cmd_x[3]]['ZeroPoint'], u.Jy)),
                      -2.5*np.log10(flxd[cmd_x[4]] / u.Quantity(jfilts.loc[cmd_x[4]]['ZeroPoint'], u.Jy)),
                     )
            dmags405.append(mags_x[4]-mags_x_star[4])
            dmags444.append(mags_x[3]-mags_x_star[3])
            dmags356.append(mags_x[2]-mags_x_star[2])
            dmags466.append(mags_x[1]-mags_x_star[1])
            dmags410.append(mags_x[0]-mags_x_star[0])
            # why would f410m change at all?
            #print(f"{col:8.1g}, {mags_x[0]:8.1f}, {mags_x_star[0]:8.1f}, {mags_x[1]:8.1f}, {mags_x_star[1]:8.1f}, {dmags410[-1]:8.1f}, {dmags466[-1]:8.1f}")
            try:
                database, mol_id = key
            except TypeError:
                mol_id = key
                database = 'ocdb'
            dmag_rows.append({'molecule': molecule, 'mol_id': mol_id, 'database': database, 'composition': consts.meta['composition'], 'density': consts.meta['density'], 'column': col, 'F356W': dmags356[-1], 'F410M': dmags410[-1],
                              'F444W': dmags444[-1], 'F466N': dmags466[-1], 'F405N': dmags405[-1]})
            t4 = time.time()
            #print(f"col={col:0.1g}; Steps took {t1-t0:0.1g}, {t2-t1:0.1g}, {t3-t2:0.1g}, {t4-t3:0.1g}")
        tt1 = time.time()
        #print(f"Done with {molecule} {key}; dt={tt1-tt0:0.3g}")
    dmag_tbl = Table(dmag_rows)
    dmag_tbl.write(f'{basepath}/tables/{mol}_ice_absorption_tables.ecsv', overwrite=True)
    dmag_tbl.add_index('database')
    dmag_tbl.add_index('mol_id')