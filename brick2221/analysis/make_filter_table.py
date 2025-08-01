"""
Make a table indicating which ices affect each of the JWST filters
"""


import importlib as imp
import icemodels
from icemodels.core import (absorbed_spectrum, absorbed_spectrum_Gaussians, convsum,
                            optical_constants_cache_dir,
                            download_all_ocdb,
                            retrieve_gerakines_co,
                            read_lida_file,
                            download_all_lida,
                            composition_to_molweight,
                            fluxes_in_filters, load_molecule, load_molecule_ocdb, atmo_model, molecule_data, read_ocdb_file)

from brick2221.analysis.analysis_setup import basepath

from astropy.table import Table
import numpy as np
import re

from astroquery.svo_fps import SvoFps


def molecule_to_latex(molecule_name):
    """
    Convert molecule name to LaTeX format with proper subscripts.

    Examples:
    h2o -> H$_2$O
    co2 -> CO$_2$
    ch4 -> CH$_4$
    """
    # Convert to uppercase for standard chemical notation
    mol = str(molecule_name).upper()

    # Handle special cases and common molecules
    replacements = {
        'H2O': r'H$_2$O',
        'CO2': r'CO$_2$',
        'CH4': r'CH$_4$',
        'N2': r'N$_2$',
        'O2': r'O$_2$',
        'NH3': r'NH$_3$',
        'N2H4': r'N$_2$H$_4$',
        'CH3OH': r'CH$_3$OH',
        'CH3OCH3': r'CH$_3$OCH$_3$',
        'HCOOCH3': r'HCOOCH$_3$',
        'CH3CHO': r'CH$_3$CHO',
        'CH3CN': r'CH$_3$CN',
        'NH4CN': r'NH$_4$CN',
        'C3H6O2': r'C$_3$H$_6$O$_2$',
        'C3H6O': r'C$_3$H$_6$O',
        'C3H6': r'C$_3$H$_6$',
        'C3H4': r'C$_3$H$_4$',
        'C3H2O': r'C$_3$H$_2$O',
        'C2H4O': r'C$_2$H$_4$O',
        'CH3CH2OH': r'CH$_3$CH$_2$OH',
        'CH3NH2': r'CH$_3$NH$_2$',
        'CH3CH2NH2': r'CH$_3$CH$_2$NH$_2$',
        'N2O': r'N$_2$O',
        'NH2CHO': r'NH$_2$CHO',
    }

    if mol in replacements:
        return replacements[mol]

    # Generic pattern for subscripts - find numbers after letters
    # This handles cases not in the replacements dict
    result = re.sub(r'([A-Z])(\d+)', r'\1$_{\2}$', mol)

    return result


def format_dmag_list(dmag_values, significant_figures=2):
    """Format dmag values to specified significant figures."""
    formatted_values = []
    for dmag in dmag_values:
        # Format to specified significant figures
        if dmag >= 1:
            formatted = f"{dmag:.{significant_figures-1}f}"
        else:
            # For values < 1, we need to handle significant figures differently
            # Find the first non-zero digit position
            if dmag == 0:
                formatted = "0.0"
            else:
                # Use scientific notation temporarily to get sig figs right
                sci_notation = f"{dmag:.{significant_figures-1}e}"
                # Convert back to decimal if reasonable
                if dmag >= 0.01:
                    decimal_places = significant_figures - 1 + abs(int(np.floor(np.log10(abs(dmag)))))
                    formatted = f"{dmag:.{decimal_places}f}".rstrip('0').rstrip('.')
                else:
                    formatted = sci_notation
        formatted_values.append(formatted)
    return formatted_values


def create_latex_table(filter_ice_table, output_file=None, column_density=1e18):
    """
    Create a LaTeX-formatted table from the filter ice table.
    """
    # Create new columns for LaTeX formatting
    latex_molecules = []
    latex_dmag = []

    for row in filter_ice_table:
        molecules = row['ice_molecules']
        dmag_vals = row['dmag_values']

        # Convert molecules to LaTeX format
        latex_mol_list = [molecule_to_latex(mol) for mol in molecules]

        # Format dmag values
        formatted_dmag = format_dmag_list(dmag_vals, significant_figures=2)

        # Create comma-separated strings
        mol_string = ', '.join(latex_mol_list)
        dmag_string = ', '.join(formatted_dmag)

        latex_molecules.append(mol_string)
        latex_dmag.append(dmag_string)

    # Create new table with LaTeX-formatted columns
    latex_table = Table({
        'Filter': filter_ice_table['filter_name'],
        'Ice Molecules': latex_molecules,
        'dmag Values': latex_dmag
    })

    # Set up LaTeX table formatting
    latex_table['Filter'].format = None
    latex_table['Ice Molecules'].format = None
    latex_table['dmag Values'].format = None

    # Write to file (required by astropy)
    if output_file is None:
        output_file = 'temp_latex_table.tex'
        temp_file = True
    else:
        temp_file = False

    # Generate LaTeX table
    latex_table.write(output_file, format='latex',
                     overwrite=True)

    # Read back the content to return/display
    with open(output_file, 'r') as f:
        latex_content = f.read()

    # Add custom LaTeX enhancements
    # Replace the table with proper formatting
    lines = latex_content.split('\n')

    # Find table start and modify
    enhanced_lines = []
    in_table = False

    log_column = int(np.log10(column_density))

    description = f'Molecules that absorb NIRCAM filters by at least 0.1 mag when their column density is 10$^{{{log_column}}}$ cm$^{{-2}}$.  This table is not comprehensive, since some molecules are potentially much more abundant (e.g., \\water), and the more complex molecules are likely to be rarer.  NIRCAM filters excluded from this table do not have significant ($>0.1$ mag) ice absorption at N(ice)=10$^{{{log_column}}}$ cm$^{{-2}}$.'
    if column_density == 1e18:
        description += 'Several molecules in the ice database are excluded because they have not been reported in the ISM, including NH$_4$CN, N$_2$H$_4$, and HC$_3$N.'
    if column_density == 1e19:
        description += 'Several molecules in the ice database are excluded because they have not been reported in the ISM, including NH$_4$CN, N$_2$H$_4$, N$_2$O, CH$_3$CH$_2$NH$_2$, CH$_3$NH$_2$, C$_6$H$_6$, C$_5$H$_5$N, C$_2$H$_6$O, C$_3$H$_8$, C$_3$H$_2$O, and HC$_3$N.'

    # the AI felt that this cruft was necessary
    for line in lines:
        if '\\begin{table}' in line:
            enhanced_lines.append('\\begin{table*}[ht]')
            enhanced_lines.append('\\centering')
            enhanced_lines.append('\\caption{Ice molecules that significantly absorb NIRCAM filters}')
            if column_density == 1e18:
                enhanced_lines.append('\\label{tab:nircam_ice_absorption}')
            else:
                enhanced_lines.append(f'\\label{{tab:nircam_ice_absorption_{{{log_column}}}}}')
            in_table = True
        elif '\\begin{tabular}' in line:
            enhanced_lines.append('\\begin{tabular}{|l|p{10cm}|p{6cm}|}')
            enhanced_lines.append('\\hline')
        elif '\\end{tabular}' in line:
            enhanced_lines.append('\\hline')
            enhanced_lines.append(line)
        elif in_table and ('Filter' in line and 'Ice Molecules' in line):
            enhanced_lines.append('\\textbf{Filter} & \\textbf{Ice Molecules} & \\textbf{$\\Delta$mag Values} \\\\')
            enhanced_lines.append('\\hline')
        elif '\\end{table}' in line:
            enhanced_lines.append(f'\\par {description}')
            enhanced_lines.append('\\end{table*}')
        else:
            enhanced_lines.append(line)

        if '\\end{table}' in line:
            in_table = False

    enhanced_content = '\n'.join(enhanced_lines)

    # Write the enhanced version back
    with open(output_file, 'w') as f:
        f.write(enhanced_content)

    if not temp_file:
        print(f"LaTeX table saved to: {output_file}")
    else:
        # Clean up temp file
        import os
        os.remove(output_file)

    return enhanced_content


def create_filter_ice_table(dmag_threshold=0.1, column_density=1e18):
    """
    Create a table showing which ice molecules significantly affect each NIRCAM filter.

    Parameters:
    -----------
    dmag_threshold : float, default=0.1
        Minimum dmag value to consider a molecule as having substantial effect
    column_density : float, default=1e19
        Column density in cm^-2 to filter the data for

    Returns:
    --------
    astropy.table.Table
        Table with columns 'filter_name' and 'ice_molecules'
        Each row represents one NIRCAM filter and its list of significant ice molecules
    """
    # Load the absorption data
    dmag_tbl = Table.read(f'{basepath}/tables/combined_ice_absorption_tables.ecsv')
    dmag_tbl = dmag_tbl[dmag_tbl['database'] != 'mymix']

    # Filter for the specified column density
    mask = dmag_tbl['column'] == column_density
    assert mask.sum() > 0, f"No ice entries found at column density {column_density:.0e} cm^-2"
    filtered_data = dmag_tbl[mask]

    #print(f"Found {len(filtered_data)} ice entries at column density {column_density:.0e} cm^-2")

    # Filter out blended molecules (compositions containing ":")
    # Only keep pure/isolated molecules
    pure_molecule_mask = ~np.array([(':' in str(comp))
                                    or ("(100" in str(comp)) or
                                    (" " in str(comp).split(" (")[0])
                                    for comp in filtered_data['composition']])
    filtered_data = filtered_data[pure_molecule_mask]
    assert len(filtered_data) > 0, "No pure molecules found"

    #print(f"After filtering for pure molecules (no blends): {len(filtered_data)} entries")

    # Filter out molecules with "under" or "over" in the name (substrate experiments)
    under_molecule_mask = ~np.array([('under' in str(mol).lower()) or ('over' in str(mol).lower()) for mol in filtered_data['molecule']])
    filtered_data = filtered_data[under_molecule_mask]

    # filter out rare molecules
    rare_molecules = ['nh4cn', 'n2h4', 'hc3n', 'n2o', 'ch3ch2nh2', 'ch3nh2', 'c3h4', 'c3h6', 'c6h6', 'c5h5n', 'c2h6o', 'c3h8', 'c3h2o', 'c2h4o']
    rare_molecule_mask = np.array([np.any([mm.lower() in mol.lower() for mm in rare_molecules]) for mol in filtered_data['molecule']])
    filtered_data = filtered_data[~rare_molecule_mask]

    #print(f"After filtering out 'under' molecules: {len(filtered_data)} entries")

    # Get filter columns from the data (shortened names like F150W)
    filter_cols = [col for col in dmag_tbl.colnames if col.startswith('F') and int(col[1:4]) < 500]

    # Helper function to normalize molecule names (handle unicode characters)
    def normalize_molecule_name(mol_name):
        """Normalize molecule names to handle unicode variants like ch₄ -> ch4"""
        mol_str = str(mol_name).lower()
        # Replace unicode subscript numbers with regular numbers
        unicode_replacements = {
            '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
            '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9'
        }
        for unicode_char, regular_char in unicode_replacements.items():
            mol_str = mol_str.replace(unicode_char, regular_char)
        return mol_str

    # Initialize results
    filter_names = []
    ice_molecule_lists = []
    dmag_value_lists = []
    author_lists = []
    molid_lists = []

    # Process each filter
    for filter_col in filter_cols:
        # Find molecules with dmag > threshold for this filter
        # Exclude NaN values
        valid_mask = ~np.isnan(filtered_data[filter_col])
        threshold_mask = filtered_data[filter_col] > dmag_threshold
        significant_mask = valid_mask & threshold_mask

        significant_molecules = [x.split(" (")[0] for x in list(filtered_data[significant_mask]['composition'])]
        significant_dmag = list(filtered_data[significant_mask][filter_col])
        significant_authors = list(filtered_data[significant_mask]['author'])
        significant_molid = list(filtered_data[significant_mask]['mol_id'])

        # Normalize molecule names and create pairs with dmag values and authors
        molecule_data_pairs = []
        for mol, dmag, author, molid in zip(significant_molecules, significant_dmag, significant_authors, significant_molid):
            normalized_mol = normalize_molecule_name(mol)
            molecule_data_pairs.append((normalized_mol, dmag, author, molid))

        # Group by unique molecule names and keep the data for maximum dmag for each
        unique_molecule_dict = {}
        for mol, dmag, author, molid in molecule_data_pairs:
            if mol in unique_molecule_dict:
                # Keep the entry with higher dmag value
                if dmag > unique_molecule_dict[mol][0]:
                    unique_molecule_dict[mol] = (dmag, author, molid)
            else:
                unique_molecule_dict[mol] = (dmag, author, molid)

        # Sort by dmag value (greatest effect first)
        sorted_molecules = sorted(unique_molecule_dict.items(), key=lambda x: x[1][0], reverse=True)

        # Separate molecules, dmag values, and authors
        unique_molecules = [mol for mol, (dmag, author, molid) in sorted_molecules]
        dmag_values = [dmag for mol, (dmag, author, molid) in sorted_molecules]
        authors = [author for mol, (dmag, author, molid) in sorted_molecules]
        molids = [str(molid) for mol, (dmag, author, molid) in sorted_molecules]

        # Only include filters that have at least one significant molecule
        if unique_molecules:
            filter_names.append(filter_col)
            ice_molecule_lists.append(unique_molecules)
            dmag_value_lists.append(dmag_values)
            author_lists.append(authors)
            molid_lists.append(molids)

    # Create the output table
    result_table = Table({
        'filter_name': filter_names,
        'ice_molecules': ice_molecule_lists,
        'dmag_values': dmag_value_lists,
        'authors': author_lists,
        'molid': molid_lists
    })

    # print(f"\nCreated table with {len(result_table)} filters having significant ice absorption")
    # print(f"Using dmag threshold: {dmag_threshold}")
    # print(f"Using column density: {column_density:.0e} cm^-2")

    return result_table


if __name__ == "__main__":
    # Create the filter-ice table with default parameters
    filter_ice_table = create_filter_ice_table(dmag_threshold=0.1, column_density=1e18)

    print(filter_ice_table[0])

    # Save the table
    output_filename = f'{basepath}/tables/nircam_filter_ice_table.ecsv'
    filter_ice_table.write(output_filename, overwrite=True)
    print(f"\nTable saved to: {output_filename}")

    # Create and save LaTeX table
    latex_output_file = f'{basepath}/tables/nircam_filter_ice_table.tex'
    latex_table = create_latex_table(filter_ice_table, output_file=latex_output_file)

    #print("\nLaTeX table preview:")
    #print(latex_table)

    filter_ice_table_sensitive = create_filter_ice_table(dmag_threshold=0.1, column_density=1e19)
    output_filename = f'{basepath}/tables/nircam_filter_ice_table_1e19.ecsv'
    filter_ice_table_sensitive.write(output_filename, overwrite=True)
    latex_output_file = f'{basepath}/tables/nircam_filter_ice_table_1e19.tex'
    latex_table = create_latex_table(filter_ice_table_sensitive, output_file=latex_output_file, column_density=1e19)

    # print()
    # filter_ice_table_higher_column = create_filter_ice_table(dmag_threshold=0.1, column_density=1e20)
    # print(f"With dmag > 0.1 at 1e20 cm^-2: {len(filter_ice_table_higher_column)} filters have significant ice absorption")
    # print(filter_ice_table_higher_column[0])

    # print()
    # filter_ice_table_lower_column = create_filter_ice_table(dmag_threshold=0.1, column_density=1e18)
    # print(f"With dmag > 0.1 at 1e18 cm^-2: {len(filter_ice_table_lower_column)} filters have significant ice absorption")
    # print(filter_ice_table_lower_column[0])

    filter_ice_table.add_index('filter_name')
    print(filter_ice_table.loc['F430M'])
    filter_ice_table_sensitive.add_index('filter_name')
    print(filter_ice_table_sensitive.loc['F430M'])

    filter_ice_table_sensitive.add_index('filter_name')
    print(filter_ice_table_sensitive.loc['F250M'])
