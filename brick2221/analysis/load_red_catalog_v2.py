"""
Improved script to load brick2221 catalog and create a sub-catalog for red stars (F356W-F444W > 0.75)
"""

from astropy.table import Table
from astropy.io import fits
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
import os
import warnings

def load_brick_catalog_efficiently(max_rows=None):
    """
    Load the brick2221 catalog efficiently with optional row limit

    Parameters:
    -----------
    max_rows : int, optional
        Maximum number of rows to load (for testing/development)

    Returns:
    --------
    basetable : astropy.table.Table
        The loaded catalog
    """
    # Set the base path
    basepath = '/orange/adamginsburg/jwst/brick'

    # Load the catalog
    catalog_path = f'{basepath}/catalogs/basic_merged_indivexp_photometry_tables_merged.fits'

    if not os.path.exists(catalog_path):
        raise FileNotFoundError(f"Catalog not found at {catalog_path}")

    print(f"Loading catalog from {catalog_path}")

    # Check file size
    size_mb = os.path.getsize(catalog_path) / (1024 * 1024)
    print(f"Catalog file size: {size_mb:.1f} MB")

    # Load the catalog with optional row limit
    try:
        if max_rows is not None:
            print(f"Loading first {max_rows} rows for testing...")
            # Load just the first few rows for testing
            with fits.open(catalog_path) as hdul:
                table_data = hdul[1].data[:max_rows]
                basetable = Table(table_data)
        else:
            print("Loading full catalog (this may take a while)...")
            basetable = Table.read(catalog_path)

        print(f"Loaded catalog with {len(basetable)} sources")

        # Check for required columns
        required_cols = ['mag_ab_f356w', 'mag_ab_f444w']
        available_cols = [col for col in required_cols if col in basetable.colnames]
        print(f"Required columns found: {available_cols}")

        # Check for coordinate columns
        coord_cols = []
        if 'skycoord_ref' in basetable.colnames:
            coord_cols.append('skycoord_ref')
        if 'skycoord_ref.ra' in basetable.colnames and 'skycoord_ref.dec' in basetable.colnames:
            coord_cols.extend(['skycoord_ref.ra', 'skycoord_ref.dec'])
        print(f"Coordinate columns found: {coord_cols}")

        return basetable

    except Exception as e:
        print(f"Error loading catalog: {e}")
        raise

def create_red_star_catalog_v2(basetable, color_threshold=0.75):
    """
    Create a sub-catalog of red stars with F356W-F444W > color_threshold

    Parameters:
    -----------
    basetable : astropy.table.Table
        The full catalog
    color_threshold : float
        The color cut threshold (default 0.75)

    Returns:
    --------
    red_catalog : astropy.table.Table
        Filtered catalog of red stars
    """

    # Check if the required columns exist
    required_cols = ['mag_ab_f356w', 'mag_ab_f444w']
    missing_cols = [col for col in required_cols if col not in basetable.colnames]

    if missing_cols:
        raise ValueError(f"Required columns not found: {missing_cols}")

    # Calculate the color F356W - F444W
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        color_356_444 = basetable['mag_ab_f356w'] - basetable['mag_ab_f444w']

    # Apply quality cuts - exclude bad measurements
    good_f356w = np.isfinite(basetable['mag_ab_f356w']) & (basetable['mag_ab_f356w'] > 0)
    good_f444w = np.isfinite(basetable['mag_ab_f444w']) & (basetable['mag_ab_f444w'] > 0)

    # Create the red star selection
    red_selection = (good_f356w & good_f444w &
                    (color_356_444 > color_threshold) &
                    np.isfinite(color_356_444))

    print(f"Selection summary:")
    print(f"  Good F356W measurements: {np.sum(good_f356w)}")
    print(f"  Good F444W measurements: {np.sum(good_f444w)}")
    print(f"  Stars with F356W-F444W > {color_threshold}: {np.sum(red_selection)}")

    red_catalog = basetable[red_selection].copy()

    # Add the color as a column
    red_catalog['color_f356w_f444w'] = color_356_444[red_selection]

    # Add coordinates if available
    coord_added = False
    if 'skycoord_ref' in red_catalog.colnames:
        # SkyCoord column already exists
        coord_added = True
        print("Using existing skycoord_ref column")
    elif 'skycoord_ref.ra' in red_catalog.colnames and 'skycoord_ref.dec' in red_catalog.colnames:
        # Create SkyCoord from RA/Dec columns
        coords = SkyCoord(ra=red_catalog['skycoord_ref.ra'],
                         dec=red_catalog['skycoord_ref.dec'],
                         unit='deg')
        red_catalog['skycoord'] = coords
        coord_added = True
        print("Created skycoord column from RA/Dec")

    if not coord_added:
        print("Warning: No coordinate information found")

    print(f"Created red star catalog with {len(red_catalog)} sources")

    return red_catalog

def load_red_stars_efficiently(color_threshold=0.75, max_rows=None):
    """
    Efficiently load and filter red star catalog

    Parameters:
    -----------
    color_threshold : float
        The color cut threshold
    max_rows : int, optional
        Maximum number of rows to load from full catalog

    Returns:
    --------
    basetable : astropy.table.Table
        The full catalog (or subset if max_rows specified)
    red_catalog : astropy.table.Table
        Filtered catalog of red stars
    """
    try:
        basetable = load_brick_catalog_efficiently(max_rows=max_rows)
        red_catalog = create_red_star_catalog_v2(basetable, color_threshold=color_threshold)
        return basetable, red_catalog
    except Exception as e:
        print(f"Error loading red stars: {e}")
        return None, None

def main():
    """
    Main function for testing
    """
    # Load a small subset for testing
    print("Testing with first 10000 rows...")
    basetable, red_catalog = load_red_stars_efficiently(color_threshold=0.75, max_rows=10000)

    if red_catalog is not None:
        print(f"\nRed star catalog summary:")
        print(f"Number of sources: {len(red_catalog)}")

        if 'color_f356w_f444w' in red_catalog.colnames:
            color_stats = red_catalog['color_f356w_f444w']
            print(f"F356W-F444W color range: {np.min(color_stats):.3f} to {np.max(color_stats):.3f}")
            print(f"Median color: {np.median(color_stats):.3f}")

        # Show some example entries
        print(f"\nFirst 5 red stars:")
        for i in range(min(5, len(red_catalog))):
            color = red_catalog['color_f356w_f444w'][i]
            f356w = red_catalog['mag_ab_f356w'][i]
            f444w = red_catalog['mag_ab_f444w'][i]
            print(f"  {i+1}: F356W={f356w:.2f}, F444W={f444w:.2f}, Color={color:.3f}")

    return basetable, red_catalog

if __name__ == "__main__":
    basetable, red_catalog = main()


"""
tb['color_356_444'] = tb['mag_ab_f356w'] - tb['mag_ab_f444w']
tb['color_200_356'] = tb['mag_ab_f200w'] - tb['mag_ab_f356w']
tb['color_200_444'] = tb['mag_ab_f200w'] - tb['mag_ab_f444w']
tb['color_115_200'] = tb['mag_ab_f115w'] - tb['mag_ab_f200w']
tb['skycoord_ref', 'color_115_200', 'color_200_444', 'color_200_356', 'color_356_444'].write('/Users/adam/Dropbox/brick2221/basic_merged_indivexp_photometry_tables_merged_filtered_widebandcolors.fits', overwrite=True)
"""