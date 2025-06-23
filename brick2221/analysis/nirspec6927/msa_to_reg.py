"""

# Load the MSA configuration JSON file
with open("MSA Plan 11 for Pointing 1 (Obs 14).json") as f:
    msa_data = json.load(f)

# Extract shutter positions
shutter_positions = []
for slitlet in msa_data['slitlets']:
    for shutter in slitlet['shutters']:
        quadrant = shutter['q']  # Quadrant
        x = shutter['x']         # Column
        y = shutter['y']         # Row
        shutter_positions.append((quadrant, x, y))

# Print first few shutters
print("Extracted shutter positions:")
for q, x, y in shutter_positions[:10]:
    print(f"Quadrant {q}, X={x}, Y={y}")


with open("msa_regions.reg", "w") as reg_file:
    reg_file.write("global color=green width=2\n")
    for q, x, y in shutter_positions:
        width, height = 0.2, 0.46  # Approximate size
        reg_file.write(f"image; box({x}, {y}, {width}, {height}, 0)\n")


import pandas as pd

# Load the MSA configuration CSV
msa_csv = "msa_shutter_map.csv"
df = pd.read_csv(msa_csv)

# Extract positions of open shutters (value = 1)
open_shutters = df[df['Shutter_State'] == 1][['Column', 'Row']]

# Print some open shutter positions
print(open_shutters.head())
"""

import json
import pandas as pd
import glob

# Load the MSA configuration JSON file
with open("MSA Plan 11 for Pointing 1 (Obs 14).json") as f:
    msa_data = json.load(f)

    
msa_file = '6927-obs14-exp1-c1e1n1-G140M-F070LP.csv'

for msa_file in glob.glob('6927-obs*.csv'):
    df = pd.read_csv(msa_file)

    # Define approximate shutter size (arcsec)
    width, height = 0.2, 0.46

    # Open file to write DS9 region definitions
    msa_regions_file = msa_file.replace('.csv', '_regions.reg')
    # with open(msa_regions_file, "w") as reg_file:
    #     reg_file.write("global color=green width=2\n")
    #     
    #     # Loop through each row in the dataframe
    #     for _, row in df.iterrows():
    #         quad = row[' Quadrant']
    #         x = row[' Column (Disp)']  # Dispersion axis (Column)
    #         y = row[' Row (Spat)']     # Spatial axis (Row)
    #         
    #         # Write each shutter as a box region in DS9 format
    #         reg_file.write(f"image; box({x}, {y}, {width}, {height}, 0)\n")
    with open(msa_regions_file, "w") as reg_file:
        reg_file.write("fk5\n")  # Use the celestial coordinate system (WCS)
        
        # Loop through each row in the dataframe
        for _, row in df.iterrows():
            ra = row[' Source RA (Degrees)']
            dec = row[' Source Dec (Degrees)']
            pa = row[' Aperture PA (Degrees)']
            
            # Write each shutter as a box region in DS9 WCS format
            reg_file.write(f"box({ra},{dec},{width}\",{height}\",{pa}) # color=green\n")


    print(f"DS9 region file saved: {msa_regions_file}")