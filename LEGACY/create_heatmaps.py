import LEGACY.get_files as fileHandler
import generate_heatmap as heatmaps
import pandas as pd
from datetime import datetime
import os


# Coil position interpolation cell size
CELL_SIZE = 0.5 # meters

# Color map and boundaries for TSS heatmaps
colorsTSS = ['blue', 'dodgerblue', 'green', 'lime', 'yellow', 'orange', 'red', 'purple', 'pink']
boundariesTSS = [-500, -100, -25, 25, 50, 75, 150, 500, 5000, 10000]  

# Color map and boundaries for Alt heatmaps
colorsALT = ['black', 'darkblue', 'green', 'lime', 'yellow', 'orange', 'red']
boundariesALT = [-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0, 2.0]  

# Record the start time
start_time = datetime.now()

# Ask user for data folder
files = fileHandler.get_files()

heatmaps_tss_cntr = 0
heatmaps_alt_cntr = 0

for i, file in enumerate(files):
    filepath = os.path.normpath(file) # Fix slashes
    df = pd.read_csv(filepath)
    filename = os.path.basename(filepath) # Get the filename safely

    # TSS heatmap generation
    required_columns = {'Easting', 'Northing', 'TSS'}
    if required_columns.issubset(df.columns):
        heatmaps.generate_TSS_heatmap(filepath, filename, df, 0, CELL_SIZE, colorsTSS, boundariesTSS)
        print(f"Generated heatmap for {filename}")
        heatmaps_tss_cntr += 1
    else:
        print(f"Skipping TSS heatmap for {filename}: Missing required columns.")

    # Altitude heatmap generation
    required_columns = {'Easting', 'Northing', 'Alt'}
    if required_columns.issubset(df.columns):
        heatmaps.generate_ALT_heatmap(filepath, filename, df, 0, CELL_SIZE, colorsALT, boundariesALT)
        print(f"Generated heatmap for {filename}")
        heatmaps_alt_cntr += 1
    else:
        print(f"Skipping Altitude heatmap for {filename}: Missing required columns.")

# Calculate the duration
end_time = datetime.now()
duration = end_time - start_time

# Print the start time, end time, and duration
#print(f"Start time: {start_time}")
#print(f"End time: {end_time}")
#print(f"Duration: {duration}")

print("Files found:", len(files))
print(files)
print("Heatmaps TSS generated:", heatmaps_tss_cntr)
print("Heatmaps Altitude generated:", heatmaps_alt_cntr)