# TSS File Converter

## Overview
This script processes `.ptr` files to extract Northing, Easting, and TSS coil values, compute estimated headings, and generate various outputs including heatmaps and peak value logs.

## Features
1. Read all the `.ptr` files in the given folder and extract the Northing, Easting, and the 3 TSS coils values.
2. Compute the estimated heading based on the difference between Easting and Northing coordinates. Note that this heading is not completely accurate as it does not account for roll and pitch angles (with corresponding lever arms) and the cross-track error of the ROV regarding the defined line. However, given that each coil is 90x90 cm and the distance between the 3 centers of the coils is 47 cm, the positional error in the side points for the port and starboard coils due to heading mismatching can be neglected.
3. Compute the position of the port and starboard coils based on the estimated heading and the coils offsets to obtain 3 different positions from each single position in the PTR file. The position of the center coil corresponds to the PTR file position.
4. Show a plot to preview the heatmap and the estimated heading before exporting the output files. This heatmap can be previewed by pressing the button “Show Map”.
5. Show a plot to check the peak values of the coils before exporting the output files. A plot for each line will be displayed by pressing the button “Show Coils”.
6. Export a `.txt` file with the computed Northing and Easting coordinates and the corresponding TSS coil measurement to drop in NaviModel and create the heatmap straight forward.
7. Export a `.csv` file with the TSS peak value, number of coil, and location for each line. This is useful to record in the Log Excel the peak value of each line.

## Necessary Software
This script is written using the Python programming language and several standard libraries. However, it is not necessary to install Python to use it because the script was exported as a `.exe` file and can be run on any Windows computer.

### If Modification is Needed
If it is necessary to modify the script, it can be edited using the following software (already installed on the OLTA Offline computers):
- **Visual Studio Code** (or any other development environment)
- **Python 3.12.6**
- **Python libraries**:
  - `pandas`: to manage the data inside the program (`pip install pandas`)
  - `matplotlib`: to plot the heatmap and the coils values (`pip install matplotlib`)
  - `pyinstaller`: to create the `.exe` file (`pip install pyinstaller`)

### Running the Script
If the Python script is modified, it can be run from Visual Studio Code. However, this is not recommended for the regular workflow because someone could modify and ruin it by mistake. It is recommended to create a new `.exe` file. This can be done by running the `.bat` script `CREATE_exe.bat` (it is necessary to have the `pyinstaller` Python library installed previously).

## Usage
1. Place the `.ptr` files in a folder.
2. Run the TSS File Converter executable.
3. Select the folder containing the `.ptr` files.
4. Configure the necessary parameters (e.g., column indices, offsets).
5. Use the provided buttons to preview the heatmap and coil plots.
6. Export the results as `.txt` and `.csv` files.

## Contact
For any issues or questions, please contact vicente.danvila@reachsubsea.com.
