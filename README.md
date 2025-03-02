# TSS Converter 4 (beta)

## Overview
This script processes the Pipetracker (`.ptr`) and VisualSoft navigation (`_Coil_X.csv`) exported files from NaviEdit to extract Northing, Easting, and TSS coil values, compute estimated headings, and generate various outputs including heatmaps, heading error analysis, and peak value logs. The new version improves accuracy, visualization, and error handling.

## Features (Version 4)
1. **Read all the pipetracker and navigation files in the same folder**:
   - It is no longer necessary to use 2 different folder for every type of file.
   - All the survey files (inputs and outputs) keep in the same folder.
2. **Improved Heading Calculation**:
   - Uses `merge_asof` to accurately match navigation data with PTR timestamps.
   - Computes course with reference to **compass North (0° to 360°)**.
   - Implements **circular mean and standard deviation** to avoid 0°/360° wrap-around errors.
   - Heading error is now correctly calculated within **±180°** range.
3. **Improved Coil Positioning**:
   - Extracts the positions of port and starboard coils using the computed points from NaviEdit in the navigation files.
   - Ensures accurate placement of each coil's data.
4. **New Heading Error Map**:
   - Displays heading error as a scatter plot with a **color scale**.
   - Adds **directional arrows** at the start of each line to indicate average course of each line.
5. **Improved Heatmaps**:
   - Magnetic field values and altitude are now plotted with **better-scaled color gradients**.
   - More accurate representation of survey data.
6. **Peak Detection for TSS Values**:
   - Detects and logs **max and min TSS values** along with their locations.
   - Saves peak information to a CSV file for further analysis.
7. **Enhanced Heading and Course Quality Control (QC) Table**:
   - Displays heading statistics in a formatted table.
   - **Color-coded cells**: White (low error), Red (high error) for quick interpretation.
8. **Error Handling Improvements**:
   - Consolidated error messages into a **single prompt**, reducing user interruptions.
   - Added time difference verification to ensure synchronization between PTR and navigation data.

## Changes from Version 3
### **1. Improved Data Handling**
✅ **v3**: Basic merging of data without detailed time synchronization.

✅ **v4**: Uses `merge_asof` for precise timestamp alignment and checks time differences against a threshold.

### **2. Advanced Heading Calculation**
✅ **v3**: Simple heading computation based on `arctan2`.

✅ **v4**: Extracts the NaviEdit computed heading values from NaviEdit.

### **3. Better Visualization & QC**
✅ **v3**: Basic scatter plots for TSS values.

✅ **v4**:
- Heatmaps for **magnetic field** and **altitude**.
- **Heading Error Map** with a color scale and directional arrows.
- **Table of heading statistics** with a color-coded format.

### **4. Error Handling Improvements**
✅ **v3**: Individual error pop-ups for missing files.

✅ **v4**: Consolidated error messages in a single prompt for better user experience.

### **5. Additional Enhancements**
- Time parsing improvements to correctly interpret PTR timestamps.
- More robust NaN handling to avoid issues in heading error calculations.
- Optimized performance by reducing redundant computations and ensuring efficient indexing.

## Necessary Software
This script is written using the Python programming language and several standard libraries. However, it is not necessary to install Python to use it because the script was exported as a `.exe` file and can be run on any Windows computer.

### If Modification is Needed
If it is necessary to modify the script, it can be edited using the following software (already installed on the OLTA Offline computers):
- **Visual Studio Code** (or any other development environment)
- **Python 3.12.6**
- **Python libraries**:
  - `pandas`: to manage the data inside the program (`pip install pandas`)
  - `matplotlib`: to plot the heatmap and the coils values (`pip install matplotlib`)
  - `scipy`: for circular statistics calculations (`pip install scipy`)
  - `pyinstaller`: to create the `.exe` file (`pip install pyinstaller`)

### Running the Script
If the Python script is modified, it can be run from Visual Studio Code. However, this is not recommended for the regular workflow because someone could modify and ruin it by mistake. It is recommended to create a new `.exe` file. This can be done by running the `.bat` script `CREATE_exe.bat` (it is necessary to have the `pyinstaller` Python library installed previously).

## Usage
1. Place the Pipetracker (`.ptr`) and the VisualSoft navigation (`_Coil_X.csv`) files from NaviEdit in the same folder.
2. Run the TSS Converter 4 executable.
3. Select the folder containing the files.
4. Configure the necessary parameters (e.g., coils columns indices, output filename).
5. Use the provided buttons to preview the heatmaps, coil plots, and heading error map.
6. Export the results as `.txt` and `.csv` files pressing ProcessFiles.

## Output Files
- `processed_data.csv`: Merged TSS and navigation data.
- `coil_peaks.csv`: Peak TSS values with corresponding locations.

## Contact
For any issues or questions, please contact vicente.danvila@reachsubsea.com.


