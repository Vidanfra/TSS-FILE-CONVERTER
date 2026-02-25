# **TSS AutoProcessor v5**

## **Overview**
TSS AutoProcessor v5 is a Python-based tool for processing and analyzing **electromagnetic survey data** from **Teledyne Pipetarcker TSS440 and Visualsoft Navigation CSV files** exported from **EIVA NaviEdit**. It extracts, merges, and analyzes **navigation and electromagnetic data** from UXO detection surveys, allowing users to:
- Export from NaviEdit the pipetracker and navigation data.
- Process raw **PTR files** and navigation data.
- Generate **heatmaps** and **quality control plots of TSS, Altitude and Depth**.
- Calculate **heading errors** and survey statistics.
- Export processed data in a structured format.
- Generate Geotiff and PNG heatmaps

---
## **Installation**
### **1. Clone the Repository**
```bash
git clone https://github.com/Vidanfra/TSS-FILE-CONVERTER 
```
### **2. Install Required Dependencies**
Ensure you have Python **3.7+** installed. Install required packages with:
```bash
pip install -r requeriments.txt
```
If `requirements.txt` is missing, install manually:
```bash
pip install pandas matplotlib numpy tk scipy rasterio pyodbc
```
---
## **Usage**
### **1. Run the Script**
Run the script from the command line or a Python environment:
```bash
python TSSAutoProcessor.py
```
Alternatively, double-click the `TSSAutoProcessor.exe` file to launch the script.

![alt text](_README_images/GUI.png)
### **2. Select Input Data**
- Click **Browse** to select the folder containing **PTR and CSV files**.
- Ensure **PTR files** and corresponding **Coil navigation CSV files** are present.
- You must use the **column numbers** for each coil in the PTR files by default **[Coil 1 = 10, Coil 2 = 11, Coil 3 = 12]**.
- You only need to change column numbers if the convention in NaviEdit or TSS DeepView is changed.

### **3. Process Data**
- **Process Files**: Extract and merge navigation and electromagnetic data.
- **Show Map**: Generate a **heatmap** of **TSS electromagnetic values** and **altitude**.
- **Show Coils**: Plot **TSS values** for each coil.
- **Heading QC**: Display heading quality control statistics.
- **Plot Altitude**: Visualize altitude data for all coils.
- **Plot Depth & Altitude**: Compare depth and altitude profiles.

### **4. Export Processed Data**
- The processed data is saved in the selected folder as **BOSSE_XXX_A.txt** (default).
- Coil peak values are saved separately in **BOSSE_XXX_A_coil_peaks.csv** (default).

### **5. WFM Export & Auto-Clicker**
The tool includes an automated workflow for exporting data from **EIVA Workflow Manager (WFM)**.
- **Enable Auto-clicker**: Check this box to automatically accept "Export settings" dialogs in WFM.
- **NE Database Settings**: Configure the connection to the NaviEdit SQL database to fetch block IDs directly.
- **Run WFM Export**: Initiates the export process for selected blocks. The auto-clicker will handle the repetitive confirmation dialogs, allowing for unattended batch processing.

### **6. Settings & Configuration**
- **Define User Offsets in NE**
  - Before starting the TSSAutoProcessor setup, introduce the User Offsets in NaviEdit. 
  - It is important to respect the sequence Coil_port > Coil_center > Coil_stbd > CRP to get the correct ID numbers: 
    - Coil_port -> ID = 1
    - Coil_center -> ID = 2 
    - Coil_stbd -> ID = 3 
    - CRP -> ID = 4 
    
    
  ![alt text](_README_images/user_offsets.png)
- **NE Database Settings**:
  - Connect to the local NaviEdit database.
  
  ![alt text](_README_images/ne_settings.png)

- **Check Correct Coil Convention**:
  - Check the correct position of the coils. The TSS 440 has a menu in the TSS DeepView software to define the Coil number (1,2,3) for Port, Center and Starboard. 
  - These numbers determine the column position for each coil in the .ptr export file.
  - It doesn't really matter what coil number convention you use, but it should match with the Online settings. The used by default order is: Coil 1 = STARBOARD, Coil 2 = CENTER, Coil 3 =PORT. 
  - If Online changes the coil number convention in the TSS software, then you have to swap the coil columns  (Coil 1 = 12, Coil 3 = 10). 
  - To assure that the coils are not swapped, you must perform a Coil position test on deck
  - By default the coils numbers convention are:
    - TSS DeepView: **[Coil 1 = Starboard, Coil 2 = Central, Coil 3 = Port]**
    - NaviEdit Offsets: **[Coil 1 = Port, Coil 2 = Central, Coil 3 = Starboard]**

  ![alt text](_README_images/coils_numbers.png)

- **Check the File Suffixes**:
  - These are the suffixes that the script will export and look for to read the data.
  - It is not necessary to be changed.

  ![alt text](_README_images/files_suffixes.png)

- **Color Heatmap Settings**:
  - Customize the color palette and value boundaries for TSS and Altitude heatmaps.
  - Adjust the **Cell Size** for grid interpolation.

  ![alt text](_README_images/heatmap_settings.png)
- **Define the Heatmap Cell Size**:
  - Define the Cell Size (m) of each TSS value point depending on the sensor. 
  - Standard value for TSS 440 is 0.5m

- **Enable Auto-clicker during the WFM Export**:
  - Auto-clicker allows to the script to accept automatically the pop up messages from WFM without human intervention. 
  - If it is necessary to click manually any setting message, you can untick the Auto-clicker to do it manually. 
- **Include CRP navigation exports**:
  - Include ROV CRP exports the navigation files and displays its data in the plots
  - It takes a bit of extra time in the WFM export process, but it is recommended to leave it enabled fro redundancy and comparison.

---

## **File Naming Conventions**
| File Type  | Naming Format        | Description |
|------------|---------------------|-------------|
| PTR File  | `Survey_XYZ.ptr`     | Raw TSS data |
| Coil 1 Nav | `Survey_XYZ_Coil_port.csv` | Navigation for Coil 1 |
| Coil 2 Nav | `Survey_XYZ__Coil_center.csv` | Navigation for Coil 2 |
| Coil 3 Nav | `Survey_XYZ_Coil_stbd.csv` | Navigation for Coil 3 |
| CRP Nav | `Survey_XYZ_CRP.csv` | Navigation for CRP |
| Output     | `BOSSE_XXX_A.txt`   | Processed TSS Data |
| Coil Peaks | `BOSSE_XXX_A_coil_peaks.csv` | Peak values per coil |

---

## **Features & Functionality**
### ✅ **Data Processing**
- Reads **pipetracker files** and corresponding **navigation CSVs**.
- Matches timestamps between **PTR** and **navigation** data.
- Swaps coil numbers to match **TSS DeepView** with **NaviEdit Offsets** conventions.
- Exports an output `.txt` file to create **TSS electromagnetic values** and **altitude** in **NaviModel**.
- Exports an output `.csv` file cointaining the **TSS peak values of each coil at each line** of the survey.

### ✅ **Analysis & Visualization**
- **Heatmaps** for **TSS electromagnetic values** and **altitude**.
![alt text](_README_images/image-1.png)
- **Coil plots** showing TSS variations over time.
![alt text](_README_images/image-2.png)
- **Heading Quality Control (QC)** with statistical analysis.
![alt text](_README_images/image-3.png)
- **Altitude & Depth Plots**: Visualize sensor altitude and depth profiles to identify anomalies or data gaps.
- **Geotiff Export**: Automatically generates **RGB** and **32-bit Float GeoTIFFs** for seamless import into NaviModel or GIS software.

### ✅ **Error Handling & Logging**
- **Logs errors and warnings** for missing or inconsistent data.
- Provides **user-friendly warnings** and **alerts** via GUI.

---

## **Troubleshooting**
### ❓ WFM export fails
✔ Review that you introduced the correct **Block IDs** (comma-separated or ranges, e.g., `100-105, 107`).

✔ Ensure **NaviEdit** is closed or not locking the database.

✔ Verify that the **SQL Server connection settings** are correct in "NE Database Settings".

### ❓ Coils position looks swapped in the heatmaps
✔ Check that the **coils numbers convention is what is expected** in both files.

✔ Check **column numbers** for TSS values.

✔ By default the coils numbers convention are:

- TSS DeepView: **[Coil 1 = Starboard, Coil 2 = Central, Coil 3 = Port]**
- NaviEdit Offsets: **[Coil 1 = Port, Coil 2 = Central, Coil 3 = Starboard]**
### ❓ No data appears after processing
✔ Ensure the **PTR and CSV files** are correctly formatted and in the selected folder.  
✔ Check **column numbers** for TSS values.

### ❓ Time mismatch errors
✔ Ensure **navigation and PTR timestamps** match within **0.25 seconds**.  
✔ Sometimes **files may contain several points not paired** on time.
✔ Check the files **if you get more than 5 or 10 points mismached**.

### ❓ Missing Navigation Files warning
✔ Make sure all **three coil navigation CSVs** exist and are named correctly for each **PTR file**.

## **Contributing**
Feel free to **fork** this repository, submit **issues**, and open **pull requests** for improvements.

👨‍💻 **Author:** Vicente Danvila Fraile

📧 **Contact:** vicente.danvila@reachsubsea.com  
