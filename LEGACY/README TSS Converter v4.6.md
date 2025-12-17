# **TSS Converter v4.6**

## **Overview**
TSS Converter v4.6 is a Python-based tool for processing and analyzing **electromagnetic survey data** from **Teledyne Pipetarcker TSS440 and Visualsoft Navigation CSV files** exported from **EIVA NaviEdit**. It extracts, merges, and analyzes **navigation and electromagnetic data** from UXO detection surveys, allowing users to:
- Process raw **PTR files** and navigation data.
- Generate **heatmaps** and **quality control plots**.
- Calculate **heading errors** and survey statistics.
- Export processed data in a structured format.

---
## **Installation**
### **1. Clone the Repository**
```bash
git clone https://github.com/Vidanfra/TSS-FILE-CONVERTER 
cd TSSConverter
```
### **2. Install Required Dependencies**
Ensure you have Python **3.7+** installed. Install required packages with:
```bash
pip install -r requeriments.txt
```
If `requirements.txt` is missing, install manually:
```bash
pip install pandas matplotlib numpy tk
```
---
## **Usage**
### **1. Run the Script**
Run the script from the command line or a Python environment:
```bash
python TSSconverter4.py
```
Alternatively, double-click the `TSSconverter v4.6.exe` file to launch the script.

![alt text](README_images/image.png)
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

### **4. Export Processed Data**
- The processed data is saved in the selected folder as **BOSSE_XXX_A.txt** (default).
- Coil peak values are saved separately in **BOSSE_XXX_A_coil_peaks.csv** (default).

---

## **File Naming Conventions**
| File Type  | Naming Format        | Description |
|------------|---------------------|-------------|
| PTR File  | `Survey_XYZ.ptr`     | Raw TSS data |
| Coil 1 Nav | `Survey_XYZ_Coil_1.csv` | Navigation for Coil 1 |
| Coil 2 Nav | `Survey_XYZ_Coil_2.csv` | Navigation for Coil 2 |
| Coil 3 Nav | `Survey_XYZ_Coil_3.csv` | Navigation for Coil 3 |
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
![alt text](README_images/image-1.png)
- **Coil plots** showing TSS variations over time.
![alt text](README_images/image-2.png)
- **Heading Quality Control (QC)** with statistical analysis.
![alt text](README_images/image-3.png)

### ✅ **Error Handling & Logging**
- **Logs errors and warnings** for missing or inconsistent data.
- Provides **user-friendly warnings** and **alerts** via GUI.

---

## **Troubleshooting**
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

---

## **Building a smaller executable**
The default requirements now omit SciPy (the only heavy dependency we previously needed solely for `circstd`).
Combined with a clean build environment and compression, the standalone executable drops from ~2.5 GB to under 150 MB.

### **Quick Build (Recommended)**
Use the automated PowerShell script that handles everything:
```powershell
.\BUILD_CLEAN_EXE.ps1
```

This script will:
1. Create a clean virtual environment (`.venv-build`)
2. Install only the minimal required dependencies
3. Build the optimized executable with PyInstaller
4. Report the final size and build time
5. Leave the environment for debugging (you can delete it manually after)

### **Manual Build (Advanced)**
If you prefer to build manually or troubleshoot:

1. **Create a clean virtual environment** so PyInstaller only sees the dependencies from `requirements.txt`:
   ```powershell
   python -m venv .venv-build
   .\.venv-build\Scripts\Activate.ps1
   pip install --upgrade pip
   pip install pandas numpy matplotlib pyinstaller --no-cache-dir
   ```

2. **Build using the optimized spec file**:
   ```powershell
   pyinstaller TSSconverter4.spec --clean --noconfirm
   ```
   
3. *(Optional but recommended)* **Install [UPX](https://upx.github.io/)** and add it to your `PATH`. PyInstaller will automatically use it to compress DLLs when available.

4. **Deactivate and remove** the temporary environment when you are done to free disk space:
   ```powershell
   deactivate
   Remove-Item .venv-build -Recurse -Force
   ```

### **Why was my executable 2.5 GB?**
Large executables typically result from:
- **Building from your main Python environment** with dozens of unnecessary packages (scipy, jupyter, pytest, etc.)
- **Debug symbols not stripped** (`strip=False` in spec file)
- **No bytecode optimization** (`optimize=0` in spec file)
- **Missing exclusions** for heavy packages like Qt, IPython, scipy

The optimized build configuration (`TSSconverter4.spec`) now:
- Enables `strip=True` to remove debug symbols
- Sets `optimize=2` for maximum bytecode optimization
- Excludes 20+ unnecessary packages that were being auto-detected
- Uses UPX compression when available

**Expected sizes:**
- Without optimization: 2-3 GB ❌
- With clean venv + optimizations: 80-150 MB ✅
- With UPX compression: 50-100 MB ✅✅---

## **Contributing**
Feel free to **fork** this repository, submit **issues**, and open **pull requests** for improvements.

👨‍💻 **Author:** Vicente Danvila Fraile

📧 **Contact:** vicente.danvila@reachsubsea.com  
