import pandas as pd #pip install pandas
import os
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt #pip install matplotlib
import numpy as np


def plotMap(folder_path, xr_col, yr_col, tss1_col, tss2_col, tss3_col, heading_col, stbd_offset, port_offset):
    # Initialize an empty list to store DataFrames
    dataframes = []
    missing_headers_files = []

    # Ensure stbd_offset and port_offset are numeric
    try:
        stbd_offset = float(stbd_offset)
        port_offset = float(port_offset)
    except ValueError:
        messagebox.showerror("Error", "stbd_offset and port_offset must be numeric values")
        return
    try:
        xr_col = int(xr_col)
        yr_col = int(yr_col)
        tss1_col = int(tss1_col)
        tss2_col = int(tss2_col)
        tss3_col = int(tss3_col)
        heading_col = int(heading_col)
    except ValueError:
        messagebox.showerror("Error", "Columns numbers must be integer numeric values")
        return

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.ptr'):
            # Construct the full file path
            file_path = os.path.join(folder_path, filename)
            
            # Read the CSV file
            df = pd.read_csv(file_path, delimiter=';', header=None)

            # Extract the required columns by their index positions
            xr = df.iloc[:, xr_col]
            yr = df.iloc[:, yr_col]
            tss1 = df.iloc[:, tss1_col]
            tss2 = df.iloc[:, tss2_col]
            tss3 = df.iloc[:, tss3_col]
            heading = df.iloc[:, tss3_col+1]

            # Create a DataFrame with the extracted columns
            df_extracted = pd.DataFrame({
                'xr': xr,
                'yr': yr,
                'tss1': tss1,
                'tss2': tss2,
                'tss3': tss3,
                'heading': heading
            })

            # Append the extracted DataFrame to the list
            dataframes.append(df_extracted)

    if missing_headers_files:
        messagebox.showerror("Error", f"The following files are missing the required headers: {', '.join(missing_headers_files)}")
        return

    # Concatenate all DataFrames into one
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Convert North heading to polar angle in radians
    print("HEADING: ", heading_col)
    polar_angle = np.radians(heading_col) 
    print("Polar angle: ", polar_angle)
    # Calculate XR and YR for TSS1
    xr1_col = combined_df['xr'] + stbd_offset * np.sin(polar_angle + np.pi/2)
    yr1_col = combined_df['yr'] + stbd_offset * np.cos(polar_angle + np.pi/2)

    # Calculate XR and YR for TSS2
    xr2_col = combined_df['xr']
    yr2_col = combined_df['yr']
    
    # Calculate XR and YR for TSS3
    xr3_col = combined_df['xr'] + port_offset * np.sin(polar_angle + np.pi/2)
    yr3_col = combined_df['yr'] + port_offset * np.cos(polar_angle + np.pi/2)
    
    #print("TEST: head: ", combined_df[heading_col][10], "polar_angle: ", np.degrees(polar_angle[10]), "angle1: ", np.degrees(polar_angle[10] + np.pi/2), "angle3: ", np.degrees(polar_angle[10] - np.pi/2))
    #print("PORT OFFSET: , ", port_offset, "STBD OFFSET: , ", stbd_offset)
    # Create scatter plots for each TSS coil
    plt.figure(figsize=(10, 6))
    scatter1 = plt.scatter(xr1_col, yr1_col, c=combined_df['tss1'], cmap='plasma', marker='o', vmin= -500, vmax= 500)
    scatter2 = plt.scatter(combined_df['xr'], combined_df['yr'], c=combined_df['tss2'], cmap='plasma', marker='o', vmin= -500, vmax= 500)
    scatter3 = plt.scatter(xr3_col, yr3_col, c=combined_df['tss3'], cmap='plasma', marker='o', vmin= -500, vmax= 500)
    #scatter1 = plt.scatter(xr1_col, yr1_col, c='r', marker='o')
    #scatter2 = plt.scatter(combined_df[xr_col], combined_df[yr_col], c='g', marker='o')
    #scatter3 = plt.scatter(xr3_col, yr3_col, c='b', marker='o')
    plt.colorbar(scatter2, label= "TSS")
    plt.xlabel("XR")
    plt.ylabel("YR")
    plt.title(f'Scatter plot of XR, YR, and TSS coils')
    plt.grid(True)
    plt.show()

def processFiles(folder_path, xr_col, yr_col, tss1_col, tss2_col, tss3_col, heading_col, stbd_offset, port_offset, output_file):
    # Initialize an empty list to store DataFrames
    dataframes = []
    missing_headers_files = []

    # Ensure stbd_offset and port_offset are numeric
    try:
        stbd_offset = float(stbd_offset)
        port_offset = float(port_offset)
    except ValueError:
        messagebox.showerror("Error", "stbd_offset and port_offset must be numeric values")
        return
    try:
        xr_col = int(xr_col)
        yr_col = int(yr_col)
        tss1_col = int(tss1_col)
        tss2_col = int(tss2_col)
        tss3_col = int(tss3_col)
        heading_col = int(heading_col)
    except ValueError:
        messagebox.showerror("Error", "Columns numbers must be integer numeric values")
        return

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.ptr'):
            # Construct the full file path
            file_path = os.path.join(folder_path, filename)
            
            # Read the CSV file
            df = pd.read_csv(file_path, delimiter=';', header=None)

            # Extract the required columns by their index positions
            xr = df.iloc[:, xr_col]
            yr = df.iloc[:, yr_col]
            tss1 = df.iloc[:, tss1_col]
            tss2 = df.iloc[:, tss2_col]
            tss3 = df.iloc[:, tss3_col]
            heading = df.iloc[:, tss3_col+1]

            # Create a DataFrame with the extracted columns
            df_extracted = pd.DataFrame({
                'xr': xr,
                'yr': yr,
                'tss1': tss1,
                'tss2': tss2,
                'tss3': tss3,
                'heading': heading
            })

            # Append the extracted DataFrame to the list
            dataframes.append(df_extracted)

    if missing_headers_files:
        messagebox.showerror("Error", f"The following files are missing the required headers: {', '.join(missing_headers_files)}")
        return

    # Concatenate all DataFrames into one
    combined_df = pd.concat(dataframes, ignore_index=True)

    print("HEADING: ", heading_col)
    polar_angle = np.radians(heading_col) 
    print("Polar angle: ", polar_angle)
    
    # Calculate XR and YR for TSS1
    xr1_col = combined_df['xr'] + stbd_offset * np.sin(polar_angle + np.pi/2)
    yr1_col = combined_df['yr'] + stbd_offset * np.cos(polar_angle + np.pi/2)

    # Calculate XR and YR for TSS2
    xr2_col = combined_df['xr']
    yr2_col = combined_df['yr']
    
    # Calculate XR and YR for TSS3
    xr3_col = combined_df['xr'] + port_offset * np.sin(polar_angle + np.pi/2)
    yr3_col = combined_df['yr'] + port_offset * np.cos(polar_angle + np.pi/2)

    # Create new DataFrames for each coil
    df_tss1 = pd.DataFrame({'xr': xr1_col, 'yr': yr1_col, 'TSS': combined_df['tss1']})
    df_tss2 = pd.DataFrame({'xr': xr2_col, 'yr': yr2_col, 'TSS': combined_df['tss2']})
    df_tss3 = pd.DataFrame({'xr': xr3_col, 'yr': yr3_col, 'TSS': combined_df['tss3']})
    
    # Concatenate the DataFrames vertically
    new_df = pd.concat([df_tss1, df_tss2, df_tss3], ignore_index=True) 

    # Save the merged DataFrame to a new CSV file
    output_file_path = os.path.join(folder_path, output_file)
    new_df.to_csv(output_file_path, index=False)

def getOutputFileName(output_file, folder_path):
    # Get the last folder name
    last_folder_name = os.path.basename(folder_path)
    # Create the output file name with the last folder name
    output_file = last_folder_name + "_" + output_file
    return output_file

def select_folder():
    folder_path = filedialog.askdirectory(title="Select a folder")
    folder_entry.delete(0, tk.END)
    folder_entry.insert(0, folder_path)

def show_map():
    folder_path = folder_entry.get()
    xr_col = xr_entry.get()
    yr_col = yr_entry.get()
    tss1_col = tss1_entry.get()
    tss2_col = tss2_entry.get()
    tss3_col = tss3_entry.get()
    heading_col = heading_entry.get()
    stbd_offset = stbd_offset_entry.get()
    port_offset = port_offset_entry.get()

    if not folder_path or not xr_col or not yr_col or not tss1_col or not tss2_col or not tss3_col or not heading_col or not stbd_offset or not port_offset:
        messagebox.showerror("Error", "Some fields are required")
        return

    plotMap(folder_path, xr_col, yr_col, tss1_col, tss2_col, tss3_col, heading_col, stbd_offset, port_offset)

def process():
    folder_path = folder_entry.get()
    xr_col = xr_entry.get()
    yr_col = yr_entry.get()
    tss1_col = tss1_entry.get()
    tss2_col = tss2_entry.get()
    tss3_col = tss3_entry.get()
    heading_col = heading_entry.get()
    stbd_offset = stbd_offset_entry.get()
    port_offset = port_offset_entry.get()
    output_file = output_entry.get()
    output_file = getOutputFileName(output_file, folder_path)

    if not folder_path or not xr_col or not yr_col or not tss1_col or not tss2_col or not tss3_col or not heading_col or not stbd_offset or not port_offset or not output_file:
        messagebox.showerror("Error", "Some fields are required")
        return

    processFiles(folder_path, xr_col, yr_col, tss1_col, tss2_col, tss3_col, heading_col, stbd_offset, port_offset, output_file)
    messagebox.showinfo("Success", f"Data has been merged and saved to {output_file}")

# Create the main window
root = tk.Tk()
root.title("CSV Processor")

# Set the font size
font = ("Helvetica", 14)

# Create and place the widgets
tk.Label(root, text="Select Folder:", font=font).grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)
folder_entry = tk.Entry(root, width=50, font=font)
folder_entry.grid(row=0, column=1, padx=10, pady=5)
folder_entry.insert(0, os.getcwd())  # Default to current working directory
tk.Button(root, text="Browse", command=select_folder, font=font).grid(row=0, column=2, padx=10, pady=5)

tk.Label(root, text="Easting Column:", font=font).grid(row=1, column=0, padx=10, pady=5, sticky=tk.W)
xr_entry = tk.Entry(root, width=50, font=font)
xr_entry.grid(row=1, column=1, padx=10, pady=5)
xr_entry.insert(0, "2")  # Default value

tk.Label(root, text="Northing Column:", font=font).grid(row=2, column=0, padx=10, pady=5, sticky=tk.W)
yr_entry = tk.Entry(root, width=50, font=font)
yr_entry.grid(row=2, column=1, padx=10, pady=5)
yr_entry.insert(0, "3")  # Default value

tk.Label(root, text="Coil 1 Column:", font=font).grid(row=3, column=0, padx=10, pady=5, sticky=tk.W)
tss1_entry = tk.Entry(root, width=50, font=font)
tss1_entry.grid(row=3, column=1, padx=10, pady=5)
tss1_entry.insert(0, "12")  # Default value

tk.Label(root, text="Coil 2 Column:", font=font).grid(row=4, column=0, padx=10, pady=5, sticky=tk.W)
tss2_entry = tk.Entry(root, width=50, font=font)
tss2_entry.grid(row=4, column=1, padx=10, pady=5)
tss2_entry.insert(0, "11")  # Default value

tk.Label(root, text="Coil 3 Column:", font=font).grid(row=5, column=0, padx=10, pady=5, sticky=tk.W)
tss3_entry = tk.Entry(root, width=50, font=font)
tss3_entry.grid(row=5, column=1, padx=10, pady=5)
tss3_entry.insert(0, "10")  # Default value

tk.Label(root, text="ROV Heading :", font=font).grid(row=6, column=0, padx=10, pady=5, sticky=tk.W)
heading_entry = tk.Entry(root, width=50, font=font)
heading_entry.grid(row=6, column=1, padx=10, pady=5)
heading_entry.insert(0, "0")  # Default value

tk.Label(root, text="Starbord Offset:", font=font).grid(row=7, column=0, padx=10, pady=5, sticky=tk.W)
stbd_offset_entry = tk.Entry(root, width=50, font=font)
stbd_offset_entry.grid(row=7, column=1, padx=10, pady=5)
stbd_offset_entry.insert(0, "0.475")  # Default value

tk.Label(root, text="Port Offset:", font=font).grid(row=7, column=2, padx=10, pady=5, sticky=tk.W)
port_offset_entry = tk.Entry(root, width=50, font=font)
port_offset_entry.grid(row=7, column=3, padx=10, pady=5)
port_offset_entry.insert(0, "-0.47")  # Default value

tk.Label(root, text="Output File Name:", font=font).grid(row=8, column=0, padx=10, pady=5, sticky=tk.W)
output_entry = tk.Entry(root, width=50, font=font)
output_entry.grid(row=8, column=1, padx=10, pady=5)
output_entry.insert(0, "combined.txt")  # Default value

tk.Button(root, text="Process Files", command=process, font=font).grid(row=9, column=0, columnspan=3, pady=10)
tk.Button(root, text="Show Map", command=show_map, font=font).grid(row=9, column=1, columnspan=3, pady=10)

# Run the main loop
root.mainloop()