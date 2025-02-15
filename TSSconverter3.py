import pandas as pd #pip install pandas
import os
import tkinter as tk
from tkinter import filedialog, messagebox, StringVar
import matplotlib.pyplot as plt #pip install matplotlib
import numpy as np
import csv

# Global variable to display ROV Heading
global_heading_avg = 0.0  # Initialize with a default value

def plotMap(folder_path, xr_col, yr_col, tss1_col, tss2_col, tss3_col, stbd_offset, port_offset):
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

            # Create a DataFrame with the extracted columns
            df_extracted = pd.DataFrame({
                'xr': xr,
                'yr': yr,
                'tss1': tss1,
                'tss2': tss2,
                'tss3': tss3,
            })

            # Append the extracted DataFrame to the list
            dataframes.append(df_extracted)

    if missing_headers_files:
        messagebox.showerror("Error", f"The following files are missing the required headers: {', '.join(missing_headers_files)}")
        return

    # Concatenate all DataFrames into one
    try:
        combined_df = pd.concat(dataframes, ignore_index=True)
    except ValueError:
        messagebox.showerror("Error", "Missing folder path. Select it using the Browse button")
        return

    # Calculate XR and YR for TSS2
    xr2_col = combined_df['xr']
    yr2_col = combined_df['yr']
    
    # Calculate the average heading angle
    heading_avg = calculate_heading(xr2_col, yr2_col)
    update_heading(heading_avg)
    polar_angle = np.radians(heading_avg) 
    
    # Calculate XR and YR for TSS1
    xr1_col = combined_df['xr'] + stbd_offset * np.sin(polar_angle + np.pi/2)
    yr1_col = combined_df['yr'] + stbd_offset * np.cos(polar_angle + np.pi/2)
    
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
    plt.title(f'Scatter plot of XR, YR, and TSS coils - Heading: {heading_avg:.2f} degrees')
    plt.grid(True)
    plt.show()

def processFiles(folder_path, xr_col, yr_col, tss1_col, tss2_col, tss3_col, stbd_offset, port_offset, output_file):
    # Initialize an empty list to store DataFrames
    dataframes = []
    missing_headers_files = []
    coil_peaks = []

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

            # Create a DataFrame with the extracted columns
            df_extracted = pd.DataFrame({
                'xr': xr,
                'yr': yr,
                'tss1': tss1,
                'tss2': tss2,
                'tss3': tss3,
            })

            # Append the extracted DataFrame to the list
            dataframes.append(df_extracted)

            # Find the highest TSS value and corresponding coil and coordinates
            max_tss1 = tss1.max()
            max_tss2 = tss2.max()
            max_tss3 = tss3.max()
            max_tss = max(max_tss1, max_tss2, max_tss3)

            if max_tss == max_tss1:
                coil = 1
                max_index = tss1.idxmax()
            elif max_tss == max_tss2:
                coil = 2
                max_index = tss2.idxmax()
            else:
                coil = 3
                max_index = tss3.idxmax()

            easting = xr[max_index]
            northing = yr[max_index]

            coil_peaks.append({
                'PTR file': filename,
                'TSS peak value': max_tss,
                'TSS coil': coil,
                'Easting': easting,
                'Northing': northing
            })
            
    print("Coil peak values:")
    print(coil_peaks)
    
    if missing_headers_files:
        messagebox.showerror("Error", f"The following files are missing the required headers: {', '.join(missing_headers_files)}")
        return

    # Concatenate all DataFrames into one
    try:
        combined_df = pd.concat(dataframes, ignore_index=True)
    except ValueError:
        messagebox.showerror("Error", "Missing folder path. Select it using the Browse button")
        return

    # Calculate XR and YR for TSS2
    xr2_col = combined_df['xr']
    yr2_col = combined_df['yr']
    
    # Calculate the average heading angle
    heading_avg = calculate_heading(xr2_col, yr2_col)
    update_heading(heading_avg)
    polar_angle = np.radians(heading_avg)
    
    # Calculate XR and YR for TSS1
    xr1_col = combined_df['xr'] + stbd_offset * np.sin(polar_angle + np.pi/2)
    yr1_col = combined_df['yr'] + stbd_offset * np.cos(polar_angle + np.pi/2)
    
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
    
    # Save the coil peaks to a new CSV file
    if '.csv' in output_file:
        coil_peaks_file_path = os.path.join(folder_path, output_file.replace('.csv', '_coil_peaks.csv'))
    elif '.txt' in output_file:
        coil_peaks_file_path = os.path.join(folder_path, output_file.replace('.txt', '_coil_peaks.csv'))
    else:
        coil_peaks_file_path = os.path.join(folder_path, output_file + '_coil_peaks.csv')
    
    with open(coil_peaks_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["PTR file", "TSS peak value", "TSS coil", "Easting", "Northing"])
        for peak in coil_peaks:
            writer.writerow([peak['PTR file'], peak['TSS peak value'], peak['TSS coil'], peak['Easting'], peak['Northing']])
    
    return heading_avg

def calculate_heading(xr2_col, yr2_col):
    heading_angles = np.degrees(np.arctan2(np.diff(xr2_col), np.diff(yr2_col)))
    #print("Diff xr2: ", np.diff(xr2_col), "Diff yr2: ", np.diff(yr2_col), "Heading Angles: ", heading_angles)
    
    # Calculate the average heading angle
    average_heading = np.mean(heading_angles)
    global_heading_avg = average_heading
    print(f"Average Heading Angle: {average_heading:.2f} degrees")
    
    return average_heading

def select_folder():
    folder_path = filedialog.askdirectory(title="Select a folder")
    folder_entry.delete(0, tk.END)
    folder_entry.insert(0, folder_path)

def show_map():
    folder_path = folder_entry.get()
    if not folder_path:
        messagebox.showerror("Error", "Missing folder path. Select it using the Browse button")
    xr_col = xr_entry.get()
    yr_col = yr_entry.get()
    tss1_col = tss1_entry.get()
    tss2_col = tss2_entry.get()
    tss3_col = tss3_entry.get()
    stbd_offset = stbd_offset_entry.get()
    port_offset = port_offset_entry.get()

    if not folder_path or not xr_col or not yr_col or not tss1_col or not tss2_col or not tss3_col or not stbd_offset or not port_offset:
        messagebox.showerror("Error", "Some fields are required")
        return

    plotMap(folder_path, xr_col, yr_col, tss1_col, tss2_col, tss3_col, stbd_offset, port_offset)

def process():
    folder_path = folder_entry.get()
    if not folder_path:
        messagebox.showerror("Error", "Missing folder path. Select it using the Browse button")
    xr_col = xr_entry.get()
    yr_col = yr_entry.get()
    tss1_col = tss1_entry.get()
    tss2_col = tss2_entry.get()
    tss3_col = tss3_entry.get()
    stbd_offset = stbd_offset_entry.get()
    port_offset = port_offset_entry.get()
    output_file = output_entry.get()

    if not folder_path or not xr_col or not yr_col or not tss1_col or not tss2_col or not tss3_col or not stbd_offset or not port_offset or not output_file:
        messagebox.showerror("Error", "Some fields are required")
        return

    heading_avg = processFiles(folder_path, xr_col, yr_col, tss1_col, tss2_col, tss3_col, stbd_offset, port_offset, output_file)
    messagebox.showinfo("Success", f"Heading: {heading_avg:.2f} degrees. Data has been merged and saved in the file:  {output_file}")

# Main program
print("TSS Converter running...")

# Create the main window
root = tk.Tk()
root.title("TSS Converter")

# Set the font size
font = ("Helvetica", 14)
font_bold = ("Helvetica", 14, "bold")

# Define a StringVar to hold the heading value
heading_var = StringVar()
heading_var.set(f"ROV Heading: {global_heading_avg:.2f} degrees")

# Function to update the heading value
def update_heading(new_heading):
    heading_var.set(f"ROV Heading: {new_heading:.2f} degrees")

# Create and place the widgets
tk.Label(root, text="Select Folder:", font=font).grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)
folder_entry = tk.Entry(root, width=50, font=font)
folder_entry.grid(row=0, column=1, padx=10, pady=5, sticky=tk.W)
folder_entry.insert(0, os.getcwd())  # Default to current working directory
tk.Button(root, text="Browse", command=select_folder, font=font_bold).grid(row=0, column=2, padx=10, pady=5, sticky=tk.W)

tk.Label(root, text="Easting Column:", font=font).grid(row=1, column=0, padx=10, pady=5, sticky=tk.W)
xr_entry = tk.Entry(root, width=20, font=font)
xr_entry.grid(row=1, column=1, padx=10, pady=5, sticky=tk.W)
xr_entry.insert(0, "2")  # Default value

tk.Label(root, text="Northing Column:", font=font).grid(row=2, column=0, padx=10, pady=5, sticky=tk.W)
yr_entry = tk.Entry(root, width=20, font=font)
yr_entry.grid(row=2, column=1, padx=10, pady=5, sticky=tk.W)
yr_entry.insert(0, "3")  # Default value

tk.Label(root, text="Coil 1 Column:", font=font).grid(row=3, column=0, padx=10, pady=5, sticky=tk.W)
tss1_entry = tk.Entry(root, width=20, font=font)
tss1_entry.grid(row=3, column=1, padx=10, pady=5, sticky=tk.W)
tss1_entry.insert(0, "12")  # Default value

tk.Label(root, text="Coil 2 Column:", font=font).grid(row=4, column=0, padx=10, pady=5, sticky=tk.W)
tss2_entry = tk.Entry(root, width=20, font=font)
tss2_entry.grid(row=4, column=1, padx=10, pady=5, sticky=tk.W)
tss2_entry.insert(0, "11")  # Default value

tk.Label(root, text="Coil 3 Column:", font=font).grid(row=5, column=0, padx=10, pady=5, sticky=tk.W)
tss3_entry = tk.Entry(root, width=20, font=font)
tss3_entry.grid(row=5, column=1, padx=10, pady=5, sticky=tk.W)
tss3_entry.insert(0, "10")  # Default value

tk.Label(root, text="Starbord Offset:", font=font).grid(row=6, column=0, padx=10, pady=5, sticky=tk.W)
stbd_offset_entry = tk.Entry(root, width=20, font=font)
stbd_offset_entry.grid(row=6, column=1, padx=10, pady=5, sticky=tk.W)
stbd_offset_entry.insert(0, "0.475")  # Default value

tk.Label(root, text="Port Offset:", font=font).grid(row=7, column=0, padx=10, pady=5, sticky=tk.W)
port_offset_entry = tk.Entry(root, width=20, font=font)
port_offset_entry.grid(row=7, column=1, padx=10, pady=5, sticky=tk.W)
port_offset_entry.insert(0, "-0.47")  # Default value

tk.Label(root, text="Output File Name:", font=font).grid(row=8, column=0, padx=10, pady=5, sticky=tk.W)
output_entry = tk.Entry(root, width=50, font=font)
output_entry.grid(row=8, column=1, padx=10, pady=5, sticky=tk.W)
output_entry.insert(0, "BOSSE_XXX_A.txt")  # Default value

# Label to display the heading value
heading_label = tk.Label(root, textvariable=heading_var, font=font_bold)
heading_label.grid(row=9, column=0, padx=10, pady=5, sticky=tk.W)

tk.Button(root, text="Process Files", command=lambda: [process(), update_heading(global_heading_avg)], font=font_bold).grid(row=10, column=0, columnspan=3, pady=10, sticky=tk.W)
tk.Button(root, text="Show Map", command=lambda: [show_map(), update_heading(global_heading_avg)], font=font_bold).grid(row=10, column=1, columnspan=3, pady=10, sticky=tk.W)

# Run the main loop
root.mainloop()