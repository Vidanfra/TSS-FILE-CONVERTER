import pandas as pd #pip install pandas
import os
import tkinter as tk
from tkinter import filedialog, messagebox, StringVar
import matplotlib.pyplot as plt #pip install matplotlib
import numpy as np
import csv
from datetime import datetime

# Global variable to display ROV Heading
global_heading_avg = 0.0  # Initialize with a default value

# Maximum time difference in seconds
MAX_TIME_DIFF_SEC = 0.15

# Column positions in the PTR file
DATE_COLUMN_POS = 0
TIME_COLUMN_POS = 1
EAST_COLUMN_POS = 2
NORTH_COLUMN_POS = 3

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
    xr1_col = combined_df['xr'] + port_offset * np.sin(polar_angle - np.pi/2)
    yr1_col = combined_df['yr'] + port_offset * np.cos(polar_angle - np.pi/2)
    
    # Calculate XR and YR for TSS3
    xr3_col = combined_df['xr'] + stbd_offset * np.sin(polar_angle - np.pi/2)
    yr3_col = combined_df['yr'] + stbd_offset * np.cos(polar_angle - np.pi/2)
    
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
    
def plotCoils(folder_path, xr_col, yr_col, tss1_col, tss2_col, tss3_col, stbd_offset, port_offset):
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

            if df is None:
                messagebox.showerror("Error", "Missing folder path. Select it using the Browse button")
                return
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
            
            plt.figure(num=filename,figsize=(10, 6))
            coils_plot = plt.plot(df_extracted['tss1'], color='r', label='TSS1')
            coils_plot = plt.plot(df_extracted['tss2'], color='b', label='TSS2')
            coils_plot = plt.plot(df_extracted['tss3'], color='g', label='TSS3')
            plt.xlabel("Time [sec]")
            plt.ylabel("TSS values [uV]")
            plt.title(f'TSS values for each coil - {filename}')
            plt.legend()
            plt.grid(True)    
    plt.show() 
    if missing_headers_files:
        messagebox.showerror("Error", f"The following files are missing the required headers: {', '.join(missing_headers_files)}")
        return
    
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
            
            min_tss1 = tss1.min()
            min_tss2 = tss2.min()
            min_tss3 = tss3.min()
            min_tss = min(min_tss1, min_tss2, min_tss3)
            
            if(abs(max_tss) > abs(min_tss)):
                abs_max_tss = max_tss
                if max_tss == max_tss1:
                    coil = 1
                    max_index = tss1.idxmax()
                elif max_tss == max_tss2:
                    coil = 2
                    max_index = tss2.idxmax()
                else:
                    coil = 3
                    max_index = tss3.idxmax()
            else:
                abs_max_tss = min_tss
                if min_tss == min_tss1:
                    coil = 1
                    max_index = tss1.idxmin()
                elif min_tss == min_tss2:
                    coil = 2
                    max_index = tss2.idxmin()
                else:
                    coil = 3
                    max_index = tss3.idxmin()

            easting = xr[max_index]
            northing = yr[max_index]

            coil_peaks.append({
                'PTR file': filename,
                'TSS peak value': abs_max_tss,
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
    xr1_col = combined_df['xr'] + port_offset * np.sin(polar_angle - np.pi/2)
    yr1_col = combined_df['yr'] + port_offset * np.cos(polar_angle - np.pi/2)
    
    # Calculate XR and YR for TSS3
    xr3_col = combined_df['xr'] + stbd_offset * np.sin(polar_angle - np.pi/2)
    yr3_col = combined_df['yr'] + stbd_offset * np.cos(polar_angle - np.pi/2)

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

def extractData2(folder_path, tss1_col, tss2_col, tss3_col):
    # Initialize an empty list to store DataFrames
    ptr_dataframe = []
    nav_coil1_dataframe = []
    nav_coil2_dataframe = []
    nav_coil3_dataframe = []

    try:
        tss1_col = int(tss1_col)
        tss2_col = int(tss2_col)
        tss3_col = int(tss3_col)
    except ValueError:
        messagebox.showerror("Error", "Columns numbers must be integer numeric values")
        return

    # Loop through all Pfiles in the folder
    for filename in os.listdir(folder_path):
        # Extract Pipetracker data
        if filename.endswith('.ptr'):
            # Construct the full file path
            file_path = os.path.join(folder_path, filename)
            
            # Read the CSV file
            df = pd.read_csv(file_path, delimiter=';', header=None)

            # Extract the required columns by their index positions
            time = df.iloc[:, 1]
            tss1 = df.iloc[:, tss1_col]
            tss2 = df.iloc[:, tss2_col]
            tss3 = df.iloc[:, tss3_col]

            # Create a DataFrame with the extracted columns
            df_extracted = pd.DataFrame({
                'time': time,
                'tss1': tss1,
                'tss2': tss2,
                'tss3': tss3,
            })

            # Append the extracted DataFrame to the list
            ptr_dataframe.append(df_extracted)
        
        # Extract Navigation data from Coil 1
        if filename.endswith('_Coil_1.csv') :
            # Construct the full file path
            file_path = os.path.join(folder_path, filename)
            
            # Read the CSV file
            df_extracted = pd.read_csv(file_path, delimiter=',')

            # Append the extracted DataFrame to the list
            nav_coil1_dataframe.append(df_extracted)

        # Extract Navigation data from Coil 2
        if filename.endswith('_Coil_2.csv') :
            # Construct the full file path
            file_path = os.path.join(folder_path, filename)
            
            # Read the CSV file
            df_extracted = pd.read_csv(file_path, delimiter=',')

            # Append the extracted DataFrame to the list
            nav_coil2_dataframe.append(df_extracted) 

        # Extract Navigation data from Coil 3
        if filename.endswith('_Coil_3.csv') :
            # Construct the full file path
            file_path = os.path.join(folder_path, filename)
            
            # Read the CSV file
            df_extracted = pd.read_csv(file_path, delimiter=',')

            # Append the extracted DataFrame to the list
            nav_coil3_dataframe.append(df_extracted)       
        
    # Concatenate all DataFrames into one
    try:
        ptr_df = pd.concat(ptr_dataframe, ignore_index=True)
        nav_coil1_df = pd.concat(nav_coil1_dataframe, ignore_index=True)
        nav_coil2_df = pd.concat(nav_coil2_dataframe, ignore_index=True)
        nav_coil3_df = pd.concat(nav_coil3_dataframe, ignore_index=True)
    except ValueError:
        messagebox.showerror("Error", "Missing folder path. Select it using the Browse button")
        return
    
    coil1_df = pd.DataFrame(columns=['Time', 'Time_diff', 'TSS1', 'Gyro', 'Alt', 'Depth'])
    coil2_df = pd.DataFrame(columns=['Time', 'Time_diff', 'TSS2', 'Gyro', 'Alt', 'Depth'])
    coil3_df = pd.DataFrame(columns=['Time', 'Time_diff', 'TSS3', 'Gyro', 'Alt', 'Depth'])
    
    for index, row in ptr_df.iterrows():
        # Convert slow time to datetime object
        slow_time = datetime.strptime(str(row['time']), '%H%M%S%f')
        # Convert fast times to datetime objects
        fast_time_df = pd.DataFrame()
        fast_time_df['Time'] = pd.to_datetime(nav_coil1_df['Time'], format='%H:%M:%S.%f')
        # Find the index of the closest time in the fast time DataFrame
        closest_index = (fast_time_df['Time'] - slow_time).abs().idxmin()

        # Calculate the difference in milliseconds
        time_diff = (slow_time - fast_time_df['Time'][closest_index]).total_seconds() * 1000
        print(f"Closest index row {index}: {closest_index} - Slow time: {slow_time} - Fast time: {fast_time_df['Time'][closest_index]} - Diff: {time_diff:.2f} ms")

        if abs(time_diff) > MAX_TIME_DIFF_MS:
            print(f"Time difference is too high: {time_diff:.2f} ms in row {index}")
            messagebox.showerror("Error", f"Time difference is too high: {time_diff:.2f} ms in row {index}")
        
        data_coil1 = {
            'Time': slow_time,
            'Time_diff': time_diff,
            'TSS1': row['tss1'],
            'Gyro': nav_coil1_df['Gyro'][closest_index],
            'Alt': nav_coil1_df['Alt'][closest_index],
            'Depth': nav_coil1_df['Depth'][closest_index],
        }
        coil1_df = coil1_df.append(data_coil1, ignore_index=True)

        data_coil2 = {
            'Time': slow_time,
            'Time_diff': time_diff,
            'TSS2': row['tss2'],
            'Gyro': nav_coil2_df.iloc[closest_index]['Gyro'],
            'Alt': nav_coil2_df.iloc[closest_index]['Alt'],
            'Depth': nav_coil2_df.iloc[closest_index]['Depth'],
        }
        coil2_df = coil2_df.append(data_coil2, ignore_index=True)

        data_coil3 = {
            'Time': slow_time,
            'Time_diff': time_diff,
            'TSS3': row['tss3'],
            'Gyro': nav_coil3_df.iloc[closest_index]['Gyro'],
            'Alt': nav_coil3_df.iloc[closest_index]['Alt'],
            'Depth': nav_coil3_df.iloc[closest_index]['Depth'],
        }
        coil3_df = coil3_df.append(data_coil3, ignore_index=True)

    merged_coil1_df = pd.concat(coil1_df, ignore_index=True)
    #merged_coil2_df = pd.concat(merged_coil2, ignore_index=True)
    #merged_coil3_df = pd.concat(merged_coil3, ignore_index=True)

    print("Merged Coil 1 DataFrame:")
    print(merged_coil1_df)

def convert_to_datetime(time_str, format_str='%H%M%S%f'):
    return datetime.strptime(str(time_str), format_str)

def read_csv_file(file_path, delimiter):
    return pd.read_csv(file_path, delimiter=delimiter)

def extractData(folder_path, tss1_col, tss2_col, tss3_col):
    ptr_dataframe = []
    nav_coil1_dataframe = []
    nav_coil2_dataframe = []
    nav_coil3_dataframe = []

    # Ensure the columns positions are numeric
    try:
        tss1_col = int(tss1_col)
        tss2_col = int(tss2_col)
        tss3_col = int(tss3_col)
    except ValueError:
        messagebox.showerror("Error", "Columns numbers must be integer numeric values")
        return

    # Check if the folder exists
    if not os.path.exists(folder_path):
        messagebox.showerror("Error", f"Folder {folder_path} does not exist")
        return

    # Check if the folder is empty
    if not os.listdir(folder_path):
        messagebox.showerror("Error", f"Folder {folder_path} is empty")
        return

    # Check all the corresponding files in the folder
    error_messages = []

    for filename in os.listdir(folder_path):

        if filename.endswith('.ptr'):
            only_name = filename.split('.')[0]
            missing_files_coils = []
            
            if not (only_name + '_Coil_1.csv') in os.listdir(folder_path):
                missing_files_coils.append("Coil 1")
            if not (only_name + '_Coil_2.csv') in os.listdir(folder_path):
                missing_files_coils.append("Coil 2")
            if not (only_name + '_Coil_3.csv') in os.listdir(folder_path):
                missing_files_coils.append("Coil 3")

            if len(missing_files_coils) == 3:
                error_messages.append(f"Missing all CSV Navigation files for PTR file: {filename}")
            elif missing_files_coils:
                error_messages.append(f"Missing CSV Navigation {', '.join(missing_files_coils)} files for PTR file: {filename}")
        
        if filename.endswith('_Coil_1.csv') or filename.endswith('_Coil_2.csv') or filename.endswith('_Coil_3.csv'):
            only_name = filename.split('_')[0]
            if not (only_name + '.ptr') in os.listdir(folder_path):
                error_messages.append(f"Missing PTR file for the CSV Navigation file: {filename}")
            if not (only_name + '_Coil_1.csv') in os.listdir(folder_path) or not (only_name + '_Coil_2.csv') in os.listdir(folder_path) or not (only_name + '_Coil_3.csv') in os.listdir(folder_path):
                error_messages.append(f"Missing the other necessary CSV Navigation files for this file: {filename}")
    
    # Show all errors in a single message box at the end
    if error_messages:
        messagebox.showerror("Error", "\n".join(error_messages))

    # Loop through all files in the folder and extract the required data
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if filename.endswith('.ptr'):           
            only_name = filename.split('.')[0]
            if not (only_name + '_Coil_1.csv') or not (only_name + '_Coil_2.csv') or not (only_name + '_Coil_3.csv') in os.listdir(folder_path):
                continue # Skip this iteration if any of the navigation required files is missing
            df = read_csv_file(file_path, ';')
            date = df.iloc[:, DATE_COLUMN_POS]
            time = df.iloc[:, TIME_COLUMN_POS]
            easting = df.iloc[:, EAST_COLUMN_POS]
            northing = df.iloc[:, NORTH_COLUMN_POS]
            tss1 = df.iloc[:, tss1_col]
            tss2 = df.iloc[:, tss2_col]
            tss3 = df.iloc[:, tss3_col]

            df_extracted = pd.DataFrame({
                'Date PTR': date,
                'Time PTR': time,
                'Easting PTR': easting,
                'Northing PTR': northing,
                'TSS1': tss1,
                'TSS2': tss2,
                'TSS3': tss3,
            })

            ptr_dataframe.append(df_extracted)

        if filename.endswith('_Coil_1.csv'):
            only_name = filename.split('_')[0]
            if not (only_name + '.ptr') in os.listdir(folder_path) or not (only_name + '_Coil_2.csv') in os.listdir(folder_path) or not (only_name + '_Coil_3.csv') in os.listdir(folder_path):
                continue # Skip this iteration if any of the PTR or navigation required files is missing
            nav_coil1_dataframe.append(read_csv_file(file_path, ','))

        if filename.endswith('_Coil_2.csv'):
            only_name = filename.split('_')[0]
            if not (only_name + '.ptr') in os.listdir(folder_path) or not (only_name + '_Coil_1.csv') in os.listdir(folder_path) or not (only_name + '_Coil_3.csv') in os.listdir(folder_path):
                continue # Skip this iteration if any of the PTR or navigation required files is missing
            nav_coil2_dataframe.append(read_csv_file(file_path, ','))

        if filename.endswith('_Coil_3.csv'):
            only_name = filename.split('_')[0]
            if not (only_name + '.ptr') in os.listdir(folder_path) or not (only_name + '_Coil_1.csv') in os.listdir(folder_path) or not (only_name + '_Coil_2.csv') in os.listdir(folder_path):
                continue # Skip this iteration if any of the PTR or navigation required files is missing
            nav_coil3_dataframe.append(read_csv_file(file_path, ','))

    # Check if any PTR or Navigation files were found
    if not ptr_dataframe:
        messagebox.showerror("Error", "No PTR files found in the folder")
        return
    if not nav_coil1_dataframe or not nav_coil2_dataframe or not nav_coil3_dataframe:
        messagebox.showerror("Error", "No Navigation files found in the folder")
        return
    
    ptr_df = pd.concat(ptr_dataframe, ignore_index=True)
    nav_coil1_df = pd.concat(nav_coil1_dataframe, ignore_index=True)
    nav_coil2_df = pd.concat(nav_coil2_dataframe, ignore_index=True)
    nav_coil3_df = pd.concat(nav_coil3_dataframe, ignore_index=True)

    # Convert the Time columns to datetime objects
    ptr_df['Time PTR'] = pd.to_datetime(ptr_df['Date PTR'] + ' ' + ptr_df['Time PTR'].astype(str), format='%d.%m.%Y %H%M%S%f')
    nav_coil1_df['Time'] = pd.to_datetime(nav_coil1_df['Date'] + ' ' + nav_coil1_df['Time'], format='%d/%m/%Y %H:%M:%S.%f')
    nav_coil2_df['Time'] = pd.to_datetime(nav_coil2_df['Date'] + ' ' + nav_coil2_df['Time'], format='%d/%m/%Y %H:%M:%S.%f')
    nav_coil3_df['Time'] = pd.to_datetime(nav_coil3_df['Date'] + ' ' + nav_coil3_df['Time'], format='%d/%m/%Y %H:%M:%S.%f')

    # Merge the DataFrames based on the closest time in the navigation data to the PTR data
    merged_coil1_df = pd.merge_asof(ptr_df, nav_coil1_df, left_on='Time PTR', right_on='Time', direction='nearest')
    merged_coil2_df = pd.merge_asof(ptr_df, nav_coil2_df, left_on='Time PTR', right_on='Time', direction='nearest')
    merged_coil3_df = pd.merge_asof(ptr_df, nav_coil3_df, left_on='Time PTR', right_on='Time', direction='nearest')

    # Check if time difference exceeds the allowed threshold
    time_diff = (merged_coil1_df['Time PTR'] - merged_coil1_df['Time']).dt.total_seconds()
    merged_coil1_df['Time_diff'] = time_diff
    if (abs(time_diff) > MAX_TIME_DIFF_SEC).any():
        messagebox.showerror("Error", f"Time difference between PTR and Coil 1 is too high: {abs(time_diff).max():.3f} seconds")

    time_diff = (merged_coil2_df['Time PTR'] - merged_coil2_df['Time']).dt.total_seconds()
    merged_coil2_df['Time_diff'] = time_diff
    if (abs(time_diff) > MAX_TIME_DIFF_SEC).any():
        messagebox.showerror("Error", f"Time difference between PTR and Coil 2 is too high: {abs(time_diff).max():.3f} seconds")

    time_diff = (merged_coil3_df['Time PTR'] - merged_coil3_df['Time']).dt.total_seconds()
    merged_coil3_df['Time_diff'] = time_diff
    if (abs(time_diff) > MAX_TIME_DIFF_SEC).any():
        messagebox.showerror("Error", f"Time difference between PTR and Coil 3 is too high: {abs(time_diff).max():.3f} seconds")   

    return merged_coil1_df, merged_coil2_df, merged_coil3_df

def mergeData(merged_coil1_df, merged_coil2_df, merged_coil3_df):
    # Drop the innecessary TSS and Date columns
    merged_coil1_df['TSS']  = merged_coil1_df['TSS1']
    del merged_coil1_df['TSS1']
    del merged_coil1_df['TSS2']
    del merged_coil1_df['TSS3']
    del merged_coil1_df['Date PTR']
    del merged_coil1_df['Date']

    merged_coil2_df['TSS']  = merged_coil2_df['TSS2']
    del merged_coil2_df['TSS1']
    del merged_coil2_df['TSS2']
    del merged_coil2_df['TSS3']
    del merged_coil2_df['Date PTR']
    del merged_coil2_df['Date']

    merged_coil3_df['TSS']  = merged_coil3_df['TSS3']
    del merged_coil3_df['TSS1']
    del merged_coil3_df['TSS2']
    del merged_coil3_df['TSS3']
    del merged_coil3_df['Date PTR']
    del merged_coil3_df['Date']

    # Ensure all dataframes have the same index length (if not, trim to the smallest length)
    min_length = min(len(merged_coil1_df), len(merged_coil2_df), len(merged_coil3_df))

    merged_coil1_df = merged_coil1_df.iloc[:min_length]
    merged_coil2_df = merged_coil2_df.iloc[:min_length]
    merged_coil3_df = merged_coil3_df.iloc[:min_length]

    # Create an interleaved dataframe
    interleaved_df = pd.concat([merged_coil1_df, merged_coil2_df, merged_coil3_df], axis=0).sort_index(kind='merge') # sort_index(kind='merge') interleaves the dataframes (first row of each, then second row of each, etc.)

    # Reset index for a clean output
    interleaved_df = interleaved_df.reset_index(drop=True)

    print("Merged Coils DataFrame:")
    print(interleaved_df)


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
    print("Folder path: " + folder_path)
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
    
def show_coils():
    folder_path = folder_entry.get()
    print("Folder path: " + folder_path)
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

    plotCoils(folder_path, xr_col, yr_col, tss1_col, tss2_col, tss3_col, stbd_offset, port_offset)

def process():
    folder_path = folder_entry.get()
    print("Folder path: " + folder_path)
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

    extractData(folder_path, tss1_col, tss2_col, tss3_col)
    #heading_avg = processFiles(folder_path, xr_col, yr_col, tss1_col, tss2_col, tss3_col, stbd_offset, port_offset, output_file)
    #messagebox.showinfo("Success", f"Heading: {heading_avg:.2f} degrees. Data has been merged and saved in the file:  {output_file}")

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

tk.Label(root, text="Coil 1 (port) Column:", font=font).grid(row=3, column=0, padx=10, pady=5, sticky=tk.W)
tss1_entry = tk.Entry(root, width=20, font=font)
tss1_entry.grid(row=3, column=1, padx=10, pady=5, sticky=tk.W)
tss1_entry.insert(0, "10")  # Default value

tk.Label(root, text="Coil 2 (center) Column:", font=font).grid(row=4, column=0, padx=10, pady=5, sticky=tk.W)
tss2_entry = tk.Entry(root, width=20, font=font)
tss2_entry.grid(row=4, column=1, padx=10, pady=5, sticky=tk.W)
tss2_entry.insert(0, "11")  # Default value

tk.Label(root, text="Coil 3 (starbord) Column:", font=font).grid(row=5, column=0, padx=10, pady=5, sticky=tk.W)
tss3_entry = tk.Entry(root, width=20, font=font)
tss3_entry.grid(row=5, column=1, padx=10, pady=5, sticky=tk.W)
tss3_entry.insert(0, "12")  # Default value

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
tk.Button(root, text="Show Coils", command=lambda: show_coils(), font=font_bold).grid(row=10, column=2, columnspan=3, pady=10, sticky=tk.W)

# Run the main loop
root.mainloop()