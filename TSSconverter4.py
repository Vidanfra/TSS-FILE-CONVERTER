import pandas as pd #pip install pandas
import os
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt #pip install matplotlib
import matplotlib.colors as mcolors
import numpy as np
import csv
from datetime import datetime
import scipy.stats as st #pip install scipy
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Name of the script version
SCRIPT_VERSION = "TSS Converter v4.4"

# Maximum time difference in seconds
MAX_TIME_DIFF_SEC = 0.25

# Column positions in the PTR file
DATE_COLUMN_POS = 0
TIME_COLUMN_POS = 1
EAST_COLUMN_POS = 2
NORTH_COLUMN_POS = 3
COLUMN_COIL_1_DEFAULT = 10
COLUMN_COIL_2_DEFAULT = 11
COLUMN_COIL_3_DEFAULT = 12

# Maximum angle error for heading and course in degrees
MAX_ANGLE_ERROR = 20

# Minimum amount of rows in a file to be considered valid
MIN_FILE_ROWS = 10

def convert_to_datetime(time_str, format_str='%H%M%S%f'):
    try:
        return datetime.strptime(str(time_str), format_str)
    except ValueError as e:
        logging.error(f"Invalid time format: {time_str}. Error: {e}")
        return None

def read_csv_file(file_path, delimiter=','):
    try:
        return pd.read_csv(file_path, delimiter=delimiter)
    except Exception as e:
        logging.error(f"Error reading CSV file {file_path}: {e}")
        return pd.DataFrame()

def get_target_path(folder_path):
    # Normalize path for cross-platform compatibility
    folder_path = os.path.normpath(folder_path)

    # Split path into parts
    path_parts = folder_path.split(os.sep)

    # Find the index of the first part containing "As"
    for i, part in enumerate(path_parts):
        if "As" in part:
            return os.sep.join(path_parts[i-1:])  # Extract from the previous folder onwards

    return folder_path  # Return full path if "As" is not found

def getCoilPeaks(merged_df):
    coil_peaks = []
    
    if 'Filename' not in merged_df or 'TSS' not in merged_df:
        logging.error("Missing required columns in DataFrame")
        return coil_peaks
    
    for line in merged_df['Filename'].unique():
        df = merged_df[merged_df['Filename'] == line]
        
        if df.empty:
            continue
        
        max_index = df['TSS'].idxmax()
        min_index = df['TSS'].idxmin()
        
        abs_max_tss = df.loc[max_index, 'TSS'] if abs(df.loc[max_index, 'TSS']) > abs(df.loc[min_index, 'TSS']) else df.loc[min_index, 'TSS']
        coil = df.loc[max_index, 'Coil']
        easting = df.loc[max_index, 'Easting']
        northing = df.loc[max_index, 'Northing']

        coil_peaks.append({
            'PTR file': line,
            'TSS peak value': abs_max_tss,
            'TSS coil': coil,
            'Easting': easting,
            'Northing': northing
        })
        
    return coil_peaks

def circular_mean(angles):
    # Computes the circular mean of angles in degrees
    angles_rad = np.radians(angles)
    mean_x = np.mean(np.cos(angles_rad))
    mean_y = np.mean(np.sin(angles_rad))
    mean_angle = np.degrees(np.arctan2(mean_y, mean_x)) % 360  # Ensure 0-360 range
    return mean_angle

def calculateHeading(merged_df):
    if merged_df.empty or not {'Filename', 'Coil', 'Easting', 'Northing', 'Gyro'}.issubset(merged_df.columns):
        logging.error("Missing required columns in DataFrame")
        return pd.DataFrame()
    
    attitude_list = []  # Use a list to store dictionaries

    # Loop through unique filenames
    for line in merged_df['Filename'].unique():
        df = merged_df[merged_df['Filename'] == line].copy()  # Copy to avoid SettingWithCopyWarning
        
        if len(df) < 2:
            continue  # If only one row, we cannot calculate diff()

        # Filter the data for Coil 2
        df_coil2 = df[df['Coil'] == 2]
        if df_coil2.empty:
            continue

        # Compute differences for course calculation
        easting_diff = df_coil2['Easting'].diff()
        northing_diff = df_coil2['Northing'].diff()

        # Compute course angle (in degrees) referenced to North:
        # Using arctan2(easting_diff, northing_diff) returns the bearing with North as 0°.
        course = (np.degrees(np.arctan2(easting_diff, northing_diff)) + 360) % 360
        course = course.bfill()  # Handle NaN in first row

        # Convert gyro heading to degrees (assumed already as compass North values)
        heading_deg = df['Gyro'] % 360  # Ensure within 0-360

        # Compute heading error (ensure result is between -180 and 180)
        heading_error = (heading_deg - course + 180) % 360 - 180

        # Use circular mean for headings and courses
        heading_avg = circular_mean(heading_deg)
        course_avg = circular_mean(course)
        heading_error_avg = np.degrees(np.angle(np.exp(1j * np.radians(heading_avg - course_avg))))
        line_direction = round(heading_avg / 5) * 5  # Round to nearest 5 degrees
        
        if line_direction == 360:
            line_direction = 0  # Ensure 360 is converted to 0

        # Compute circular standard deviation (in radians) and then convert to degrees.
        heading_std_rad = st.circstd(np.radians(heading_deg), high=2*np.pi, low=0)
        heading_std = np.degrees(heading_std_rad)

        course_std_rad = st.circstd(np.radians(course), high=2*np.pi, low=0)
        course_std = np.degrees(course_std_rad)

        # Store values in a dictionary
        attitude_list.append({
            'Filename': line,
            'Line Direction': line_direction,
            'Heading Avg': heading_avg,
            'Course Avg': course_avg,
            'Heading Error Avg': heading_error_avg,
            'Heading Std': heading_std,
            'Course Std': course_std,
            'Heading Error Std': heading_error.std(),
            'Heading': heading_deg.tolist(),
            'Course': course.tolist(),
            'Heading Error': heading_error.tolist(),     
        })


    # Convert list of dictionaries to a DataFrame
    attitude_df = pd.DataFrame(attitude_list)

    if not attitude_df.empty:
        # Get the most frequent survey direction
        survey_direction = attitude_df['Line Direction'].mode()
        attitude_df['Survey Direction'] = survey_direction[0] if not survey_direction.empty else np.nan

        # Calculate global averages using circular mean
        attitude_df['Global Heading Avg'] = circular_mean(attitude_df['Heading Avg'])
        attitude_df['Global Heading Error Avg'] = attitude_df['Heading Error Avg'].mean()

    return attitude_df

def extractData(folder_path, tss1_col, tss2_col, tss3_col):
    ptr_dataframe = []
    nav_coil1_dataframe = []
    nav_coil2_dataframe = []
    nav_coil3_dataframe = []

    # Validate column numbers
    try:
        tss1_col = int(tss1_col)
        tss2_col = int(tss2_col)
        tss3_col = int(tss3_col)
    except ValueError:
        logging.error("Columns numbers must be integer numeric values")
        messagebox.showerror("Error", "Columns numbers must be integer numeric values")
        return

    # Check if the folder exists
    if not os.path.exists(folder_path):
        logging.error(f"Folder {folder_path} does not exist")
        messagebox.showerror("Error", f"Folder {folder_path} does not exist")
        return

    # Check if the folder is empty
    if not os.listdir(folder_path):
        logging.error(f"Folder {folder_path} is empty")
        messagebox.showerror("Error", f"Folder {folder_path} is empty")
        return
    
    #logging.info(f"Files in the folder: {os.listdir(folder_path)}") #DEBUG

    # Check all the corresponding files in the folder
    error_messages = []

    for filename in os.listdir(folder_path):

        if filename.endswith('.ptr'):
            only_name = filename.removesuffix('.ptr')
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
            only_name = filename.removesuffix('_Coil_1.csv').removesuffix('_Coil_2.csv').removesuffix('_Coil_3.csv')
            if not (only_name + '.ptr') in os.listdir(folder_path):
                error_messages.append(f"Missing PTR file for the CSV Navigation file: {filename}")
            if not (only_name + '_Coil_1.csv') in os.listdir(folder_path) or not (only_name + '_Coil_2.csv') in os.listdir(folder_path) or not (only_name + '_Coil_3.csv') in os.listdir(folder_path):
                error_messages.append(f"Missing the other necessary CSV Navigation files for this file: {filename}")
    
    # Show all errors in a single message box at the end
    if error_messages:
        logging.error("\n".join(error_messages))
        messagebox.showerror("Error", "\n".join(error_messages))
    
    # Get all files in the folder (store in a set for efficient lookup)
    existing_files = set(os.listdir(folder_path))

    # Loop through all files in the folder and extract the required data
    for filename in  existing_files:
        file_path = os.path.join(folder_path, filename)

        if filename.endswith('.ptr'):           
            only_name = filename.removesuffix('.ptr')
             # Check if all required files exist
            required_files = {f"{only_name}_Coil_1.csv", f"{only_name}_Coil_2.csv", f"{only_name}_Coil_3.csv"}

            if not required_files.issubset(existing_files):
                logging.warning(f"Skipping PTR file {filename} due to missing navigation files")
                continue  # Skip this iteration if any required navigation file is missing
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()

                # Check if the first line contains text (headers)
                if any(c.isalpha() for c in first_line):
                    df = pd.read_csv(file_path, delimiter=';', skiprows=1, header=None)
                else:
                    df = pd.read_csv(file_path, delimiter=';', header=None)

            except Exception as e:
                logging.error(f"Error reading PTR file {file_path}: {e}")
                continue

            if len(df) < MIN_FILE_ROWS:
                messagebox.showwarning("Warning", f"PTR file has less than 10 lines: {filename}")
                logging.warning(f"PTR file has less than 10 lines: {filename}")

            try:
                df_extracted = pd.DataFrame({
                    'Filename': filename,
                    'Date_PTR': df.iloc[:, DATE_COLUMN_POS],
                    'Time_PTR': df.iloc[:, TIME_COLUMN_POS],
                    'Easting_PTR': df.iloc[:, EAST_COLUMN_POS],
                    'Northing_PTR': df.iloc[:, NORTH_COLUMN_POS],
                    'TSS1': df.iloc[:, tss1_col], # Teledyne TSS DeepView coils numbers convention: COIL 1 = STARBOARD
                    'TSS2': df.iloc[:, tss2_col], # Teledyne TSS DeepView coils numbers convention: COIL 2 = CENTER
                    'TSS3': df.iloc[:, tss3_col], # Teledyne TSS DeepView coils numbers convention: COIL 3 = PORT
                })
                ptr_dataframe.append(df_extracted)
            except IndexError:
                logging.error(f"Invalid column index in {filename}")
                continue

        if filename.endswith('_Coil_1.csv'): # NaviEdit User Offsets coils numbers convention: COIL 1 = PORT
            try:
                only_name = filename.removesuffix('_Coil_1.csv')
                if not (only_name + '.ptr') in os.listdir(folder_path) or not (only_name + '_Coil_2.csv') in os.listdir(folder_path) or not (only_name + '_Coil_3.csv') in os.listdir(folder_path):
                    continue # Skip this iteration if any of the PTR or navigation required files is missing
                nav1_df = read_csv_file(file_path, ',')
                if len(nav1_df) < MIN_FILE_ROWS:
                    messagebox.showwarning("Warning", f"Coil 1 navigation file has less than 10 lines: {filename}")
                    logging.warning(f"Coil 1 navigation file has less than 10 lines: {filename}")
                nav_coil1_dataframe.append(nav1_df)
            except Exception as e:
                logging.error(f"Error reading coil 1 navigation file {file_path}: {e}")
                continue

        if filename.endswith('_Coil_2.csv'): # NaviEdit User Offsets coils numbers convention: COIL 2 = CENTER
            try:
                only_name = filename.removesuffix('_Coil_2.csv')
                if not (only_name + '.ptr') in os.listdir(folder_path) or not (only_name + '_Coil_1.csv') in os.listdir(folder_path) or not (only_name + '_Coil_3.csv') in os.listdir(folder_path):
                    continue # Skip this iteration if any of the PTR or navigation required files is missing
                nav2_df = read_csv_file(file_path, ',')
                if len(nav2_df) < MIN_FILE_ROWS:
                    messagebox.showwarning("Warning", f"Coil 2 navigation file has less than 10 lines: {filename}")
                    logging.warning(f"Coil 2 navigation file has less than 10 lines: {filename}")
                nav_coil2_dataframe.append(nav2_df)
            except Exception as e:
                logging.error(f"Error reading coil 2 navigation file {file_path}: {e}")
                continue


        if filename.endswith('_Coil_3.csv'): # NaviEdit User Offsets coils numbers convention: COIL 3 = STARBOARD
            try:
                only_name = filename.removesuffix('_Coil_3.csv')
                if not (only_name + '.ptr') in os.listdir(folder_path) or not (only_name + '_Coil_1.csv') in os.listdir(folder_path) or not (only_name + '_Coil_2.csv') in os.listdir(folder_path):
                    continue # Skip this iteration if any of the PTR or navigation required files is missing
                nav3_df = read_csv_file(file_path, ',')
                if len(nav3_df) < MIN_FILE_ROWS:
                    messagebox.showwarning("Warning", f"Coil 3 navigation file has less than 10 lines: {filename}")
                    logging.warning(f"Coil 3 navigation file has less than 10 lines: {filename}")
                nav_coil3_dataframe.append(nav3_df)
            except Exception as e:
                logging.error(f"Error reading coil 3 navigation file {file_path}: {e}")
                continue

    # Check if any PTR or Navigation files were found
    if not ptr_dataframe:
        logging.error("No PTR files extracted")
        #messagebox.showerror("Error", "No PTR files extracted")
        return
    if not nav_coil1_dataframe or not nav_coil2_dataframe or not nav_coil3_dataframe:
        logging.error("No Navigation files extracted")
        #messagebox.showerror("Error", "No Navigation files extracted")
        return
    
    # Concatenate data
    ptr_df = pd.concat(ptr_dataframe, ignore_index=True)
    nav_coil1_df = pd.concat(nav_coil1_dataframe, ignore_index=True)
    nav_coil2_df = pd.concat(nav_coil2_dataframe, ignore_index=True)
    nav_coil3_df = pd.concat(nav_coil3_dataframe, ignore_index=True)

    # Ensure the Time PTR column is formatted correctly
    ptr_df['Time_PTR'] = ptr_df['Time_PTR'].astype(str).str.zfill(9)  # Ensure the time string is 9 chars
    ptr_df['Time_PTR'] = ptr_df['Time_PTR'].str[:6] + '.' + ptr_df['Time_PTR'].str[6:]  # Insert decimal before ms

    # Convert the Time columns to datetime objects
    ptr_df['Time_PTR'] = pd.to_datetime(ptr_df['Date_PTR'] + ' ' + ptr_df['Time_PTR'], format='%d.%m.%Y %H%M%S.%f')
    nav_coil1_df['Time'] = pd.to_datetime(nav_coil1_df['Date'] + ' ' + nav_coil1_df['Time'], format='%d/%m/%Y %H:%M:%S.%f')
    nav_coil2_df['Time'] = pd.to_datetime(nav_coil2_df['Date'] + ' ' + nav_coil2_df['Time'], format='%d/%m/%Y %H:%M:%S.%f')
    nav_coil3_df['Time'] = pd.to_datetime(nav_coil3_df['Date'] + ' ' + nav_coil3_df['Time'], format='%d/%m/%Y %H:%M:%S.%f')

    # Ensure both DataFrames are sorted by the key columns
    ptr_df = ptr_df.sort_values(by='Time_PTR')
    nav_coil1_df = nav_coil1_df.sort_values(by='Time')
    nav_coil2_df = nav_coil2_df.sort_values(by='Time')
    nav_coil3_df = nav_coil3_df.sort_values(by='Time')
    
    # Swapping of COIL 1 and COIL 3 to match both coils numbers conventions. PTR files numbers convention will be used as reference.
    # - PTR files: TSS DeepView coils numbers convention, Coil 1 = STARBOARD, Coil 2 = CENTRAL, Coil 3 = PORT [DEFAULT]
    # - Navigation files: NaviEdit User Offsets coils numbers convention, Coil 1 = PORT, Coil 2 = CENTER, Coil 3 = STARBOARD
    nav_coil1_df_swapped = nav_coil3_df.copy() # Swapped COIL 1 and COIL 3 to match PTR files numbers convention
    nav_coil2_df_swapped = nav_coil2_df.copy()
    nav_coil3_df_swapped = nav_coil1_df.copy() # Swapped COIL 3 and COIL 1 to match PTR files numbers convention
    
    # Merge the DataFrames based on the closest time in the navigation data to the PTR data
    merged_coil1_df = pd.merge_asof(ptr_df, nav_coil1_df_swapped, left_on='Time_PTR', right_on='Time', direction='nearest') 
    merged_coil2_df = pd.merge_asof(ptr_df, nav_coil2_df_swapped, left_on='Time_PTR', right_on='Time', direction='nearest')
    merged_coil3_df = pd.merge_asof(ptr_df, nav_coil3_df_swapped, left_on='Time_PTR', right_on='Time', direction='nearest') 

    # Check if time difference exceeds the allowed threshold
    time_diff = (merged_coil1_df['Time_PTR'] - merged_coil1_df['Time']).dt.total_seconds()
    merged_coil1_df['Time_diff'] = time_diff
    high_time_diff = abs(time_diff) > MAX_TIME_DIFF_SEC
    if (abs(time_diff) > MAX_TIME_DIFF_SEC).any():
        logging.warning(f"Time difference between PTR and Navigation timestamp is too high in {high_time_diff.sum()} points. Max value :  {abs(time_diff).max():.3f} seconds")
        messagebox.showwarning("Warning", f"Time difference between PTR and Navigation timestamp is too high in {high_time_diff.sum()} points. Max value :  {abs(time_diff).max():.3f} seconds")

    time_diff = (merged_coil2_df['Time_PTR'] - merged_coil2_df['Time']).dt.total_seconds()
    merged_coil2_df['Time_diff'] = time_diff
    #if (abs(time_diff) > MAX_TIME_DIFF_SEC).any():
        #messagebox.showerror("Error", f"Time difference between PTR and Coil 2 is too high: {abs(time_diff).max():.3f} seconds")

    time_diff = (merged_coil3_df['Time_PTR'] - merged_coil3_df['Time']).dt.total_seconds()
    merged_coil3_df['Time_diff'] = time_diff
    #if (abs(time_diff) > MAX_TIME_DIFF_SEC).any():
        #messagebox.showerror("Error", f"Time difference between PTR and Coil 3 is too high: {abs(time_diff).max():.3f} seconds")   

    return merged_coil1_df, merged_coil2_df, merged_coil3_df

def mergeData(merged_coil1_df, merged_coil2_df, merged_coil3_df):
    # Drop the innecessary TSS and Date columns
    merged_coil1_df['TSS']  = merged_coil1_df['TSS1']
    merged_coil1_df['Coil'] = 1
    del merged_coil1_df['TSS1']
    del merged_coil1_df['TSS2']
    del merged_coil1_df['TSS3']
    del merged_coil1_df['Date_PTR']
    del merged_coil1_df['Date']

    merged_coil2_df['TSS']  = merged_coil2_df['TSS2']
    merged_coil2_df['Coil'] = 2
    del merged_coil2_df['TSS1']
    del merged_coil2_df['TSS2']
    del merged_coil2_df['TSS3']
    del merged_coil2_df['Date_PTR']
    del merged_coil2_df['Date']

    merged_coil3_df['TSS']  = merged_coil3_df['TSS3']
    merged_coil3_df['Coil'] = 3
    del merged_coil3_df['TSS1']
    del merged_coil3_df['TSS2']
    del merged_coil3_df['TSS3']
    del merged_coil3_df['Date_PTR']
    del merged_coil3_df['Date']

    # Ensure all dataframes have the same index length (if not, trim to the smallest length)
    min_length = min(len(merged_coil1_df), len(merged_coil2_df), len(merged_coil3_df))

    merged_coil1_df = merged_coil1_df.iloc[:min_length]
    merged_coil2_df = merged_coil2_df.iloc[:min_length]
    merged_coil3_df = merged_coil3_df.iloc[:min_length]

    # Create an interleaved dataframe
    interleaved_df = pd.concat([merged_coil1_df, merged_coil2_df, merged_coil3_df], axis=0).sort_index(kind='merge') # sort_index(kind='merge') interleaves the dataframes (first row of each df, then second row of each df, etc.)

    # Reset index for a clean output
    interleaved_df = interleaved_df.reset_index(drop=True)

    return interleaved_df

def plotMaps(folder_path, tss1_col, tss2_col, tss3_col):
    try:
        # Extract the data from the PTR and Navigation files in the selected folder
        coil1_df, coil2_df, coil3_df = extractData(folder_path, tss1_col, tss2_col, tss3_col)
        merged_df = mergeData(coil1_df, coil2_df, coil3_df)
        
        # Validate merged_df
        required_columns = {'Easting', 'Northing', 'TSS', 'Alt'}
        if not required_columns.issubset(merged_df.columns):
            missing = required_columns - set(merged_df.columns)
            logging.error(f"Missing required columns in merged data: {missing}")
            raise ValueError(f"Missing required columns in merged data: {missing}")

        # Calculate the average heading for each line
        attitude_df = calculateHeading(merged_df)
        
        # Ensure survey direction data exists
        if 'Survey Direction' not in attitude_df.columns or attitude_df.empty:
            logging.error("Survey direction data is missing or could not be computed.")
            raise ValueError("Survey direction data is missing or could not be computed.")
        
        survey_direction = attitude_df['Survey Direction'].iloc[0]
        
        # Get the target path for the output files
        target_path = get_target_path(folder_path)

        # Create a custom TSS colormap
        cmap_tss = mcolors.ListedColormap(['blue', 'green', 'lime', 'yellow', 'red', 'purple'])
        bounds_tss = [-500, -25, 25, 50, 75, 500, 10000]
        norm_tss = mcolors.BoundaryNorm(bounds_tss, cmap_tss.N)

        # Create scatter plots for TSS values
        plt.figure(num= target_path + " Magnetic", figsize=(7, 6))
        scatter = plt.scatter(merged_df['Easting'], merged_df['Northing'], c=merged_df['TSS'], cmap=cmap_tss, norm=norm_tss, marker='o')
        plt.colorbar(scatter, label="TSS [uV]", boundaries=bounds_tss, ticks=[-500, -25, 0, 25, 50, 75, 500, 10000])
        plt.xlabel("Easting [m]")
        plt.ylabel("Northing [m]")
        plt.suptitle(f"Heatmap of TSS Magnetic Values")
        plt.title(f"Survey Direction: {survey_direction:.0f} º")
        plt.gca().set_aspect('equal', adjustable='box')  # Set aspect ratio to 1:1
        plt.grid(True)

        # Create a custom Altitude colormap
        cmap_alt = mcolors.ListedColormap(['lime','pink','magenta', 'purple', 'blue', 'dodgerblue', 'gold', 'orange', 'red'])
        bounds_alt = [-1, -0.5, -0.25, -0.1, 0, 0.25, 0.5, 0.75, 1]
        norm_alt = mcolors.BoundaryNorm(bounds_alt, cmap_alt.N)

        # Create scatter plots for flying altitude
        plt.figure(num=target_path + " Altitude", figsize=(7, 6))
        scatter = plt.scatter(merged_df['Easting'], merged_df['Northing'], c=merged_df['Alt'], cmap=cmap_alt, norm=norm_alt, marker='o')
        plt.colorbar(scatter, label="Altitude [m]", boundaries=bounds_alt, ticks=[-1, -0.5, -0.25, -0.1, 0, 0.25, 0.5, 0.75, 1])
        plt.xlabel("Easting [m]")
        plt.ylabel("Northing [m]")
        plt.suptitle(f'Heatmap of TSS Flying Altitude')
        plt.title(f"Survey Direction: {survey_direction:.0f} º")
        plt.gca().set_aspect('equal', adjustable='box')  # Set aspect ratio to 1:1
        plt.grid(True)

        plt.show()
        
        # Log success message
        logging.info("Magnetic and altitude heatmaps plotted succesfully.")
        
    except Exception as e:
        logging.error(f"Error plotting maps: {e}")
        messagebox.showerror("Error", f"Error plotting maps: {e}")
    
def plotCoils(folder_path, tss1_col, tss2_col, tss3_col):
    try:
        # Extract the data from the PTR and Navigation files in the selected folder
        coil1_df, coil2_df, coil3_df = extractData(folder_path, tss1_col, tss2_col, tss3_col)
        
        # Validate extracted data
        if coil1_df.empty or coil2_df.empty or coil3_df.empty:
            logging.error("Extracted data is empty. Please check the input files.")
            raise ValueError("Extracted data is empty. Please check the input files.")

        required_columns = {'Filename', 'TSS1', 'TSS2', 'TSS3'}
        for df in [coil1_df, coil2_df, coil3_df]:
            if not required_columns.issubset(df.columns):             
                missing = required_columns - set(df.columns)
                logging.error(f"Missing required columns in extracted data: {missing}")
                raise ValueError(f"Missing required columns in extracted data: {missing}")

        # Split the data into individual line files
        line_files = []
        for filename in coil1_df['Filename'].unique():
            line_files.append({
                'filename': filename,
                'TSS1': coil1_df[coil1_df['Filename'] == filename]['TSS1'],
                'TSS2': coil2_df[coil2_df['Filename'] == filename]['TSS2'],
                'TSS3': coil3_df[coil3_df['Filename'] == filename]['TSS3']
            })

        # Loop through all line files in the folder
        for line in line_files: 
            fig_name = line['filename'].removesuffix('.ptr')  
             
            plt.figure(num=fig_name,figsize=(10, 6))
            plt.plot(line['TSS1'], color='r', label='Coil 1 - STBD')
            plt.plot(line['TSS2'], color='b', label='Coil 2 - CENTER')
            plt.plot(line['TSS3'], color='g', label='Coil 3 - PORT')

            # Function to annotate min and max points
            def annotate_peaks(tss_data, color, label):
                max_idx = tss_data.idxmax()  # Index of max value
                min_idx = tss_data.idxmin()  # Index of min value
                max_val = tss_data[max_idx]  # Max value
                min_val = tss_data[min_idx]  # Min value

                # Annotate max
                plt.annotate(f'Max: {max_val:.2f}', (max_idx, max_val),
                            textcoords="offset points", xytext=(0,10), ha='center',
                            color=color, fontsize=10, fontweight='bold')

                # Annotate min
                plt.annotate(f'Min: {min_val:.2f}', (min_idx, min_val),
                            textcoords="offset points", xytext=(0,-15), ha='center',
                            color=color, fontsize=10, fontweight='bold')

            # Annotate peaks for each coil
            annotate_peaks(line['TSS1'], 'r', 'Coil 1')
            annotate_peaks(line['TSS2'], 'b', 'Coil 2')
            annotate_peaks(line['TSS3'], 'g', 'Coil 3')
            plt.xlabel("Time [sec]")
            plt.ylabel("TSS values [uV]")
            plt.title(f'TSS values for each coil - {line['filename']}')
            plt.legend()
            plt.grid(True)    
        plt.show() 
        
        # Log success message
        logging.info("Coil data plotted succesfully.")
        
    except Exception as e:
        logging.error(f"Error plotting coil data: {e}")
        messagebox.showerror("Error", f"Error plotting coil data: {e}")

def plotHeading(folder_path, tss1_col, tss2_col, tss3_col):
    try:
        # Extract the data from the PTR and Navigation files in the selected folder
        coil1_df, coil2_df, coil3_df = extractData(folder_path, tss1_col, tss2_col, tss3_col)
        
        if coil1_df.empty or coil2_df.empty or coil3_df.empty:
            logging.error("One or more extracted DataFrames are empty. Check input data.")
            return

        # Merge the DataFrames into a single DataFrame
        merged_df = mergeData(coil1_df, coil2_df, coil3_df)
        if merged_df.empty:
            logging.error("Merged DataFrame is empty. Exiting function.")
            return
        
        # Calculate the average heading for each line
        attitude_df = calculateHeading(merged_df)
        if attitude_df.empty:
            logging.error("Attitude DataFrame is empty. Exiting function.")
            return

        # Get the target path for the output files
        target_path = get_target_path(folder_path)
        
        # Extract the survey direction
        survey_direction = attitude_df['Survey Direction'].iloc[0]

        # Scatter plot with color scale
        headings_err = np.concatenate(attitude_df['Heading Error'].values)  # Flatten the list of lists into a 1D NumPy array
        headings_err = pd.Series(headings_err)  # Convert back to a pandas Series for NaN handling
        headings_err = headings_err.dropna()  # Drop NaN values
        headings_err = pd.concat([headings_err, headings_err, headings_err], axis=0).sort_index(kind='merge')
        headings_err = headings_err.abs()  # Take the absolute value
        # Normalize the error to be between 0 and 90 degrees (for color scaling, not in the table)
        headings_err = np.where(headings_err > 90, np.abs(180 - headings_err), headings_err)

        # Scatter plot with color scale
        fig, ax = plt.subplots(num= target_path + " Heading Error", figsize=(7, 6))
        scatter = ax.scatter(merged_df['Easting'], merged_df['Northing'], c= headings_err, cmap='coolwarm', vmin=0, vmax= MAX_ANGLE_ERROR, edgecolors='k')

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Abs Heading Error [0º, 90º]", fontsize=12)

        ax.set_xlabel("Easting [m]")
        ax.set_ylabel("Northing [m]")
        ax.set_aspect('equal', adjustable='box')  # Set aspect ratio to 1:1
        fig.suptitle("Scatter Plot of Heading - Course Error")
        ax.set_title(f"Survey Direction: {survey_direction:.0f} º")
        ax.grid(True, linestyle='--', alpha=0.6)

        # 🔹 Add arrows for each line
        for line in merged_df['Filename'].unique():
            df_line = merged_df[merged_df['Filename'] == line].copy()
            
            if len(df_line) < 2:
                logging.warning("Adding arrows Heading QC heatmap: Skipping line %s due to insufficient data points.", line)
                continue  # Skip if not enough points

            # Get first two points
            x1, y1 = df_line.iloc[1][['Easting', 'Northing']]
            x2, y2 = df_line.iloc[len(df_line)-2][['Easting', 'Northing']]
            dx = x2 - x1
            dy = y2 - y1 

            # Draw arrow at the first point
            ax.arrow(x1, y1, dx, dy, head_width=0.5, head_length=0.5, fc='black', ec='black')

        # Select the columns to display
        columns_to_display = [
            "Filename", 
            "Line Direction",
            "Heading Avg", 
            "Course Avg", 
            "Heading Error Avg", 
            "Heading Std", 
            "Course Std"
        ]
        table_data = attitude_df[columns_to_display].copy()

        # Round numerical values for better readability
        for col in columns_to_display[1:]:
            table_data[col] = table_data[col].round(2)

        # Define function for color scaling
        def get_color(value, max_error_angle = MAX_ANGLE_ERROR):
            """Returns a color based on the value (white near 0, red near ±X)."""
            abs_val = min(abs(value), max_error_angle) 
            red_intensity = int((abs_val / max_error_angle) * 255)
            red = 255
            green = 255 - red_intensity
            blue = 255 - red_intensity
            return f"#{red:02X}{green:02X}{blue:02X}"

        # Create figure
        num_rows = len(table_data)
        fig, ax = plt.subplots(num= target_path + " Heading Stats",figsize=(10, num_rows * 0.5 + 2))
        ax.axis('off')  # Hide the axes
        ax.axis('tight')

        # Create table
        table = ax.table(
            cellText=table_data.values,
            colLabels=table_data.columns,
            loc='center',
            cellLoc='center',
            cellColours=[["white"] * len(columns_to_display) for _ in range(num_rows)]  # Default colors
        )

        # Apply color formatting to specific columns
        color_cols = ["Heading Error Avg", "Heading Std", "Course Std"]
        for row_idx, row in table_data.iterrows():
            for col_idx, col in enumerate(columns_to_display):
                if col in color_cols:
                    cell = table[row_idx + 1, col_idx]  # Row +1 because index 0 is for headers
                    cell.set_facecolor(get_color(row[col]))

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)  # Adjust table scaling

        plt.title("Heading and Course Statistics per Line [º]")

        # Show plot
        plt.show()
        
        # Log success message
        logging.info("Heading QC stats and heatmap plotted succesfully.")
        
    except Exception as e:
        logging.error(f"Error plotting heading: {e}")
        messagebox.showerror("Error", f"Error plotting heading: {e}")

def processFiles(folder_path, tss1_col, tss2_col, tss3_col, output_file):
    try:
        if not os.path.isdir(folder_path):
            raise ValueError("Invalid folder path.")
        
        # Extract the data from the PTR and Navigation files in the selected folder
        coil1_df, coil2_df, coil3_df = extractData(folder_path, tss1_col, tss2_col, tss3_col)
        
        if coil1_df.empty or coil2_df.empty or coil3_df.empty:
            raise ValueError("Extracted data is empty. Check input files and column names.")
        
        # Merge the DataFrames into a single DataFrame
        merged_df = mergeData(coil1_df, coil2_df, coil3_df)
        
        # Find the absolute maximum TSS value for each coil and its corresponding position
        coil_peaks = getCoilPeaks(merged_df)
        logging.info("Coil peak values computed successfully.")
        
        # Save the merged DataFrame to a new CSV file
        output_file_path = os.path.join(folder_path, output_file)
        merged_df.to_csv(output_file_path, index=False)
        logging.info(f"Merged data saved to {output_file_path}")
        
        # Save the coil peaks to a new CSV file
        coil_peaks_file_path = os.path.splitext(output_file)[0] + '_coil_peaks.csv'
        coil_peaks_file_path = os.path.join(folder_path, coil_peaks_file_path)
        
        with open(coil_peaks_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["PTR file", "TSS peak value", "TSS coil", "Easting", "Northing"])
            for peak in coil_peaks:
                writer.writerow([peak.get('PTR file', ''), peak.get('TSS peak value', ''), 
                                 peak.get('TSS coil', ''), peak.get('Easting', ''), peak.get('Northing', '')])
        logging.info(f"Coil peaks saved to {coil_peaks_file_path}")
        
        # Log and show success message
        logging.info("Files processed successfully.")
        messagebox.showinfo("Success", "Files processed successfully")
        
    except Exception as e:
        logging.error(f"Error processing files: {e}")
        messagebox.showerror("Error", f"Error processing files: {e}")

def select_folder():
    folder_path = filedialog.askdirectory(title="Select a folder")
    if folder_path:
        folder_entry.delete(0, tk.END)
        folder_entry.insert(0, folder_path)
        logging.info(f"Selected folder path: {folder_path}")
    
def validate_inputs(folder_path, tss1_col, tss2_col, tss3_col, output_file=None):
    if not folder_path:
        raise ValueError("Missing folder path. Select it using the Browse button.")
    if not all([tss1_col, tss2_col, tss3_col]):
        raise ValueError("TSS column values are required.")
    if output_file is not None and not output_file.strip():
        raise ValueError("Output file name is required.")

def show_heading():
    try:
        folder_path = folder_entry.get()
        tss1_col, tss2_col, tss3_col = tss1_entry.get(), tss2_entry.get(), tss3_entry.get()
        validate_inputs(folder_path, tss1_col, tss2_col, tss3_col)
        plotHeading(folder_path, tss1_col, tss2_col, tss3_col)

    except ValueError as e:
        logging.error(f"Error plotting heading: {e}")
        messagebox.showerror("Error", str(e))

def show_maps():
    try:
        folder_path = folder_entry.get()
        tss1_col, tss2_col, tss3_col = tss1_entry.get(), tss2_entry.get(), tss3_entry.get()
        validate_inputs(folder_path, tss1_col, tss2_col, tss3_col)
        plotMaps(folder_path, tss1_col, tss2_col, tss3_col)
        
    except ValueError as e:
        logging.error(f"Error plotting maps: {e}")
        messagebox.showerror("Error", str(e))
    
def show_coils():
    try:
        folder_path = folder_entry.get()
        tss1_col, tss2_col, tss3_col = tss1_entry.get(), tss2_entry.get(), tss3_entry.get()
        validate_inputs(folder_path, tss1_col, tss2_col, tss3_col)
        plotCoils(folder_path, tss1_col, tss2_col, tss3_col)
 
    except ValueError as e:
        logging.error(f"Error plotting coils: {e}")
        messagebox.showerror("Error", str(e))

def process():
    try:
        folder_path = folder_entry.get()
        tss1_col, tss2_col, tss3_col = tss1_entry.get(), tss2_entry.get(), tss3_entry.get()
        output_file = output_entry.get()
        validate_inputs(folder_path, tss1_col, tss2_col, tss3_col, output_file)
        processFiles(folder_path, tss1_col, tss2_col, tss3_col, output_file)

    except ValueError as e:
        logging.error(f"Error processing files: {e}")
        messagebox.showerror("Error", str(e))

# Main program
logging.info(f"{SCRIPT_VERSION} started.")

# Create the main window
root = tk.Tk()
root.title(SCRIPT_VERSION)

# Set the font size
font = ("Helvetica", 14)
font_bold = ("Helvetica", 14, "bold")

# Create and place the widgets
tk.Label(root, text="Select Folder:", font=font).grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)
folder_entry = tk.Entry(root, width=50, font=font)
folder_entry.grid(row=0, column=1, padx=10, pady=5, sticky=tk.W)
folder_entry.insert(0, os.getcwd())  # Default to current working directory
tk.Button(root, text="Browse", command=select_folder, font=font_bold).grid(row=0, column=2, padx=10, pady=5, sticky=tk.W)

tk.Label(root, text="Coil 1 (starboard) Column PTR file:", font=font).grid(row=3, column=0, padx=10, pady=5, sticky=tk.W)
tss1_entry = tk.Entry(root, width=20, font=font)
tss1_entry.grid(row=3, column=1, padx=10, pady=5, sticky=tk.W)
tss1_entry.insert(0, COLUMN_COIL_1_DEFAULT)  # Default value

tk.Label(root, text="Coil 2 (center) Column PTR file:", font=font).grid(row=4, column=0, padx=10, pady=5, sticky=tk.W)
tss2_entry = tk.Entry(root, width=20, font=font)
tss2_entry.grid(row=4, column=1, padx=10, pady=5, sticky=tk.W)
tss2_entry.insert(0, COLUMN_COIL_2_DEFAULT)  # Default value

tk.Label(root, text="Coil 3 (port) Column PTR file:", font=font).grid(row=5, column=0, padx=10, pady=5, sticky=tk.W)
tss3_entry = tk.Entry(root, width=20, font=font)
tss3_entry.grid(row=5, column=1, padx=10, pady=5, sticky=tk.W)
tss3_entry.insert(0, COLUMN_COIL_3_DEFAULT)  # Default value

tk.Label(root, text="Output File Name:", font=font).grid(row=8, column=0, padx=10, pady=5, sticky=tk.W)
output_entry = tk.Entry(root, width=50, font=font)
output_entry.grid(row=8, column=1, padx=10, pady=5, sticky=tk.W)
output_entry.insert(0, "BOSSE_XXX_A.txt")  # Default value

tk.Button(root, text="Heading QC", command=lambda: show_heading(), font=font_bold).grid(row=9, column=0, columnspan=3, pady=10, sticky=tk.W)
tk.Button(root, text="Process Files", command=lambda: process(), font=font_bold).grid(row=10, column=0, columnspan=3, pady=10, sticky=tk.W)
tk.Button(root, text="Show Map", command=lambda: show_maps(), font=font_bold).grid(row=10, column=1, columnspan=3, pady=10, sticky=tk.W)
tk.Button(root, text="Show Coils", command=lambda: show_coils(), font=font_bold).grid(row=10, column=2, columnspan=3, pady=10, sticky=tk.W)

logging.info("Graphic User Interface created. Main loop started.")

# Run the main loop
root.mainloop()