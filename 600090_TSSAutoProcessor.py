import pandas as pd #pip install pandas
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, colorchooser

# IMPORTANT: Set matplotlib backend BEFORE importing pyplot to avoid Tkinter threading issues
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt #pip install matplotlib
import matplotlib.colors as mcolors
import numpy as np
import csv
from datetime import datetime
import logging
import json
import generate_heatmap as heatmaps
import subprocess
import threading
import re
import atexit

# Import altitude extraction module for DVL Altitude Fixer
from altitudeFromSQL import extract_altitude_from_sql, extract_altitude_for_block_ids_direct

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Register cleanup function for exit to prevent Tkinter threading errors
def _cleanup_matplotlib():
    """Close all matplotlib figures on exit to prevent threading issues."""
    try:
        plt.close('all')
    except:
        pass

atexit.register(_cleanup_matplotlib)

# Name of the script version
SCRIPT_VERSION = "600090 TSS AutoProcessor v.4"

# Maximum time difference in seconds
MAX_TIME_DIFF_SEC = 0.25

# Cached datetime format detection for efficient parsing
_datetime_format_cache = {}

def parse_datetime_smart(date_str, time_str, cache_key=None):
    """
    Smart datetime parser with cached format detection.
    Detects the format once per cache_key (e.g., file name) and reuses it for all rows.
    
    Args:
        date_str: Date string (e.g., '20241231' or '31/12/2024')
        time_str: Time string (e.g., '12:34:56.789')
        cache_key: Optional key to cache format detection (e.g., filename or 'nav_file_format')
    
    Returns:
        datetime object or None if parsing fails
    """
    global _datetime_format_cache
    
    combined = f"{date_str} {time_str}"
    
    # If we have a cached format for this key, use it directly
    if cache_key and cache_key in _datetime_format_cache:
        try:
            return datetime.strptime(combined, _datetime_format_cache[cache_key])
        except ValueError:
            # Cache might be stale, fall through to detection
            pass
    
    # Format patterns to try
    formats = [
        '%Y%m%d %H:%M:%S.%f',      # 20241231 12:34:56.789
        '%d/%m/%Y %H:%M:%S.%f',    # 31/12/2024 12:34:56.789
        '%Y-%m-%d %H:%M:%S.%f',    # 2024-12-31 12:34:56.789
        '%Y%m%d %H:%M:%S',          # 20241231 12:34:56 (no ms)
        '%d/%m/%Y %H:%M:%S',        # 31/12/2024 12:34:56 (no ms)
    ]
    
    for fmt in formats:
        try:
            result = datetime.strptime(combined, fmt)
            # Cache the successful format
            if cache_key:
                _datetime_format_cache[cache_key] = fmt
            return result
        except ValueError:
            continue
    
    return None

def parse_datetime_column_smart(df, date_col='Date', time_col='Time', cache_key=None):
    """
    Parse datetime from Date and Time columns using smart format detection.
    Detects format from first valid row and applies to entire DataFrame.
    
    Args:
        df: DataFrame with Date and Time columns
        date_col: Name of date column
        time_col: Name of time column
        cache_key: Optional key to cache format detection
    
    Returns:
        Series of datetime objects
    """
    global _datetime_format_cache
    
    if date_col not in df.columns or time_col not in df.columns:
        return pd.Series([None] * len(df))
    
    # Create combined datetime string column
    combined = df[date_col].astype(str) + ' ' + df[time_col].astype(str)
    
    # If we have a cached format, use vectorized parsing
    if cache_key and cache_key in _datetime_format_cache:
        try:
            return pd.to_datetime(combined, format=_datetime_format_cache[cache_key], errors='coerce')
        except:
            pass
    
    # Detect format from first valid row
    formats = [
        '%Y%m%d %H:%M:%S.%f',
        '%d/%m/%Y %H:%M:%S.%f',
        '%Y-%m-%d %H:%M:%S.%f',
        '%Y%m%d %H:%M:%S',
        '%d/%m/%Y %H:%M:%S',
    ]
    
    # Try first non-null row to detect format
    sample = combined.dropna().iloc[0] if len(combined.dropna()) > 0 else None
    
    if sample:
        for fmt in formats:
            try:
                datetime.strptime(sample, fmt)
                # Found working format - cache it and use vectorized parsing
                if cache_key:
                    _datetime_format_cache[cache_key] = fmt
                return pd.to_datetime(combined, format=fmt, errors='coerce')
            except ValueError:
                continue
    
    # Fallback to pandas auto-detection
    return pd.to_datetime(combined, errors='coerce')

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

# File suffix conventions for navigation CSV files
COIL_PORT_SUFFIX = "_Coil_port.csv"      # Port coil (NaviEdit convention: Coil 1 = PORT)
COIL_CENTER_SUFFIX = "_Coil_center.csv"    # Center coil
COIL_STBD_SUFFIX = "_Coil_stbd.csv"      # Starboard coil (NaviEdit convention: Coil 3 = STARBOARD)
CRP_SUFFIX = "_CRP.csv"               # ROV CRP navigation

# File suffix conventions for DVL Altitude Fixer navigation files (.nav)
DVL_COIL_PORT_SUFFIX = "_Coil_port.nav"
DVL_COIL_CENTER_SUFFIX = "_Coil_center.nav"
DVL_COIL_STBD_SUFFIX = "_Coil_stbd.nav"
DVL_CRP_SUFFIX = "_CRP.nav"

# WFM Export XML template filename for DVL Altitude Fixer
WFM_DEPTH_EXPORT_XML = "WFM_DepthExport.xml"

# Script directory (where the script is located)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Color map and boundaries for TSS heatmaps
COLORS_TSS = ['blue', 'dodgerblue', 'green', 'lime', 'yellow', 'orange', 'red', 'purple', 'pink']
BOUNDARIES_TSS = [-500, -100, -25, 25, 50, 75, 150, 500, 5000, 10000]

# Color map and boundaries for Alt heatmaps
COLORS_ALT = ['black', 'darkblue', 'green', 'limegreen','lime', 'yellow', 'orange', 'red','magenta']
BOUNDARIES_ALT = [-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5]  

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

def normalize_nav_time_column(df):
    """
    Normalize the time column name in navigation dataframes.
    Handles both old NaviEdit header 'Time' and new header 'HH:MM:SS.SSS'.
    Returns the dataframe with the time column renamed to 'Time' if needed.
    """
    if 'Time' in df.columns:
        return df  # Already has the standard 'Time' column
    elif 'HH:MM:SS.SSS' in df.columns:
        df = df.rename(columns={'HH:MM:SS.SSS': 'Time'})
        logging.info("Renamed 'HH:MM:SS.SSS' column to 'Time' (new NaviEdit format detected)")
        return df
    else:
        logging.warning(f"Time column not found. Available columns: {df.columns.tolist()}")
        return df

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
        
        if abs(df.loc[max_index, 'TSS']) > abs(df.loc[min_index, 'TSS']): # Positive peak
            abs_max_tss = df.loc[max_index, 'TSS']
            coil = df.loc[max_index, 'Coil']
            easting = df.loc[max_index, 'Easting']
            northing = df.loc[max_index, 'Northing']
        else: # Negative peak
            abs_max_tss = df.loc[min_index, 'TSS']
            coil = df.loc[min_index, 'Coil']
            easting = df.loc[min_index, 'Easting']
            northing = df.loc[min_index, 'Northing']

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

def circular_std(angles):
    """Compute circular standard deviation (degrees) without SciPy."""
    if angles is None:
        return np.nan

    angles_series = pd.Series(angles).dropna()
    if angles_series.empty:
        return np.nan

    angles_rad = np.radians(angles_series.to_numpy())
    sin_sum = np.sin(angles_rad).sum()
    cos_sum = np.cos(angles_rad).sum()
    r = np.sqrt(sin_sum ** 2 + cos_sum ** 2) / len(angles_rad)

    # Clamp r to avoid log(0) while preserving behaviour near 0
    r = np.clip(r, 1e-12, 1.0)
    std_rad = np.sqrt(-2.0 * np.log(r))
    return np.degrees(std_rad)

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

        # Compute circular standard deviation in degrees using NumPy-based helper
        heading_std = circular_std(heading_deg)
        course_std = circular_std(course)

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

def extractData(folder_path, tss1_col, tss2_col, tss3_col, use_crp=True):
    ptr_dataframe = []
    nav_coil1_dataframe = []
    nav_coil2_dataframe = []
    nav_coil3_dataframe = []
    nav_crp_dataframe = []

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
            
            if not (only_name + COIL_PORT_SUFFIX) in os.listdir(folder_path):
                missing_files_coils.append("Coil Port")
            if not (only_name + COIL_CENTER_SUFFIX) in os.listdir(folder_path):
                missing_files_coils.append("Coil Center")
            if not (only_name + COIL_STBD_SUFFIX) in os.listdir(folder_path):
                missing_files_coils.append("Coil Stbd")
            if use_crp and not (only_name + CRP_SUFFIX) in os.listdir(folder_path):
                missing_files_coils.append("CRP")

            if len(missing_files_coils) == 4:
                error_messages.append(f"Missing all CSV Navigation files for PTR file: {filename}")
            elif missing_files_coils:
                # CRP is optional - only warn, don't add to error messages
                coil_missing = [c for c in missing_files_coils if c != "CRP"]
                crp_missing = "CRP" in missing_files_coils
                if coil_missing:
                    error_messages.append(f"Missing CSV Navigation {', '.join(coil_missing)} files for PTR file: {filename}")
                if crp_missing and use_crp:
                    logging.warning(f"Missing CRP navigation file for PTR file: {filename} - CRP data will not be included")
        
        if filename.endswith(COIL_PORT_SUFFIX) or filename.endswith(COIL_CENTER_SUFFIX) or filename.endswith(COIL_STBD_SUFFIX) or (use_crp and filename.endswith(CRP_SUFFIX)):
            only_name = filename.removesuffix(COIL_PORT_SUFFIX).removesuffix(COIL_CENTER_SUFFIX).removesuffix(COIL_STBD_SUFFIX).removesuffix(CRP_SUFFIX)
            if not (only_name + '.ptr') in os.listdir(folder_path):
                error_messages.append(f"Missing PTR file for the CSV Navigation file: {filename}")
            # Check for required coil files (CRP is optional)
            if not (only_name + COIL_PORT_SUFFIX) in os.listdir(folder_path) or not (only_name + COIL_CENTER_SUFFIX) in os.listdir(folder_path) or not (only_name + COIL_STBD_SUFFIX) in os.listdir(folder_path):
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
             # Check if all required coil files exist (CRP is optional)
            required_files = {f"{only_name}{COIL_PORT_SUFFIX}", f"{only_name}{COIL_CENTER_SUFFIX}", f"{only_name}{COIL_STBD_SUFFIX}"}

            if not required_files.issubset(existing_files):
                logging.warning(f"Skipping PTR file {filename} due to missing navigation files")
                continue  # Skip this iteration if any required navigation file is missing
            
            # Check if CRP file exists (optional)
            crp_file = f"{only_name}{CRP_SUFFIX}"
            if use_crp and crp_file not in existing_files:
                logging.warning(f"CRP navigation file missing for {filename} - CRP data will not be included")
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

        if filename.endswith(COIL_PORT_SUFFIX): # NaviEdit User Offsets coils numbers convention: COIL 1 = PORT
            try:
                only_name = filename.removesuffix(COIL_PORT_SUFFIX)
                if not (only_name + '.ptr') in os.listdir(folder_path) or not (only_name + COIL_CENTER_SUFFIX) in os.listdir(folder_path) or not (only_name + COIL_STBD_SUFFIX) in os.listdir(folder_path):
                    continue # Skip this iteration if any of the PTR or navigation required files is missing
                nav1_df = read_csv_file(file_path, ',')
                if len(nav1_df) < MIN_FILE_ROWS:
                    messagebox.showwarning("Warning", f"Coil Port navigation file has less than 10 lines: {filename}")
                    logging.warning(f"Coil Port navigation file has less than 10 lines: {filename}")
                nav_coil1_dataframe.append(nav1_df)
            except Exception as e:
                logging.error(f"Error reading coil port navigation file {file_path}: {e}")
                continue

        if filename.endswith(COIL_CENTER_SUFFIX): # NaviEdit User Offsets coils numbers convention: COIL 2 = CENTER
            try:
                only_name = filename.removesuffix(COIL_CENTER_SUFFIX)
                if not (only_name + '.ptr') in os.listdir(folder_path) or not (only_name + COIL_PORT_SUFFIX) in os.listdir(folder_path) or not (only_name + COIL_STBD_SUFFIX) in os.listdir(folder_path):
                    continue # Skip this iteration if any of the PTR or navigation required files is missing
                nav2_df = read_csv_file(file_path, ',')
                if len(nav2_df) < MIN_FILE_ROWS:
                    messagebox.showwarning("Warning", f"Coil Center navigation file has less than 10 lines: {filename}")
                    logging.warning(f"Coil Center navigation file has less than 10 lines: {filename}")
                nav_coil2_dataframe.append(nav2_df)
            except Exception as e:
                logging.error(f"Error reading coil center navigation file {file_path}: {e}")
                continue


        if filename.endswith(COIL_STBD_SUFFIX): # NaviEdit User Offsets coils numbers convention: COIL 3 = STARBOARD
            try:
                only_name = filename.removesuffix(COIL_STBD_SUFFIX)
                if not (only_name + '.ptr') in os.listdir(folder_path) or not (only_name + COIL_PORT_SUFFIX) in os.listdir(folder_path) or not (only_name + COIL_CENTER_SUFFIX) in os.listdir(folder_path):
                    continue # Skip this iteration if any of the PTR or navigation required files is missing
                nav3_df = read_csv_file(file_path, ',')
                if len(nav3_df) < MIN_FILE_ROWS:
                    messagebox.showwarning("Warning", f"Coil Stbd navigation file has less than 10 lines: {filename}")
                    logging.warning(f"Coil Stbd navigation file has less than 10 lines: {filename}")
                nav_coil3_dataframe.append(nav3_df)
            except Exception as e:
                logging.error(f"Error reading coil stbd navigation file {file_path}: {e}")
                continue

        if use_crp and filename.endswith(CRP_SUFFIX): # ROV CRP (Center Reference Point) navigation
            try:
                only_name = filename.removesuffix(CRP_SUFFIX)
                if not (only_name + '.ptr') in os.listdir(folder_path) or not (only_name + COIL_PORT_SUFFIX) in os.listdir(folder_path) or not (only_name + COIL_CENTER_SUFFIX) in os.listdir(folder_path) or not (only_name + COIL_STBD_SUFFIX) in os.listdir(folder_path):
                    continue # Skip this iteration if any of the PTR or navigation required files is missing
                crp_df = read_csv_file(file_path, ',')
                if len(crp_df) < MIN_FILE_ROWS:
                    messagebox.showwarning("Warning", f"CRP navigation file has less than 10 lines: {filename}")
                    logging.warning(f"CRP navigation file has less than 10 lines: {filename}")
                nav_crp_dataframe.append(crp_df)
            except Exception as e:
                logging.error(f"Error reading CRP navigation file {file_path}: {e}")
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
    
    # Check if CRP data is available
    has_crp_data = len(nav_crp_dataframe) > 0
    if use_crp and not has_crp_data:
        logging.warning("No CRP navigation files found - CRP data will not be included in output")
        messagebox.showwarning("Warning", "No CRP navigation files found - CRP data will not be included in output")
    
    # Concatenate data
    ptr_df = pd.concat(ptr_dataframe, ignore_index=True)
    nav_coil1_df = pd.concat(nav_coil1_dataframe, ignore_index=True)
    nav_coil2_df = pd.concat(nav_coil2_dataframe, ignore_index=True)
    nav_coil3_df = pd.concat(nav_coil3_dataframe, ignore_index=True)
    nav_crp_df = pd.concat(nav_crp_dataframe, ignore_index=True) if has_crp_data else None

    # Normalize the time column name for navigation dataframes (handles both old 'Time' and new 'HH:MM:SS.SSS' headers)
    nav_coil1_df = normalize_nav_time_column(nav_coil1_df)
    nav_coil2_df = normalize_nav_time_column(nav_coil2_df)
    nav_coil3_df = normalize_nav_time_column(nav_coil3_df)
    if has_crp_data:
        nav_crp_df = normalize_nav_time_column(nav_crp_df)

    # Ensure the Time PTR column is formatted correctly
    ptr_df['Time_PTR'] = ptr_df['Time_PTR'].astype(str).str.zfill(9)  # Ensure the time string is 9 chars
    ptr_df['Time_PTR'] = ptr_df['Time_PTR'].str[:6] + '.' + ptr_df['Time_PTR'].str[6:]  # Insert decimal before ms

    # Convert the Time columns to datetime objects
    ptr_df['Time_PTR'] = pd.to_datetime(ptr_df['Date_PTR'] + ' ' + ptr_df['Time_PTR'], format='%d.%m.%Y %H%M%S.%f')
    nav_coil1_df['Time'] = pd.to_datetime(nav_coil1_df['Date'] + ' ' + nav_coil1_df['Time'], format='%d/%m/%Y %H:%M:%S.%f')
    nav_coil2_df['Time'] = pd.to_datetime(nav_coil2_df['Date'] + ' ' + nav_coil2_df['Time'], format='%d/%m/%Y %H:%M:%S.%f')
    nav_coil3_df['Time'] = pd.to_datetime(nav_coil3_df['Date'] + ' ' + nav_coil3_df['Time'], format='%d/%m/%Y %H:%M:%S.%f')
    if has_crp_data:
        nav_crp_df['Time'] = pd.to_datetime(nav_crp_df['Date'] + ' ' + nav_crp_df['Time'], format='%d/%m/%Y %H:%M:%S.%f')

    # Ensure both DataFrames are sorted by the key columns
    ptr_df = ptr_df.sort_values(by='Time_PTR')
    nav_coil1_df = nav_coil1_df.sort_values(by='Time')
    nav_coil2_df = nav_coil2_df.sort_values(by='Time')
    nav_coil3_df = nav_coil3_df.sort_values(by='Time')
    if has_crp_data:
        nav_crp_df = nav_crp_df.sort_values(by='Time')
    
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
    merged_crp_df = pd.merge_asof(ptr_df, nav_crp_df, left_on='Time_PTR', right_on='Time', direction='nearest') if has_crp_data else None 

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

    if has_crp_data and merged_crp_df is not None:
        time_diff = (merged_crp_df['Time_PTR'] - merged_crp_df['Time']).dt.total_seconds()
        merged_crp_df['Time_diff'] = time_diff
        high_time_diff_crp = abs(time_diff) > MAX_TIME_DIFF_SEC
        if high_time_diff_crp.any():
            logging.warning(f"Time difference between PTR and CRP Navigation timestamp is too high in {high_time_diff_crp.sum()} points. Max value: {abs(time_diff).max():.3f} seconds")

    return merged_coil1_df, merged_coil2_df, merged_coil3_df, merged_crp_df

def mergeData(merged_coil1_df, merged_coil2_df, merged_coil3_df, merged_crp_df):
    # Helper function to clean up a coil dataframe
    def cleanup_coil_df(df, coil_number, tss_column):
        df['TSS'] = df[tss_column]
        df['Coil'] = coil_number
        # Remove individual TSS columns
        for col in ['TSS1', 'TSS2', 'TSS3']:
            if col in df.columns:
                del df[col]
        # Remove Date_PTR and Time_PTR (we keep Date and Time from navigation)
        if 'Date_PTR' in df.columns:
            del df['Date_PTR']
        if 'Time_PTR' in df.columns:
            del df['Time_PTR']
        # Remove PTR position columns (we use ROV/Navigation Easting and Northing instead)
        if 'Easting_PTR' in df.columns:
            del df['Easting_PTR']
        if 'Northing_PTR' in df.columns:
            del df['Northing_PTR']
        # Remove unnecessary columns
        for col in ['NavQ', 'PipeZ', 'PipeX', 'Time_diff']:
            if col in df.columns:
                del df[col]
        return df
    
    # Clean up each coil dataframe
    merged_coil1_df = cleanup_coil_df(merged_coil1_df, 1, 'TSS1')
    merged_coil2_df = cleanup_coil_df(merged_coil2_df, 2, 'TSS2')
    merged_coil3_df = cleanup_coil_df(merged_coil3_df, 3, 'TSS3')

    # Check if CRP data is available
    has_crp_data = merged_crp_df is not None and not merged_crp_df.empty

    # Ensure all dataframes have the same index length (if not, trim to the smallest length)
    if has_crp_data:
        min_length = min(len(merged_coil1_df), len(merged_coil2_df), len(merged_coil3_df), len(merged_crp_df))
        merged_crp_df = merged_crp_df.iloc[:min_length].reset_index(drop=True)
    else:
        min_length = min(len(merged_coil1_df), len(merged_coil2_df), len(merged_coil3_df))

    merged_coil1_df = merged_coil1_df.iloc[:min_length].copy()
    merged_coil2_df = merged_coil2_df.iloc[:min_length].copy()
    merged_coil3_df = merged_coil3_df.iloc[:min_length].copy()

    # Add CRP columns to each coil dataframe before interleaving
    if has_crp_data:
        merged_coil1_df['Easting_CRP'] = merged_crp_df['Easting'].values
        merged_coil1_df['Northing_CRP'] = merged_crp_df['Northing'].values
        merged_coil2_df['Easting_CRP'] = merged_crp_df['Easting'].values
        merged_coil2_df['Northing_CRP'] = merged_crp_df['Northing'].values
        merged_coil3_df['Easting_CRP'] = merged_crp_df['Easting'].values
        merged_coil3_df['Northing_CRP'] = merged_crp_df['Northing'].values

    # Create an interleaved dataframe
    interleaved_df = pd.concat([merged_coil1_df, merged_coil2_df, merged_coil3_df], axis=0).sort_index(kind='merge') # sort_index(kind='merge') interleaves the dataframes (first row of each df, then second row of each df, etc.)

    # Reset index for a clean output
    interleaved_df = interleaved_df.reset_index(drop=True)
    
    # Reorder columns: Filename, Date, Time, Easting, Northing, Coil, TSS, GeographicalEast, GeographicalNorth, then the rest, with Easting_CRP and Northing_CRP at the end
    priority_columns = [
        'Filename', 'Date', 'Time', 'Easting', 'Northing', 'Coil', 'TSS',
        'GeographicalEast', 'GeographicalNorth'
    ]
    end_columns = ['Easting_CRP', 'Northing_CRP']
    
    # Get columns that exist in the dataframe
    existing_priority = [col for col in priority_columns if col in interleaved_df.columns]
    existing_end = [col for col in end_columns if col in interleaved_df.columns]
    # Get remaining columns not in priority or end lists
    remaining_columns = [col for col in interleaved_df.columns if col not in priority_columns and col not in end_columns]
    # Combine in desired order: priority first, then remaining, then CRP at the end
    ordered_columns = existing_priority + remaining_columns + existing_end
    interleaved_df = interleaved_df[ordered_columns]

    return interleaved_df

def reorganizeData(merged_df):
    """
    Reorganizes the merged dataframe to have one row per timestamp with all three coils' data.
    Keeps metadata from coil 2 (center coil).
    """
    if merged_df.empty:
        logging.error("Input DataFrame is empty")
        return pd.DataFrame()
    
    # Check if CRP data is available in the merged dataframe
    has_crp_data = 'Easting_CRP' in merged_df.columns and 'Northing_CRP' in merged_df.columns
    
    # Separate data by coil
    coil1_df = merged_df[merged_df['Coil'] == 1].copy()
    coil2_df = merged_df[merged_df['Coil'] == 2].copy()
    coil3_df = merged_df[merged_df['Coil'] == 3].copy()
    
    # Reset indices for proper alignment
    coil1_df = coil1_df.reset_index(drop=True)
    coil2_df = coil2_df.reset_index(drop=True)
    coil3_df = coil3_df.reset_index(drop=True)
    
    # Ensure all dataframes have the same length
    min_length = min(len(coil1_df), len(coil2_df), len(coil3_df))
    
    coil1_df = coil1_df.iloc[:min_length]
    coil2_df = coil2_df.iloc[:min_length]
    coil3_df = coil3_df.iloc[:min_length]
    
    # Create reorganized dataframe with coil 2 as base (to keep its metadata)
    reorganized_df = pd.DataFrame({
        'Filename': coil2_df['Filename'],
        'Time_PTR': coil2_df['Time_PTR'],
        
        # Coil 1 (Starboard) position and TSS
        'Easting_Coil1': coil1_df['Easting'],
        'Northing_Coil1': coil1_df['Northing'],
        'TSS_Coil1': coil1_df['TSS'],
        
        # Coil 2 (Center) position and TSS
        'Easting_Coil2': coil2_df['Easting'],
        'Northing_Coil2': coil2_df['Northing'],
        'TSS_Coil2': coil2_df['TSS'],
        
        # Coil 3 (Port) position and TSS
        'Easting_Coil3': coil3_df['Easting'],
        'Northing_Coil3': coil3_df['Northing'],
        'TSS_Coil3': coil3_df['TSS'],
        
        # Metadata from Coil 2
        'Kp': coil2_df['Kp'],
        'Dcc': coil2_df['Dcc'],
        'Gyro': coil2_df['Gyro'],
        'Alt': coil2_df['Alt'],
        'Depth': coil2_df['Depth'],
        'GeographicalEast': coil2_df['GeographicalEast'],
        'GeographicalNorth': coil2_df['GeographicalNorth'],
        'Tide': coil2_df['Tide'],
    })
    
    # Add ROV CRP position columns if available (take from coil2 since all coils have the same CRP values)
    if has_crp_data:
        reorganized_df['Easting_CRP'] = coil2_df['Easting_CRP'].values
        reorganized_df['Northing_CRP'] = coil2_df['Northing_CRP'].values
    
    return reorganized_df

def plotMaps(folder_path, tss1_col, tss2_col, tss3_col, use_crp=True):
    try:
        # Extract the data from the PTR and Navigation files in the selected folder
        coil1_df, coil2_df, coil3_df, crp_df = extractData(folder_path, tss1_col, tss2_col, tss3_col, use_crp)
        merged_df = mergeData(coil1_df, coil2_df, coil3_df, crp_df)
        
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
        cmap_tss = mcolors.ListedColormap(COLORS_TSS)
        bounds_tss = BOUNDARIES_TSS
        norm_tss = mcolors.BoundaryNorm(bounds_tss, cmap_tss.N)

        # Create scatter plots for TSS values
        plt.figure(num= target_path + " Magnetic", figsize=(7, 6))
        scatter = plt.scatter(merged_df['Easting'], merged_df['Northing'], c=merged_df['TSS'], cmap=cmap_tss, norm=norm_tss, marker='o')
        plt.colorbar(scatter, label="TSS [uV]", boundaries=bounds_tss, ticks=bounds_tss)
        plt.xlabel("Easting [m]")
        plt.ylabel("Northing [m]")
        plt.suptitle(f"Heatmap of TSS Magnetic Values")
        plt.title(f"Survey Direction: {survey_direction:.0f} º")
        plt.gca().set_aspect('equal', adjustable='box')  # Set aspect ratio to 1:1
        plt.grid(True)

        # Create a custom Altitude colormap
        cmap_alt = mcolors.ListedColormap(COLORS_ALT)
        bounds_alt = BOUNDARIES_ALT
        norm_alt = mcolors.BoundaryNorm(bounds_alt, cmap_alt.N)

        # Create scatter plots for flying altitude
        plt.figure(num=target_path + " Altitude", figsize=(7, 6))
        scatter = plt.scatter(merged_df['Easting'], merged_df['Northing'], c=merged_df['Alt'], cmap=cmap_alt, norm=norm_alt, marker='o')
        plt.colorbar(scatter, label="Altitude [m]", boundaries=bounds_alt, ticks=bounds_alt)
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
    
def plotCoils(folder_path, tss1_col, tss2_col, tss3_col, use_crp=True):
    try:
        # Extract the data from the PTR and Navigation files in the selected folder
        coil1_df, coil2_df, coil3_df, crp_df = extractData(folder_path, tss1_col, tss2_col, tss3_col, use_crp)
        
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

def plotAltitude(folder_path, tss1_col, tss2_col, tss3_col, use_crp=True):
    try:
        # Extract the data from the PTR and Navigation files in the selected folder
        coil1_df, coil2_df, coil3_df, crp_df = extractData(folder_path, tss1_col, tss2_col, tss3_col, use_crp)
        
        # Validate extracted data
        if coil1_df.empty or coil2_df.empty or coil3_df.empty:
            logging.error("Extracted data is empty. Please check the input files.")
            raise ValueError("Extracted data is empty. Please check the input files.")

        required_columns = {'Filename', 'Alt'}
        for df in [coil1_df, coil2_df, coil3_df]:
            if not required_columns.issubset(df.columns):             
                missing = required_columns - set(df.columns)
                logging.error(f"Missing required columns in extracted data: {missing}")
                raise ValueError(f"Missing required columns in extracted data: {missing}")

        # Check if CRP data is available
        has_crp_data = crp_df is not None and not crp_df.empty and 'Alt' in crp_df.columns

        # Split the data into individual line files
        line_files = []
        for filename in coil1_df['Filename'].unique():
            line_data = {
                'filename': filename,
                'Alt1': coil1_df[coil1_df['Filename'] == filename]['Alt'],
                'Alt2': coil2_df[coil2_df['Filename'] == filename]['Alt'],
                'Alt3': coil3_df[coil3_df['Filename'] == filename]['Alt']
            }
            if has_crp_data:
                line_data['Alt_CRP'] = crp_df[crp_df['Filename'] == filename]['Alt'] if 'Filename' in crp_df.columns else None
            line_files.append(line_data)

        # Loop through all line files in the folder
        for line in line_files: 
            fig_name = line['filename'].removesuffix('.ptr') + " - Altitude"
             
            plt.figure(num=fig_name, figsize=(10, 6))
            plt.plot(line['Alt1'].values, color='r', label='Coil 1 - STBD')
            plt.plot(line['Alt2'].values, color='b', label='Coil 2 - CENTER')
            plt.plot(line['Alt3'].values, color='g', label='Coil 3 - PORT')
            
            # Plot CRP altitude if available
            if has_crp_data and line.get('Alt_CRP') is not None and not line['Alt_CRP'].empty:
                plt.plot(line['Alt_CRP'].values, color='purple', linestyle='--', label='CRP')

            # Function to annotate min and max points
            def annotate_peaks(alt_data, color, label):
                if alt_data.empty:
                    return
                max_idx = alt_data.values.argmax()
                min_idx = alt_data.values.argmin()
                max_val = alt_data.values[max_idx]
                min_val = alt_data.values[min_idx]

                # Annotate max
                plt.annotate(f'Max: {max_val:.2f}', (max_idx, max_val),
                            textcoords="offset points", xytext=(0,10), ha='center',
                            color=color, fontsize=10, fontweight='bold')

                # Annotate min
                plt.annotate(f'Min: {min_val:.2f}', (min_idx, min_val),
                            textcoords="offset points", xytext=(0,-15), ha='center',
                            color=color, fontsize=10, fontweight='bold')

            # Annotate peaks for each coil
            annotate_peaks(line['Alt1'], 'r', 'Coil 1')
            annotate_peaks(line['Alt2'], 'b', 'Coil 2')
            annotate_peaks(line['Alt3'], 'g', 'Coil 3')
            if has_crp_data and line.get('Alt_CRP') is not None and not line['Alt_CRP'].empty:
                annotate_peaks(line['Alt_CRP'], 'purple', 'CRP')
            
            plt.xlabel("Time [sec]")
            plt.ylabel("Altitude [m]")
            plt.title(f'Altitude values for each coil - {line["filename"]}')
            plt.legend()
            plt.grid(True)    
        plt.show() 
        
        # Log success message
        logging.info("Altitude data plotted succesfully.")
        
    except Exception as e:
        logging.error(f"Error plotting altitude data: {e}")
        messagebox.showerror("Error", f"Error plotting altitude data: {e}")

def plotHeading(folder_path, tss1_col, tss2_col, tss3_col, use_crp=True):
    try:
        # Extract the data from the PTR and Navigation files in the selected folder
        coil1_df, coil2_df, coil3_df, crp_df = extractData(folder_path, tss1_col, tss2_col, tss3_col, use_crp)
        
        if coil1_df.empty or coil2_df.empty or coil3_df.empty:
            logging.error("One or more extracted DataFrames are empty. Check input data.")
            return

        # Merge the DataFrames into a single DataFrame
        merged_df = mergeData(coil1_df, coil2_df, coil3_df, crp_df)
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
        scatter = ax.scatter(merged_df['Easting'], merged_df['Northing'], c= headings_err, cmap='coolwarm', vmin=0, vmax= MAX_ANGLE_ERROR)

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

def processFiles(folder_path, tss1_col, tss2_col, tss3_col, output_file, silent=False, use_crp=True):
    try:
        if not os.path.isdir(folder_path):
            raise ValueError("Invalid folder path.")
        
        # Extract the data from the PTR and Navigation files in the selected folder
        coil1_df, coil2_df, coil3_df, crp_df = extractData(folder_path, tss1_col, tss2_col, tss3_col, use_crp)
        
        if coil1_df.empty or coil2_df.empty or coil3_df.empty:
            raise ValueError("Extracted data is empty. Check input files and column names.")
        
        # Merge the DataFrames into a single DataFrame
        merged_df = mergeData(coil1_df, coil2_df, coil3_df, crp_df)
        #merged_df = reorganizeData(merged_df) # Optional: reorganize data to have one row per timestamp with all three coils' data
        
        # Find the absolute maximum TSS value for each coil and its corresponding position
        coil_peaks = getCoilPeaks(merged_df)
        logging.info("Coil peak values computed successfully.")
        
        # Format datetime columns to 'HH:MM:SS.SSS' for better readability in output
        output_df = merged_df.copy()
        if 'Time_PTR' in output_df.columns:
            output_df['Time_PTR'] = output_df['Time_PTR'].dt.strftime('%H:%M:%S.%f').str[:-3]  # Remove last 3 digits to get milliseconds
        if 'Time' in output_df.columns:
            output_df['Time'] = output_df['Time'].dt.strftime('%H:%M:%S.%f').str[:-3]  # Remove last 3 digits to get milliseconds
        
        # Create 'Line' column by extracting everything after the first underscore from Filename
        # Example: "251204171430_S1-5_A_L4.ptr" -> "S1-5_A_L4"
        if 'Filename' in output_df.columns:
            output_df['Line'] = output_df['Filename'].apply(
                lambda x: x.split('_', 1)[1].replace('.ptr', '') if '_' in str(x) else str(x).replace('.ptr', '')
            )
            # Reorder columns to place 'Line' right after 'Filename'
            cols = output_df.columns.tolist()
            filename_idx = cols.index('Filename')
            cols.remove('Line')
            cols.insert(filename_idx + 1, 'Line')
            output_df = output_df[cols]
        
        # Remove unwanted columns from the output file
        for col in ['Tide', 'Kp', 'Dcc']:
            if col in output_df.columns:
                output_df.drop(columns=[col], inplace=True)

        # Save the merged DataFrame to a new CSV file
        output_file_path = os.path.join(folder_path, output_file)
        output_df.to_csv(output_file_path, index=False)
        logging.info(f"Merged data saved to {output_file_path}")
        
        merged_df_TSS = merged_df[['Easting', 'Northing', 'TSS']]
        merged_df_ALT = merged_df[['Easting', 'Northing', 'Alt']]
        merged_df_TSS.to_csv(os.path.join(folder_path, os.path.splitext(output_file)[0] + '_TSS.txt'), index=False)
        merged_df_ALT.to_csv(os.path.join(folder_path, os.path.splitext(output_file)[0] + '_ALT.txt'), index=False)
        
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
        if not silent:
            messagebox.showinfo("Success", "Files processed successfully")
        
        return merged_df

    except Exception as e:
        logging.error(f"Error processing files: {e}")
        messagebox.showerror("Error", f"Error processing files: {e}")
        return None

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'settings.json')
CELL_SIZE = 0.5 # Default

def load_settings():
    global COLORS_TSS, BOUNDARIES_TSS, COLORS_ALT, BOUNDARIES_ALT
    defaults = {
        "folder_path": os.getcwd(),
        "ne_path": "",
        "tss1_col": COLUMN_COIL_1_DEFAULT,
        "tss2_col": COLUMN_COIL_2_DEFAULT,
        "tss3_col": COLUMN_COIL_3_DEFAULT,
        "output_file": "TARGET_XXX_AF.txt",
        "coil_port_suffix": COIL_PORT_SUFFIX,
        "coil_center_suffix": COIL_CENTER_SUFFIX,
        "coil_stbd_suffix": COIL_STBD_SUFFIX,
        "crp_suffix": CRP_SUFFIX,
        "cell_size": 0.5,
        "use_crp": True,
        "colors_tss": COLORS_TSS,
        "boundaries_tss": BOUNDARIES_TSS,
        "colors_alt": COLORS_ALT,
        "boundaries_alt": BOUNDARIES_ALT,
        "sql_db_path": "",
        "z_dvl_offset": "0.0",
        "sql_server_name": "RS-GOEL-PVE03",
        "folder_filter": "04_NAVISCAN",
        "wfm_ne_db_name": "",
        "wfm_ne_db_server": "localhost"
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                settings = json.load(f)
                defaults.update(settings)
                # Update global color variables from saved settings
                if "colors_tss" in settings:
                    COLORS_TSS = settings["colors_tss"]
                if "boundaries_tss" in settings:
                    BOUNDARIES_TSS = settings["boundaries_tss"]
                if "colors_alt" in settings:
                    COLORS_ALT = settings["colors_alt"]
                if "boundaries_alt" in settings:
                    BOUNDARIES_ALT = settings["boundaries_alt"]
        except Exception as e:
            logging.error(f"Error loading settings: {e}")
    return defaults

def save_settings():
    # Preserve NE Database settings that are managed by the NE Database Settings dialog
    existing_sql_settings = {}
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                existing = json.load(f)
                for key in ['sql_db_path', 'z_dvl_offset', 'sql_server_name', 'folder_filter', 'wfm_ne_db_name', 'wfm_ne_db_server']:
                    if key in existing:
                        existing_sql_settings[key] = existing[key]
        except:
            pass
    
    settings = {
        "folder_path": folder_entry.get(),
        "ne_path": ne_path_entry.get(),
        "tss1_col": tss1_entry.get(),
        "tss2_col": tss2_entry.get(),
        "tss3_col": tss3_entry.get(),
        "output_file": output_entry.get(),
        "coil_port_suffix": coil_port_suffix_entry.get(),
        "coil_center_suffix": coil_center_suffix_entry.get(),
        "coil_stbd_suffix": coil_stbd_suffix_entry.get(),
        "crp_suffix": crp_suffix_entry.get(),
        "cell_size": cell_size_entry.get(),
        "use_crp": use_crp_var.get(),
        "auto_clicker_enabled": auto_clicker_var.get(),
        "colors_tss": COLORS_TSS,
        "boundaries_tss": BOUNDARIES_TSS,
        "colors_alt": COLORS_ALT,
        "boundaries_alt": BOUNDARIES_ALT
    }
    
    # Merge with existing SQL settings
    settings.update(existing_sql_settings)
    
    try:
        os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
        with open(CONFIG_FILE, 'w') as f:
            json.dump(settings, f, indent=4)
        logging.info("Settings saved.")
    except Exception as e:
        logging.error(f"Error saving settings: {e}")

# ============================================================================
# DVL Altitude Fixer Functions
# ============================================================================

# ============================================================================
# WFM Dialog Auto-Accepter (for automatic channel selection)
# ============================================================================

def start_wfm_dialog_auto_accepter(duration_seconds=120, stop_event=None, output_folder=None, expected_file_count=0):
    """
    Start a background thread that automatically accepts WFM "Export settings" dialog boxes.
    Monitors VS Nav and PTR file exports and stops when complete or timeout reached.
    
    Args:
        duration_seconds: Maximum time to run (default 2 minutes)
        stop_event: Optional threading.Event to signal stop from outside
        output_folder: Folder to monitor for exported files (.nav, .ptr)
        expected_file_count: Number of files expected (used to detect completion)
    
    Returns:
        (stop_event, thread) - call stop_event.set() to stop the thread
    """
    if stop_event is None:
        stop_event = threading.Event()
    
    def auto_accept_loop():
        import time
        import ctypes
        import glob
        from ctypes import c_bool, c_void_p
        
        # Windows API constants
        BM_CLICK = 0x00F5
        
        # Properly define HWND and LPARAM for 64-bit
        if ctypes.sizeof(c_void_p) == 8:
            HWND = ctypes.c_uint64
            LPARAM = ctypes.c_int64
        else:
            HWND = ctypes.c_uint32
            LPARAM = ctypes.c_int32
        
        WNDENUMPROC = ctypes.WINFUNCTYPE(c_bool, HWND, LPARAM)
        
        user32 = ctypes.windll.user32
        user32.EnumWindows.argtypes = [WNDENUMPROC, LPARAM]
        user32.EnumWindows.restype = c_bool
        user32.EnumChildWindows.argtypes = [HWND, WNDENUMPROC, LPARAM]
        user32.EnumChildWindows.restype = c_bool
        user32.GetWindowTextW.argtypes = [HWND, ctypes.c_wchar_p, ctypes.c_int]
        user32.GetWindowTextW.restype = ctypes.c_int
        user32.GetWindowTextLengthW.argtypes = [HWND]
        user32.GetWindowTextLengthW.restype = ctypes.c_int
        user32.GetClassNameW.argtypes = [HWND, ctypes.c_wchar_p, ctypes.c_int]
        user32.GetClassNameW.restype = ctypes.c_int
        user32.IsWindowVisible.argtypes = [HWND]
        user32.IsWindowVisible.restype = c_bool
        user32.SendMessageW.argtypes = [HWND, ctypes.c_uint, ctypes.c_ulonglong, ctypes.c_longlong]
        user32.SendMessageW.restype = ctypes.c_longlong
        
        start_time = time.time()
        clicks_made = 0
        started_accepting = False
        
        # File monitoring state
        file_cutoff_time = start_time - 2  # Only count files created after start
        last_file_count = 0
        stable_file_checks = 0
        required_stable_checks = 3  # Files must be stable for 3 consecutive checks (~1.5 sec)
        
        logging.info("WFM Auto-Accepter: Waiting for 'Export settings' dialogs...")
        if output_folder and expected_file_count > 0:
            logging.info(f"WFM Auto-Accepter: Monitoring {output_folder} for {expected_file_count} files")
        
        def get_window_text(hwnd):
            try:
                length = user32.GetWindowTextLengthW(hwnd) + 1
                buffer = ctypes.create_unicode_buffer(length)
                user32.GetWindowTextW(hwnd, buffer, length)
                return buffer.value
            except:
                return ""
        
        def get_class_name(hwnd):
            try:
                buffer = ctypes.create_unicode_buffer(256)
                user32.GetClassNameW(hwnd, buffer, 256)
                return buffer.value
            except:
                return ""
        
        def find_and_click_ok(hwnd):
            """Find OK button and click it"""
            clicked = [False]
            
            @WNDENUMPROC
            def enum_child_callback(child_hwnd, lparam):
                if clicked[0]:
                    return False
                try:
                    class_name = get_class_name(child_hwnd)
                    text = get_window_text(child_hwnd)
                    
                    if 'Button' in class_name:
                        text_lower = text.lower().strip().replace('&', '')
                        if text_lower in ['ok', 'yes', 'accept']:
                            user32.SendMessageW(child_hwnd, BM_CLICK, 0, 0)
                            clicked[0] = True
                            return False
                except:
                    pass
                return True
            
            try:
                user32.EnumChildWindows(hwnd, enum_child_callback, 0)
            except:
                pass
            return clicked[0]
        
        def count_new_export_files():
            """Count .nav and .ptr files created after start_time"""
            if not output_folder or not os.path.exists(output_folder):
                return 0
            
            count = 0
            try:
                for pattern in ['*.nav', '*.ptr']:
                    for f in glob.glob(os.path.join(output_folder, pattern)):
                        try:
                            if os.path.getmtime(f) > file_cutoff_time:
                                count += 1
                        except OSError:
                            continue
            except Exception:
                pass
            return count
        
        while not stop_event.is_set() and (time.time() - start_time) < duration_seconds:
            try:
                export_dialogs = []
                
                @WNDENUMPROC
                def enum_windows_callback(hwnd, lparam):
                    try:
                        if user32.IsWindowVisible(hwnd):
                            title = get_window_text(hwnd)
                            if 'export settings' in title.lower():
                                export_dialogs.append((hwnd, title))
                    except:
                        pass
                    return True
                
                user32.EnumWindows(enum_windows_callback, 0)
                
                # Auto-accept Export settings dialogs
                for hwnd, title in export_dialogs:
                    if not started_accepting:
                        started_accepting = True
                        logging.info("WFM Auto-Accepter: Started accepting dialogs")
                    
                    if find_and_click_ok(hwnd):
                        clicks_made += 1
                        logging.info(f"WFM Auto-Accepter: Accepted dialog #{clicks_made}")
                        time.sleep(0.4)
                
                # Check if export is complete by monitoring files
                if output_folder and expected_file_count > 0 and started_accepting:
                    current_file_count = count_new_export_files()
                    
                    if current_file_count >= expected_file_count:
                        # Check if file count is stable (no new files being added)
                        if current_file_count == last_file_count:
                            stable_file_checks += 1
                            if stable_file_checks >= required_stable_checks:
                                logging.info(f"WFM Auto-Accepter: Export complete! {current_file_count}/{expected_file_count} files detected")
                                break
                        else:
                            stable_file_checks = 0
                    
                    last_file_count = current_file_count
                
            except:
                pass
            
            time.sleep(0.5)
        
        elapsed = time.time() - start_time
        final_file_count = count_new_export_files() if output_folder else 0
        logging.info(f"WFM Auto-Accepter: Finished after {elapsed:.1f}s. Dialogs accepted: {clicks_made}, Files detected: {final_file_count}")
    
    thread = threading.Thread(target=auto_accept_loop, daemon=True)
    thread.start()
    
    return stop_event, thread


def stop_wfm_auto_accepter(stop_event):
    """Stop the WFM dialog auto-accepter thread."""
    if stop_event:
        stop_event.set()
        logging.info("WFM Auto-Accepter: Stop signal sent")


def parse_block_ids(block_ids_str):
    """
    Parse block IDs from a string in the format '100-105, 107'.
    Returns a sorted list of unique block IDs.
    """
    block_ids = []
    try:
        parts = [p.strip() for p in block_ids_str.split(',')]
        for part in parts:
            if '-' in part:
                start, end = map(int, part.split('-'))
                block_ids.extend(range(start, end + 1))
            else:
                block_ids.append(int(part))
        block_ids = sorted(list(set(block_ids)))  # Remove duplicates and sort
    except ValueError:
        logging.error("Invalid Block ID format")
        return []
    return block_ids


def get_expected_wfm_output_files(navdepth_folder, block_ids):
    """
    Get list of expected output files from WFM depth export.
    WFM exports 4 files per block: _CRP.nav, _Coil_port.nav, _Coil_center.nav, _Coil_stbd.nav
    
    Args:
        navdepth_folder: Path to navdepth output folder
        block_ids: List of block IDs being exported
    
    Returns:
        List of expected file paths (CRP files only - these are sufficient to detect completion)
    """
    expected_files = []
    for bid in block_ids:
        # We only check for CRP files as the primary indicator
        # WFM creates files with the block name from the database
        # Pattern is like: BlockName_CRP.nav
        crp_pattern = os.path.join(navdepth_folder, f"*{DVL_CRP_SUFFIX}")
        expected_files.append((bid, crp_pattern))
    return expected_files


def wait_for_wfm_completion(navdepth_folder, block_ids, timeout_seconds=300, poll_interval=2, 
                            progress_callback=None, cancel_flag=None, start_time=None):
    """
    Monitor navdepth folder for WFM output file completion.
    
    Args:
        navdepth_folder: Path to the navdepth output folder
        block_ids: List of block IDs being exported
        timeout_seconds: Maximum time to wait (default 5 minutes)
        poll_interval: How often to check (default 2 seconds)
        progress_callback: Optional callback function(status_message, found_count, expected_count)
        cancel_flag: Optional list [bool] that can be set to [True] to cancel waiting
        start_time: Timestamp when the export was initiated. Only files modified after this time
                    will be considered. If None, uses current time (for backwards compatibility).
    
    Returns:
        (success: bool, found_files: list, message: str)
    """
    import time
    import glob
    
    process_start_time = time.time()
    
    # Use provided start_time or default to now (minus a small buffer for clock skew)
    if start_time is None:
        file_cutoff_time = process_start_time - 2  # 2 second buffer
    else:
        file_cutoff_time = start_time - 2  # 2 second buffer for clock skew
    
    expected_count = len(block_ids)
    
    # Track files and their sizes for stability checking
    file_sizes = {}
    stable_count = 0
    required_stable_checks = 2  # File must have same size for 2 consecutive checks
    
    logging.info(f"Waiting for WFM to export {expected_count} blocks to {navdepth_folder}...")
    logging.info(f"Only considering files modified after: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(file_cutoff_time))}")
    
    while True:
        elapsed = time.time() - process_start_time
        
        # Check for cancellation
        if cancel_flag and cancel_flag[0]:
            return False, [], "Cancelled by user"
        
        # Check for timeout
        if elapsed > timeout_seconds:
            all_files = glob.glob(os.path.join(navdepth_folder, f"*{DVL_CRP_SUFFIX}"))
            new_files = [f for f in all_files if os.path.getmtime(f) > file_cutoff_time]
            return False, new_files, f"Timeout after {timeout_seconds}s. Found {len(new_files)} new files of {expected_count} expected."
        
        # Find all CRP files in navdepth folder
        all_crp_files = glob.glob(os.path.join(navdepth_folder, f"*{DVL_CRP_SUFFIX}"))
        
        # Filter to only files modified AFTER the start time
        current_files = []
        for f in all_crp_files:
            try:
                file_mtime = os.path.getmtime(f)
                if file_mtime > file_cutoff_time:
                    current_files.append(f)
            except OSError:
                continue
        
        found_count = len(current_files)
        
        # Check file stability (sizes not changing)
        all_stable = True
        for f in current_files:
            try:
                current_size = os.path.getsize(f)
                prev_size = file_sizes.get(f, -1)
                
                if current_size != prev_size:
                    file_sizes[f] = current_size
                    all_stable = False
            except OSError:
                all_stable = False
        
        # Update progress
        if progress_callback:
            status = f"Found {found_count}/{expected_count} NEW files ({int(elapsed)}s elapsed)"
            if all_stable and found_count >= expected_count:
                status += " - Verifying stability..."
            progress_callback(status, found_count, expected_count)
        
        # Check if we have all expected files and they are stable
        if found_count >= expected_count:
            if all_stable:
                stable_count += 1
                if stable_count >= required_stable_checks:
                    logging.info(f"WFM export complete: {found_count} new files found and stable")
                    return True, current_files, f"Found {found_count} files"
            else:
                stable_count = 0  # Reset if files changed
        
        time.sleep(poll_interval)
    
    return False, [], "Unknown error"


def run_wfm_depth_export(block_ids_str, output_folder):
    """
    Run the WFM Depth Export script with the specified block IDs.
    Exports files to the 'navdepth' folder in the specified output folder.
    """
    # Load settings for WFM database configuration
    settings = load_settings()
    wfm_db_name = settings.get('wfm_ne_db_name', '')
    wfm_db_server = settings.get('wfm_ne_db_server', 'localhost')
    
    # Validate WFM database settings
    if not wfm_db_name:
        messagebox.showerror("Error", "Please configure the WFM NE Database Name in NE Database Settings.")
        return False
    
    # Parse block IDs
    block_ids = parse_block_ids(block_ids_str)
    if not block_ids:
        messagebox.showerror("Error", "Invalid Block ID format. Use comma separated numbers or ranges (e.g. 100-105, 107).")
        return False
    
    # Create navdepth output folder
    if output_folder and os.path.exists(output_folder):
        navdepth_folder = os.path.join(output_folder, "navdepth")
    else:
        navdepth_folder = os.path.join(SCRIPT_DIR, "navdepth")
        
    if not os.path.exists(navdepth_folder):
        try:
            os.makedirs(navdepth_folder)
            logging.info(f"Created navdepth folder: {navdepth_folder}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not create navdepth folder: {e}")
            return False
    
    # Path to the XML template
    xml_path = os.path.join(SCRIPT_DIR, WFM_DEPTH_EXPORT_XML)
    
    if not os.path.exists(xml_path):
        messagebox.showerror("Error", f"WFM XML template not found at {xml_path}")
        return False
    
    try:
        with open(xml_path, 'r') as f:
            xml_content = f.read()
        
        # Replace InputTask with SetPropertyTask for OutputDirectory
        target_output_str = 'name="Select OUTPUT folder (for Depth Nav exports)" output="OutputDirectory" AskForInput="true" Message="Select OUTPUT folder (for Depth Nav exports)"/>'
        replacement_output_str = f'name="Set Output Directory" input="{navdepth_folder}" output="OutputDirectory" level="2"/>'
        
        if target_output_str in xml_content:
            xml_content = xml_content.replace(
                f'<InputTask po="5" {target_output_str}',
                f'<SetPropertyTask po="5" {replacement_output_str}'
            )
        
        # Comment out redundant SetPropertyTask po="6"
        target_po6 = '<SetPropertyTask po="6" name="Define output directory" input="{OutputDirectory}" output="OutputDirectory" level="2"/>'
        if target_po6 in xml_content:
            xml_content = xml_content.replace(target_po6, f'<!-- {target_po6} -->')
        
        # --- INJECT WFM DATABASE SETTINGS ---
        # Replace SqlServer
        xml_content = xml_content.replace('<SqlServer>localhost</SqlServer>', f'<SqlServer>{wfm_db_server}</SqlServer>')
        # Replace DatabaseName (the placeholder in the template)
        # First, try to find the existing database name pattern and replace it
        import re
        xml_content = re.sub(r'<DatabaseName>[^<]*</DatabaseName>', f'<DatabaseName>{wfm_db_name}</DatabaseName>', xml_content)
        
        logging.info(f"WFM Database: {wfm_db_name} on server {wfm_db_server}")
        
        # --- DYNAMICALLY GENERATE EXPORT TASKS FOR EACH BLOCK ID ---
        start_marker = '<FindBlocksTask po="4" name="Iterate Blocks in Job" output="Block">'
        end_marker = '</FindBlocksTask>'
        
        start_idx = xml_content.find(start_marker)
        end_idx = xml_content.find(end_marker)
        
        if start_idx != -1 and end_idx != -1:
            template_start = xml_content.find('<GroupTask po="1" name="Export {Block.name}">', start_idx)
            last_group_task_end = xml_content.rfind('</GroupTask>', start_idx, end_idx) + len('</GroupTask>')
            
            block_template = xml_content[template_start:last_group_task_end]
            
            generated_tasks = []
            for i, bid in enumerate(block_ids):
                task_xml = block_template
                task_xml = task_xml.replace('{Block.id}', str(bid))
                task_xml = task_xml.replace('{Block.name}', f'Block_{bid}')
                task_xml = task_xml.replace('po="1" name="Export', f'po="{i+4}" name="Export')
                generated_tasks.append(task_xml)
            
            new_content_block = '\n'.join(generated_tasks)
            full_match_string = xml_content[start_idx:end_idx + len(end_marker)]
            xml_content = xml_content.replace(full_match_string, new_content_block)
        
        # Replace relative log paths with absolute paths
        logs_dir = os.path.join(SCRIPT_DIR, "LOGS_WFM_DVL_Fixer")
        if not os.path.exists(logs_dir):
            try:
                os.makedirs(logs_dir)
            except Exception as e:
                logging.warning(f"Could not create logs directory: {e}")
        
        xml_content = xml_content.replace(r'.\LOGS_WFM_DVL_Fixer', logs_dir)
        
        # Save to temp file
        temp_xml_path = os.path.join(SCRIPT_DIR, "temp_wfm_depth_export.xml")
        with open(temp_xml_path, "w") as f:
            f.write(xml_content)
        
        # Run WFM
        wfm_exe = r"C:\Eiva\WorkFlowManager\WorkflowManager.exe"
        if not os.path.exists(wfm_exe):
            messagebox.showerror("Error", f"Workflow Manager executable not found at {wfm_exe}")
            return False
        
        logging.info(f"Starting WFM Depth Export for Block IDs: {block_ids}")
        logging.info(f"Output folder: {navdepth_folder}")
        
        subprocess.Popen([wfm_exe, "-run", temp_xml_path])
        
        logging.info(f"WFM Depth Export started for Block IDs: {block_ids}")
        return True
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to run Depth Export script: {e}")
        logging.error(f"WFM Depth Export error: {e}")
        return False


def read_sql_altitude_csv(file_path):
    """
    Read the SQL Altitude CSV file and extract the 'Altitude' column.
    Returns a DataFrame with CRP Altitude data with parsed datetime.
    """
    if not os.path.exists(file_path):
        logging.error(f"SQL Altitude CSV file not found: {file_path}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(file_path)
        if 'Altitude' not in df.columns:
            logging.error(f"'Altitude' column not found in {file_path}")
            logging.info(f"Available columns: {df.columns.tolist()}")
            return pd.DataFrame()
        
        relevant_cols = ['ID', 'Name', 'Folder', 'Time', 'RelTime', 'Depth', 'Altitude']
        existing_cols = [col for col in relevant_cols if col in df.columns]
        df = df[existing_cols].copy()
        df.rename(columns={'Altitude': 'CRP_Altitude', 'Depth': 'SQL_Depth'}, inplace=True)
        
        if 'Name' in df.columns:
            df['Name'] = df['Name'].astype(str)
        
        if 'Time' in df.columns:
            df['DateTime'] = pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S.%f')
            df = df.sort_values(by='DateTime')
        
        logging.info(f"Successfully read SQL Altitude CSV: {len(df)} rows")
        return df
    except Exception as e:
        logging.error(f"Error reading SQL Altitude CSV: {e}")
        return pd.DataFrame()


def read_nav_file_with_time_dvl(file_path):
    """
    Read a VisualSoft .nav file and extract the 'DEPTH' column with time parsing.
    Returns a DataFrame with depth data and parsed datetime.
    """
    if not os.path.exists(file_path):
        logging.warning(f"Nav file not found: {file_path}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(file_path, delimiter=',')
        df.columns = df.columns.str.strip().str.upper()
        
        if 'DEPTH' not in df.columns:
            logging.warning(f"'DEPTH' column not found in {file_path}")
            logging.info(f"Available columns: {df.columns.tolist()}")
            return pd.DataFrame()
        
        date_col = 'DATE' if 'DATE' in df.columns else None
        time_col = None
        if 'TIME' in df.columns:
            time_col = 'TIME'
        elif 'HH:MM:SS.SSS' in df.columns:
            time_col = 'HH:MM:SS.SSS'
        
        if date_col and time_col:
            try:
                df['DateTime'] = pd.to_datetime(
                    df[date_col].astype(str) + ' ' + df[time_col].astype(str), 
                    format='%Y%m%d %H:%M:%S.%f'
                )
            except:
                try:
                    df['DateTime'] = pd.to_datetime(
                        df[date_col].astype(str) + ' ' + df[time_col].astype(str), 
                        format='%d/%m/%Y %H:%M:%S.%f'
                    )
                except Exception as e:
                    logging.warning(f"Could not parse datetime in {file_path}: {e}")
                    return pd.DataFrame()
            
            df = df.sort_values(by='DateTime')
            df['Date'] = df['DateTime'].dt.strftime('%d/%m/%Y')
        
        df['Source_File'] = os.path.basename(file_path)
        
        logging.info(f"Successfully read nav file: {os.path.basename(file_path)} - {len(df)} rows")
        return df
    except Exception as e:
        logging.error(f"Error reading nav file {file_path}: {e}")
        return pd.DataFrame()


def find_nav_files_dvl(navdepth_folder, suffix):
    """
    Find all nav files in the navdepth folder with the given suffix.
    Returns a list of file paths.
    """
    if not os.path.exists(navdepth_folder):
        logging.error(f"Navdepth folder not found: {navdepth_folder}")
        return []
    
    if not os.path.isdir(navdepth_folder):
        logging.error(f"Path is not a directory: {navdepth_folder}")
        return []
    
    nav_files = []
    for filename in os.listdir(navdepth_folder):
        if filename.endswith(suffix):
            nav_files.append(os.path.join(navdepth_folder, filename))
    
    logging.info(f"Found {len(nav_files)} files with suffix '{suffix}' in {navdepth_folder}")
    return nav_files


def get_block_name_from_file(filename, suffix):
    """Extract block name from a nav file name by removing the suffix."""
    return filename.replace(suffix, '')


def read_nav_files_by_block(navdepth_folder):
    """
    Read all nav files and organize them by block name.
    Returns a dictionary: {block_name: {'CRP': df, 'Coil_Port': df, 'Coil_Center': df, 'Coil_Stbd': df}}
    """
    blocks = {}
    
    crp_files = find_nav_files_dvl(navdepth_folder, DVL_CRP_SUFFIX)
    
    for crp_file in crp_files:
        block_name = get_block_name_from_file(os.path.basename(crp_file), DVL_CRP_SUFFIX)
        
        blocks[block_name] = {
            'CRP': pd.DataFrame(),
            'Coil_Port': pd.DataFrame(),
            'Coil_Center': pd.DataFrame(),
            'Coil_Stbd': pd.DataFrame()
        }
        
        crp_df = read_nav_file_with_time_dvl(crp_file)
        if not crp_df.empty:
            blocks[block_name]['CRP'] = crp_df
        
        port_file = os.path.join(navdepth_folder, block_name + DVL_COIL_PORT_SUFFIX)
        if os.path.exists(port_file):
            port_df = read_nav_file_with_time_dvl(port_file)
            if not port_df.empty:
                blocks[block_name]['Coil_Port'] = port_df
        
        center_file = os.path.join(navdepth_folder, block_name + DVL_COIL_CENTER_SUFFIX)
        if os.path.exists(center_file):
            center_df = read_nav_file_with_time_dvl(center_file)
            if not center_df.empty:
                blocks[block_name]['Coil_Center'] = center_df
        
        stbd_file = os.path.join(navdepth_folder, block_name + DVL_COIL_STBD_SUFFIX)
        if os.path.exists(stbd_file):
            stbd_df = read_nav_file_with_time_dvl(stbd_file)
            if not stbd_df.empty:
                blocks[block_name]['Coil_Stbd'] = stbd_df
    
    logging.info(f"Found {len(blocks)} blocks in navdepth folder")
    return blocks


def match_altitude_with_nav(sql_altitude_df, nav_blocks, z_dvl_offset, expected_block_ids=None):
    """
    Match SQL Altitude data with nav depth data by time using CRP as reference.
    Calculates Coil Altitude = (CRP Altitude + DVL offset) + CRP Depth - Coil Depth
    Returns a dictionary: {block_name: merged_dataframe}
    
    Args:
        sql_altitude_df: DataFrame with SQL altitude data
        nav_blocks: Dictionary of nav data organized by block name
        z_dvl_offset: Z offset value for DVL altitude calculation
        expected_block_ids: Optional list of expected block IDs (for validation/error messages)
    """
    results = {}
    unmatched_nav_blocks = []  # Nav files without SQL data
    
    if sql_altitude_df.empty or 'DateTime' not in sql_altitude_df.columns:
        logging.error("SQL Altitude DataFrame is empty or missing DateTime column")
        return results
    
    if 'Name' not in sql_altitude_df.columns and 'ID' not in sql_altitude_df.columns:
        logging.error("SQL Altitude DataFrame is missing 'Name' or 'ID' column for block matching")
        return results
    
    # Get available block names/IDs from SQL data for comparison
    sql_block_names = set()
    sql_block_ids = set()
    if 'Name' in sql_altitude_df.columns:
        sql_block_names = set(sql_altitude_df['Name'].astype(str).unique())
    if 'ID' in sql_altitude_df.columns:
        sql_block_ids = set(sql_altitude_df['ID'].unique())
    
    logging.info(f"SQL data contains {len(sql_block_names)} unique block names: {sorted(sql_block_names)[:10]}{'...' if len(sql_block_names) > 10 else ''}")
    logging.info(f"Nav files contain {len(nav_blocks)} blocks: {sorted(nav_blocks.keys())[:10]}{'...' if len(nav_blocks) > 10 else ''}")
    
    for block_name, nav_data in nav_blocks.items():
        logging.info(f"Processing block: {block_name}")
        
        crp_df = nav_data['CRP']
        if crp_df.empty:
            logging.info(f"  - Skipping: CRP data empty")
            continue
        
        if 'DateTime' not in crp_df.columns:
            logging.info(f"  - Skipping: CRP data missing DateTime column")
            continue
        
        # Try to match by Name first (since nav files usually use the timestamp name)
        sql_block_df = pd.DataFrame()
        if 'Name' in sql_altitude_df.columns:
             sql_block_df = sql_altitude_df[sql_altitude_df['Name'].astype(str) == str(block_name)].copy()
        
        # If no match by Name, and block_name is numeric, try matching by ID
        if sql_block_df.empty and 'ID' in sql_altitude_df.columns and block_name.isdigit():
            sql_block_df = sql_altitude_df[sql_altitude_df['ID'] == int(block_name)].copy()
        
        if sql_block_df.empty:
            logging.warning(f"  - Warning: No matching SQL Altitude data found for block '{block_name}'")
            unmatched_nav_blocks.append(block_name)
            continue
        
        logging.info(f"  - Found {len(sql_block_df)} SQL Altitude records for this block")
        
        crp_df = crp_df.sort_values(by='DateTime').reset_index(drop=True)
        sql_block_df_sorted = sql_block_df.sort_values(by='DateTime').reset_index(drop=True)
        
        merged_crp = pd.merge_asof(
            crp_df,
            sql_block_df_sorted[['DateTime', 'CRP_Altitude']],
            on='DateTime',
            direction='nearest'
        )
        
        matched_count = merged_crp['CRP_Altitude'].notna().sum()
        logging.info(f"  - Matched {matched_count} of {len(merged_crp)} records with SQL Altitude")
        
        if merged_crp['CRP_Altitude'].isna().all():
            logging.warning(f"  - Warning: No time-matched SQL Altitude data found for this block")
            continue
        
        merged_crp = merged_crp.rename(columns={'DEPTH': 'CRP_Depth'})
        
        if 'Date' not in merged_crp.columns:
            merged_crp['Date'] = merged_crp['DateTime'].dt.strftime('%d/%m/%Y')
        
        coil_dfs = []
        coil_types = [
            ('Coil_Port', 'Port', 1),
            ('Coil_Center', 'Center', 2),
            ('Coil_Stbd', 'Stbd', 3)
        ]
        
        for coil_key, coil_name, coil_num in coil_types:
            coil_df = nav_data[coil_key]
            if coil_df.empty or 'DateTime' not in coil_df.columns:
                logging.info(f"  - Skipping {coil_name}: data empty or missing DateTime")
                continue
            
            coil_df = coil_df.sort_values(by='DateTime').reset_index(drop=True)
            
            merged_coil = pd.merge_asof(
                merged_crp[['DateTime', 'Date', 'CRP_Depth', 'CRP_Altitude']].copy(),
                coil_df[['DateTime', 'DEPTH']].rename(columns={'DEPTH': 'Coil_Depth'}),
                on='DateTime',
                direction='nearest',
                tolerance=pd.Timedelta(seconds=MAX_TIME_DIFF_SEC)
            )
            
            merged_coil['Coil'] = coil_num
            merged_coil['Coil_Name'] = coil_name
            merged_coil['Coil_Altitude'] = ((merged_coil['CRP_Altitude'] + z_dvl_offset) + merged_coil['CRP_Depth'] - merged_coil['Coil_Depth']).round(2)
            merged_coil['CRP_Altitude'] = (merged_coil['CRP_Altitude'] + z_dvl_offset).round(2)
            merged_coil['Time'] = merged_coil['DateTime'].dt.strftime('%H:%M:%S.%f').str[:-3]
            
            coil_dfs.append(merged_coil)
        
        if coil_dfs:
            min_len = min(len(df) for df in coil_dfs)
            coil_dfs = [df.iloc[:min_len].reset_index(drop=True) for df in coil_dfs]
            
            interleaved_df = pd.concat(coil_dfs, axis=0).sort_index(kind='merge')
            interleaved_df = interleaved_df.reset_index(drop=True)
            
            column_order = ['Date', 'Time', 'Coil', 'Coil_Name', 'Coil_Depth', 'CRP_Altitude', 'CRP_Depth', 'Coil_Altitude']
            existing_cols = [col for col in column_order if col in interleaved_df.columns]
            remaining_cols = [col for col in interleaved_df.columns if col not in column_order and col != 'DateTime']
            interleaved_df = interleaved_df[existing_cols + remaining_cols]
            
            results[block_name] = interleaved_df
            logging.info(f"  - Successfully processed: {len(interleaved_df)} rows")
        else:
            logging.info(f"  - No coil data matched for this block")
    
    # Report summary of matching results
    if unmatched_nav_blocks:
        logging.warning(f"")
        logging.warning(f"=== BLOCK MATCHING SUMMARY ===")
        logging.warning(f"Nav files found for blocks: {sorted(nav_blocks.keys())}")
        logging.warning(f"Could NOT find SQL data for: {sorted(unmatched_nav_blocks)}")
        if sql_block_names:
            logging.warning(f"SQL database contains these block names: {sorted(sql_block_names)[:20]}{'...' if len(sql_block_names) > 20 else ''}")
        logging.warning(f"")
        logging.warning(f"POSSIBLE CAUSES:")
        logging.warning(f"  1. Wrong Block IDs entered (check the Block IDs field)")
        logging.warning(f"  2. Nav files in folder don't match the entered Block IDs")
        logging.warning(f"  3. Block names in nav files don't match names in database")
        logging.warning(f"")
    
    return results


def save_calculated_altitude_csvs(results, output_folder):
    """Save the calculated altitude dataframes to CSV files, one per block."""
    if not os.path.exists(output_folder):
        try:
            os.makedirs(output_folder)
            logging.info(f"Created output folder: {output_folder}")
        except Exception as e:
            logging.error(f"Could not create output folder: {e}")
            return False
    
    for block_name, df in results.items():
        output_file = os.path.join(output_folder, f"{block_name}_calculated_altitude.csv")
        try:
            df.to_csv(output_file, index=False)
            logging.info(f"Saved CSV: {output_file}")
        except Exception as e:
            logging.error(f"Error saving CSV {output_file}: {e}")
    
    return True


def get_file_type_and_block(filename):
    """
    Determine block name and coil type from filename.
    Returns (block_name, coil_type) or (None, None).
    """
    name_without_ext = os.path.splitext(filename)[0]
    name_lower = name_without_ext.lower()
    
    block_name = None
    coil_type = None
    
    if name_lower.endswith('_coil_center'):
        block_name = name_without_ext[:-12]
        coil_type = 'Center'
    elif name_lower.endswith('_coil_port'):
        block_name = name_without_ext[:-10]
        coil_type = 'Port'
    elif name_lower.endswith('_coil_stbd'):
        block_name = name_without_ext[:-10]
        coil_type = 'Stbd'
    elif name_lower.endswith('_crp'):
        block_name = name_without_ext[:-4]
        coil_type = 'CRP'
    
    if block_name:
        if block_name.lower().endswith('_manual'):
            block_name = block_name[:-7]
            
    return block_name, coil_type


def build_nav_file_index(nav_update_path, target_blocks=None):
    """
    Pre-build an index of navigation files organized by block name.
    
    Args:
        nav_update_path: Root path to scan for navigation files
        target_blocks: Optional set of block names to filter for (optimization)
    
    Returns:
        Dictionary: {block_name: [(file_path, coil_type), ...]}
    """
    file_index = {}
    
    if not nav_update_path or not os.path.exists(nav_update_path):
        return file_index
    
    for root, dirs, files in os.walk(nav_update_path):
        for filename in files:
            if not filename.lower().endswith('.csv') and not filename.lower().endswith('.nav'):
                continue
            
            block_name, coil_type = get_file_type_and_block(filename)
            
            if not block_name or not coil_type:
                continue
            
            # Skip if not in target blocks (optimization)
            if target_blocks and block_name not in target_blocks:
                continue
            
            file_path = os.path.join(root, filename)
            
            if block_name not in file_index:
                file_index[block_name] = []
            file_index[block_name].append((file_path, coil_type))
    
    logging.info(f"Built file index: {len(file_index)} blocks, {sum(len(v) for v in file_index.values())} files")
    return file_index


def update_nav_files_batch(nav_update_path, calculated_data):
    """
    Update VisualSoft Nav CSV files with calculated Altitude and Depth values.
    
    OPTIMIZED VERSION:
    - Pre-builds file index before processing
    - Uses smart datetime parsing with format caching
    - Only processes files for blocks in calculated_data
    """
    if not nav_update_path or not os.path.exists(nav_update_path):
        logging.error("Invalid VisualSoft Nav CSV folder")
        return 0, 0
    
    # Pre-build file index for only the blocks we need
    target_blocks = set(calculated_data.keys())
    file_index = build_nav_file_index(nav_update_path, target_blocks)
    
    if not file_index:
        logging.warning("No matching navigation files found")
        return 0, 0
    
    files_updated = 0
    files_skipped = 0
    
    # Process files by block (more efficient - calculated_data is already loaded per block)
    for block_name, file_list in file_index.items():
        if block_name not in calculated_data:
            continue
        
        calc_df = calculated_data[block_name]
        
        for file_path, coil_type in file_list:
            filename = os.path.basename(file_path)
            logging.info(f"  - Updating: {filename} ({coil_type})... ")
            
            try:
                # Read file with auto-detect separator
                try:
                    target_df = pd.read_csv(file_path, sep=',')
                    if len(target_df.columns) < 2:
                        target_df = pd.read_csv(file_path, sep=';')
                except:
                    target_df = pd.read_csv(file_path, sep=None, engine='python')
                
                target_df.columns = target_df.columns.str.strip()
                
                if 'Alt' not in target_df.columns or 'Depth' not in target_df.columns:
                    logging.info("Skipped (missing Alt/Depth columns)")
                    files_skipped += 1
                    continue
                
                if 'Date' not in target_df.columns or 'Time' not in target_df.columns:
                    logging.info("Skipped (missing Date/Time columns)")
                    files_skipped += 1
                    continue
                
                # Use smart datetime parsing with caching (per file type)
                cache_key = f"navfile_{block_name}"
                target_df['DateTime'] = parse_datetime_column_smart(
                    target_df, 'Date', 'Time', cache_key
                )
                
                # Check if parsing was mostly successful
                if target_df['DateTime'].isna().sum() > len(target_df) * 0.5:
                    logging.warning(f"Poor datetime parsing for {filename}")
                    files_skipped += 1
                    continue
                
                # Prepare source data based on coil type
                if coil_type == 'CRP':
                    source_df = calc_df[['DateTime', 'CRP_Altitude', 'CRP_Depth']].copy()
                    source_df = source_df.rename(columns={'CRP_Altitude': 'New_Alt', 'CRP_Depth': 'New_Depth'})
                    source_df = source_df.drop_duplicates(subset=['DateTime'])
                else:
                    source_df = calc_df[calc_df['Coil_Name'] == coil_type][['DateTime', 'Coil_Altitude', 'Coil_Depth']].copy()
                    source_df = source_df.rename(columns={'Coil_Altitude': 'New_Alt', 'Coil_Depth': 'New_Depth'})
                
                if source_df.empty:
                    logging.info(f"Skipped (no calculated data for {coil_type})")
                    files_skipped += 1
                    continue
                
                # Sort for merge_asof
                target_df = target_df.sort_values(by='DateTime').reset_index(drop=True)
                source_df = source_df.sort_values(by='DateTime').reset_index(drop=True)
                
                # Merge by nearest time
                merged = pd.merge_asof(
                    target_df,
                    source_df[['DateTime', 'New_Alt', 'New_Depth']],
                    on='DateTime',
                    direction='nearest',
                    tolerance=pd.Timedelta(seconds=0.5)
                )
                
                # Update values where we have matches
                mask = merged['New_Alt'].notna()
                target_df.loc[mask, 'Alt'] = merged.loc[mask, 'New_Alt']
                target_df.loc[mask, 'Depth'] = merged.loc[mask, 'New_Depth']
                
                # Save back (without DateTime column)
                target_df.drop(columns=['DateTime'], inplace=True)
                target_df.to_csv(file_path, index=False)
                
                logging.info(f"Done ({mask.sum()} rows updated)")
                files_updated += 1
                    
            except Exception as e:
                logging.error(f"Error: {e}")
                files_skipped += 1

    return files_updated, files_skipped


def process_dvl_correction(sql_db_path, z_dvl_offset, output_folder, sql_server_name, folder_filter, block_ids_str=None):
    """
    Main processing function for DVL altitude correction.
    Returns calculated_data dictionary if successful, None otherwise.
    
    OPTIMIZED VERSION:
    - Queries SQL directly for specific block IDs (no full-database fetch)
    - Skips CSV file I/O - uses DataFrame directly
    - Uses smart datetime parsing with format caching
    
    Args:
        block_ids_str: String of block IDs (e.g., '100-105, 107') to filter SQL data.
                       REQUIRED for optimized processing.
    """
    if not sql_db_path:
        logging.error("Please select NaviEdit SQL Database path.")
        return None
    
    if not output_folder:
        logging.error("Please select an Output Folder.")
        return None
    
    try:
        z_offset = float(z_dvl_offset)
    except ValueError:
        logging.error("Z DVL Offset must be a numeric value.")
        return None
    
    logging.info(f"Z DVL Offset: {z_offset} m")
    logging.info(f"SQL Server: {sql_server_name}")
    logging.info(f"Folder Filter: {folder_filter}")
    logging.info(f"Formula: Coil_Altitude = (CRP_Altitude + {z_offset}) + CRP_Depth - Coil_Depth")
    
    # Parse block IDs (required for optimized path)
    target_block_ids = None
    if block_ids_str:
        target_block_ids = parse_block_ids(block_ids_str)
        if target_block_ids:
            logging.info(f"Querying SQL for Block IDs: {target_block_ids}")
    
    # Step 1: Extract altitude data from SQL database
    logging.info("Step 1: Extracting altitude data from NaviEdit database...")
    
    if target_block_ids:
        # OPTIMIZED PATH: Query SQL directly for specific blocks
        sql_altitude_df = extract_altitude_for_block_ids_direct(sql_db_path, target_block_ids, sql_server_name)
        
        if sql_altitude_df.empty:
            # Check if there's detailed error info attached to the DataFrame
            error_info = sql_altitude_df.attrs.get('error_info', {})
            error_type = error_info.get('error_type', 'unknown')
            
            if error_type == 'block_ids_not_found':
                not_found = error_info.get('not_found_ids', [])
                available_ids = error_info.get('available_block_ids', [])
                available_names = error_info.get('available_block_names', {})
                
                # Build user-friendly error message
                error_msg = "SQL Altitude Extraction Error\n\n"
                error_msg += f"Requested Block IDs: {target_block_ids}\n\n"
                
                if not_found:
                    error_msg += f"❌ Block IDs NOT found in database:\n   {not_found}\n\n"
                
                if available_ids:
                    # Show available blocks with names (limited to 15 for readability)
                    error_msg += "Available Block IDs in database:\n"
                    for i, bid in enumerate(available_ids[:15]):
                        name = available_names.get(bid, "")
                        error_msg += f"   {bid}: {name}\n"
                    if len(available_ids) > 15:
                        error_msg += f"   ... and {len(available_ids) - 15} more blocks\n"
                else:
                    error_msg += "No blocks with altitude data found in database.\n"
                
                error_msg += "\nPossible causes:\n"
                error_msg += "  • Wrong Block ID number entered\n"
                error_msg += "  • Block exists but has no bathy/altitude data\n"
                error_msg += "  • Wrong database selected\n"
                
                logging.error(error_msg)
                messagebox.showerror("SQL Extraction Failed", error_msg)
            
            elif error_type == 'connection_failed':
                error_msg = "Database Connection Error\n\n"
                error_msg += error_info.get('message', 'Could not connect to database')
                error_msg += "\n\nPossible causes:\n"
                error_msg += "  • Database file is locked by another application\n"
                error_msg += "  • SQL Server service is not running\n"
                error_msg += "  • Wrong database path selected\n"
                logging.error(error_msg)
                messagebox.showerror("Database Connection Failed", error_msg)
            
            elif error_type == 'database_not_found':
                error_msg = f"Database Not Found\n\n{error_info.get('message', '')}"
                logging.error(error_msg)
                messagebox.showerror("Database Not Found", error_msg)
            
            else:
                # Generic error
                error_msg = error_info.get('message', f"No altitude data found for Block IDs: {target_block_ids}")
                logging.error(error_msg)
                messagebox.showerror("SQL Extraction Failed", error_msg)
            
            return None
        
        logging.info(f"  - Extracted {len(sql_altitude_df)} altitude records directly from database")
    else:
        # FALLBACK: Legacy path - extract all then filter (slower for large DBs)
        logging.warning("No block IDs provided - using slower legacy extraction method")
        
        if output_folder and os.path.exists(output_folder):
            altitude_csv_path = os.path.join(output_folder, "TSS_Altitude.csv")
        else:
            altitude_csv_path = os.path.join(SCRIPT_DIR, "TSS_Altitude.csv")
        
        sql_altitude_df = extract_altitude_from_sql(sql_db_path, altitude_csv_path, folder_filter, sql_server_name)
        
        if sql_altitude_df.empty:
            logging.error("Failed to extract altitude data from database or no data found.")
            return None
        
        logging.info(f"  - Extracted {len(sql_altitude_df)} altitude records from database")
        
        # Read back with proper formatting (legacy path)
        sql_altitude_df = read_sql_altitude_csv(altitude_csv_path)
        if sql_altitude_df.empty:
            logging.error("Failed to read generated altitude CSV.")
            return None
    
    # Step 2: Read nav depth files
    navdepth_folder = os.path.join(output_folder, "navdepth") if output_folder and os.path.exists(output_folder) else os.path.join(SCRIPT_DIR, "navdepth")
    
    logging.info(f"Step 2: Reading nav depth files from: {navdepth_folder}")
    
    if not os.path.exists(navdepth_folder):
        logging.error(f"Navdepth folder not found: {navdepth_folder}")
        return None
    
    nav_blocks = read_nav_files_by_block(navdepth_folder)
    
    if not nav_blocks:
        logging.error("No nav files found in navdepth folder.")
        return None
    
    logging.info(f"  - Found {len(nav_blocks)} blocks in navdepth folder")
    
    # Step 3: Match and calculate altitude
    logging.info("Step 3: Matching data and calculating Coil Altitude...")
    
    results = match_altitude_with_nav(sql_altitude_df, nav_blocks, z_offset, target_block_ids)
    
    if not results:
        # Provide detailed error message about the mismatch
        nav_block_names = sorted(nav_blocks.keys())
        entered_block_ids = target_block_ids if target_block_ids else []
        
        error_msg = "Block ID Mismatch Error:\n\n"
        error_msg += f"Nav files found in folder for blocks:\n  {nav_block_names}\n\n"
        error_msg += f"Block IDs entered:\n  {entered_block_ids}\n\n"
        error_msg += "No matching data could be found.\n\n"
        error_msg += "Please check:\n"
        error_msg += "  1. Are the Block IDs correct?\n"
        error_msg += "  2. Do the nav files match those Block IDs?\n"
        error_msg += "  3. Is the correct folder selected?"
        
        logging.error(error_msg)
        messagebox.showerror("Block ID Mismatch", error_msg)
        return None
    
    # Step 4: Filter and save CSV files
    logging.info(f"Step 4: Saving CSV files to: {output_folder}")
    
    existing_blocks = set()
    if os.path.exists(output_folder):
        for filename in os.listdir(output_folder):
            if filename.lower().endswith('.csv') or filename.lower().endswith('.nav'):
                b_name, _ = get_file_type_and_block(filename)
                if b_name:
                    existing_blocks.add(b_name)
    
    if existing_blocks:
        filtered_results = {k: v for k, v in results.items() if k in existing_blocks}
    else:
        filtered_results = results
    
    if not filtered_results:
        logging.warning("No matching blocks found in output folder.")
        return None

    save_calculated_altitude_csvs(filtered_results, output_folder)
    
    # Convert results for update_nav_files_batch (needs DateTime column)
    calculated_data = {}
    for block_name, df in filtered_results.items():
        df_copy = df.copy()
        if 'Date' in df_copy.columns and 'Time' in df_copy.columns:
            # Use smart datetime parsing with caching
            df_copy['DateTime'] = parse_datetime_column_smart(df_copy, 'Date', 'Time', f'calc_{block_name}')
            df_copy = df_copy.dropna(subset=['DateTime'])
        calculated_data[block_name] = df_copy
    
    logging.info("DVL Altitude correction processing completed.")
    return calculated_data


def open_sql_settings_dialog():
    """Open a dialog to configure NE Database settings."""
    settings = load_settings()
    
    dialog = tk.Toplevel(root)
    dialog.title("NE Database Settings")
    dialog.geometry("550x480")
    dialog.transient(root)
    dialog.grab_set()
    
    # Center the dialog
    root_x = root.winfo_x()
    root_y = root.winfo_y()
    root_w = root.winfo_width()
    root_h = root.winfo_height()
    x = root_x + (root_w // 2) - 275
    y = root_y + (root_h // 2) - 240
    dialog.geometry(f"+{x}+{y}")
    
    frame = ttk.Frame(dialog, padding="20")
    frame.pack(fill=tk.BOTH, expand=True)
    
    # === WFM Database Settings Section ===
    ttk.Label(frame, text="WFM Database Settings", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 10))
    
    # WFM NE Database Name
    ttk.Label(frame, text="WFM NE Database Name:").grid(row=1, column=0, sticky=tk.W, pady=5)
    wfm_db_name_entry = ttk.Entry(frame, width=40)
    wfm_db_name_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
    wfm_db_name_entry.insert(0, settings.get('wfm_ne_db_name', ''))
    ttk.Label(frame, text="(NaviEdit database name for WFM)", font=("Segoe UI", 8, "italic")).grid(row=2, column=1, sticky=tk.W, padx=5)
    
    # WFM NE Database Server
    ttk.Label(frame, text="WFM NE Database Server:").grid(row=3, column=0, sticky=tk.W, pady=5)
    wfm_db_server_entry = ttk.Entry(frame, width=40)
    wfm_db_server_entry.grid(row=3, column=1, sticky="ew", padx=5, pady=5)
    wfm_db_server_entry.insert(0, settings.get('wfm_ne_db_server', 'localhost'))
    ttk.Label(frame, text="(e.g. localhost, RS-GOEL-PVE03)", font=("Segoe UI", 8, "italic")).grid(row=4, column=1, sticky=tk.W, padx=5)
    
    ttk.Separator(frame, orient='horizontal').grid(row=5, column=0, columnspan=3, sticky="ew", pady=15)
    
    # === SQL Altitude Extraction Settings Section ===
    ttk.Label(frame, text="SQL Altitude Extraction Settings", font=("Segoe UI", 10, "bold")).grid(row=6, column=0, columnspan=3, sticky=tk.W, pady=(0, 10))
    
    # NaviEdit SQL Database Path
    ttk.Label(frame, text="NaviEdit SQL Database:").grid(row=7, column=0, sticky=tk.W, pady=5)
    sql_db_entry = ttk.Entry(frame, width=40)
    sql_db_entry.grid(row=7, column=1, sticky="ew", padx=5, pady=5)
    sql_db_entry.insert(0, settings.get('sql_db_path', ''))
    
    def browse_sql_db():
        file_path = filedialog.askopenfilename(
            title="Select NaviEdit SQL Server Database File",
            filetypes=[("SQL Server Database", "*.mdf"), ("All files", "*.*")]
        )
        if file_path:
            sql_db_entry.delete(0, tk.END)
            sql_db_entry.insert(0, file_path)
    
    ttk.Button(frame, text="Browse", command=browse_sql_db).grid(row=7, column=2, padx=5, pady=5)
    ttk.Label(frame, text="(Path to NaviEdit .mdf database file)", font=("Segoe UI", 8, "italic")).grid(row=8, column=1, sticky=tk.W, padx=5)
    
    # Z DVL Offset
    ttk.Label(frame, text="Z DVL Offset (m):").grid(row=9, column=0, sticky=tk.W, pady=5)
    z_offset_entry = ttk.Entry(frame, width=20)
    z_offset_entry.grid(row=9, column=1, sticky=tk.W, padx=5, pady=5)
    z_offset_entry.insert(0, settings.get('z_dvl_offset', '0.0'))
    ttk.Label(frame, text="(Offset to apply to DVL altitude)", font=("Segoe UI", 8, "italic")).grid(row=10, column=1, sticky=tk.W, padx=5)
    
    # SQL Server Name
    ttk.Label(frame, text="SQL Server Name:").grid(row=11, column=0, sticky=tk.W, pady=5)
    server_entry = ttk.Entry(frame, width=40)
    server_entry.grid(row=11, column=1, sticky="ew", padx=5, pady=5)
    server_entry.insert(0, settings.get('sql_server_name', 'RS-GOEL-PVE03'))
    ttk.Label(frame, text="(e.g. RS-GOEL-PVE03, localhost, .)", font=("Segoe UI", 8, "italic")).grid(row=12, column=1, sticky=tk.W, padx=5)
    
    # Folder Filter
    ttk.Label(frame, text="Folder Filter:").grid(row=13, column=0, sticky=tk.W, pady=5)
    filter_entry = ttk.Entry(frame, width=40)
    filter_entry.grid(row=13, column=1, sticky="ew", padx=5, pady=5)
    filter_entry.insert(0, settings.get('folder_filter', '04_NAVISCAN'))
    ttk.Label(frame, text="(Parent folder name to extract from)", font=("Segoe UI", 8, "italic")).grid(row=14, column=1, sticky=tk.W, padx=5)
    
    frame.columnconfigure(1, weight=1)
    
    def save_and_close():
        try:
            current_settings = {}
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r') as f:
                    current_settings = json.load(f)
            
            current_settings['wfm_ne_db_name'] = wfm_db_name_entry.get()
            current_settings['wfm_ne_db_server'] = wfm_db_server_entry.get()
            current_settings['sql_db_path'] = sql_db_entry.get()
            current_settings['z_dvl_offset'] = z_offset_entry.get()
            current_settings['sql_server_name'] = server_entry.get()
            current_settings['folder_filter'] = filter_entry.get()
            
            os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
            with open(CONFIG_FILE, 'w') as f:
                json.dump(current_settings, f, indent=4)
            logging.info("NE Database Settings saved successfully")
            messagebox.showinfo("Success", "NE Database Settings saved.")
            dialog.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Could not save settings: {e}")
    
    ttk.Button(frame, text="Save", command=save_and_close).grid(row=15, column=1, pady=20, sticky=tk.E)
    
    dialog.wait_window()


def recalculate_tss_altitude():
    """
    Combined function that runs:
    1. Export Depth from NaviEdit (WFM) - with automatic completion detection
    2. Calculate TSS Altitude
    3. Update VisualSoft Navigation files
    
    OPTIMIZED VERSION:
    - Automatic WFM completion monitoring (no user interaction needed)
    - Progress window with step tracking
    - Direct SQL query for specific blocks
    - Pre-indexed file lookup
    """
    try:
        # Get settings
        settings = load_settings()
        block_ids_str = ne_path_entry.get()
        output_folder = folder_entry.get()
        sql_db_path = settings.get('sql_db_path', '')
        z_dvl_offset = settings.get('z_dvl_offset', '0.0')
        sql_server_name = settings.get('sql_server_name', 'RS-GOEL-PVE03')
        folder_filter = settings.get('folder_filter', '04_NAVISCAN')
        
        # Validate inputs
        if not block_ids_str:
            messagebox.showerror("Error", "Please enter Block IDs.")
            return
        
        if not output_folder:
            messagebox.showerror("Error", "Please select a Processed Data Folder.")
            return
        
        if not sql_db_path:
            messagebox.showerror("Error", "Please configure the NaviEdit SQL Database path in SQL Settings.")
            return
        
        # Parse block IDs
        block_ids = parse_block_ids(block_ids_str)
        if not block_ids:
            messagebox.showerror("Error", "Invalid Block ID format.")
            return
        
        navdepth_folder = os.path.join(output_folder, "navdepth")
        
        # Ask user to confirm
        result = messagebox.askyesno("Recalculate TSS Altitude", 
            f"This will:\n\n"
            f"1. Export Depth data from NaviEdit for blocks: {block_ids_str}\n"
            f"2. Calculate TSS Altitude using offset: {z_dvl_offset}m\n"
            f"3. Update VisualSoft Navigation files in:\n   {output_folder}\n\n"
            f"Continue?")
        
        if not result:
            return
        
        # Create progress dialog
        progress_dialog = tk.Toplevel(root)
        progress_dialog.title("TSS Altitude Recalculation Progress")
        progress_dialog.geometry("500x330")
        progress_dialog.transient(root)
        progress_dialog.grab_set()
        
        # Center the dialog
        root_x = root.winfo_x()
        root_y = root.winfo_y()
        root_w = root.winfo_width()
        root_h = root.winfo_height()
        x = root_x + (root_w // 2) - 250
        y = root_y + (root_h // 2) - 150
        progress_dialog.geometry(f"+{x}+{y}")
        
        # Cancel flag for WFM monitoring
        cancel_flag = [False]
        
        def on_cancel():
            cancel_flag[0] = True
            cancel_button.config(state=tk.DISABLED, text="Cancelling...")
        
        def on_dialog_close():
            if step_status[0] < 3:  # Still processing
                on_cancel()
        
        progress_dialog.protocol("WM_DELETE_WINDOW", on_dialog_close)
        
        frame = ttk.Frame(progress_dialog, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(frame, text="Processing TSS Altitude Recalculation", 
                                font=('TkDefaultFont', 11, 'bold'))
        title_label.pack(pady=(0, 15))
        
        # Step labels
        step_labels = []
        step_frames = []
        steps = [
            "Step 1: Export Depth from NaviEdit (WFM)",
            "Step 2: Calculate TSS Altitude from SQL",
            "Step 3: Update VisualSoft Navigation Files"
        ]
        
        for i, step_text in enumerate(steps):
            step_frame = ttk.Frame(frame)
            step_frame.pack(fill=tk.X, pady=5)
            
            status_icon = ttk.Label(step_frame, text="⬜", width=3)
            status_icon.pack(side=tk.LEFT)
            
            step_label = ttk.Label(step_frame, text=step_text)
            step_label.pack(side=tk.LEFT)
            
            step_labels.append((status_icon, step_label))
            step_frames.append(step_frame)
        
        # Status message
        status_label = ttk.Label(frame, text="Initializing...", wraplength=450)
        status_label.pack(pady=15)
        
        # Progress bar
        progress_var = tk.DoubleVar(value=0)
        progress_bar = ttk.Progressbar(frame, variable=progress_var, maximum=100, length=400)
        progress_bar.pack(pady=10)
        
        # Cancel button
        cancel_button = ttk.Button(frame, text="Cancel", command=on_cancel)
        cancel_button.pack(pady=20)
        
        # Track current step
        step_status = [0]  # [current_step_index]
        
        def update_step(step_idx, status='in_progress'):
            """Update step status: 'pending', 'in_progress', 'done', 'error'"""
            icons = {'pending': '⬜', 'in_progress': '🔄', 'done': '✅', 'error': '❌'}
            if step_idx < len(step_labels):
                step_labels[step_idx][0].config(text=icons.get(status, '⬜'))
            progress_dialog.update()
        
        def update_status(message):
            status_label.config(text=message)
            progress_dialog.update()
        
        def update_progress(pct):
            progress_var.set(pct)
            progress_dialog.update()
        
        # ===== Step 1: Run WFM Depth Export =====
        logging.info("=== Step 1: Exporting Depth from NaviEdit ===")
        update_step(0, 'in_progress')
        update_status("Starting Workflow Manager...")
        update_progress(5)
        
        # Capture timestamp BEFORE launching WFM - only files modified after this will be considered
        import time
        wfm_start_timestamp = time.time()
        
        success = run_wfm_depth_export(block_ids_str, output_folder)
        
        if not success:
            update_step(0, 'error')
            update_status("Failed to start WFM export")
            progress_dialog.destroy()
            return
        
        # Monitor for WFM completion
        update_status(f"Waiting for WFM to export {len(block_ids)} blocks...")
        
        def wfm_progress_callback(status_message, found, expected):
            if not cancel_flag[0]:
                pct = 10 + (30 * found / max(expected, 1))  # 10-40%
                update_status(status_message)
                update_progress(pct)
        
        wfm_success, found_files, wfm_message = wait_for_wfm_completion(
            navdepth_folder, 
            block_ids,
            timeout_seconds=300,  # 5 minutes timeout
            poll_interval=2,
            progress_callback=wfm_progress_callback,
            cancel_flag=cancel_flag,
            start_time=wfm_start_timestamp  # Only consider files newer than this
        )
        
        if cancel_flag[0]:
            update_step(0, 'error')
            update_status("Cancelled by user")
            progress_dialog.after(2000, progress_dialog.destroy)
            return
        
        if not wfm_success:
            update_step(0, 'error')
            update_status(f"WFM export issue: {wfm_message}")
            result = messagebox.askyesno("WFM Export Warning", 
                f"{wfm_message}\n\nDo you want to continue anyway with available files?")
            if not result:
                progress_dialog.destroy()
                return
        
        update_step(0, 'done')
        update_progress(40)
        
        # ===== Step 2: Calculate TSS Altitude =====
        logging.info("=== Step 2: Calculating TSS Altitude ===")
        update_step(1, 'in_progress')
        update_status("Extracting altitude data from SQL database...")
        update_progress(45)
        
        calculated_data = process_dvl_correction(sql_db_path, z_dvl_offset, output_folder, 
                                                  sql_server_name, folder_filter, block_ids_str)
        
        if calculated_data is None:
            update_step(1, 'error')
            update_status("Failed to calculate TSS Altitude")
            progress_dialog.after(2000, progress_dialog.destroy)
            messagebox.showerror("Error", "Failed to calculate TSS Altitude. Check the log for details.")
            return
        
        update_step(1, 'done')
        update_progress(70)
        
        # ===== Step 3: Update VisualSoft Navigation files =====
        logging.info("=== Step 3: Updating VisualSoft Navigation Files ===")
        update_step(2, 'in_progress')
        update_status(f"Updating navigation files in {output_folder}...")
        update_progress(75)
        
        files_updated, files_skipped = update_nav_files_batch(output_folder, calculated_data)
        
        update_step(2, 'done')
        update_progress(100)
        step_status[0] = 3  # Mark as complete
        
        # Update UI for completion
        cancel_button.config(state=tk.DISABLED, text="Complete")
        update_status(f"✅ Done! Updated {files_updated} files, skipped {files_skipped}")
        
        # Close progress dialog after delay and show summary
        progress_dialog.after(1500, progress_dialog.destroy)
        
        # Summary
        messagebox.showinfo("TSS Altitude Recalculation Complete", 
            f"Process completed successfully!\n\n"
            f"Files updated: {files_updated}\n"
            f"Files skipped: {files_skipped}\n\n"
            f"Calculated altitude CSVs saved to:\n{output_folder}")
        
        logging.info(f"=== TSS Altitude Recalculation Complete ===")
        logging.info(f"Files updated: {files_updated}, Files skipped: {files_skipped}")
        
    except Exception as e:
        logging.error(f"Error during TSS Altitude recalculation: {e}")
        messagebox.showerror("Error", f"Error during TSS Altitude recalculation: {e}")


def plot_altitude_depth(source_folder):
    """
    Display a plot with CRP and Center Coil data.
    X-axis: Time
    Left Y-axis (inverted): Depth values (CRP and Center Coil)
    Right Y-axis (normal): Altitude values (CRP and Center Coil)
    
    The 0m altitude line aligns with the seabed depth (CRP_Depth + CRP_Altitude from first row).
    Includes zoom and pan functionality via matplotlib navigation toolbar.
    """
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    
    if not source_folder or not os.path.exists(source_folder):
        messagebox.showerror("Error", f"Source folder not found: {source_folder}\nPlease run 'Recalculate TSS Altitude' first.")
        return
    
    csv_files = [f for f in os.listdir(source_folder) if f.endswith('_calculated_altitude.csv')]
    
    if not csv_files:
        messagebox.showerror("Error", "No calculated altitude CSV files found.\nPlease run 'Recalculate TSS Altitude' first.")
        return
    
    logging.info(f"Generating Depth & Altitude plots for {len(csv_files)} files...")
    
    for csv_file in csv_files:
        # Read the CSV file
        csv_path = os.path.join(source_folder, csv_file)
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logging.error(f"Error reading {csv_file}: {e}")
            continue
        
        # Parse Date and Time to create a datetime for plotting
        if 'Date' in df.columns and 'Time' in df.columns:
            df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S.%f')
        else:
            logging.warning(f"Skipping {csv_file}: missing Date or Time columns.")
            continue
        
        # Get block name for title
        block_name = csv_file.replace('_calculated_altitude.csv', '')
        
        # Get CRP data (from Center coil rows, since CRP values are repeated for all coils)
        center_data = df[df['Coil_Name'] == 'Center'].copy()
        
        if center_data.empty:
            logging.warning(f"Skipping {csv_file}: No Center Coil data found.")
            continue
        
        # Calculate seabed depth from average CRP_Depth + average CRP_Altitude
        avg_crp_depth = center_data['CRP_Depth'].mean()
        avg_crp_altitude = center_data['CRP_Altitude'].mean()
        seabed_depth = avg_crp_depth + avg_crp_altitude
        
        # Create the plot window
        plot_window = tk.Toplevel()
        plot_window.title(f"Depth & Altitude - {block_name}")
        plot_window.geometry("1300x750")
        
        # Create figure
        fig, ax1 = plt.subplots(figsize=(14, 7))
        
        # Colors
        crp_depth_color = 'red'
        crp_alt_color = 'red'
        coil_depth_color = 'blue'
        coil_alt_color = 'blue'
        
        # Left Y-axis: Depth (inverted)
        ax1.set_xlabel('Time', fontsize=11)
        ax1.set_ylabel('Depth (m)', fontsize=11, color='#2c3e50')
        
        # Plot CRP Depth (solid line)
        ax1.plot(center_data['DateTime'], center_data['CRP_Depth'], 
                color=crp_depth_color, linestyle='-', linewidth=2.0, alpha=0.9,
                label='CRP Depth')
        
        # Plot Center Coil Depth (solid line)
        ax1.plot(center_data['DateTime'], center_data['Coil_Depth'], 
                color=coil_depth_color, linestyle='-', linewidth=1.5, alpha=0.8,
                label='Center Coil Depth')
        
        # Calculate ranges for both Depth and Altitude to ensure same scale
        all_depths = pd.concat([center_data['Coil_Depth'], center_data['CRP_Depth']]).dropna()
        depth_min, depth_max = all_depths.min(), all_depths.max()
        
        all_altitudes = pd.concat([center_data['Coil_Altitude'], center_data['CRP_Altitude']]).dropna()
        alt_min, alt_max = all_altitudes.min(), all_altitudes.max()
        
        # Convert altitude limits to "virtual depth" limits relative to seabed
        virt_depth_from_alt_max = seabed_depth - alt_max
        virt_depth_from_alt_min = seabed_depth - alt_min
        
        # Determine global depth range covering both datasets
        global_depth_min = min(depth_min, virt_depth_from_alt_max)
        global_depth_max = max(depth_max, virt_depth_from_alt_min)
        
        # Add padding
        span = global_depth_max - global_depth_min
        if span == 0: span = 1.0
        padding = span * 0.1
        
        plot_depth_min = global_depth_min - padding
        plot_depth_max = global_depth_max + padding
        
        # Set Depth Axis (Left, Inverted)
        ax1.invert_yaxis()
        ax1.set_ylim(plot_depth_max, plot_depth_min)
        ax1.tick_params(axis='y', labelcolor='#2c3e50')
        ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Right Y-axis: Altitude (aligned so 0m = seabed depth)
        ax2 = ax1.twinx()
        ax2.set_ylabel('Altitude (m)', fontsize=11, color='#c0392b')
        
        # Plot CRP Altitude (dotted line)
        ax2.plot(center_data['DateTime'], center_data['CRP_Altitude'], 
                color=crp_alt_color, linestyle=':', linewidth=2.0, alpha=0.9,
                label='CRP Altitude')
        
        # Plot Center Coil Altitude (dotted line)
        ax2.plot(center_data['DateTime'], center_data['Coil_Altitude'], 
                color=coil_alt_color, linestyle=':', linewidth=2.0, alpha=0.9,
                label='Center Coil Altitude')
                
        # Set Altitude limits based on the Depth limits to maintain 1:1 scale
        plot_alt_min = seabed_depth - plot_depth_max
        plot_alt_max = seabed_depth - plot_depth_min
        
        ax2.set_ylim(plot_alt_min, plot_alt_max)
        ax2.tick_params(axis='y', labelcolor='#c0392b')
        
        # Add horizontal line at 0 altitude (seabed reference)
        ax2.axhline(y=0, color='#7f8c8d', linestyle='-', linewidth=1, alpha=0.5, label='Seabed (0m)')
        
        # Format x-axis
        fig.autofmt_xdate()
        
        # Create combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10, 
                   framealpha=0.9, edgecolor='#bdc3c7')
        
        # Title with seabed depth info
        plt.title(f'CRP & Center Coil - Block {block_name}\nSeabed Depth: {seabed_depth:.2f}m (CRP Depth + CRP Altitude)', 
                  fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        # Embed in Tkinter window with navigation toolbar for zoom/pan
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        
        # Add navigation toolbar for zoom and pan functionality
        toolbar_frame = ttk.Frame(plot_window)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()
        
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Proper cleanup function to close matplotlib figure before destroying window
        def on_close():
            try:
                plt.close(fig)  # Close the matplotlib figure first
            except:
                pass
            plot_window.destroy()
        
        # Add close button with proper cleanup
        ttk.Button(plot_window, text="Close", command=on_close).pack(pady=10)
        
        # Also handle window close button (X) with proper cleanup
        plot_window.protocol("WM_DELETE_WINDOW", on_close)
        
        logging.info(f"Plot displayed for block: {block_name}")


def show_depth_altitude():
    """Show Depth & Altitude plot for calculated altitude CSV files."""
    try:
        folder_path = folder_entry.get()
        if not folder_path:
            messagebox.showerror("Error", "Please select a Processed Data Folder first.")
            return
        plot_altitude_depth(folder_path)
    except Exception as e:
        logging.error(f"Error plotting depth & altitude: {e}")
        messagebox.showerror("Error", str(e))

# ============================================================================
# End of DVL Altitude Fixer Functions
# ============================================================================

def update_globals():
    global COIL_PORT_SUFFIX, COIL_CENTER_SUFFIX, COIL_STBD_SUFFIX, CRP_SUFFIX, CELL_SIZE
    COIL_PORT_SUFFIX = coil_port_suffix_entry.get()
    COIL_CENTER_SUFFIX = coil_center_suffix_entry.get()
    COIL_STBD_SUFFIX = coil_stbd_suffix_entry.get()
    CRP_SUFFIX = crp_suffix_entry.get()
    try:
        CELL_SIZE = float(cell_size_entry.get())
    except ValueError:
        CELL_SIZE = 0.5 # Default fallback

def open_heatmap_settings():
    """Open the Heatmap Settings window."""
    global COLORS_TSS, BOUNDARIES_TSS, COLORS_ALT, BOUNDARIES_ALT
    
    settings_window = tk.Toplevel(root)
    settings_window.title("Heatmap Settings")
    settings_window.geometry("650x550")
    settings_window.transient(root)
    settings_window.grab_set()
    
    # Create notebook for tabs
    notebook = ttk.Notebook(settings_window)
    notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # TSS Tab
    tss_frame = ttk.Frame(notebook, padding="10")
    notebook.add(tss_frame, text="TSS Color Scale")
    
    # ALT Tab
    alt_frame = ttk.Frame(notebook, padding="10")
    notebook.add(alt_frame, text="ALT Color Scale")
    
    # Store entry widgets for later retrieval
    tss_entries = []
    alt_entries = []
    
    def create_color_table(parent_frame, colors, boundaries, entries_list):
        """Create a color table with From, To, Color columns."""
        # Headers
        ttk.Label(parent_frame, text="From", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, padx=5, pady=5)
        ttk.Label(parent_frame, text="To", font=("Segoe UI", 10, "bold")).grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(parent_frame, text="Color", font=("Segoe UI", 10, "bold")).grid(row=0, column=2, padx=5, pady=5)
        
        for i in range(10):
            row_data = {}
            
            # From entry
            from_entry = ttk.Entry(parent_frame, width=12)
            from_entry.grid(row=i+1, column=0, padx=5, pady=2)
            if i == 0:
                # First row: editable From
                if len(boundaries) > 0:
                    from_entry.insert(0, str(boundaries[0]))
            else:
                # Other rows: readonly, auto-populated
                from_entry.configure(state='readonly')
            row_data['from'] = from_entry
            
            # To entry
            to_entry = ttk.Entry(parent_frame, width=12)
            to_entry.grid(row=i+1, column=1, padx=5, pady=2)
            if i < len(boundaries) - 1:
                to_entry.insert(0, str(boundaries[i+1]))
            row_data['to'] = to_entry
            
            # Color button
            color_btn = tk.Button(parent_frame, text="", width=15, relief="solid")
            color_btn.grid(row=i+1, column=2, padx=5, pady=2)
            if i < len(colors):
                try:
                    color_btn.configure(bg=colors[i])
                except:
                    color_btn.configure(bg="white")
            row_data['color_btn'] = color_btn
            row_data['color'] = colors[i] if i < len(colors) else ""
            
            # Color picker function
            def pick_color(btn=color_btn, row_idx=i, elist=entries_list):
                color = colorchooser.askcolor(title=f"Choose color for row {row_idx + 1}")
                if color[1]:
                    btn.configure(bg=color[1])
                    elist[row_idx]['color'] = color[1]
            
            color_btn.configure(command=pick_color)
            
            # Auto-update next row's From when To changes
            def on_to_change(event, row_idx=i, elist=entries_list):
                if row_idx + 1 < len(elist):
                    next_from = elist[row_idx + 1]['from']
                    to_val = elist[row_idx]['to'].get()
                    next_from.configure(state='normal')
                    next_from.delete(0, tk.END)
                    next_from.insert(0, to_val)
                    next_from.configure(state='readonly')
            
            to_entry.bind('<FocusOut>', on_to_change)
            to_entry.bind('<Return>', on_to_change)
            
            entries_list.append(row_data)
        
        # Initialize From values for rows > 0
        for i in range(1, min(len(boundaries), 10)):
            if i < len(entries_list):
                entries_list[i]['from'].configure(state='normal')
                entries_list[i]['from'].delete(0, tk.END)
                entries_list[i]['from'].insert(0, str(boundaries[i]))
                entries_list[i]['from'].configure(state='readonly')
    
    create_color_table(tss_frame, COLORS_TSS, BOUNDARIES_TSS, tss_entries)
    create_color_table(alt_frame, COLORS_ALT, BOUNDARIES_ALT, alt_entries)
    
    def save_heatmap_settings():
        """Extract values from tables and save to global variables."""
        global COLORS_TSS, BOUNDARIES_TSS, COLORS_ALT, BOUNDARIES_ALT
        
        def extract_values(entries_list):
            colors = []
            boundaries = []
            
            for i, row in enumerate(entries_list):
                from_val = row['from'].get().strip()
                to_val = row['to'].get().strip()
                color_val = row['color']
                
                # Skip empty rows
                if not to_val and i > 0:
                    break
                
                if i == 0:
                    if from_val:
                        try:
                            boundaries.append(float(from_val))
                        except ValueError:
                            messagebox.showerror("Error", f"Invalid 'From' value in row {i+1}")
                            return None, None
                
                if to_val:
                    try:
                        boundaries.append(float(to_val))
                    except ValueError:
                        messagebox.showerror("Error", f"Invalid 'To' value in row {i+1}")
                        return None, None
                    
                    if color_val:
                        colors.append(color_val)
                    else:
                        messagebox.showerror("Error", f"Please select a color for row {i+1}")
                        return None, None
            
            return colors, boundaries
        
        new_tss_colors, new_tss_boundaries = extract_values(tss_entries)
        if new_tss_colors is None:
            return
        
        new_alt_colors, new_alt_boundaries = extract_values(alt_entries)
        if new_alt_colors is None:
            return
        
        # Validate that we have at least 2 boundaries (1 color interval)
        if len(new_tss_boundaries) < 2:
            messagebox.showerror("Error", "TSS scale needs at least one complete interval (From and To values)")
            return
        if len(new_alt_boundaries) < 2:
            messagebox.showerror("Error", "ALT scale needs at least one complete interval (From and To values)")
            return
        
        # Update global variables
        COLORS_TSS = new_tss_colors
        BOUNDARIES_TSS = new_tss_boundaries
        COLORS_ALT = new_alt_colors
        BOUNDARIES_ALT = new_alt_boundaries
        
        # Save to file
        save_settings()
        
        logging.info(f"Heatmap settings saved. TSS: {len(COLORS_TSS)} colors, ALT: {len(COLORS_ALT)} colors")
        messagebox.showinfo("Success", "Heatmap settings saved successfully!")
        settings_window.destroy()
    
    # Buttons frame
    btn_frame = ttk.Frame(settings_window)
    btn_frame.pack(fill=tk.X, padx=10, pady=10)
    
    ttk.Button(btn_frame, text="Save Settings", command=save_heatmap_settings).pack(side=tk.RIGHT, padx=5)
    ttk.Button(btn_frame, text="Cancel", command=settings_window.destroy).pack(side=tk.RIGHT, padx=5)

def select_folder():
    folder_path = filedialog.askdirectory(title="Select a folder")
    if folder_path:
        folder_entry.delete(0, tk.END)
        folder_entry.insert(0, folder_path)
        logging.info(f"Selected folder path: {folder_path}")

def run_export_script():
    output_path = folder_entry.get().replace('/', '\\')
    block_ids_str = ne_path_entry.get()
    
    if not output_path:
        messagebox.showerror("Error", "Please select Output folder.")
        return
    
    if not block_ids_str:
        messagebox.showerror("Error", "Please enter Block IDs.")
        return

    # Parse Block IDs
    block_ids = []
    try:
        parts = [p.strip() for p in block_ids_str.split(',')]
        for part in parts:
            if '-' in part:
                start, end = map(int, part.split('-'))
                block_ids.extend(range(start, end + 1))
            else:
                block_ids.append(int(part))
        block_ids = sorted(list(set(block_ids))) # Remove duplicates and sort
    except ValueError:
        messagebox.showerror("Error", "Invalid Block ID format. Use comma separated numbers or ranges (e.g. 100-105, 107).")
        return

    if not block_ids:
        messagebox.showerror("Error", "No valid Block IDs found.")
        return

    # Path to the XML template
    xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', '600090_WFM_Export.xml')
    
    if not os.path.exists(xml_path):
        messagebox.showerror("Error", f"WFM XML template not found at {xml_path}")
        return

    try:
        with open(xml_path, 'r') as f:
            xml_content = f.read()
            
        # Replace InputTask with SetPropertyTask for OutputDirectory
        target_output_str = 'name="Select OUTPUT folder (for PTR and Nav exports)" output="OutputDirectory" AskForInput="true" Message="Select OUTPUT folder (for PTR and Nav exports)"/>'
        replacement_output_str = f'name="Set Output Directory" input="{output_path}" output="OutputDirectory" level="2"/>'
        
        if target_output_str in xml_content:
            xml_content = xml_content.replace(f'<InputTask po="5" {target_output_str}', f'<SetPropertyTask po="5" {replacement_output_str}')

        # Comment out redundant SetPropertyTask po="6"
        target_po6 = '<SetPropertyTask po="6" name="Define output directory" input="{OutputDirectory}" output="OutputDirectory" level="2"/>'
        if target_po6 in xml_content:
             xml_content = xml_content.replace(target_po6, f'<!-- {target_po6} -->')

        # --- DYNAMICALLY GENERATE EXPORT TASKS FOR EACH BLOCK ID ---
        
        # 1. Extract the template for a single block export (inside FindBlocksTask)
        # We look for the content between <FindBlocksTask ...> and </FindBlocksTask>
        # And specifically the <GroupTask ... name="Export {Block.name}"> part.
        
        start_marker = '<FindBlocksTask po="4" name="Iterate Blocks in Job" output="Block">'
        end_marker = '</FindBlocksTask>'
        
        start_idx = xml_content.find(start_marker)
        end_idx = xml_content.find(end_marker)
        
        if start_idx != -1 and end_idx != -1:
            # Extract the inner content (the GroupTask)
            # We need to find the start of the inner GroupTask
            inner_content_start = xml_content.find('<GroupTask po="1" name="Export {Block.name}">', start_idx)
            # The end of the inner content is before the end_marker (minus indentation/newlines)
            # But simpler: just take the whole block and replace it with our generated list
            
            # Let's extract the template string for one block
            # We assume the structure is consistent with the file we read
            template_start = xml_content.find('<GroupTask po="1" name="Export {Block.name}">', start_idx)
            # Find the matching closing tag for this GroupTask. 
            # Since it's nested, we can't just search for </GroupTask>.
            # However, we know it ends right before </FindBlocksTask>
            template_end = end_idx 
            
            # Refine template_end to exclude the closing </FindBlocksTask> tag's indentation if possible, 
            # but taking everything up to end_idx is safe enough if we strip trailing whitespace from the extracted part.
            
            # Actually, let's just grab the text and assume it's the last child.
            # A safer way is to manually construct the string based on what we know is there, 
            # OR use the file content we read.
            
            # Let's try to extract the exact string from the file content
            block_template = xml_content[template_start:template_end].strip()
            
            # Remove the last </GroupTask> if it was captured (it should be part of the block template)
            # Wait, the structure is:
            # <FindBlocksTask>
            #    <NaviEditTask>...</NaviEditTask>
            #    <GroupTask ...> ... </GroupTask>
            # </FindBlocksTask>
            
            # So we need to extract just the <GroupTask ...> ... </GroupTask> part.
            # The <NaviEditTask> part is for FindBlocks, we don't need it for direct ID export.
            
            # Let's find the end of the GroupTask.
            # It is the last </GroupTask> before </FindBlocksTask>
            last_group_task_end = xml_content.rfind('</GroupTask>', start_idx, end_idx) + len('</GroupTask>')
            
            block_template = xml_content[template_start:last_group_task_end]
            
            # Generate the new XML block
            generated_tasks = []
            for i, bid in enumerate(block_ids):
                # Create a modified copy of the template
                task_xml = block_template
                
                # Replace placeholders
                # {Block.id} -> The actual ID
                task_xml = task_xml.replace('{Block.id}', str(bid))
                
                # {Block.name} -> "Block_ID" (since we don't have the name)
                task_xml = task_xml.replace('{Block.name}', f'Block_{bid}')
                
                # Update the 'po' (process order) attribute of the GroupTask to be sequential
                # The original is po="1". We can change it to i+4 to avoid conflict with previous tasks (po=1, po=2)
                task_xml = task_xml.replace('po="1" name="Export', f'po="{i+4}" name="Export')
                
                generated_tasks.append(task_xml)
            
            # Join all generated tasks
            new_content_block = '\n'.join(generated_tasks)
            
            # Replace the entire FindBlocksTask block with the new content
            # We replace from start_marker to end_marker + len(end_marker)
            full_match_string = xml_content[start_idx:end_idx + len(end_marker)]
            xml_content = xml_content.replace(full_match_string, new_content_block)

        # Replace relative log paths with absolute paths
        logs_dir = os.path.join(os.getcwd(), "LOGS WFM")
        temp_dir = os.path.join(logs_dir, "temp")
        if not os.path.exists(logs_dir):
            try:
                os.makedirs(logs_dir)
            except Exception as e:
                logging.warning(f"Could not create logs directory: {e}")
        if not os.path.exists(temp_dir):
            try:
                os.makedirs(temp_dir)
            except Exception as e:
                logging.warning(f"Could not create temp directory: {e}")

        xml_content = xml_content.replace(r'.\LOGS_WFM_600090_import_export', logs_dir)

        # Save to temp file in the temp subfolder
        temp_xml_path = os.path.join(temp_dir, "temp_wfm_export.xml")
        with open(temp_xml_path, "w") as f:
            f.write(xml_content)
            
        # Run WFM
        wfm_exe = r"C:\Eiva\WorkFlowManager\WorkflowManager.exe"
        if not os.path.exists(wfm_exe):
             messagebox.showerror("Error", f"Workflow Manager executable not found at {wfm_exe}")
             return
        
        # Start the auto-accepter to handle channel selection dialogs for VS Nav exports
        # Only if auto-clicker is enabled in settings
        if auto_clicker_var.get():
            # Each block exports 4 VS Nav files (CRP, PORT, CENTER, STBD) + 1 PTR file = 5 files per block
            # The auto-accepter will monitor for these files and stop when export is complete
            expected_files_per_block = 5  # 4 nav files + 1 ptr file
            total_expected_files = len(block_ids) * expected_files_per_block
            
            auto_stop_event, auto_thread = start_wfm_dialog_auto_accepter(
                duration_seconds=120,  # Max 2 minutes
                output_folder=output_path,
                expected_file_count=total_expected_files
            )
            logging.info(f"Started WFM dialog auto-accepter (max 2 min, expecting {total_expected_files} files)")
        else:
            logging.info("Auto-clicker disabled - user will need to click OK on WFM dialogs manually")
    
        subprocess.Popen([wfm_exe, "-run", temp_xml_path])
        # messagebox.showinfo("Info", f"Export Script started for Block IDs: {block_ids}. Please wait for it to finish.")
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to run Export script: {e}")

def validate_inputs(folder_path, tss1_col, tss2_col, tss3_col, output_file=None):
    if not folder_path:
        raise ValueError("Missing folder path. Select it using the Browse button.")
    if not all([tss1_col, tss2_col, tss3_col]):
        raise ValueError("TSS column values are required.")
    if output_file is not None and not output_file.strip():
        raise ValueError("Output file name is required.")

def show_heading():
    try:
        update_globals()
        save_settings()
        folder_path = folder_entry.get()
        tss1_col, tss2_col, tss3_col = tss1_entry.get(), tss2_entry.get(), tss3_entry.get()
        use_crp = use_crp_var.get()
        validate_inputs(folder_path, tss1_col, tss2_col, tss3_col)
        plotHeading(folder_path, tss1_col, tss2_col, tss3_col, use_crp)
    except ValueError as e:
        logging.error(f"Error plotting heading: {e}")
        messagebox.showerror("Error", str(e))

def show_maps():
    try:
        update_globals()
        save_settings()
        folder_path = folder_entry.get()
        tss1_col, tss2_col, tss3_col = tss1_entry.get(), tss2_entry.get(), tss3_entry.get()
        use_crp = use_crp_var.get()
        validate_inputs(folder_path, tss1_col, tss2_col, tss3_col)
        plotMaps(folder_path, tss1_col, tss2_col, tss3_col, use_crp)
    except ValueError as e:
        logging.error(f"Error plotting maps: {e}")
        messagebox.showerror("Error", str(e))
    
def show_coils():
    try:
        update_globals()
        save_settings()
        folder_path = folder_entry.get()
        tss1_col, tss2_col, tss3_col = tss1_entry.get(), tss2_entry.get(), tss3_entry.get()
        use_crp = use_crp_var.get()
        validate_inputs(folder_path, tss1_col, tss2_col, tss3_col)
        plotCoils(folder_path, tss1_col, tss2_col, tss3_col, use_crp)
    except ValueError as e:
        logging.error(f"Error plotting coils: {e}")
        messagebox.showerror("Error", str(e))

def show_altitude():
    try:
        update_globals()
        save_settings()
        folder_path = folder_entry.get()
        tss1_col, tss2_col, tss3_col = tss1_entry.get(), tss2_entry.get(), tss3_entry.get()
        use_crp = use_crp_var.get()
        validate_inputs(folder_path, tss1_col, tss2_col, tss3_col)
        plotAltitude(folder_path, tss1_col, tss2_col, tss3_col, use_crp)
    except ValueError as e:
        logging.error(f"Error plotting altitude: {e}")
        messagebox.showerror("Error", str(e))
        
def close_plots():
    try:
        # Close all open matplotlib plots FIRST before destroying Tkinter windows
        plt.close('all')
        
        # Also close any Tkinter Toplevel windows that might be holding plots
        # We iterate through all children of root and destroy Toplevels that have "Depth & Altitude" in title
        for widget in list(root.winfo_children()):  # Use list() to avoid modification during iteration
            if isinstance(widget, tk.Toplevel):
                try:
                    title = widget.title()
                    if "Depth & Altitude" in title:
                        widget.destroy()
                except tk.TclError:
                    pass  # Window already destroyed
                    
    except Exception as e:
        logging.error(f"Error closing plots: {e}")

def process():
    try:
        update_globals()
        save_settings()
        folder_path = folder_entry.get()
        tss1_col, tss2_col, tss3_col = tss1_entry.get(), tss2_entry.get(), tss3_entry.get()
        output_file = output_entry.get()
        use_crp = use_crp_var.get()
        validate_inputs(folder_path, tss1_col, tss2_col, tss3_col, output_file)
        processFiles(folder_path, tss1_col, tss2_col, tss3_col, output_file, use_crp=use_crp)
    except ValueError as e:
        logging.error(f"Error processing files: {e}")
        messagebox.showerror("Error", str(e))

def create_heatmaps_action():
    import gc  # Import garbage collector for cleanup
    try:
        update_globals()
        save_settings()
        folder_path = folder_entry.get()
        tss1_col, tss2_col, tss3_col = tss1_entry.get(), tss2_entry.get(), tss3_entry.get()
        output_file = output_entry.get()
        use_crp = use_crp_var.get()
        validate_inputs(folder_path, tss1_col, tss2_col, tss3_col, output_file)
        
        # Run processFiles silently to get the dataframe
        merged_df = processFiles(folder_path, tss1_col, tss2_col, tss3_col, output_file, silent=True, use_crp=use_crp)
        
        if merged_df is not None:
            # Generate heatmaps
            
            output_file_path = os.path.join(folder_path, output_file)
            filename = os.path.basename(output_file_path)
            
            # TSS heatmap generation
            required_columns = {'Easting', 'Northing', 'TSS'}
            if required_columns.issubset(merged_df.columns):
                heatmaps.generate_TSS_heatmap(output_file_path, filename, merged_df, 0, CELL_SIZE, COLORS_TSS, BOUNDARIES_TSS)
                logging.info(f"Generated TSS heatmap for {filename}")
            else:
                logging.warning(f"Skipping TSS heatmap for {filename}: Missing required columns.")

            # Altitude heatmap generation
            required_columns = {'Easting', 'Northing', 'Alt'}
            if required_columns.issubset(merged_df.columns):
                heatmaps.generate_ALT_heatmap(output_file_path, filename, merged_df, 0, CELL_SIZE, COLORS_ALT, BOUNDARIES_ALT)
                logging.info(f"Generated Altitude heatmap for {filename}")
            else:
                logging.warning(f"Skipping Altitude heatmap for {filename}: Missing required columns.")
            
            # Force garbage collection after heatmap generation to clean up any remaining matplotlib objects
            gc.collect()
            
            messagebox.showinfo("Success", "Files processed and heatmaps generated successfully")

    except ValueError as e:
        logging.error(f"Error creating heatmaps: {e}")
        messagebox.showerror("Error", str(e))
    except Exception as e:
        logging.error(f"Unexpected error creating heatmaps: {e}")
        messagebox.showerror("Error", f"Unexpected error: {e}")
    finally:
        # Always run garbage collection to clean up matplotlib objects on the main thread
        try:
            gc.collect()
        except:
            pass

class HeatmapSettingsWindow(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Heatmap Settings")
        self.geometry("600x500")
        
        self.tss_rows = []
        self.alt_rows = []
        
        # Create tabs
        tab_control = ttk.Notebook(self)
        self.tab_tss = ttk.Frame(tab_control)
        self.tab_alt = ttk.Frame(tab_control)
        tab_control.add(self.tab_tss, text='TSS Settings')
        tab_control.add(self.tab_alt, text='ALT Settings')
        tab_control.pack(expand=1, fill="both")
        
        self.create_table(self.tab_tss, COLORS_TSS, BOUNDARIES_TSS, self.tss_rows)
        self.create_table(self.tab_alt, COLORS_ALT, BOUNDARIES_ALT, self.alt_rows)
        
        save_btn = ttk.Button(self, text="Save Settings", command=self.save_and_close)
        save_btn.pack(pady=10)

    def create_table(self, parent, colors, boundaries, rows_list):
        # Headers
        ttk.Label(parent, text="From").grid(row=0, column=0, padx=5, pady=5)
        ttk.Label(parent, text="To").grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(parent, text="Color").grid(row=0, column=2, padx=5, pady=5)
        
        for i in range(10):
            # From Entry
            from_var = tk.StringVar()
            from_entry = ttk.Entry(parent, textvariable=from_var, width=15)
            from_entry.grid(row=i+1, column=0, padx=5, pady=2)
            if i > 0:
                from_entry.configure(state='readonly')
            
            # To Entry
            to_var = tk.StringVar()
            to_entry = ttk.Entry(parent, textvariable=to_var, width=15)
            to_entry.grid(row=i+1, column=1, padx=5, pady=2)
            
            # Color Button
            color_var = tk.StringVar()
            color_btn = tk.Button(parent, text="Pick Color", width=15)
            color_btn.grid(row=i+1, column=2, padx=5, pady=2)
            
            # Closure to capture current variables
            def pick_color(btn=color_btn, var=color_var):
                color = colorchooser.askcolor(color=var.get())[1]
                if color:
                    var.set(color)
                    btn.configure(bg=color)
            
            color_btn.configure(command=pick_color)
            
            # Logic to update next row's From
            def update_next(event, idx=i, var=to_var, r_list=rows_list):
                if idx + 1 < len(r_list):
                    r_list[idx+1]['from_var'].set(var.get())

            to_entry.bind('<KeyRelease>', update_next)
            to_entry.bind('<FocusOut>', update_next)

            rows_list.append({
                'from_var': from_var,
                'to_var': to_var,
                'color_var': color_var,
                'color_btn': color_btn
            })

        # Populate
        for i in range(min(len(colors), 10)):
            if i < len(boundaries) - 1:
                rows_list[i]['from_var'].set(boundaries[i])
                rows_list[i]['to_var'].set(boundaries[i+1])
                rows_list[i]['color_var'].set(colors[i])
                rows_list[i]['color_btn'].configure(bg=colors[i])
                
                if i + 1 < 10:
                    rows_list[i+1]['from_var'].set(boundaries[i+1])

    def save_and_close(self):
        global COLORS_TSS, BOUNDARIES_TSS, COLORS_ALT, BOUNDARIES_ALT
        
        def extract_values(rows_list):
            colors = []
            boundaries = []
            
            # Get first boundary
            try:
                first_val = float(rows_list[0]['from_var'].get())
                boundaries.append(first_val)
            except ValueError:
                return None, None # Invalid start

            for row in rows_list:
                to_val_str = row['to_var'].get()
                color_val = row['color_var'].get()
                
                if not to_val_str or not color_val:
                    break # Stop at empty row
                
                try:
                    to_val = float(to_val_str)
                    boundaries.append(to_val)
                    colors.append(color_val)
                except ValueError:
                    messagebox.showerror("Error", "Invalid numeric value in table.")
                    return None, None
            
            if not colors:
                return None, None
                
            return colors, boundaries

        new_tss_colors, new_tss_boundaries = extract_values(self.tss_rows)
        new_alt_colors, new_alt_boundaries = extract_values(self.alt_rows)
        
        if new_tss_colors and new_tss_boundaries:
            COLORS_TSS = new_tss_colors
            BOUNDARIES_TSS = new_tss_boundaries
            
        if new_alt_colors and new_alt_boundaries:
            COLORS_ALT = new_alt_colors
            BOUNDARIES_ALT = new_alt_boundaries
            
        save_settings()
        self.destroy()
        messagebox.showinfo("Settings Saved", "Heatmap settings updated successfully.")

def open_heatmap_settings():
    HeatmapSettingsWindow(root)

# Main program
logging.info(f"{SCRIPT_VERSION} started.")

# Create the main window
root = tk.Tk()
root.title(SCRIPT_VERSION)
root.geometry("1200x635")

# Load settings
settings = load_settings()

# Style configuration
style = ttk.Style()
style.theme_use('clam') # 'clam', 'alt', 'default', 'classic'

# Define colors
bg_color = "#f5f6f7"
accent_color = "#0078d7"
text_color = "#333333"
entry_bg = "#ffffff"

root.configure(bg=bg_color)

style.configure(".", background=bg_color, foreground=text_color, font=("Segoe UI", 10))
style.configure("TLabel", background=bg_color, foreground=text_color)
style.configure("TButton", font=("Segoe UI", 10, "bold"))
style.configure("TEntry", fieldbackground=entry_bg)
style.configure("TLabelframe", background=bg_color, foreground=accent_color)
style.configure("TLabelframe.Label", background=bg_color, foreground=accent_color, font=("Segoe UI", 11, "bold"))
style.configure("TCheckbutton", background=bg_color)

# Main container
main_frame = ttk.Frame(root, padding="10", style="TFrame")
main_frame.pack(fill=tk.BOTH, expand=True)

# --- Left Frame: Settings ---
left_frame = ttk.LabelFrame(main_frame, text="Settings", padding="10")
left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

# Block IDs Entry
ttk.Label(left_frame, text="Block IDs (e.g. 100-105, 107):").grid(row=0, column=0, sticky=tk.W, pady=(5, 2))
ne_path_entry = ttk.Entry(left_frame)
ne_path_entry.grid(row=1, column=0, columnspan=2, sticky="ew", padx=(0, 5), pady=2)
ne_path_entry.insert(0, settings.get("ne_path", ""))

# Folder Selection
ttk.Label(left_frame, text="Select Processed Data Folder:").grid(row=2, column=0, sticky=tk.W, pady=(5, 2))
folder_entry = ttk.Entry(left_frame)
folder_entry.grid(row=3, column=0, sticky="ew", padx=(0, 5), pady=2)
folder_entry.insert(0, settings["folder_path"])
ttk.Button(left_frame, text="Browse", command=select_folder).grid(row=3, column=1, sticky="e", pady=2)
left_frame.columnconfigure(0, weight=1) # Make entry expand

# Export Data from NE Button (renamed from Run Export Script)
ttk.Button(left_frame, text="Export Data from NE", command=run_export_script).grid(row=4, column=0, columnspan=2, sticky="ew", pady=(5, 2))

# Recalculate TSS Altitude Button
ttk.Button(left_frame, text="Recalculate TSS Altitude", command=recalculate_tss_altitude).grid(row=5, column=0, columnspan=2, sticky="ew", pady=(2, 10))

# Column Settings
ttk.Label(left_frame, text="Coil 1 (Starboard) Column:").grid(row=6, column=0, sticky=tk.W, pady=2)
tss1_entry = ttk.Entry(left_frame, width=10)
tss1_entry.grid(row=6, column=1, sticky=tk.W, pady=2)
tss1_entry.insert(0, settings["tss1_col"])

ttk.Label(left_frame, text="Coil 2 (Center) Column:").grid(row=7, column=0, sticky=tk.W, pady=2)
tss2_entry = ttk.Entry(left_frame, width=10)
tss2_entry.grid(row=7, column=1, sticky=tk.W, pady=2)
tss2_entry.insert(0, settings["tss2_col"])

ttk.Label(left_frame, text="Coil 3 (Port) Column:").grid(row=8, column=0, sticky=tk.W, pady=2)
tss3_entry = ttk.Entry(left_frame, width=10)
tss3_entry.grid(row=8, column=1, sticky=tk.W, pady=2)
tss3_entry.insert(0, settings["tss3_col"])

# Suffix Settings
ttk.Separator(left_frame, orient='horizontal').grid(row=9, column=0, columnspan=2, sticky="ew", pady=10)
ttk.Label(left_frame, text="File Suffixes:").grid(row=10, column=0, sticky=tk.W, pady=2)

ttk.Label(left_frame, text="Coil Port Suffix:").grid(row=11, column=0, sticky=tk.W, pady=2)
coil_port_suffix_entry = ttk.Entry(left_frame, width=20)
coil_port_suffix_entry.grid(row=11, column=1, sticky=tk.W, pady=2)
coil_port_suffix_entry.insert(0, settings["coil_port_suffix"])

ttk.Label(left_frame, text="Coil Center Suffix:").grid(row=12, column=0, sticky=tk.W, pady=2)
coil_center_suffix_entry = ttk.Entry(left_frame, width=20)
coil_center_suffix_entry.grid(row=12, column=1, sticky=tk.W, pady=2)
coil_center_suffix_entry.insert(0, settings["coil_center_suffix"])

ttk.Label(left_frame, text="Coil Stbd Suffix:").grid(row=13, column=0, sticky=tk.W, pady=2)
coil_stbd_suffix_entry = ttk.Entry(left_frame, width=20)
coil_stbd_suffix_entry.grid(row=13, column=1, sticky=tk.W, pady=2)
coil_stbd_suffix_entry.insert(0, settings["coil_stbd_suffix"])

ttk.Label(left_frame, text="CRP Suffix:").grid(row=14, column=0, sticky=tk.W, pady=2)
crp_suffix_entry = ttk.Entry(left_frame, width=20)
crp_suffix_entry.grid(row=14, column=1, sticky=tk.W, pady=2)
crp_suffix_entry.insert(0, settings["crp_suffix"])

# CRP Checkbox
use_crp_var = tk.BooleanVar(value=settings.get("use_crp", True))
ttk.Checkbutton(left_frame, text="Include CRP Navigation", variable=use_crp_var).grid(row=15, column=0, columnspan=2, sticky=tk.W, pady=5)

# Auto-clicker Checkbox (for WFM Export dialogs)
auto_clicker_var = tk.BooleanVar(value=settings.get("auto_clicker_enabled", True))
ttk.Checkbutton(left_frame, text="Enable Auto-clicker (WFM Export)", variable=auto_clicker_var).grid(row=16, column=0, columnspan=2, sticky=tk.W, pady=5)

# NE Database Settings Button at bottom of left frame
ttk.Separator(left_frame, orient='horizontal').grid(row=17, column=0, columnspan=2, sticky="ew", pady=10)
ttk.Button(left_frame, text="NE Database Settings", command=open_sql_settings_dialog).grid(row=18, column=0, columnspan=2, sticky="ew", pady=5)


# --- Middle Frame: QC Tools ---
middle_frame = ttk.LabelFrame(main_frame, text="QC Tools", padding="10")
middle_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

ttk.Button(middle_frame, text="Heading QC", command=show_heading).pack(fill=tk.X, pady=5)
ttk.Button(middle_frame, text="Show Map", command=show_maps).pack(fill=tk.X, pady=5)
ttk.Button(middle_frame, text="Plot TSS", command=show_coils).pack(fill=tk.X, pady=5)
ttk.Button(middle_frame, text="Plot Altitude", command=show_altitude).pack(fill=tk.X, pady=5)
ttk.Button(middle_frame, text="Plot Depth & Altitude", command=show_depth_altitude).pack(fill=tk.X, pady=5)
ttk.Separator(middle_frame, orient='horizontal').pack(fill=tk.X, pady=10)
ttk.Button(middle_frame, text="Close Plots", command=close_plots).pack(fill=tk.X, pady=5)


# --- Right Frame: Creation ---
right_frame = ttk.LabelFrame(main_frame, text="Creation", padding="10")
right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

ttk.Button(right_frame, text="Process Files", command=process).pack(fill=tk.X, pady=5)
ttk.Separator(right_frame, orient='horizontal').pack(fill=tk.X, pady=10)
ttk.Button(right_frame, text="Create Heatmaps", command=create_heatmaps_action).pack(fill=tk.X, pady=5)
ttk.Separator(right_frame, orient='horizontal').pack(fill=tk.X, pady=10)

# Color Heatmap Settings section (renamed from Heatmap Settings)
ttk.Button(right_frame, text="Color Heatmap Settings", command=open_heatmap_settings).pack(fill=tk.X, pady=5)

# Output File Name (moved from left frame)
ttk.Label(right_frame, text="Output File Name:").pack(anchor=tk.W, pady=(10, 2))
output_entry = ttk.Entry(right_frame, width=30)
output_entry.pack(fill=tk.X, pady=2)
output_entry.insert(0, settings["output_file"])

# Heatmap Cell Size (moved from left frame)
ttk.Label(right_frame, text="Heatmap Cell Size (m):").pack(anchor=tk.W, pady=(5, 2))
cell_size_entry = ttk.Entry(right_frame, width=10)
cell_size_entry.pack(anchor=tk.W, pady=2)
cell_size_entry.insert(0, settings["cell_size"])

# Proper cleanup on exit to prevent Tkinter threading issues
def on_app_closing():
    """Clean up matplotlib figures and close application properly."""
    try:
        plt.close('all')  # Close all matplotlib figures before exiting
    except:
        pass
    try:
        root.quit()
        root.destroy()
    except:
        pass

root.protocol("WM_DELETE_WINDOW", on_app_closing)

logging.info("Graphic User Interface created. Main loop started.")

# Run the main loop
root.mainloop()