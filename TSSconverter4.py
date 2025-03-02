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

# Maximum time difference in seconds
MAX_TIME_DIFF_SEC = 0.15

# Column positions in the PTR file
DATE_COLUMN_POS = 0
TIME_COLUMN_POS = 1
EAST_COLUMN_POS = 2
NORTH_COLUMN_POS = 3

# Maximum angle error for heading and course in degrees
MAX_ANGLE_ERROR = 20

def convert_to_datetime(time_str, format_str='%H%M%S%f'):
    return datetime.strptime(str(time_str), format_str)

def read_csv_file(file_path, delimiter):
    return pd.read_csv(file_path, delimiter=delimiter)

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
    for line in merged_df['Filename'].unique():
        df = merged_df[merged_df['Filename'] == line]
        max_index = df['TSS'].idxmax()
        min_index = df['TSS'].idxmin()

        if abs(df['TSS'][max_index]) > abs(df['TSS'][min_index]):
            abs_max_tss = df['TSS'][max_index]
        else:
            abs_max_tss = df['TSS'][min_index]
    
        coil = df['Coil'][max_index]
        easting = df['Easting'][max_index]
        northing = df['Northing'][max_index]

        coil_peaks.append({
            'PTR file': line,
            'TSS peak value': abs_max_tss,
            'TSS coil': coil,
            'Easting': easting,
            'Northing': northing
        })
        
    return coil_peaks

def circular_mean(angles):
    """Computes the circular mean of angles in degrees."""
    angles_rad = np.radians(angles)
    mean_x = np.mean(np.cos(angles_rad))
    mean_y = np.mean(np.sin(angles_rad))
    mean_angle = np.degrees(np.arctan2(mean_y, mean_x)) % 360  # Ensure 0-360 range
    return mean_angle

def calculateHeading(merged_df):
    attitude_list = []  # Use a list to store dictionaries

    # Loop through unique filenames
    for line in merged_df['Filename'].unique():
        df = merged_df[merged_df['Filename'] == line].copy()  # Copy to avoid SettingWithCopyWarning
        
        if len(df) < 2:
            continue  # If only one row, we cannot calculate diff()

        # Filter the data for Coil 2
        df_coil2 = df[df['Coil'] == 2]

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
        heading_error_avg = heading_avg - course_avg
        line_direction = round(heading_avg / 5) * 5  # Round to nearest 5 degrees

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
            df = pd.read_csv(file_path, delimiter=';', header=None)
            date = df.iloc[:, DATE_COLUMN_POS]
            time = df.iloc[:, TIME_COLUMN_POS]
            easting = df.iloc[:, EAST_COLUMN_POS]
            northing = df.iloc[:, NORTH_COLUMN_POS]
            tss1 = df.iloc[:, tss1_col]
            tss2 = df.iloc[:, tss2_col]
            tss3 = df.iloc[:, tss3_col]

            df_extracted = pd.DataFrame({
                'Filename': filename,
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

    # Ensure the Time PTR column is formatted correctly
    ptr_df['Time PTR'] = ptr_df['Time PTR'].astype(str).str.zfill(9)  # Ensure the time string is 9 chars
    ptr_df['Time PTR'] = ptr_df['Time PTR'].str[:6] + '.' + ptr_df['Time PTR'].str[6:]  # Insert decimal before ms

    # Convert the Time columns to datetime objects
    ptr_df['Time PTR'] = pd.to_datetime(ptr_df['Date PTR'] + ' ' + ptr_df['Time PTR'], format='%d.%m.%Y %H%M%S.%f')
    nav_coil1_df['Time'] = pd.to_datetime(nav_coil1_df['Date'] + ' ' + nav_coil1_df['Time'], format='%d/%m/%Y %H:%M:%S.%f')
    nav_coil2_df['Time'] = pd.to_datetime(nav_coil2_df['Date'] + ' ' + nav_coil2_df['Time'], format='%d/%m/%Y %H:%M:%S.%f')
    nav_coil3_df['Time'] = pd.to_datetime(nav_coil3_df['Date'] + ' ' + nav_coil3_df['Time'], format='%d/%m/%Y %H:%M:%S.%f')

    # Ensure both DataFrames are sorted by the key columns
    ptr_df = ptr_df.sort_values(by='Time PTR')
    nav_coil1_df = nav_coil1_df.sort_values(by='Time')
    nav_coil2_df = nav_coil2_df.sort_values(by='Time')
    nav_coil3_df = nav_coil3_df.sort_values(by='Time')

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
    merged_coil1_df['Coil'] = 1
    del merged_coil1_df['TSS1']
    del merged_coil1_df['TSS2']
    del merged_coil1_df['TSS3']
    del merged_coil1_df['Date PTR']
    del merged_coil1_df['Date']

    merged_coil2_df['TSS']  = merged_coil2_df['TSS2']
    merged_coil2_df['Coil'] = 2
    del merged_coil2_df['TSS1']
    del merged_coil2_df['TSS2']
    del merged_coil2_df['TSS3']
    del merged_coil2_df['Date PTR']
    del merged_coil2_df['Date']

    merged_coil3_df['TSS']  = merged_coil3_df['TSS3']
    merged_coil3_df['Coil'] = 3
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

    return interleaved_df

def plotMaps(folder_path, tss1_col, tss2_col, tss3_col):
    # Extract the data from the PTR and Navigation files in the selected folder
    coil1_df, coil2_df, coil3_df = extractData(folder_path, tss1_col, tss2_col, tss3_col)
    merged_df = mergeData(coil1_df, coil2_df, coil3_df)

    # Calculate the average heading for each line
    attitude_df = calculateHeading(merged_df)

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
    plt.title(f"Survey Direction: {attitude_df['Survey Direction'].iloc[0]:.0f} º")
    plt.gca().set_aspect('equal', adjustable='box')  # Set aspect ratio to 1:1
    plt.grid(True)

    # Create a custom Altitude colormap
    cmap_alt = mcolors.ListedColormap(['purple', 'magenta', 'blue', 'cyan', 'yellow', 'orange', 'red'])
    bounds_alt = [-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
    norm_alt = mcolors.BoundaryNorm(bounds_alt, cmap_alt.N)

    # Create scatter plots for flying altitude
    plt.figure(num=target_path + " Altitude", figsize=(7, 6))
    scatter = plt.scatter(merged_df['Easting'], merged_df['Northing'], c=merged_df['Alt'], cmap=cmap_alt, norm=norm_alt, marker='o')
    plt.colorbar(scatter, label="Altitude [m]", boundaries=bounds_alt, ticks=[-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    plt.xlabel("Easting [m]")
    plt.ylabel("Northing [m]")
    plt.suptitle(f'Heatmap of TSS Flying Altitude')
    plt.title(f"Survey Direction: {attitude_df['Survey Direction'].iloc[0]:.0f} º")
    plt.gca().set_aspect('equal', adjustable='box')  # Set aspect ratio to 1:1
    plt.grid(True)

    plt.show()
    
def plotCoils(folder_path, tss1_col, tss2_col, tss3_col):
    # Extract the data from the PTR and Navigation files in the selected folder
    coil1_df, coil2_df, coil3_df = extractData(folder_path, tss1_col, tss2_col, tss3_col)

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
        plt.figure(num=line['filename'],figsize=(10, 6))
        plt.plot(line['TSS1'], color='r', label='Coil 1')
        plt.plot(line['TSS2'], color='b', label='Coil 2')
        plt.plot(line['TSS3'], color='g', label='Coil 3')

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

def plotHeading(folder_path, tss1_col, tss2_col, tss3_col):
    # Extract the data from the PTR and Navigation files in the selected folder
    coil1_df, coil2_df, coil3_df = extractData(folder_path, tss1_col, tss2_col, tss3_col)

    # Merge the DataFrames into a single DataFrame
    merged_df = mergeData(coil1_df, coil2_df, coil3_df)

    # Calculate the average heading for each line
    attitude_df = calculateHeading(merged_df)

    # Get the target path for the output files
    target_path = get_target_path(folder_path)

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
    cbar.set_label("Abs Heading Error (°)", fontsize=12)

    ax.set_xlabel("Easting [m]")
    ax.set_ylabel("Northing [m]")
    ax.set_aspect('equal', adjustable='box')  # Set aspect ratio to 1:1
    ax.set_title("Scatter Plot of Heading Error")
    ax.grid(True, linestyle='--', alpha=0.6)

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

def processFiles(folder_path, tss1_col, tss2_col, tss3_col, output_file):
    # Extract the data from the PTR and Navigation files in the selected folder
    coil1_df, coil2_df, coil3_df = extractData(folder_path, tss1_col, tss2_col, tss3_col)

    # Merge the DataFrames into a single DataFrame
    merged_df = mergeData(coil1_df, coil2_df, coil3_df)

    # Find the absolute maximum TSS value for each coil and its corresponding position
    coil_peaks = getCoilPeaks(merged_df)  
    print("Coil peak values:")
    print(coil_peaks)

    # Save the merged DataFrame to a new CSV file
    output_file_path = os.path.join(folder_path, output_file)
    merged_df.to_csv(output_file_path, index=False)
    
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

def select_folder():
    folder_path = filedialog.askdirectory(title="Select a folder")
    folder_entry.delete(0, tk.END)
    folder_entry.insert(0, folder_path)

def show_heading():
    folder_path = folder_entry.get()
    print("Folder path: " + folder_path)
    if not folder_path:
        messagebox.showerror("Error", "Missing folder path. Select it using the Browse button")
        return
    
    tss1_col = tss1_entry.get()
    tss2_col = tss2_entry.get()
    tss3_col = tss3_entry.get()

    if not folder_path or not tss1_col or not tss2_col or not tss3_col:
        messagebox.showerror("Error", "Some fields are required")
        return
    
    plotHeading(folder_path, tss1_col, tss2_col, tss3_col)

def show_maps():
    folder_path = folder_entry.get()
    print("Folder path: " + folder_path)
    if not folder_path:
        messagebox.showerror("Error", "Missing folder path. Select it using the Browse button")
        return

    tss1_col = tss1_entry.get()
    tss2_col = tss2_entry.get()
    tss3_col = tss3_entry.get()

    if not folder_path or not tss1_col or not tss2_col or not tss3_col:
        messagebox.showerror("Error", "Some fields are required")
        return

    plotMaps(folder_path, tss1_col, tss2_col, tss3_col)
    
def show_coils():
    folder_path = folder_entry.get()
    print("Folder path: " + folder_path)
    if not folder_path:
        messagebox.showerror("Error", "Missing folder path. Select it using the Browse button")
    tss1_col = tss1_entry.get()
    tss2_col = tss2_entry.get()
    tss3_col = tss3_entry.get()

    if not folder_path or not tss1_col or not tss2_col or not tss3_col:
        messagebox.showerror("Error", "Some fields are required")
        return

    plotCoils(folder_path, tss1_col, tss2_col, tss3_col)

def process():
    folder_path = folder_entry.get()
    tss1_col = tss1_entry.get()
    tss2_col = tss2_entry.get()
    tss3_col = tss3_entry.get()
    output_file = output_entry.get()

    if not folder_path:
        messagebox.showerror("Error", "Missing folder path. Select it using the Browse button")
        return

    if not folder_path or not tss1_col or not tss2_col or not tss3_col or not output_file:
        messagebox.showerror("Error", "Some fields are required")
        return

    processFiles(folder_path, tss1_col, tss2_col, tss3_col, output_file)

# Main program
print("TSS Converter running...")

# Create the main window
root = tk.Tk()
root.title("TSS Converter 4")

# Set the font size
font = ("Helvetica", 14)
font_bold = ("Helvetica", 14, "bold")

# Create and place the widgets
tk.Label(root, text="Select Folder:", font=font).grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)
folder_entry = tk.Entry(root, width=50, font=font)
folder_entry.grid(row=0, column=1, padx=10, pady=5, sticky=tk.W)
folder_entry.insert(0, os.getcwd())  # Default to current working directory
tk.Button(root, text="Browse", command=select_folder, font=font_bold).grid(row=0, column=2, padx=10, pady=5, sticky=tk.W)

tk.Label(root, text="Coil 1 (port) Column PTR file:", font=font).grid(row=3, column=0, padx=10, pady=5, sticky=tk.W)
tss1_entry = tk.Entry(root, width=20, font=font)
tss1_entry.grid(row=3, column=1, padx=10, pady=5, sticky=tk.W)
tss1_entry.insert(0, "12")  # Default value

tk.Label(root, text="Coil 2 (center) Column PTR file:", font=font).grid(row=4, column=0, padx=10, pady=5, sticky=tk.W)
tss2_entry = tk.Entry(root, width=20, font=font)
tss2_entry.grid(row=4, column=1, padx=10, pady=5, sticky=tk.W)
tss2_entry.insert(0, "11")  # Default value

tk.Label(root, text="Coil 3 (starbord) Column PTR file:", font=font).grid(row=5, column=0, padx=10, pady=5, sticky=tk.W)
tss3_entry = tk.Entry(root, width=20, font=font)
tss3_entry.grid(row=5, column=1, padx=10, pady=5, sticky=tk.W)
tss3_entry.insert(0, "10")  # Default value

tk.Label(root, text="Output File Name:", font=font).grid(row=8, column=0, padx=10, pady=5, sticky=tk.W)
output_entry = tk.Entry(root, width=50, font=font)
output_entry.grid(row=8, column=1, padx=10, pady=5, sticky=tk.W)
output_entry.insert(0, "BOSSE_XXX_A.txt")  # Default value

tk.Button(root, text="Heading QC", command=lambda: show_heading(), font=font_bold).grid(row=9, column=0, columnspan=3, pady=10, sticky=tk.W)
tk.Button(root, text="Process Files", command=lambda: process(), font=font_bold).grid(row=10, column=0, columnspan=3, pady=10, sticky=tk.W)
tk.Button(root, text="Show Map", command=lambda: show_maps(), font=font_bold).grid(row=10, column=1, columnspan=3, pady=10, sticky=tk.W)
tk.Button(root, text="Show Coils", command=lambda: show_coils(), font=font_bold).grid(row=10, column=2, columnspan=3, pady=10, sticky=tk.W)


# Run the main loop
root.mainloop()