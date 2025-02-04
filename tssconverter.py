import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog

def processFiles(folder_path, output_file):
    # Initialize an empty list to store DataFrames
    dataframes = []

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            # Construct the full file path
            file_path = os.path.join(folder_path, filename)
            
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Extract the required columns
            df_extracted = df[['XR', 'YR', 'TSS1', 'TSS2', 'TSS3']]
            
            # Append the extracted DataFrame to the list
            dataframes.append(df_extracted)

    # Concatenate all DataFrames into one
    merged_df = pd.concat(dataframes, ignore_index=True)

    # Initialize an empty list to store the reorganized data
    reorganized_data = []

    # Loop through the merged DataFrame and reorganize the data for coil 1
    for index, row in merged_df.iterrows():
        xr = row['XR']
        yr = row['YR']
        
        # Append XR, YR, and TSS1
        reorganized_data.append([xr, yr, row['TSS1']])
        
    # Loop through the merged DataFrame and reorganize the data for coil 2
    for index, row in merged_df.iterrows():
        xr = row['XR']
        yr = row['YR']
        
        # Append XR, YR, and TSS1
        reorganized_data.append([xr, yr, row['TSS2']])

    # Loop through the merged DataFrame and reorganize the data for coil 3
    for idenx, row in merged_df.iterrows():
        xr = row['XR']
        yr = row['YR']
        
        # Append XR, YR, and TSS1
        reorganized_data.append([xr, yr, row['TSS3']])

    # Create a new DataFrame from the reorganized data
    reorganized_df = pd.DataFrame(reorganized_data, columns=['XR', 'YR', 'TSS'])

    # Save the reorganized DataFrame to a new Excel file
    script_path = os.getcwd()
    output_file_path = os.path.join(script_path, output_file)
    reorganized_df.to_csv(output_file_path, index=False)

    print(f"Data has been merged and reorganized. Saved to {output_file_path}")

def findFolder() -> str:
    
    # Get the current working directory 
    script_path = os.getcwd()

    # Initialize the Tkinter window (root window)
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    root.call('wm', 'attributes', '.', '-topmost', True)  # Keep the file dialog on top

    # Open the file dialog and let the user select a folder, starting in the script's directory
    folder_path = filedialog.askdirectory(
        title="Select a folder",
        initialdir=script_path  # Start in the script's directory
    )

    # Check if a folder was selected
    if folder_path:
        folder_name = os.path.basename(folder_path)  # Get the base name of the folder
        print(f"Selected folder: ", folder_name)  # Print the folder name
        print(f"Full path of the selected folder: ", folder_path)  # Print the full path
        return folder_path  # Return the full path of the selected folder
    else:
        print("No folder selected.")  # Print error if no folder was selected
        return -1  # Return -1 to indicate failure

# Example usage:
selected_folder = findFolder()  # Opens the dialog and returns the selected file path
processFiles(selected_folder, "output.csv")