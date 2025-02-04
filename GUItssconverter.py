import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt

def plotCSVData(file_path: str):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Check if the required columns are present
    if not {'XR', 'YR', 'TSS'}.issubset(df.columns):
        print("CSV file does not contain the required columns: 'XR', 'YR', 'TSS'")
        return
    
    # Create a scatter plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df['XR'], df['YR'], c=df['TSS'], cmap='viridis', marker='o')
    plt.colorbar(scatter, label='TSS')
    plt.xlabel('XR')
    plt.ylabel('YR')
    plt.title('Scatter plot of XR, YR, and TSS')
    plt.grid(True)
    plt.show()

def plotCSVCoils(file_path: str, xr_col: str, yr_col: str, tss1_col: str, tss2_col: str, tss3_col: str):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Check if the required columns are present
    if not {xr_col, yr_col, tss1_col, tss2_col, tss3_col}.issubset(df.columns):
        print(f"CSV file does not contain the required columns: {xr_col}, {yr_col}, {tss1_col}, {tss2_col}, {tss3_col}")
        return
    
    # Create scatter plots for each TSS coil
    for tss in [tss1_col, tss2_col, tss3_col]:
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(df[xr_col], df[yr_col], c=df[tss], cmap='viridis', marker='o')
        plt.colorbar(scatter, label=tss)
        plt.xlabel(xr_col)
        plt.ylabel(yr_col)
        plt.title(f'Scatter plot of {xr_col}, {yr_col}, and {tss}')
        plt.grid(True)
    plt.show()

def processFiles(folder_path, xr_col, yr_col, tss1_col, tss2_col, tss3_col, output_file):
    # Initialize an empty list to store DataFrames
    dataframes = []
    missing_headers_files = []

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            # Construct the full file path
            file_path = os.path.join(folder_path, filename)
            
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Check if the required columns are present
            if not {xr_col, yr_col, tss1_col, tss2_col, tss3_col}.issubset(df.columns):
                missing_headers_files.append(filename)
                continue
            
            # Extract the required columns
            df_extracted = df[[xr_col, yr_col, tss1_col, tss2_col, tss3_col]]
            
            # Append the extracted DataFrame to the list
            dataframes.append(df_extracted)

    if missing_headers_files:
        messagebox.showerror("Error", f"The following files are missing the required headers: {', '.join(missing_headers_files)}")
        return

    # Concatenate all DataFrames into one
    merged_df = pd.concat(dataframes, ignore_index=True)

    # Save the merged DataFrame to a new CSV file
    output_file_path = os.path.join(folder_path, output_file)
    merged_df.to_csv(output_file_path, index=False)

    print(f"Data has been merged and saved to {output_file_path}")

def select_folder():
    folder_path = filedialog.askdirectory(title="Select a folder")
    folder_entry.delete(0, tk.END)
    folder_entry.insert(0, folder_path)

def process():
    folder_path = folder_entry.get()
    xr_col = xr_entry.get()
    yr_col = yr_entry.get()
    tss1_col = tss1_entry.get()
    tss2_col = tss2_entry.get()
    tss3_col = tss3_entry.get()
    output_file = output_entry.get()

    if not folder_path or not xr_col or not yr_col or not tss1_col or not tss2_col or not tss3_col or not output_file:
        messagebox.showerror("Error", "All fields are required")
        return

    processFiles(folder_path, xr_col, yr_col, tss1_col, tss2_col, tss3_col, output_file)
    messagebox.showinfo("Success", f"Data has been merged and saved to {output_file}")

def show_plot():
    file_path = filedialog.askopenfilename(title="Select a CSV file", filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
    if not file_path:
        return

    xr_col = xr_entry.get()
    yr_col = yr_entry.get()
    tss1_col = tss1_entry.get()
    tss2_col = tss2_entry.get()
    tss3_col = tss3_entry.get()

    plotCSVCoils(file_path, xr_col, yr_col, tss1_col, tss2_col, tss3_col)

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
xr_entry.insert(0, "XR")  # Default value

tk.Label(root, text="Northing Column:", font=font).grid(row=2, column=0, padx=10, pady=5, sticky=tk.W)
yr_entry = tk.Entry(root, width=50, font=font)
yr_entry.grid(row=2, column=1, padx=10, pady=5)
yr_entry.insert(0, "YR")  # Default value

tk.Label(root, text="Coil 1 Column:", font=font).grid(row=3, column=0, padx=10, pady=5, sticky=tk.W)
tss1_entry = tk.Entry(root, width=50, font=font)
tss1_entry.grid(row=3, column=1, padx=10, pady=5)
tss1_entry.insert(0, "TSS1")  # Default value

tk.Label(root, text="Coil 2 Column:", font=font).grid(row=4, column=0, padx=10, pady=5, sticky=tk.W)
tss2_entry = tk.Entry(root, width=50, font=font)
tss2_entry.grid(row=4, column=1, padx=10, pady=5)
tss2_entry.insert(0, "TSS2")  # Default value

tk.Label(root, text="Coil 3 Column:", font=font).grid(row=5, column=0, padx=10, pady=5, sticky=tk.W)
tss3_entry = tk.Entry(root, width=50, font=font)
tss3_entry.grid(row=5, column=1, padx=10, pady=5)
tss3_entry.insert(0, "TSS3")  # Default value

tk.Label(root, text="Output File Name:", font=font).grid(row=6, column=0, padx=10, pady=5, sticky=tk.W)
output_entry = tk.Entry(root, width=50, font=font)
output_entry.grid(row=6, column=1, padx=10, pady=5)
output_entry.insert(0, "combined.csv")  # Default value

tk.Button(root, text="Process Files", command=process, font=font).grid(row=7, column=0, columnspan=3, pady=10)
tk.Button(root, text="Show Plot", command=show_plot, font=font).grid(row=8, column=0, columnspan=3, pady=10)

# Run the main loop
root.mainloop()