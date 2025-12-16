"""
TSSAltitudeFixer - DVL Altitude Correction Tool
This script corrects DVL Altitude values by comparing SQL Altitude data 
with depth values from VisualSoft navigation exports.
"""

import os
import sys
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json

# Import altitude extraction module
from altitudeFromSQL import extract_altitude_from_sql

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Script version
SCRIPT_VERSION = "TSSAltitudeFixer v2.0"

# Script directory (where the script is located)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# File suffix conventions for navigation files
COIL_PORT_SUFFIX = "_Coil_port.nav"
COIL_CENTER_SUFFIX = "_Coil_center.nav"
COIL_STBD_SUFFIX = "_Coil_stbd.nav"
CRP_SUFFIX = "_CRP.nav"

# WFM Export XML template filename
WFM_EXPORT_XML = "WFM_DepthExport.xml"

# Config folder and settings file
CONFIG_FOLDER = os.path.join(SCRIPT_DIR, "config")
SETTINGS_FILE = os.path.join(CONFIG_FOLDER, "settings.json")

# Maximum time difference in seconds for matching
MAX_TIME_DIFF_SEC = 0.25

# Global list to track open plot windows
open_plot_windows = []


def load_settings():
    """Load settings from the config/settings.json file."""
    default_settings = {
        'sql_db_path': '',
        'block_ids': '',
        'z_dvl_offset': '0.0',
        'nav_update_path': '',
        'sql_server_name': 'RS-GOEL-PVE03',
        'folder_filter': '04_NAVISCAN'
    }
    
    if not os.path.exists(SETTINGS_FILE):
        return default_settings
    
    try:
        with open(SETTINGS_FILE, 'r') as f:
            settings = json.load(f)
        # Merge with defaults to handle missing keys
        for key in default_settings:
            if key not in settings:
                settings[key] = default_settings[key]
        return settings
    except Exception as e:
        logging.warning(f"Could not load settings: {e}")
        return default_settings


def save_settings(sql_db_path, block_ids, z_dvl_offset, nav_update_path='', sql_server_name='RS-GOEL-PVE03', folder_filter='04_NAVISCAN'):
    """Save settings to the config/settings.json file."""
    # Create config folder if it doesn't exist
    if not os.path.exists(CONFIG_FOLDER):
        try:
            os.makedirs(CONFIG_FOLDER)
            logging.info(f"Created config folder: {CONFIG_FOLDER}")
        except Exception as e:
            logging.warning(f"Could not create config folder: {e}")
            return False
    
    settings = {
        'sql_db_path': sql_db_path,
        'block_ids': block_ids,
        'z_dvl_offset': z_dvl_offset,
        'nav_update_path': nav_update_path,
        'sql_server_name': sql_server_name,
        'folder_filter': folder_filter
    }
    
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=4)
        logging.info("Settings saved successfully")
        return True
    except Exception as e:
        logging.warning(f"Could not save settings: {e}")
        return False


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


def run_wfm_depth_export(block_ids_str, output_folder, output_text):
    """
    Run the WFM Depth Export script with the specified block IDs.
    Exports files to the 'navdepth' folder in the specified output folder.
    """
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
    xml_path = os.path.join(SCRIPT_DIR, WFM_EXPORT_XML)
    
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
        
        # --- DYNAMICALLY GENERATE EXPORT TASKS FOR EACH BLOCK ID ---
        start_marker = '<FindBlocksTask po="4" name="Iterate Blocks in Job" output="Block">'
        end_marker = '</FindBlocksTask>'
        
        start_idx = xml_content.find(start_marker)
        end_idx = xml_content.find(end_marker)
        
        if start_idx != -1 and end_idx != -1:
            # Extract the template for a single block export
            template_start = xml_content.find('<GroupTask po="1" name="Export {Block.name}">', start_idx)
            last_group_task_end = xml_content.rfind('</GroupTask>', start_idx, end_idx) + len('</GroupTask>')
            
            block_template = xml_content[template_start:last_group_task_end]
            
            # Generate the new XML block for each block ID
            generated_tasks = []
            for i, bid in enumerate(block_ids):
                task_xml = block_template
                
                # Replace placeholders
                task_xml = task_xml.replace('{Block.id}', str(bid))
                task_xml = task_xml.replace('{Block.name}', f'Block_{bid}')
                
                # Update the 'po' (process order) attribute
                task_xml = task_xml.replace('po="1" name="Export', f'po="{i+4}" name="Export')
                
                generated_tasks.append(task_xml)
            
            # Join all generated tasks
            new_content_block = '\n'.join(generated_tasks)
            
            # Replace the entire FindBlocksTask block with the new content
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
        
        output_text.insert(tk.END, f"\nStarting WFM Depth Export for Block IDs: {block_ids}\n")
        output_text.insert(tk.END, f"Output folder: {navdepth_folder}\n")
        output_text.update()
        
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
        
        # Keep relevant columns
        relevant_cols = ['ID', 'Name', 'Folder', 'Time', 'RelTime', 'Depth', 'Altitude']
        existing_cols = [col for col in relevant_cols if col in df.columns]
        df = df[existing_cols].copy()
        df.rename(columns={'Altitude': 'CRP_Altitude', 'Depth': 'SQL_Depth'}, inplace=True)
        
        # Convert Name to string for matching with nav file names
        if 'Name' in df.columns:
            df['Name'] = df['Name'].astype(str)
        
        # Parse Time column to datetime
        if 'Time' in df.columns:
            df['DateTime'] = pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S.%f')
            df = df.sort_values(by='DateTime')
        
        logging.info(f"Successfully read SQL Altitude CSV: {len(df)} rows")
        return df
    except Exception as e:
        logging.error(f"Error reading SQL Altitude CSV: {e}")
        return pd.DataFrame()


def read_nav_file_with_time(file_path):
    """
    Read a VisualSoft .nav file and extract the 'DEPTH' column with time parsing.
    Returns a DataFrame with depth data and parsed datetime.
    """
    if not os.path.exists(file_path):
        logging.warning(f"Nav file not found: {file_path}")
        return pd.DataFrame()
    
    try:
        # Nav files are typically comma-separated
        df = pd.read_csv(file_path, delimiter=',')
        
        # Normalize column names (handle various header formats) - convert to uppercase for consistency
        df.columns = df.columns.str.strip().str.upper()
        
        if 'DEPTH' not in df.columns:
            logging.warning(f"'DEPTH' column not found in {file_path}")
            logging.info(f"Available columns: {df.columns.tolist()}")
            return pd.DataFrame()
        
        # Parse Date and Time columns to datetime
        # Handle different date formats: YYYYMMDD or DD/MM/YYYY
        date_col = 'DATE' if 'DATE' in df.columns else None
        time_col = None
        if 'TIME' in df.columns:
            time_col = 'TIME'
        elif 'HH:MM:SS.SSS' in df.columns:
            time_col = 'HH:MM:SS.SSS'
        
        if date_col and time_col:
            # Try YYYYMMDD format first (nav export format)
            try:
                df['DateTime'] = pd.to_datetime(
                    df[date_col].astype(str) + ' ' + df[time_col].astype(str), 
                    format='%Y%m%d %H:%M:%S.%f'
                )
            except:
                # Try DD/MM/YYYY format (alternative format)
                try:
                    df['DateTime'] = pd.to_datetime(
                        df[date_col].astype(str) + ' ' + df[time_col].astype(str), 
                        format='%d/%m/%Y %H:%M:%S.%f'
                    )
                except Exception as e:
                    logging.warning(f"Could not parse datetime in {file_path}: {e}")
                    return pd.DataFrame()
            
            df = df.sort_values(by='DateTime')
            # Keep Date column in a readable format
            df['Date'] = df['DateTime'].dt.strftime('%d/%m/%Y')
        
        # Extract filename for block identification
        df['Source_File'] = os.path.basename(file_path)
        
        logging.info(f"Successfully read nav file: {os.path.basename(file_path)} - {len(df)} rows")
        return df
    except Exception as e:
        logging.error(f"Error reading nav file {file_path}: {e}")
        return pd.DataFrame()


def read_nav_file(file_path):
    """
    Read a VisualSoft .nav file and extract the 'DEPTH' column.
    Returns a DataFrame with depth data.
    """
    if not os.path.exists(file_path):
        logging.warning(f"Nav file not found: {file_path}")
        return pd.DataFrame()
    
    try:
        # Nav files are typically comma-separated
        df = pd.read_csv(file_path, delimiter=',')
        
        # Normalize column names (handle various header formats)
        df.columns = df.columns.str.strip()
        
        if 'DEPTH' not in df.columns:
            # Try lowercase
            if 'Depth' in df.columns:
                df.rename(columns={'Depth': 'DEPTH'}, inplace=True)
            elif 'depth' in df.columns:
                df.rename(columns={'depth': 'DEPTH'}, inplace=True)
            else:
                logging.warning(f"'DEPTH' column not found in {file_path}")
                logging.info(f"Available columns: {df.columns.tolist()}")
                return pd.DataFrame()
        
        logging.info(f"Successfully read nav file: {os.path.basename(file_path)} - {len(df)} rows")
        return df
    except Exception as e:
        logging.error(f"Error reading nav file {file_path}: {e}")
        return pd.DataFrame()


def find_nav_files(navdepth_folder, suffix):
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


def read_all_nav_depths(navdepth_folder):
    """
    Read depth data from all navigation files in the navdepth folder.
    Returns a dictionary with DataFrames for each coil type.
    """
    results = {
        'CRP': pd.DataFrame(),
        'Coil_Port': pd.DataFrame(),
        'Coil_Center': pd.DataFrame(),
        'Coil_Stbd': pd.DataFrame()
    }
    
    # Read CRP files
    crp_files = find_nav_files(navdepth_folder, CRP_SUFFIX)
    crp_dfs = []
    for f in crp_files:
        df = read_nav_file(f)
        if not df.empty:
            df['Source_File'] = os.path.basename(f)
            crp_dfs.append(df)
    if crp_dfs:
        results['CRP'] = pd.concat(crp_dfs, ignore_index=True)
    
    # Read Coil Port files
    port_files = find_nav_files(navdepth_folder, COIL_PORT_SUFFIX)
    port_dfs = []
    for f in port_files:
        df = read_nav_file(f)
        if not df.empty:
            df['Source_File'] = os.path.basename(f)
            port_dfs.append(df)
    if port_dfs:
        results['Coil_Port'] = pd.concat(port_dfs, ignore_index=True)
    
    # Read Coil Center files
    center_files = find_nav_files(navdepth_folder, COIL_CENTER_SUFFIX)
    center_dfs = []
    for f in center_files:
        df = read_nav_file(f)
        if not df.empty:
            df['Source_File'] = os.path.basename(f)
            center_dfs.append(df)
    if center_dfs:
        results['Coil_Center'] = pd.concat(center_dfs, ignore_index=True)
    
    # Read Coil Stbd files
    stbd_files = find_nav_files(navdepth_folder, COIL_STBD_SUFFIX)
    stbd_dfs = []
    for f in stbd_files:
        df = read_nav_file(f)
        if not df.empty:
            df['Source_File'] = os.path.basename(f)
            stbd_dfs.append(df)
    if stbd_dfs:
        results['Coil_Stbd'] = pd.concat(stbd_dfs, ignore_index=True)
    
    return results


def get_block_name_from_file(filename, suffix):
    """
    Extract block name from a nav file name by removing the suffix.
    Example: '251128140017_CRP.nav' -> '251128140017'
    """
    return filename.replace(suffix, '')


def read_nav_files_by_block(navdepth_folder):
    """
    Read all nav files and organize them by block name.
    Returns a dictionary: {block_name: {'CRP': df, 'Coil_Port': df, 'Coil_Center': df, 'Coil_Stbd': df}}
    """
    blocks = {}
    
    # Find all CRP files first (these define the blocks)
    crp_files = find_nav_files(navdepth_folder, CRP_SUFFIX)
    
    for crp_file in crp_files:
        block_name = get_block_name_from_file(os.path.basename(crp_file), CRP_SUFFIX)
        
        # Initialize block entry
        blocks[block_name] = {
            'CRP': pd.DataFrame(),
            'Coil_Port': pd.DataFrame(),
            'Coil_Center': pd.DataFrame(),
            'Coil_Stbd': pd.DataFrame()
        }
        
        # Read CRP file
        crp_df = read_nav_file_with_time(crp_file)
        if not crp_df.empty:
            blocks[block_name]['CRP'] = crp_df
        
        # Read corresponding Coil Port file
        port_file = os.path.join(navdepth_folder, block_name + COIL_PORT_SUFFIX)
        if os.path.exists(port_file):
            port_df = read_nav_file_with_time(port_file)
            if not port_df.empty:
                blocks[block_name]['Coil_Port'] = port_df
        
        # Read corresponding Coil Center file
        center_file = os.path.join(navdepth_folder, block_name + COIL_CENTER_SUFFIX)
        if os.path.exists(center_file):
            center_df = read_nav_file_with_time(center_file)
            if not center_df.empty:
                blocks[block_name]['Coil_Center'] = center_df
        
        # Read corresponding Coil Stbd file
        stbd_file = os.path.join(navdepth_folder, block_name + COIL_STBD_SUFFIX)
        if os.path.exists(stbd_file):
            stbd_df = read_nav_file_with_time(stbd_file)
            if not stbd_df.empty:
                blocks[block_name]['Coil_Stbd'] = stbd_df
    
    logging.info(f"Found {len(blocks)} blocks in navdepth folder")
    return blocks


def match_altitude_with_nav(sql_altitude_df, nav_blocks, z_dvl_offset, output_text):
    """
    Match SQL Altitude data with nav depth data by time using CRP as reference.
    Uses the 'Name' column from SQL Altitude to match with block names from nav files.
    Creates an interleaved dataframe with one row per coil.
    Calculates Coil Altitude = (CRP Altitude + DVL offset) + CRP Depth - Coil Depth
    
    Returns a dictionary: {block_name: merged_dataframe}
    """
    results = {}
    
    if sql_altitude_df.empty or 'DateTime' not in sql_altitude_df.columns:
        logging.error("SQL Altitude DataFrame is empty or missing DateTime column")
        output_text.insert(tk.END, "  - ERROR: SQL Altitude DataFrame is empty or missing DateTime column\n")
        return results
    
    if 'Name' not in sql_altitude_df.columns:
        logging.error("SQL Altitude DataFrame is missing 'Name' column for block matching")
        output_text.insert(tk.END, "  - ERROR: SQL Altitude DataFrame is missing 'Name' column\n")
        return results
    
    for block_name, nav_data in nav_blocks.items():
        output_text.insert(tk.END, f"\nProcessing block: {block_name}\n")
        output_text.update()
        
        crp_df = nav_data['CRP']
        if crp_df.empty:
            output_text.insert(tk.END, f"  - Skipping: CRP data empty\n")
            continue
        
        if 'DateTime' not in crp_df.columns:
            output_text.insert(tk.END, f"  - Skipping: CRP data missing DateTime column\n")
            output_text.insert(tk.END, f"    Available columns: {crp_df.columns.tolist()}\n")
            continue
        
        # Filter SQL Altitude data for this specific block using the 'Name' column
        sql_block_df = sql_altitude_df[sql_altitude_df['Name'] == block_name].copy()
        
        if sql_block_df.empty:
            output_text.insert(tk.END, f"  - Warning: No matching SQL Altitude data found for block '{block_name}'\n")
            # Show available names in SQL for debugging
            available_names = sql_altitude_df['Name'].unique()[:5]
            output_text.insert(tk.END, f"    Available names in SQL (first 5): {list(available_names)}\n")
            continue
        
        output_text.insert(tk.END, f"  - Found {len(sql_block_df)} SQL Altitude records for this block\n")
        
        # Debug: Show time ranges
        output_text.insert(tk.END, f"  - Nav CRP time range: {crp_df['DateTime'].min()} to {crp_df['DateTime'].max()}\n")
        output_text.insert(tk.END, f"  - SQL Altitude time range: {sql_block_df['DateTime'].min()} to {sql_block_df['DateTime'].max()}\n")
        
        # Use CRP nav as reference for time matching with SQL Altitude
        crp_df = crp_df.sort_values(by='DateTime').reset_index(drop=True)
        sql_block_df_sorted = sql_block_df.sort_values(by='DateTime').reset_index(drop=True)
        
        # Merge SQL Altitude with CRP nav data using nearest time match
        # Use a wider tolerance since we're matching by block name already
        merged_crp = pd.merge_asof(
            crp_df,
            sql_block_df_sorted[['DateTime', 'CRP_Altitude']],
            on='DateTime',
            direction='nearest'
            # No tolerance - just find nearest match since we already filtered by block name
        )
        
        # Check if merge succeeded
        matched_count = merged_crp['CRP_Altitude'].notna().sum()
        output_text.insert(tk.END, f"  - Matched {matched_count} of {len(merged_crp)} records with SQL Altitude\n")
        
        if merged_crp['CRP_Altitude'].isna().all():
            output_text.insert(tk.END, f"  - Warning: No time-matched SQL Altitude data found for this block\n")
            continue
        
        # Rename CRP depth column
        merged_crp = merged_crp.rename(columns={'DEPTH': 'CRP_Depth'})
        
        # Ensure Date column exists
        if 'Date' not in merged_crp.columns:
            merged_crp['Date'] = merged_crp['DateTime'].dt.strftime('%d/%m/%Y')
        
        # Process each coil type
        coil_dfs = []
        coil_types = [
            ('Coil_Port', 'Port', 1),
            ('Coil_Center', 'Center', 2),
            ('Coil_Stbd', 'Stbd', 3)
        ]
        
        for coil_key, coil_name, coil_num in coil_types:
            coil_df = nav_data[coil_key]
            if coil_df.empty or 'DateTime' not in coil_df.columns:
                output_text.insert(tk.END, f"  - Skipping {coil_name}: data empty or missing DateTime\n")
                continue
            
            coil_df = coil_df.sort_values(by='DateTime').reset_index(drop=True)
            
            # Match coil data with CRP reference using nearest time
            merged_coil = pd.merge_asof(
                merged_crp[['DateTime', 'Date', 'CRP_Depth', 'CRP_Altitude']].copy(),
                coil_df[['DateTime', 'DEPTH']].rename(columns={'DEPTH': 'Coil_Depth'}),
                on='DateTime',
                direction='nearest',
                tolerance=pd.Timedelta(seconds=MAX_TIME_DIFF_SEC)
            )
            
            # Add coil identifier
            merged_coil['Coil'] = coil_num
            merged_coil['Coil_Name'] = coil_name
            
            # Calculate Coil Altitude: (CRP Altitude + DVL offset) + CRP Depth - Coil Depth
            # Round to 2 decimal places
            merged_coil['Coil_Altitude'] = ((merged_coil['CRP_Altitude'] + z_dvl_offset) + merged_coil['CRP_Depth'] - merged_coil['Coil_Depth']).round(2)
            
            # Apply DVL offset to CRP Altitude for output CSV
            merged_coil['CRP_Altitude'] = (merged_coil['CRP_Altitude'] + z_dvl_offset).round(2)
            
            # Add Time column from DateTime
            merged_coil['Time'] = merged_coil['DateTime'].dt.strftime('%H:%M:%S.%f').str[:-3]
            
            coil_dfs.append(merged_coil)
        
        if coil_dfs:
            # Interleave the dataframes (row 1 of each coil, then row 2 of each coil, etc.)
            # First, ensure all have the same length
            min_len = min(len(df) for df in coil_dfs)
            coil_dfs = [df.iloc[:min_len].reset_index(drop=True) for df in coil_dfs]
            
            # Interleave by concatenating and sorting by index
            interleaved_df = pd.concat(coil_dfs, axis=0).sort_index(kind='merge')
            interleaved_df = interleaved_df.reset_index(drop=True)
            
            # Reorder columns (exclude DateTime from output)
            column_order = ['Date', 'Time', 'Coil', 'Coil_Name', 'Coil_Depth', 'CRP_Altitude', 'CRP_Depth', 'Coil_Altitude']
            existing_cols = [col for col in column_order if col in interleaved_df.columns]
            remaining_cols = [col for col in interleaved_df.columns if col not in column_order and col != 'DateTime']
            interleaved_df = interleaved_df[existing_cols + remaining_cols]
            
            results[block_name] = interleaved_df
            output_text.insert(tk.END, f"  - Successfully processed: {len(interleaved_df)} rows\n")
        else:
            output_text.insert(tk.END, f"  - No coil data matched for this block\n")
    
    return results


def save_calculated_altitude_csvs(results, output_folder, output_text):
    """
    Save the calculated altitude dataframes to CSV files, one per block.
    """
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
            output_text.insert(tk.END, f"  - Saved: {os.path.basename(output_file)}\n")
            logging.info(f"Saved CSV: {output_file}")
        except Exception as e:
            output_text.insert(tk.END, f"  - Error saving {block_name}: {e}\n")
            logging.error(f"Error saving CSV {output_file}: {e}")
    
    return True


def process_dvl_correction(sql_db_path, z_dvl_offset, output_folder, output_text):
    """
    Main processing function for DVL altitude correction.
    Extracts altitude from NaviEdit SQL database, matches with nav depth data,
    calculates Coil Altitude, and saves CSV files per block.
    """
    output_text.delete(1.0, tk.END)
    output_text.insert(tk.END, f"=== {SCRIPT_VERSION} - DVL Altitude Correction ===\n\n")
    
    # Load settings to get SQL config
    settings = load_settings()
    sql_server_name = settings.get('sql_server_name', 'RS-GOEL-PVE03')
    folder_filter = settings.get('folder_filter', '04_NAVISCAN')
    
    # Validate inputs
    if not sql_db_path:
        messagebox.showerror("Error", "Please select NaviEdit SQL Database path.")
        return
    
    if not output_folder:
        messagebox.showerror("Error", "Please select an Output Folder (VisualSoft Nav CSV Path).")
        return
    
    try:
        z_offset = float(z_dvl_offset)
    except ValueError:
        messagebox.showerror("Error", "Z DVL Offset must be a numeric value.")
        return
    
    output_text.insert(tk.END, f"Z DVL Offset: {z_offset} m\n")
    output_text.insert(tk.END, f"SQL Server: {sql_server_name}\n")
    output_text.insert(tk.END, f"Folder Filter: {folder_filter}\n")
    output_text.insert(tk.END, f"Formula: Coil_Altitude = (CRP_Altitude + {z_offset}) + CRP_Depth - Coil_Depth\n\n")
    
    # Step 1: Extract altitude data from SQL database
    output_text.insert(tk.END, "Step 1: Extracting altitude data from NaviEdit database...\n")
    output_text.update()
    
    # Generate the TSS_Altitude.csv in the output folder if available
    if output_folder and os.path.exists(output_folder):
        altitude_csv_path = os.path.join(output_folder, "TSS_Altitude.csv")
    else:
        altitude_csv_path = os.path.join(SCRIPT_DIR, "TSS_Altitude.csv")
    
    sql_altitude_df = extract_altitude_from_sql(sql_db_path, altitude_csv_path, folder_filter, sql_server_name)
    
    if sql_altitude_df.empty:
        messagebox.showerror("Error", "Failed to extract altitude data from database or no data found.")
        return
    
    output_text.insert(tk.END, f"  - Extracted {len(sql_altitude_df)} altitude records\n")
    output_text.insert(tk.END, f"  - Saved to: {altitude_csv_path}\n")
    
    # Now read the CSV as before (for compatibility with existing code)
    sql_altitude_df = read_sql_altitude_csv(altitude_csv_path)
    if sql_altitude_df.empty:
        messagebox.showerror("Error", "Failed to read generated altitude CSV.")
        return
    
    output_text.insert(tk.END, f"  - CRP Altitude records: {len(sql_altitude_df)}\n")
    
    if 'CRP_Altitude' in sql_altitude_df.columns:
        output_text.insert(tk.END, f"  - Altitude range: {sql_altitude_df['CRP_Altitude'].min():.3f} to {sql_altitude_df['CRP_Altitude'].max():.3f} m\n")
    
    # Show unique block names in SQL data
    if 'Name' in sql_altitude_df.columns:
        unique_names = sql_altitude_df['Name'].unique()
        output_text.insert(tk.END, f"  - Block names in SQL data: {len(unique_names)}\n")
        for name in unique_names[:10]:  # Show first 10
            output_text.insert(tk.END, f"    - {name}\n")
        if len(unique_names) > 10:
            output_text.insert(tk.END, f"    ... and {len(unique_names) - 10} more\n")
    
    # Construct navdepth folder path
    if output_folder and os.path.exists(output_folder):
        navdepth_folder = os.path.join(output_folder, "navdepth")
    else:
        navdepth_folder = os.path.join(SCRIPT_DIR, "navdepth")
        
    output_text.insert(tk.END, f"\nStep 2: Reading nav depth files from: {navdepth_folder}\n")
    output_text.update()
    
    if not os.path.exists(navdepth_folder):
        messagebox.showerror("Error", f"Navdepth folder not found: {navdepth_folder}\nPlease run 'Export Depth from NaviEdit' first.")
        return
    
    # Read nav files organized by block
    nav_blocks = read_nav_files_by_block(navdepth_folder)
    
    if not nav_blocks:
        messagebox.showerror("Error", "No nav files found in navdepth folder.")
        return
    
    output_text.insert(tk.END, f"  - Found {len(nav_blocks)} blocks in navdepth folder\n")
    for block_name in nav_blocks.keys():
        output_text.insert(tk.END, f"    - {block_name}\n")
    
    # Match and calculate altitude
    output_text.insert(tk.END, f"\nStep 3: Matching data and calculating Coil Altitude...\n")
    output_text.update()
    
    results = match_altitude_with_nav(sql_altitude_df, nav_blocks, z_offset, output_text)
    
    if not results:
        messagebox.showerror("Error", "No data could be matched. Check time synchronization between files.")
        return
    
    # Save CSV files
    output_text.insert(tk.END, f"\nStep 4: Saving CSV files to: {output_folder}\n")
    output_text.update()
    
    # Filter results based on blocks present in output_folder
    existing_blocks = set()
    if os.path.exists(output_folder):
        for filename in os.listdir(output_folder):
            # We only care about nav files to identify blocks
            if filename.lower().endswith('.csv') or filename.lower().endswith('.nav'):
                b_name, _ = get_file_type_and_block(filename)
                if b_name:
                    existing_blocks.add(b_name)
    
    if not existing_blocks:
        output_text.insert(tk.END, f"  - Warning: No existing block files found in {output_folder}. No CSVs will be saved.\n")
    
    # Filter results
    filtered_results = {k: v for k, v in results.items() if k in existing_blocks}
    
    skipped_count = len(results) - len(filtered_results)
    if skipped_count > 0:
        output_text.insert(tk.END, f"  - Skipped {skipped_count} blocks not found in output folder.\n")
    
    if not filtered_results:
        messagebox.showwarning("Warning", "No matching blocks found in output folder. Nothing saved.")
        return

    save_calculated_altitude_csvs(filtered_results, output_folder, output_text)
    
    # Summary statistics
    output_text.insert(tk.END, "\n=== Summary Statistics ===\n")
    
    for block_name, df in filtered_results.items():
        output_text.insert(tk.END, f"\nBlock: {block_name}\n")
        output_text.insert(tk.END, f"  - Total rows: {len(df)}\n")
        
        if 'Coil_Altitude' in df.columns:
            output_text.insert(tk.END, f"  - Coil Altitude: min={df['Coil_Altitude'].min():.3f}, max={df['Coil_Altitude'].max():.3f}, mean={df['Coil_Altitude'].mean():.3f} m\n")
        
        if 'CRP_Depth' in df.columns:
            output_text.insert(tk.END, f"  - CRP Depth: min={df['CRP_Depth'].min():.3f}, max={df['CRP_Depth'].max():.3f}, mean={df['CRP_Depth'].mean():.3f} m\n")
    
    output_text.insert(tk.END, f"\n=== Processing Complete ===\n")
    output_text.insert(tk.END, f"Output folder: {output_folder}\n")
    
    messagebox.showinfo("Success", f"DVL Altitude correction completed!\n\nCSV files saved to:\n{output_folder}")
    
    logging.info("DVL Altitude correction processing completed.")


def import_altitude_from_sql(sql_db_path, output_folder, output_text):
    """
    Import altitude data from NaviEdit SQL database and save to TSS_Altitude.csv.
    This is a standalone operation that can be run before the full DVL correction.
    """
    output_text.delete(1.0, tk.END)
    output_text.insert(tk.END, f"=== {SCRIPT_VERSION} - Import Altitude from SQL ===\n\n")
    
    # Validate inputs
    if not sql_db_path:
        messagebox.showerror("Error", "Please select NaviEdit SQL Database path.")
        return False
    
    if not os.path.exists(sql_db_path):
        messagebox.showerror("Error", f"Database file not found: {sql_db_path}")
        return False
    
    output_text.insert(tk.END, f"Database: {sql_db_path}\n")
    output_text.insert(tk.END, f"Extracting altitude data from '04_NAVISCAN' folder...\n")
    output_text.update()
    
    # Generate the TSS_Altitude.csv in the output folder if available
    if output_folder and os.path.exists(output_folder):
        altitude_csv_path = os.path.join(output_folder, "TSS_Altitude.csv")
    else:
        altitude_csv_path = os.path.join(SCRIPT_DIR, "TSS_Altitude.csv")
    
    try:
        sql_altitude_df = extract_altitude_from_sql(sql_db_path, altitude_csv_path, "04_NAVISCAN")
        
        if sql_altitude_df.empty:
            messagebox.showerror("Error", "Failed to extract altitude data from database or no data found.")
            return False
        
        output_text.insert(tk.END, f"\n=== Import Successful ===\n")
        output_text.insert(tk.END, f"Extracted {len(sql_altitude_df)} altitude records\n")
        output_text.insert(tk.END, f"Saved to: {altitude_csv_path}\n")
        
        # Show statistics
        if 'Altitude' in sql_altitude_df.columns:
            output_text.insert(tk.END, f"\nAltitude range: {sql_altitude_df['Altitude'].min():.3f} to {sql_altitude_df['Altitude'].max():.3f} m\n")
        
        if 'Name' in sql_altitude_df.columns:
            unique_names = sql_altitude_df['Name'].unique()
            output_text.insert(tk.END, f"\nBlocks found: {len(unique_names)}\n")
            for name in unique_names[:15]:  # Show first 15
                block_count = len(sql_altitude_df[sql_altitude_df['Name'] == name])
                output_text.insert(tk.END, f"  - {name} ({block_count} records)\n")
            if len(unique_names) > 15:
                output_text.insert(tk.END, f"  ... and {len(unique_names) - 15} more blocks\n")
        
        messagebox.showinfo("Success", f"Altitude data imported successfully!\n\n{len(sql_altitude_df)} records saved to:\n{altitude_csv_path}")
        return True
        
    except Exception as e:
        output_text.insert(tk.END, f"\nError: {e}\n")
        messagebox.showerror("Error", f"Failed to import altitude data: {e}")
        logging.error(f"Import altitude error: {e}")
        return False


def select_nav_export_folder(entry):
    """Open folder dialog and set the nav export path."""
    folder_path = filedialog.askdirectory(title="Select VisualSoft Nav Exports Folder")
    if folder_path:
        entry.delete(0, tk.END)
        entry.insert(0, folder_path)


def select_sql_database_file(entry):
    """Open file dialog and set the NaviEdit SQL Server database path."""
    file_path = filedialog.askopenfilename(
        title="Select NaviEdit SQL Server Database File",
        filetypes=[("SQL Server Database", "*.mdf"), ("All files", "*.*")]
    )
    if file_path:
        entry.delete(0, tk.END)
        entry.insert(0, file_path)


def plot_altitude_depth(source_folder, output_text):
    """
    Display a plot with CRP and Center Coil data.
    X-axis: Time
    Left Y-axis (inverted): Depth values (CRP and Center Coil)
    Right Y-axis (normal): Altitude values (CRP and Center Coil)
    
    The 0m altitude line aligns with the seabed depth (CRP_Depth + CRP_Altitude from first row).
    """
    if not source_folder or not os.path.exists(source_folder):
        messagebox.showerror("Error", f"Source folder not found: {source_folder}\nPlease run 'Process DVL Correction' first.")
        return
    
    csv_files = [f for f in os.listdir(source_folder) if f.endswith('_calculated_altitude.csv')]
    
    if not csv_files:
        messagebox.showerror("Error", "No calculated altitude CSV files found.\nPlease run 'Process DVL Correction' first.")
        return
    
    output_text.insert(tk.END, f"\nGenerating plots for {len(csv_files)} files...\n")
    
    for csv_file in csv_files:
        # Read the CSV file
        csv_path = os.path.join(source_folder, csv_file)
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            output_text.insert(tk.END, f"  - Error reading {csv_file}: {e}\n")
            continue
        
        # Parse Date and Time to create a datetime for plotting
        if 'Date' in df.columns and 'Time' in df.columns:
            df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S.%f')
        else:
            output_text.insert(tk.END, f"  - Skipping {csv_file}: missing Date or Time columns.\n")
            continue
        
        # Get block name for title
        block_name = csv_file.replace('_calculated_altitude.csv', '')
        
        # Get CRP data (from Center coil rows, since CRP values are repeated for all coils)
        center_data = df[df['Coil_Name'] == 'Center'].copy()
        
        if center_data.empty:
            output_text.insert(tk.END, f"  - Skipping {csv_file}: No Center Coil data found.\n")
            continue
        
        # Calculate seabed depth from first row (CRP_Depth + CRP_Altitude)
        # Note: CRP_Altitude in CSV already has offset applied, so we use it directly
        first_row = center_data.iloc[0]
        seabed_depth = first_row['CRP_Depth'] + first_row['CRP_Altitude']
        
        # Create the plot window
        plot_window = tk.Toplevel()
        plot_window.title(f"CRP & Center Coil - {block_name}")
        plot_window.geometry("1300x700")
        
        # Add to global list
        open_plot_windows.append(plot_window)
        
        # Create figure
        fig, ax1 = plt.subplots(figsize=(14, 7))
        
        # Colors: CRP in purple/orange tones, Coil in blue/cyan tones
        crp_depth_color = 'red'      # Purple for CRP Depth
        crp_alt_color = 'red'        # Orange for CRP Altitude
        coil_depth_color = 'blue'     # Blue for Coil Depth
        coil_alt_color = 'blue'       # Teal for Coil Altitude
        
        # Left Y-axis: Depth (inverted)
        ax1.set_xlabel('Time', fontsize=11)
        ax1.set_ylabel('Depth (m)', fontsize=11, color='#2c3e50')
        
        # Plot CRP Depth (thicker, dashed line)
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
        # Relationship: depth = seabed_depth - altitude
        virt_depth_from_alt_max = seabed_depth - alt_max  # Corresponds to highest altitude (shallowest depth)
        virt_depth_from_alt_min = seabed_depth - alt_min  # Corresponds to lowest altitude (deepest depth)
        
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
        
        # Plot CRP Altitude (thicker, dash-dot line)
        ax2.plot(center_data['DateTime'], center_data['CRP_Altitude'], 
                color=crp_alt_color, linestyle=':', linewidth=2.0, alpha=0.9,
                label='CRP Altitude')
        
        # Plot Center Coil Altitude (dotted line)
        ax2.plot(center_data['DateTime'], center_data['Coil_Altitude'], 
                color=coil_alt_color, linestyle=':', linewidth=2.0, alpha=0.9,
                label='Center Coil Altitude')
                
        # Set Altitude limits based on the Depth limits to maintain 1:1 scale
        # altitude = seabed_depth - depth
        plot_alt_min = seabed_depth - plot_depth_max  # Corresponds to bottom of plot
        plot_alt_max = seabed_depth - plot_depth_min  # Corresponds to top of plot
        
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
        
        # Embed in Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add close button
        ttk.Button(plot_window, text="Close", command=plot_window.destroy).pack(pady=10)
        
        output_text.insert(tk.END, f"  - Plot displayed for block: {block_name}\n")
        logging.info(f"Plot displayed for block: {block_name}")


def close_all_plots():
    """Close all open plot windows."""
    global open_plot_windows
    count = 0
    for window in open_plot_windows:
        try:
            window.destroy()
            count += 1
        except:
            pass
    open_plot_windows = []
    plt.close('all')
    logging.info(f"Closed {count} plot windows")


def get_file_type_and_block(filename):
    """
    Determine block name and coil type from filename.
    Returns (block_name, coil_type) or (None, None).
    coil_type is one of: 'Center', 'Port', 'Stbd', 'CRP'
    """
    # Check for suffixes (ignoring extension)
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
        # Handle '_manual' suffix in block name (e.g. 251126174143_manual)
        if block_name.lower().endswith('_manual'):
            block_name = block_name[:-7]
            
    return block_name, coil_type


def update_nav_files_batch(nav_update_path, output_text):
    """
    Update VisualSoft Nav CSV files in the specified folder and subfolders
    with calculated Altitude and Depth values.
    """
    output_text.delete(1.0, tk.END)
    output_text.insert(tk.END, f"=== {SCRIPT_VERSION} - Update Nav Files ===\n\n")
    
    if not nav_update_path or not os.path.exists(nav_update_path):
        messagebox.showerror("Error", "Please select a valid VisualSoft Nav CSV folder.")
        return
    
    # 1. Load all calculated data
    # First check the nav_update_path for calculated files (new workflow)
    calculated_folder = nav_update_path
    calc_files = [f for f in os.listdir(calculated_folder) if f.endswith('_calculated_altitude.csv')]
    
    # If not found, check the default folder (backward compatibility)
    if not calc_files:
        default_folder = os.path.join(SCRIPT_DIR, "calculated_coil_altitude")
        if os.path.exists(default_folder):
            calc_files_default = [f for f in os.listdir(default_folder) if f.endswith('_calculated_altitude.csv')]
            if calc_files_default:
                calculated_folder = default_folder
                calc_files = calc_files_default
    
    if not calc_files:
        messagebox.showerror("Error", f"No calculated altitude CSV files found in {nav_update_path} or default folder.")
        return
    
    output_text.insert(tk.END, f"Step 1: Loading calculated data from {calculated_folder}...\n")
    output_text.update()
    
    calculated_data = {} # Key: block_name, Value: DataFrame
        
    for f in calc_files:
        block_name = f.replace('_calculated_altitude.csv', '')
        try:
            df = pd.read_csv(os.path.join(calculated_folder, f))
            # Create DateTime column for matching
            if 'Date' in df.columns and 'Time' in df.columns:
                # Handle potential format issues
                df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S.%f', errors='coerce')
                # Drop rows with invalid datetime
                df = df.dropna(subset=['DateTime'])
                calculated_data[block_name] = df
                output_text.insert(tk.END, f"  - Loaded data for block {block_name} ({len(df)} rows)\n")
            else:
                output_text.insert(tk.END, f"  - Warning: Skipping {f} (missing Date/Time columns)\n")
        except Exception as e:
            output_text.insert(tk.END, f"  - Error loading {f}: {e}\n")
            
    output_text.insert(tk.END, f"Loaded data for {len(calculated_data)} blocks.\n\n")
    
    # 2. Walk through nav folder
    output_text.insert(tk.END, f"Step 2: Scanning folder: {nav_update_path}\n")
    output_text.update()
    
    files_updated = 0
    files_skipped = 0
    
    for root, dirs, files in os.walk(nav_update_path):
        for filename in files:
            if not filename.lower().endswith('.csv') and not filename.lower().endswith('.nav'):
                continue
                
            block_name, coil_type = get_file_type_and_block(filename)
            
            if not block_name or not coil_type:
                continue
                
            if block_name not in calculated_data:
                # output_text.insert(tk.END, f"  - Skipping {filename}: No calculated data for block {block_name}\n")
                continue
            
            file_path = os.path.join(root, filename)
            output_text.insert(tk.END, f"  - Updating: {filename} ({coil_type})... ")
            output_text.update()
            
            try:
                # Read target file
                # Try to detect delimiter (VisualSoft often uses comma)
                try:
                    target_df = pd.read_csv(file_path, sep=',')
                    if len(target_df.columns) < 2: # Try semicolon if comma failed to split
                        target_df = pd.read_csv(file_path, sep=';')
                except:
                    target_df = pd.read_csv(file_path, sep=None, engine='python')
                
                # Normalize columns
                original_columns = target_df.columns.tolist()
                target_df.columns = target_df.columns.str.strip()
                
                if 'Alt' not in target_df.columns or 'Depth' not in target_df.columns:
                    output_text.insert(tk.END, "Skipped (missing Alt/Depth columns)\n")
                    files_skipped += 1
                    continue
                
                # Parse DateTime in target file
                # VisualSoft format usually: Date (YYYYMMDD) Time (HH:MM:SS.sss)
                # Or Date (DD/MM/YYYY)
                if 'Date' in target_df.columns and 'Time' in target_df.columns:
                    try:
                        # Try standard format first
                        target_df['DateTime'] = pd.to_datetime(target_df['Date'].astype(str) + ' ' + target_df['Time'].astype(str), format='%Y%m%d %H:%M:%S.%f', errors='coerce')
                        
                        # If many NaT, try alternative format
                        if target_df['DateTime'].isna().sum() > len(target_df) * 0.5:
                             target_df['DateTime'] = pd.to_datetime(target_df['Date'].astype(str) + ' ' + target_df['Time'].astype(str), format='%d/%m/%Y %H:%M:%S.%f', errors='coerce')
                    except:
                         target_df['DateTime'] = pd.to_datetime(target_df['Date'].astype(str) + ' ' + target_df['Time'].astype(str), errors='coerce')
                else:
                    output_text.insert(tk.END, "Skipped (missing Date/Time columns)\n")
                    files_skipped += 1
                    continue
                
                # Get calculated data for this block
                calc_df = calculated_data[block_name]
                
                # Filter calculated data for this coil type
                if coil_type == 'CRP':
                    # For CRP, we can use any row for a given timestamp, as CRP data is repeated
                    # But we should probably just take the CRP columns
                    source_df = calc_df[['DateTime', 'CRP_Altitude', 'CRP_Depth']].copy()
                    source_df = source_df.rename(columns={'CRP_Altitude': 'New_Alt', 'CRP_Depth': 'New_Depth'})
                    # Drop duplicates on DateTime to avoid merge explosion
                    source_df = source_df.drop_duplicates(subset=['DateTime'])
                else:
                    # For Coils, filter by Coil_Name
                    source_df = calc_df[calc_df['Coil_Name'] == coil_type][['DateTime', 'Coil_Altitude', 'Coil_Depth']].copy()
                    source_df = source_df.rename(columns={'Coil_Altitude': 'New_Alt', 'Coil_Depth': 'New_Depth'})
                
                if source_df.empty:
                    output_text.insert(tk.END, f"Skipped (no calculated data for {coil_type})\n")
                    files_skipped += 1
                    continue
                
                # Sort both by time
                target_df = target_df.sort_values(by='DateTime').reset_index(drop=True)
                source_df = source_df.sort_values(by='DateTime').reset_index(drop=True)
                
                # Merge to update values
                # We want to keep all rows from target_df
                merged = pd.merge_asof(
                    target_df,
                    source_df[['DateTime', 'New_Alt', 'New_Depth']],
                    on='DateTime',
                    direction='nearest',
                    tolerance=pd.Timedelta(seconds=0.5)
                )
                
                # Update values where we have a match
                mask = merged['New_Alt'].notna()
                
                # Update Alt and Depth
                # Note: VisualSoft might expect specific column names. We assume 'Alt' and 'Depth' exist as checked above.
                target_df.loc[mask, 'Alt'] = merged.loc[mask, 'New_Alt']
                target_df.loc[mask, 'Depth'] = merged.loc[mask, 'New_Depth']
                
                # Restore original column names if needed (simplified here)
                # Save back to file
                target_df.drop(columns=['DateTime'], inplace=True)
                
                # Use original separator if possible, default to comma
                target_df.to_csv(file_path, index=False)
                
                output_text.insert(tk.END, f"Done ({mask.sum()} rows updated)\n")
                files_updated += 1
                    
            except Exception as e:
                output_text.insert(tk.END, f"Error: {e}\n")
                files_skipped += 1

    output_text.insert(tk.END, f"\n=== Update Complete ===\n")
    output_text.insert(tk.END, f"Files updated: {files_updated}\n")
    output_text.insert(tk.END, f"Files skipped/error: {files_skipped}\n")
    messagebox.showinfo("Success", f"Nav files update completed.\nUpdated {files_updated} files.")


def select_nav_update_folder(entry):
    """Open folder dialog and set the nav update path."""
    folder_path = filedialog.askdirectory(title="Select VisualSoft Nav CSV Folder to Update")
    if folder_path:
        entry.delete(0, tk.END)
        entry.insert(0, folder_path)


def open_sql_settings_dialog(root):
    """Open a dialog to configure SQL Server settings."""
    settings = load_settings()
    
    dialog = tk.Toplevel(root)
    dialog.title("SQL Settings")
    dialog.geometry("500x250")
    dialog.transient(root)
    dialog.grab_set()
    
    # Center the dialog
    root_x = root.winfo_x()
    root_y = root.winfo_y()
    root_w = root.winfo_width()
    root_h = root.winfo_height()
    x = root_x + (root_w // 2) - 250
    y = root_y + (root_h // 2) - 125
    dialog.geometry(f"+{x}+{y}")
    
    frame = ttk.Frame(dialog, padding="20")
    frame.pack(fill=tk.BOTH, expand=True)
    
    # SQL Server Name
    ttk.Label(frame, text="SQL Server Name:").grid(row=0, column=0, sticky=tk.W, pady=5)
    server_entry = ttk.Entry(frame, width=40)
    server_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
    server_entry.insert(0, settings.get('sql_server_name', 'RS-GOEL-PVE03'))
    ttk.Label(frame, text="(e.g. RS-GOEL-PVE03, localhost, .)", font=("Segoe UI", 8, "italic")).grid(row=1, column=1, sticky=tk.W, padx=5)
    
    # Folder Filter
    ttk.Label(frame, text="Folder Filter:").grid(row=2, column=0, sticky=tk.W, pady=5)
    filter_entry = ttk.Entry(frame, width=40)
    filter_entry.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
    filter_entry.insert(0, settings.get('folder_filter', '04_NAVISCAN'))
    ttk.Label(frame, text="(Parent folder name to extract from)", font=("Segoe UI", 8, "italic")).grid(row=3, column=1, sticky=tk.W, padx=5)
    
    def save_and_close():
        # We need to read the other settings from the main window entries, but they are not accessible here directly.
        # However, load_settings() gets the saved ones.
        # The main window save_settings() overwrites everything.
        # So we should update the settings file directly, but preserve other values.
        
        current_settings = load_settings()
        current_settings['sql_server_name'] = server_entry.get()
        current_settings['folder_filter'] = filter_entry.get()
        
        # Save back
        try:
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(current_settings, f, indent=4)
            logging.info("SQL Settings saved successfully")
            messagebox.showinfo("Success", "SQL Settings saved.")
            dialog.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Could not save settings: {e}")
    
    ttk.Button(frame, text="Save", command=save_and_close).grid(row=4, column=1, pady=20, sticky=tk.E)
    
    dialog.wait_window()


def create_gui():
    """Create and run the main GUI."""
    # Create the main window
    root = tk.Tk()
    root.title(SCRIPT_VERSION)
    root.geometry("1050x800")
    
    # Style configuration
    style = ttk.Style()
    style.theme_use('clam')
    
    # Define colors
    bg_color = "#f5f6f7"
    accent_color = "#0078d7"
    text_color = "#333333"
    entry_bg = "#ffffff"
    
    root.configure(bg=bg_color)
    
    style.configure(".", background=bg_color, foreground=text_color, font=("Segoe UI", 10))
    style.configure("TLabel", background=bg_color, foreground=text_color)
    style.configure("TButton", font=("Segoe UI", 10, "bold"))
    style.configure("Big.TButton", font=("Segoe UI", 11, "bold"))
    style.configure("TEntry", fieldbackground=entry_bg)
    style.configure("TLabelframe", background=bg_color, foreground=accent_color)
    style.configure("TLabelframe.Label", background=bg_color, foreground=accent_color, font=("Segoe UI", 11, "bold"))
    
    # Main container
    main_frame = ttk.Frame(root, padding="10")
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # --- Input Settings Frame ---
    input_frame = ttk.LabelFrame(main_frame, text="Input Settings", padding="10")
    input_frame.pack(fill=tk.X, padx=5, pady=5)
    
    # NaviEdit SQL Database Path
    ttk.Label(input_frame, text="NaviEdit SQL Database:").grid(row=0, column=0, sticky=tk.W, pady=2)
    sql_db_entry = ttk.Entry(input_frame, width=60)
    sql_db_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
    ttk.Button(input_frame, text="Browse", command=lambda: select_sql_database_file(sql_db_entry)).grid(row=0, column=2, pady=2)
    
    # Coil Depth Export Block ID
    ttk.Label(input_frame, text="Block IDs (e.g. 100-105, 107):").grid(row=1, column=0, sticky=tk.W, pady=2)
    block_id_entry = ttk.Entry(input_frame, width=60)
    block_id_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=2)
    
    # Z DVL Offset
    ttk.Label(input_frame, text="Z DVL Offset (m):").grid(row=2, column=0, sticky=tk.W, pady=2)
    z_offset_entry = ttk.Entry(input_frame, width=20)
    z_offset_entry.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
    z_offset_entry.insert(0, "0.0")
    
    # VisualSoft Nav CSV Update Path
    ttk.Label(input_frame, text="VisualSoft Nav CSV Path:").grid(row=3, column=0, sticky=tk.W, pady=2)
    nav_update_entry = ttk.Entry(input_frame, width=60)
    nav_update_entry.grid(row=3, column=1, sticky="ew", padx=5, pady=2)
    ttk.Button(input_frame, text="Browse", command=lambda: select_nav_update_folder(nav_update_entry)).grid(row=3, column=2, pady=2)
    
    # Info label about file locations
    ttk.Label(input_frame, text="Note: 'navdepth' folder and 'TSS_Altitude.csv' will be created in the selected VisualSoft Nav CSV Path.", font=("Segoe UI", 8, "italic")).grid(row=4, column=0, columnspan=3, sticky=tk.W, pady=(10, 2))
    
    input_frame.columnconfigure(1, weight=1)
    
    # --- Actions Frame ---
    actions_frame = ttk.LabelFrame(main_frame, text="Actions", padding="10")
    actions_frame.pack(fill=tk.X, padx=5, pady=5)
    
    # Output text area
    output_frame = ttk.LabelFrame(main_frame, text="Output", padding="10")
    output_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    output_text = tk.Text(output_frame, wrap=tk.WORD, font=("Consolas", 9))
    output_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
    
    scrollbar = ttk.Scrollbar(output_frame, orient=tk.VERTICAL, command=output_text.yview)
    scrollbar.pack(fill=tk.Y, side=tk.RIGHT)
    output_text.configure(yscrollcommand=scrollbar.set)
    
    # Helper to save settings from main window
    def save_current_settings():
        # We need to preserve sql_server_name and folder_filter which are not in the main UI
        current = load_settings()
        save_settings(
            sql_db_entry.get(), 
            block_id_entry.get(), 
            z_offset_entry.get(), 
            nav_update_entry.get(),
            current.get('sql_server_name', 'RS-GOEL-PVE03'),
            current.get('folder_filter', '04_NAVISCAN')
        )

    # WFM Export button
    def run_wfm_export():
        block_ids_str = block_id_entry.get()
        if not block_ids_str:
            messagebox.showerror("Error", "Please enter Block IDs.")
            return
        save_current_settings()
        output_text.delete(1.0, tk.END)
        output_text.insert(tk.END, f"=== {SCRIPT_VERSION} - WFM Depth Export ===\n\n")
        run_wfm_depth_export(block_ids_str, nav_update_entry.get(), output_text)
    
    # Process button
    def run_process():
        save_current_settings()
        process_dvl_correction(
            sql_db_entry.get(),
            z_offset_entry.get(),
            nav_update_entry.get(),
            output_text
        )
    
    # Plot button callback
    def run_plot():
        plot_altitude_depth(nav_update_entry.get(), output_text)
        
    # Update Nav Files button callback
    def run_update_nav():
        save_current_settings()
        update_nav_files_batch(nav_update_entry.get(), output_text)
        
    # Open SQL Settings
    def run_sql_settings():
        open_sql_settings_dialog(root)

    # --- Main Row (Big Buttons) ---
    main_row_frame = ttk.Frame(actions_frame)
    main_row_frame.pack(fill=tk.X, pady=5)
    
    # 1. Export Depth from NE
    btn_export = ttk.Button(main_row_frame, text="1. Export Depth from NE", command=run_wfm_export, style="Big.TButton")
    btn_export.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
    
    # 2. Calculate TSS Altitude
    btn_calc = ttk.Button(main_row_frame, text="2. Calculate TSS Altitude", command=run_process, style="Big.TButton")
    btn_calc.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
    
    # 3. Update VisualSoft Navigation
    btn_update = ttk.Button(main_row_frame, text="3. Update VisualSoft Navigation", command=run_update_nav, style="Big.TButton")
    btn_update.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
    
    # --- Secondary Row (Normal Buttons) ---
    sec_row_frame = ttk.Frame(actions_frame)
    sec_row_frame.pack(fill=tk.X, pady=5)
    
    # Settings SQL
    ttk.Button(sec_row_frame, text="Settings SQL", command=run_sql_settings).pack(side=tk.LEFT, padx=5, pady=5)
    
    # View Plots
    ttk.Button(sec_row_frame, text="View Plots", command=run_plot).pack(side=tk.LEFT, padx=5, pady=5)
    
    # Close All Plots
    ttk.Button(sec_row_frame, text="Close All Plots", command=close_all_plots).pack(side=tk.LEFT, padx=5, pady=5)
    
    # Load saved settings
    settings = load_settings()
    
    # Apply saved settings
    if settings['sql_db_path']:
        sql_db_entry.insert(0, settings['sql_db_path'])
    
    if settings['block_ids']:
        block_id_entry.insert(0, settings['block_ids'])
    
    # Update Z offset (clear default first, then insert saved value)
    if settings['z_dvl_offset']:
        z_offset_entry.delete(0, tk.END)
        z_offset_entry.insert(0, settings['z_dvl_offset'])
        
    if settings['nav_update_path']:
        nav_update_entry.insert(0, settings['nav_update_path'])
    
    # Save settings on window close
    def on_closing():
        save_current_settings()
        plt.close('all')
        root.destroy()
        sys.exit()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    logging.info(f"{SCRIPT_VERSION} started.")
    
    # Run the main loop
    root.mainloop()


if __name__ == "__main__":
    create_gui()
