"""
altitudeFromSQL - Extract TSS Altitude data from NaviEdit SQL Server database

This module reads altitude data from the NaviEdit SQL Server database (.mdf file)
and exports it to a CSV file compatible with the TSSAltitudeFixer tool.
"""

import os
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def try_connect_to_mdf(mdf_path, server_name=None):
    """
    Try different methods to connect to the MDF file.
    Checks if the file is attached to a local SQL Server instance first.
    """
    import pyodbc
    
    abs_path = os.path.abspath(mdf_path)
    
    # 1. Try to find the database on known servers (including the one from the user's image)
    servers_to_check = ["RS-GOEL-PVE03", "localhost", "(local)", "."]
    
    # Add user provided server if available
    if server_name and server_name not in servers_to_check:
        servers_to_check.insert(0, server_name)
    
    # Determine best available driver
    drivers = [d for d in pyodbc.drivers() if "SQL Server" in d]
    best_driver = "SQL Server" # Fallback
    
    # Preference order
    preferred_drivers = [
        "ODBC Driver 18 for SQL Server",
        "ODBC Driver 17 for SQL Server",
        "SQL Server Native Client 11.0",
        "ODBC Driver 13 for SQL Server",
        "SQL Server"
    ]
    
    for pd in preferred_drivers:
        if pd in drivers:
            best_driver = pd
            break
            
    logging.info(f"Using ODBC Driver: {best_driver}")

    # Try to find if the file is already attached to a running server
    for server in servers_to_check:
        try:
            # Connect to master
            conn_str = f"Driver={{{best_driver}}};Server={server};Database=master;Trusted_Connection=yes;"
            if "Driver 18" in best_driver:
                conn_str += "TrustServerCertificate=yes;"
                
            # logging.info(f"Checking server: {server}")
            with pyodbc.connect(conn_str, timeout=2) as conn:
                cursor = conn.cursor()
                # Check if file exists in sys.master_files
                # We check for the filename at the end of the path to be more robust against drive mapping differences
                filename = os.path.basename(abs_path)
                cursor.execute("SELECT DB_NAME(database_id), physical_name FROM sys.master_files WHERE physical_name LIKE ?", (f"%{filename}",))
                rows = cursor.fetchall()
                
                target_db = None
                for row in rows:
                    db_name, phys_name = row
                    # If exact match or just filename match (if we want to be loose)
                    # Let's try to be reasonably sure it's the right file
                    if os.path.abspath(phys_name).lower() == abs_path.lower():
                        target_db = db_name
                        break
                    if os.path.basename(phys_name).lower() == filename.lower():
                        # Fallback: if filename matches, assume it's the one (user selected it)
                        target_db = db_name
                
                if target_db:
                    logging.info(f"Found file attached as database '{target_db}' on {server}")
                    # Connect to the specific database
                    db_conn_str = f"Driver={{{best_driver}}};Server={server};Database={target_db};Trusted_Connection=yes;"
                    if "Driver 18" in best_driver:
                        db_conn_str += "TrustServerCertificate=yes;"
                    return pyodbc.connect(db_conn_str)
        except Exception:
            pass

    # 2. If not found attached, try LocalDB AttachDbFilename
    connection_templates = [
        (
            r"Driver={SQL Server Native Client 11.0};"
            r"Server=(LocalDB)\MSSQLLocalDB;"
            f"AttachDbFilename={abs_path};"
            r"Trusted_Connection=yes;"
        ),
        (
            r"Driver={ODBC Driver 17 for SQL Server};"
            r"Server=(LocalDB)\MSSQLLocalDB;"
            f"AttachDbFilename={abs_path};"
            r"Trusted_Connection=yes;"
        ),
        (
            r"Driver={ODBC Driver 18 for SQL Server};"
            r"Server=(LocalDB)\MSSQLLocalDB;"
            f"AttachDbFilename={abs_path};"
            r"Trusted_Connection=yes;"
            r"TrustServerCertificate=yes;"
        ),
        (
            r"Driver={SQL Server};"
            r"Server=(LocalDB)\MSSQLLocalDB;"
            f"AttachDbFilename={abs_path};"
            r"Trusted_Connection=yes;"
        ),
    ]
    
    for conn_str in connection_templates:
        try:
            # logging.info(f"Trying LocalDB attach: {conn_str[:50]}...")
            conn = pyodbc.connect(conn_str, timeout=5)
            logging.info("LocalDB Connection successful!")
            return conn
        except pyodbc.Error:
            continue
    
    return None


def extract_altitude_from_sql(db_path, output_csv_path, parent_folder_filter="04_NAVISCAN", server_name=None):
    """
    Extract altitude data from NaviEdit SQL Server database.
    
    Reads from blocks in the specified parent folder and all its subfolders.
    
    Args:
        db_path: Path to the NaviEdit .mdf SQL Server database file
        output_csv_path: Path where the TSS_Altitude.csv will be saved
        parent_folder_filter: Parent folder name to filter (default: "04_NAVISCAN")
        server_name: Optional SQL Server name to connect to
    
    Returns:
        pandas DataFrame with the extracted data, or empty DataFrame on failure
    """
    if not os.path.exists(db_path):
        logging.error(f"Database file not found: {db_path}")
        return pd.DataFrame()
    
    try:
        import pyodbc
    except ImportError:
        logging.error("pyodbc module not installed. Please install it with: pip install pyodbc")
        return pd.DataFrame()
    
    conn = None
    try:
        # Try to connect to the MDF database
        conn = try_connect_to_mdf(db_path, server_name)
        
        if conn is None:
            logging.error("Could not connect to database with any available method")
            return pd.DataFrame()
        
        # Build the SQL query for SQL Server using the user provided structure
        # We use LEFT JOINs for parents to handle varying depth, and CONCAT_WS to build the path
        query = """
        SELECT 
            b.ID,
            b.Name,
            CONCAT_WS('\\', p4.Name, p3.Name, p2.Name, p1.Name, f.Name) AS Folder,
            DATEADD(ms, b.StartTimeMs + bat.Time, convert(datetime2(3), b.StartTime)) AS Time,
            bat.Time AS RelTime,
            bat.Depth,
            bat.Altitude
        FROM Block b
        INNER JOIN Folder f ON f.ID = b.FolderID
        LEFT JOIN Folder p1 ON p1.Id = f.ParentID
        LEFT JOIN Folder p2 ON p2.Id = p1.ParentID
        LEFT JOIN Folder p3 ON p3.Id = p2.ParentID
        LEFT JOIN Folder p4 ON p4.Id = p3.ParentID
        INNER JOIN Bathy bat ON bat.BlockID = b.ID AND bat.seq = 0
        WHERE b.BlockType = 256
        ORDER BY b.StartTime
        """
        
        # Execute query
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            logging.warning("No altitude data found in database")
            return pd.DataFrame()
            
        # Debug: Check what folders we actually got
        if 'Folder' in df.columns:
            unique_folders = df['Folder'].unique()
            logging.info(f"Database contains {len(unique_folders)} unique folders. First 5: {unique_folders[:5]}")
        
        # Filter by parent folder if specified
        if parent_folder_filter:
            # Filter rows where Folder starts with the parent folder name
            # Note: The SQL query now constructs Folder as "Parent\Child". 
            # If parent_folder_filter is "04_NAVISCAN", we check if it's in the string.
            mask = df['Folder'].str.contains(parent_folder_filter, case=False, na=False)
            df = df[mask].copy()
            
            if df.empty:
                logging.warning(f"No data found for folder filter: {parent_folder_filter}")
                if 'unique_folders' in locals():
                     logging.info(f"Available folders: {unique_folders}")
                return pd.DataFrame()
        
        # Format Time column to string format expected by TSSAltitudeFixer
        if 'Time' in df.columns:
            df['Time'] = pd.to_datetime(df['Time']).dt.strftime('%Y-%m-%d %H:%M:%S.%f')
        
        logging.info(f"Extracted {len(df)} altitude records from database")
        
        # Save to CSV
        df.to_csv(output_csv_path, index=False)
        logging.info(f"Saved altitude data to: {output_csv_path}")
        
        return df
        
    except Exception as e:
        logging.error(f"Error extracting altitude data: {e}")
        if conn:
            try:
                conn.close()
            except:
                pass
        return pd.DataFrame()


def extract_altitude_for_blocks(db_path, output_csv_path, block_ids=None):
    """
    Extract altitude data from NaviEdit SQL Server database for specific blocks.
    
    Args:
        db_path: Path to the NaviEdit .mdf SQL Server database file
        output_csv_path: Path where the TSS_Altitude.csv will be saved
        block_ids: List of block IDs to filter (optional). If None, extracts all blocks.
    
    Returns:
        pandas DataFrame with the extracted data, or empty DataFrame on failure
    """
    if not os.path.exists(db_path):
        logging.error(f"Database file not found: {db_path}")
        return pd.DataFrame()
    
    try:
        import pyodbc
    except ImportError:
        logging.error("pyodbc module not installed. Please install it with: pip install pyodbc")
        return pd.DataFrame()
    
    conn = None
    try:
        conn = try_connect_to_mdf(db_path)
        
        if conn is None:
            logging.error("Could not connect to database")
            return pd.DataFrame()
        
        # Build the SQL query for SQL Server
        query = """
        SELECT 
            b.ID,
            b.Name,
            CONCAT_WS('\\', p4.Name, p3.Name, p2.Name, p1.Name, f.Name) AS Folder,
            DATEADD(ms, b.StartTimeMs + bat.Time, convert(datetime2(3), b.StartTime)) AS Time,
            bat.Time AS RelTime,
            bat.Depth,
            bat.Altitude
        FROM Block b
        INNER JOIN Folder f ON f.ID = b.FolderID
        LEFT JOIN Folder p1 ON p1.Id = f.ParentID
        LEFT JOIN Folder p2 ON p2.Id = p1.ParentID
        LEFT JOIN Folder p3 ON p3.Id = p2.ParentID
        LEFT JOIN Folder p4 ON p4.Id = p3.ParentID
        INNER JOIN Bathy bat ON bat.BlockID = b.ID AND bat.seq = 0
        WHERE b.BlockType = 256
        """
        
        # Add block ID filter if specified
        if block_ids:
            block_ids_str = ', '.join(str(bid) for bid in block_ids)
            query += f" AND b.ID IN ({block_ids_str})"
        
        query += " ORDER BY b.StartTime"
        
        # Execute query
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            logging.warning("No altitude data found in database")
            return pd.DataFrame()
        
        # Format Time column
        if 'Time' in df.columns:
            df['Time'] = pd.to_datetime(df['Time']).dt.strftime('%Y-%m-%d %H:%M:%S.%f')
        
        logging.info(f"Extracted {len(df)} altitude records from database")
        
        # Save to CSV
        df.to_csv(output_csv_path, index=False)
        logging.info(f"Saved altitude data to: {output_csv_path}")
        
        return df
        
    except Exception as e:
        logging.error(f"Error extracting altitude data: {e}")
        if conn:
            try:
                conn.close()
            except:
                pass
        return pd.DataFrame()


def get_available_folders(db_path):
    """
    Get list of available parent folders in the database.
    
    Args:
        db_path: Path to the NaviEdit .mdf SQL Server database file
    
    Returns:
        List of folder names
    """
    if not os.path.exists(db_path):
        logging.error(f"Database file not found: {db_path}")
        return []
    
    try:
        import pyodbc
    except ImportError:
        logging.error("pyodbc module not installed")
        return []
    
    conn = None
    try:
        conn = try_connect_to_mdf(db_path)
        
        if conn is None:
            return []
        
        query = """
        SELECT DISTINCT p1.Name
        FROM Block b
        INNER JOIN Folder f ON f.ID = b.FolderID
        LEFT JOIN Folder p1 ON p1.Id = f.ParentID
        WHERE b.BlockType = 256 AND p1.Name IS NOT NULL
        ORDER BY p1.Name
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df['Name'].tolist()
        
    except Exception as e:
        logging.error(f"Error getting folders: {e}")
        if conn:
            try:
                conn.close()
            except:
                pass
        return []


def extract_altitude_for_block_ids_direct(db_path, block_ids, server_name=None):
    """
    Extract altitude data directly from NaviEdit SQL Server database for specific block IDs.
    Optimized version that filters in SQL query and returns DataFrame directly without CSV I/O.
    
    Args:
        db_path: Path to the NaviEdit .mdf SQL Server database file
        block_ids: List of block IDs to fetch (required)
        server_name: Optional SQL Server name to connect to
    
    Returns:
        pandas DataFrame with the extracted data (already formatted), or empty DataFrame on failure.
        If the DataFrame is empty, the 'error_info' attribute will contain diagnostic information.
    """
    if not os.path.exists(db_path):
        logging.error(f"Database file not found: {db_path}")
        result = pd.DataFrame()
        result.attrs['error_info'] = {
            'error_type': 'database_not_found',
            'message': f"Database file not found: {db_path}",
            'requested_block_ids': list(block_ids) if block_ids else [],
            'available_block_ids': []
        }
        return result
    
    if not block_ids:
        logging.error("No block IDs provided")
        result = pd.DataFrame()
        result.attrs['error_info'] = {
            'error_type': 'no_block_ids',
            'message': "No Block IDs provided",
            'requested_block_ids': [],
            'available_block_ids': []
        }
        return result
    
    try:
        import pyodbc
    except ImportError:
        logging.error("pyodbc module not installed. Please install it with: pip install pyodbc")
        result = pd.DataFrame()
        result.attrs['error_info'] = {
            'error_type': 'import_error',
            'message': "pyodbc module not installed. Please install it with: pip install pyodbc",
            'requested_block_ids': list(block_ids),
            'available_block_ids': []
        }
        return result
    
    conn = None
    try:
        conn = try_connect_to_mdf(db_path, server_name)
        
        if conn is None:
            logging.error("Could not connect to database")
            result = pd.DataFrame()
            result.attrs['error_info'] = {
                'error_type': 'connection_failed',
                'message': f"Could not connect to database: {db_path}\nServer: {server_name or 'auto-detect'}",
                'requested_block_ids': list(block_ids),
                'available_block_ids': []
            }
            return result
        
        # First, query what Block IDs actually exist in the database
        available_blocks_query = """
        SELECT DISTINCT b.ID, b.Name 
        FROM Block b
        INNER JOIN Bathy bat ON bat.BlockID = b.ID AND bat.seq = 0
        WHERE b.BlockType = 256
        ORDER BY b.ID
        """
        
        try:
            available_df = pd.read_sql_query(available_blocks_query, conn)
            available_block_ids = available_df['ID'].tolist() if not available_df.empty else []
            available_block_names = dict(zip(available_df['ID'], available_df['Name'])) if not available_df.empty else {}
        except Exception as e:
            logging.warning(f"Could not query available blocks: {e}")
            available_block_ids = []
            available_block_names = {}
        
        # Build the SQL query with block ID filter in WHERE clause
        block_ids_str = ', '.join(str(bid) for bid in block_ids)
        
        query = f"""
        SELECT 
            b.ID,
            b.Name,
            CONCAT_WS('\\', p4.Name, p3.Name, p2.Name, p1.Name, f.Name) AS Folder,
            DATEADD(ms, b.StartTimeMs + bat.Time, convert(datetime2(3), b.StartTime)) AS Time,
            bat.Time AS RelTime,
            bat.Depth,
            bat.Altitude
        FROM Block b
        INNER JOIN Folder f ON f.ID = b.FolderID
        LEFT JOIN Folder p1 ON p1.Id = f.ParentID
        LEFT JOIN Folder p2 ON p2.Id = p1.ParentID
        LEFT JOIN Folder p3 ON p3.Id = p2.ParentID
        LEFT JOIN Folder p4 ON p4.Id = p3.ParentID
        INNER JOIN Bathy bat ON bat.BlockID = b.ID AND bat.seq = 0
        WHERE b.BlockType = 256 AND b.ID IN ({block_ids_str})
        ORDER BY b.StartTime
        """
        
        # Execute query
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            # Determine which requested IDs were not found
            requested_set = set(block_ids)
            available_set = set(available_block_ids)
            not_found_ids = sorted(requested_set - available_set)
            found_in_db_but_no_bathy = sorted(requested_set & available_set)
            
            # Build detailed error message
            error_msg = f"No altitude data found for Block IDs: {list(block_ids)}\n\n"
            
            if not_found_ids:
                error_msg += f"Block IDs NOT in database: {not_found_ids}\n"
                error_msg += "  → These Block IDs do not exist in the NaviEdit database.\n\n"
            
            if available_block_ids:
                # Show some available IDs as examples (first 20)
                example_ids = available_block_ids[:20]
                error_msg += f"Available Block IDs in database (first 20): {example_ids}\n"
                if len(available_block_ids) > 20:
                    error_msg += f"  ... and {len(available_block_ids) - 20} more\n"
            else:
                error_msg += "No Block IDs with altitude data found in database.\n"
            
            logging.warning(error_msg)
            
            result = pd.DataFrame()
            result.attrs['error_info'] = {
                'error_type': 'block_ids_not_found',
                'message': error_msg,
                'requested_block_ids': list(block_ids),
                'available_block_ids': available_block_ids,
                'available_block_names': available_block_names,
                'not_found_ids': not_found_ids
            }
            return result
        
        # Format Time column to string format expected by rest of the pipeline
        if 'Time' in df.columns:
            df['Time'] = pd.to_datetime(df['Time']).dt.strftime('%Y-%m-%d %H:%M:%S.%f')
        
        logging.info(f"Extracted {len(df)} altitude records for {len(block_ids)} blocks from database")
        
        # Add pre-parsed DateTime column for efficiency (avoid re-parsing later)
        if 'Time' in df.columns:
            df['DateTime'] = pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S.%f')
        
        # Rename columns to match expected format
        df.rename(columns={'Altitude': 'CRP_Altitude', 'Depth': 'SQL_Depth'}, inplace=True)
        
        # Ensure Name is string
        if 'Name' in df.columns:
            df['Name'] = df['Name'].astype(str)
        
        return df
        
    except Exception as e:
        logging.error(f"Error extracting altitude data: {e}")
        if conn:
            try:
                conn.close()
            except:
                pass
        result = pd.DataFrame()
        result.attrs['error_info'] = {
            'error_type': 'exception',
            'message': f"Error extracting altitude data: {e}",
            'requested_block_ids': list(block_ids) if block_ids else [],
            'available_block_ids': []
        }
        return result


def get_block_count_by_folder(db_path):
    """
    Get count of blocks per parent folder.
    
    Args:
        db_path: Path to the NaviEdit .mdf SQL Server database file
    
    Returns:
        Dictionary {folder_name: block_count}
    """
    if not os.path.exists(db_path):
        logging.error(f"Database file not found: {db_path}")
        return {}
    
    try:
        import pyodbc
    except ImportError:
        logging.error("pyodbc module not installed")
        return {}
    
    conn = None
    try:
        conn = try_connect_to_mdf(db_path)
        
        if conn is None:
            return {}
        
        query = """
        SELECT p1.Name, COUNT(DISTINCT b.ID) as BlockCount
        FROM Block b
        INNER JOIN Folder f ON f.ID = b.FolderID
        LEFT JOIN Folder p1 ON p1.Id = f.ParentID
        WHERE b.BlockType = 256 AND p1.Name IS NOT NULL
        GROUP BY p1.Name
        ORDER BY p1.Name
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return dict(zip(df['Name'], df['BlockCount']))
        
    except Exception as e:
        logging.error(f"Error getting block counts: {e}")
        if conn:
            try:
                conn.close()
            except:
                pass
        return {}


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python altitudeFromSQL.py <database_path> [output_csv_path] [folder_filter]")
        print("Example: python altitudeFromSQL.py project.mdf TSS_Altitude.csv 04_NAVISCAN")
        sys.exit(1)
    
    db_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "TSS_Altitude.csv"
    folder_filter = sys.argv[3] if len(sys.argv) > 3 else "04_NAVISCAN"
    
    print(f"Extracting altitude data from: {db_path}")
    print(f"Output CSV: {output_path}")
    print(f"Folder filter: {folder_filter}")
    
    df = extract_altitude_from_sql(db_path, output_path, folder_filter)
    
    if not df.empty:
        print(f"\nExtracted {len(df)} records")
        print(f"Columns: {df.columns.tolist()}")
        print(f"\nFirst 5 rows:")
        print(df.head())
    else:
        print("No data extracted")
