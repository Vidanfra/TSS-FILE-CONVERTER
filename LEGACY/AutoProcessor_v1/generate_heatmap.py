import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
import os
import csv
import rasterio
from rasterio.transform import from_origin

def getCoilPeaks(merged_df, column):
    coil_peaks = []
    
    if 'Filename' not in merged_df or column not in merged_df:
        return coil_peaks
    
    for line in merged_df['Filename'].unique():
        df = merged_df[merged_df['Filename'] == line]
        
        if df.empty:
            continue
        
        max_index = df[column].idxmax()
        min_index = df[column].idxmin()
        
        if abs(df.loc[max_index, column]) > abs(df.loc[min_index, column]): # Positive peak
            abs_max_value = df.loc[max_index, column]
            coil = df.loc[max_index, 'Coil']
            easting = df.loc[max_index, 'Easting']
            northing = df.loc[max_index, 'Northing']
        else: # Negative peak
            abs_max_value = df.loc[min_index, column]
            coil = df.loc[min_index, 'Coil']
            easting = df.loc[min_index, 'Easting']
            northing = df.loc[min_index, 'Northing']
        
        coil_peaks.append({
            'PTR file': line,
            'Peak value': abs_max_value,
            'Coil': coil,
            'Easting': easting,
            'Northing': northing
        })
    coil_peaks_df = pd.DataFrame(coil_peaks)
    return coil_peaks_df

def generate_TSS_heatmap(output_folder, filename, df, class_id, cell_size, colors, boundaries):
    """
    Generate heatmap from TSS data.
    
    Args:
        output_folder: User-specified output folder for generated files
        filename: Name of the file
        df: DataFrame with TSS data
        class_id: Class ID for YOLO labeling
        cell_size: Cell size for grid interpolation
    """
    # Get coil peak data
    coil_peaks_df = getCoilPeaks(df, column='TSS')
    max_peak_idx = coil_peaks_df['Peak value'].abs().idxmax()
    max_peak_easting = coil_peaks_df.loc[max_peak_idx, 'Easting']
    max_peak_northing = coil_peaks_df.loc[max_peak_idx, 'Northing']

    # Get data to create heatmap
    easting = df['Easting']
    northing = df['Northing']
    tss_intensity = df['TSS']
    
    # Define grid based on cell_size
    margin = cell_size
    
    # Snap grid boundaries to cell_size multiples to ensure consistent alignment
    # regardless of data extent. This prevents color shifts when comparing
    # heatmaps from lines with different lengths.
    west = np.floor((min(easting) - margin) / cell_size) * cell_size
    north = np.ceil((max(northing) + margin) / cell_size) * cell_size
    east = np.ceil((max(easting) + margin) / cell_size) * cell_size
    south = np.floor((min(northing) - margin) / cell_size) * cell_size
    
    # Calculate grid dimensions
    nx = int(round((east - west) / cell_size))
    ny = int(round((north - south) / cell_size))
    
    # Generate coordinates for pixel centers using arange for exact spacing
    # X: Left to Right
    x_coords = np.arange(west + cell_size/2, east, cell_size)
    # Y: Top to Bottom (North to South) - must be descending for image coordinates
    y_coords = np.arange(north - cell_size/2, south, -cell_size)
    
    # Ensure we have the expected number of coordinates
    nx = len(x_coords)
    ny = len(y_coords)
    
    # Calculate actual grid extent
    east_grid = west + nx * cell_size
    south_grid = north - ny * cell_size
    
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)

    # Interpolate intensity values onto the grid
    grid_z = griddata((easting, northing), tss_intensity, (grid_x, grid_y), "nearest") # Options: 'linear', 'nearest', 'cubic'

    # Build a KDTree to efficiently find the distance from each grid point to the nearest data point
    tree = cKDTree(np.column_stack((easting, northing)))
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    distances, _ = tree.query(grid_points)
    distances = distances.reshape(grid_x.shape)

    # Mask grid cells that are too far from any original data point
    grid_z_masked = np.where(distances <= cell_size, grid_z, np.nan)

    # Create a continuous colormap from the discrete colors.
    cmap_faded = mcolors.LinearSegmentedColormap.from_list('faded_cmap', colors, N=256) # We use many bins (here, 256) so the colors will fade continuously.

    # Custom normalization that preserves your fixed color stops.
    class FadedBoundaryNorm(mcolors.Normalize):
        def __init__(self, boundaries, ncolors, clip=False):
            self.boundaries = np.array(boundaries)
            self.ncolors = ncolors  # this should match len(colors)
            # Set vmin and vmax to the first and last boundary values
            super().__init__(vmin=boundaries[0], vmax=boundaries[-1], clip=clip)
        
        def __call__(self, value, clip=None):
            # For each value, find its normalized position.
            # Then we divide by (ncolors - 1) to map to the [0,1] range.
            normed = np.interp(value, self.boundaries,
                            np.linspace(0, self.ncolors - 1, len(self.boundaries))) # The mapping: boundaries[i] maps to i and boundaries[i+1] maps to i+1.
            return normed / (self.ncolors - 1)

    # Create the norm instance
    norm_faded = FadedBoundaryNorm(boundaries, ncolors=len(colors))

    # Ensure square figure for YOLO (640x640)
    fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=100)  # 640x640 pixels

    # Plot heatmap
    # grid_z_masked is (Y, X) from meshgrid, so no transpose needed
    # Use interpolation='nearest' to prevent blurry output when grid resolution differs from image size
    ax.imshow(grid_z_masked, extent=(west, east_grid, south_grid, north), cmap=cmap_faded, norm=norm_faded, interpolation='nearest')
    ax.axis('off')

    # Ensure the directory exists in the user-specified output folder
    if os.path.isfile(output_folder):
        output_folder = os.path.dirname(output_folder)

    rgb_geotiffs_dir = os.path.join(output_folder, "heatmaps", "RGBgeotiffs")
    float32_geotiffs_dir = os.path.join(output_folder, "heatmaps", "32bitgeotiffs")
    peaks_dir = os.path.join(output_folder, "heatmaps", "pngs")
    
    os.makedirs(rgb_geotiffs_dir, exist_ok=True)
    os.makedirs(float32_geotiffs_dir, exist_ok=True)
    os.makedirs(peaks_dir, exist_ok=True)
    
    rgb_tif_name = os.path.join(rgb_geotiffs_dir, filename.removesuffix(".txt") + "_TSS.tif")
    float32_tif_name = os.path.join(float32_geotiffs_dir, filename.removesuffix(".txt") + "_TSS.tif")
    peaks_name = os.path.join(peaks_dir, filename.removesuffix(".txt") + "_TSS.png")
    
    # Save GeoTIFFs
    # Data is already (Y, X) from meshgrid
    data_grid = grid_z_masked
    
    transform = from_origin(west, north, cell_size, cell_size)
    
    # Save RGB GeoTIFF (RGBA uint8)
    rgba = cmap_faded(norm_faded(data_grid))
    rgba_uint8 = (rgba * 255).astype(np.uint8)
    
    with rasterio.open(
        rgb_tif_name,
        'w',
        driver='GTiff',
        height=data_grid.shape[0],
        width=data_grid.shape[1],
        count=4,
        dtype='uint8',
        crs=None, 
        transform=transform,
    ) as dst:
        dst.write(np.moveaxis(rgba_uint8, 2, 0))
    
    # Save 32-bit float GeoTIFF (actual TSS values)
    # Replace NaN with nodata value for proper GeoTIFF handling
    data_grid_float32 = data_grid.astype(np.float32)
    nodata_value = -9999.0
    data_grid_float32 = np.where(np.isnan(data_grid_float32), nodata_value, data_grid_float32)
    
    with rasterio.open(
        float32_tif_name,
        'w',
        driver='GTiff',
        height=data_grid.shape[0],
        width=data_grid.shape[1],
        count=1,
        dtype='float32',
        crs=None,
        transform=transform,
        nodata=nodata_value,
    ) as dst:
        dst.write(data_grid_float32, 1)
    
    # Compute bounding box for the peak
    min_x, max_x = min(easting), max(easting)
    min_y, max_y = min(northing), max(northing)

    peak_x_norm = (max_peak_easting - min_x) / (max_x - min_x)  # Normalize [0,1]
    peak_y_norm = (max_peak_northing - min_y) / (max_y - min_y)  # Normalize [0,1]

    box_size = 0.05  # Bounding box size relative to image dimensions
    class_id = 0  # Class ID for the bounding
    x_center = peak_x_norm
    y_center = peak_y_norm
    width = box_size
    height = box_size

    # Define bounding box size (adjust as needed)
    box_size = 1  # Meters
    x_min = max_peak_easting - box_size / 2
    y_min = max_peak_northing - box_size / 2

    rect = patches.Rectangle((x_min, y_min), box_size, box_size, linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(rect)
    ax.text(max_peak_easting, max_peak_northing, s=f"{coil_peaks_df.loc[max_peak_idx, 'Peak value']}", color='black', fontsize=10, ha='center', va='center')

    # Create a dummy mappable for the colorbar to show the colors evenly spaced
    dummy_norm = mcolors.Normalize(vmin=0, vmax=1)
    dummy_mappable = plt.cm.ScalarMappable(norm=dummy_norm, cmap=cmap_faded)

    # Add colorbar
    cbar = fig.colorbar(dummy_mappable, ax=ax, location='left')
    cbar.set_label('TSS (uV)')
    
    # Set ticks and labels
    ticks = np.linspace(0, 1, len(boundaries))
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(boundaries)

    # Save and show the plot
    plt.savefig(peaks_name, bbox_inches='tight', pad_inches=0)

    # Close the figure
    plt.close()

    '''
    label_str = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
    with open(txt_name, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=' ')  # Ensure space as a separator
        writer.writerow(label_str.split())  # Convert string to list. writerow() expects an iterable.
        print(f"Saved {txt_name} : {label_str}")
    '''

def generate_ALT_heatmap(output_folder, filename, df, class_id, cell_size, colors, boundaries):
    """
    Generate heatmap from Alt data.
    
    Args:
        output_folder: User-specified output folder for generated files
        filename: Name of the file
        df: DataFrame with Alt data
        class_id: Class ID for YOLO labeling
        cell_size: Cell size for grid interpolation
    """
    # Get coil peak data
    coil_peaks_df = getCoilPeaks(df, column='Alt')
    max_peak_idx = coil_peaks_df['Peak value'].abs().idxmax()
    max_peak_easting = coil_peaks_df.loc[max_peak_idx, 'Easting']
    max_peak_northing = coil_peaks_df.loc[max_peak_idx, 'Northing']

    # Get data to create heatmap
    easting = df['Easting']
    northing = df['Northing']
    altitude = df['Alt']
    
    # Define grid based on cell_size
    margin = cell_size
    
    # Snap grid boundaries to cell_size multiples to ensure consistent alignment
    # regardless of data extent. This prevents color shifts when comparing
    # heatmaps from lines with different lengths.
    west = np.floor((min(easting) - margin) / cell_size) * cell_size
    north = np.ceil((max(northing) + margin) / cell_size) * cell_size
    east = np.ceil((max(easting) + margin) / cell_size) * cell_size
    south = np.floor((min(northing) - margin) / cell_size) * cell_size
    
    # Calculate grid dimensions
    nx = int(round((east - west) / cell_size))
    ny = int(round((north - south) / cell_size))
    
    # Generate coordinates for pixel centers using arange for exact spacing
    # X: Left to Right
    x_coords = np.arange(west + cell_size/2, east, cell_size)
    # Y: Top to Bottom (North to South) - must be descending for image coordinates
    y_coords = np.arange(north - cell_size/2, south, -cell_size)
    
    # Ensure we have the expected number of coordinates
    nx = len(x_coords)
    ny = len(y_coords)
    
    # Calculate actual grid extent
    east_grid = west + nx * cell_size
    south_grid = north - ny * cell_size
    
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)

    # Interpolate intensity values onto the grid
    grid_z = griddata((easting, northing), altitude, (grid_x, grid_y), "nearest") # Options: 'linear', 'nearest', 'cubic'

    # Build a KDTree to efficiently find the distance from each grid point to the nearest data point
    tree = cKDTree(np.column_stack((easting, northing)))
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    distances, _ = tree.query(grid_points)
    distances = distances.reshape(grid_x.shape)

    # Mask grid cells that are too far from any original data point
    grid_z_masked = np.where(distances <= cell_size, grid_z, np.nan)

    # Create a continuous colormap from the discrete colors.
    cmap_faded = mcolors.LinearSegmentedColormap.from_list('faded_cmap', colors, N=256) # We use many bins (here, 256) so the colors will fade continuously.

    # Custom normalization that preserves your fixed color stops.
    class FadedBoundaryNorm(mcolors.Normalize):
        def __init__(self, boundaries, ncolors, clip=False):
            self.boundaries = np.array(boundaries)
            self.ncolors = ncolors  # this should match len(colors)
            # Set vmin and vmax to the first and last boundary values
            super().__init__(vmin=boundaries[0], vmax=boundaries[-1], clip=clip)
        
        def __call__(self, value, clip=None):
            # For each value, find its normalized position.
            # Then we divide by (ncolors - 1) to map to the [0,1] range.
            normed = np.interp(value, self.boundaries,
                            np.linspace(0, self.ncolors - 1, len(self.boundaries))) # The mapping: boundaries[i] maps to i and boundaries[i+1] maps to i+1.
            return normed / (self.ncolors - 1)

    # Create the norm instance
    norm_faded = FadedBoundaryNorm(boundaries, ncolors=len(colors))

    # Ensure square figure for YOLO (640x640)
    fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=100)  # 640x640 pixels

    # Plot heatmap
    # grid_z_masked is (Y, X) from meshgrid, so no transpose needed
    # Use interpolation='nearest' to prevent blurry output when grid resolution differs from image size
    ax.imshow(grid_z_masked, extent=(west, east_grid, south_grid, north), cmap=cmap_faded, norm=norm_faded, interpolation='nearest')
    ax.axis('off')

    # Ensure the directory exists in the user-specified output folder
    if os.path.isfile(output_folder):
        output_folder = os.path.dirname(output_folder)

    rgb_geotiffs_dir = os.path.join(output_folder, "heatmaps", "RGBgeotiffs")
    float32_geotiffs_dir = os.path.join(output_folder, "heatmaps", "32bitgeotiffs")
    peaks_dir = os.path.join(output_folder, "heatmaps", "pngs")
    
    os.makedirs(rgb_geotiffs_dir, exist_ok=True)
    os.makedirs(float32_geotiffs_dir, exist_ok=True)
    os.makedirs(peaks_dir, exist_ok=True)
    
    rgb_tif_name = os.path.join(rgb_geotiffs_dir, filename.removesuffix(".txt") + "_ALT.tif")
    float32_tif_name = os.path.join(float32_geotiffs_dir, filename.removesuffix(".txt") + "_ALT.tif")
    peaks_name = os.path.join(peaks_dir, filename.removesuffix(".txt") + "_ALT.png")
    
    # Save GeoTIFFs
    # Data is already (Y, X) from meshgrid
    data_grid = grid_z_masked
    
    transform = from_origin(west, north, cell_size, cell_size)
    
    # Save RGB GeoTIFF (RGBA uint8)
    rgba = cmap_faded(norm_faded(data_grid))
    rgba_uint8 = (rgba * 255).astype(np.uint8)
    
    with rasterio.open(
        rgb_tif_name,
        'w',
        driver='GTiff',
        height=data_grid.shape[0],
        width=data_grid.shape[1],
        count=4,
        dtype='uint8',
        crs=None, 
        transform=transform,
    ) as dst:
        dst.write(np.moveaxis(rgba_uint8, 2, 0))
    
    # Save 32-bit float GeoTIFF (actual Alt values)
    # Replace NaN with nodata value for proper GeoTIFF handling
    data_grid_float32 = data_grid.astype(np.float32)
    nodata_value = -9999.0
    data_grid_float32 = np.where(np.isnan(data_grid_float32), nodata_value, data_grid_float32)
    
    with rasterio.open(
        float32_tif_name,
        'w',
        driver='GTiff',
        height=data_grid.shape[0],
        width=data_grid.shape[1],
        count=1,
        dtype='float32',
        crs=None,
        transform=transform,
        nodata=nodata_value,
    ) as dst:
        dst.write(data_grid_float32, 1)
    
    # Compute bounding box for the peak
    min_x, max_x = min(easting), max(easting)
    min_y, max_y = min(northing), max(northing)

    peak_x_norm = (max_peak_easting - min_x) / (max_x - min_x)  # Normalize [0,1]
    peak_y_norm = (max_peak_northing - min_y) / (max_y - min_y)  # Normalize [0,1]

    box_size = 0.05  # Bounding box size relative to image dimensions
    class_id = 0  # Class ID for the bounding
    x_center = peak_x_norm
    y_center = peak_y_norm
    width = box_size
    height = box_size

    # Define bounding box size (adjust as needed)
    box_size = 1  # Meters
    x_min = max_peak_easting - box_size / 2
    y_min = max_peak_northing - box_size / 2

    rect = patches.Rectangle((x_min, y_min), box_size, box_size, linewidth=2, edgecolor='white', facecolor='none')
    ax.add_patch(rect)
    ax.text(max_peak_easting, max_peak_northing, s=f"{coil_peaks_df.loc[max_peak_idx, 'Peak value']}", color='white', fontsize=10, ha='center', va='center')

    # Create a dummy mappable for the colorbar to show the colors evenly spaced
    dummy_norm = mcolors.Normalize(vmin=0, vmax=1)
    dummy_mappable = plt.cm.ScalarMappable(norm=dummy_norm, cmap=cmap_faded)

    # Add colorbar
    cbar = fig.colorbar(dummy_mappable, ax=ax, location='left')
    cbar.set_label('Altitude (meters)')
    
    # Set ticks and labels
    ticks = np.linspace(0, 1, len(boundaries))
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(boundaries)

    # Save and show the plot
    plt.savefig(peaks_name, bbox_inches='tight', pad_inches=0)

    # Close the figure
    plt.close()

    '''
    label_str = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
    with open(txt_name, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=' ')  # Ensure space as a separator
        writer.writerow(label_str.split())  # Convert string to list. writerow() expects an iterable.
        print(f"Saved {txt_name} : {label_str}")
    '''