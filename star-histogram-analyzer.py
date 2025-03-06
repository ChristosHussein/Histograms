import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import ipywidgets as widgets
from IPython.display import display, clear_output
import os
import re
import glob

def load_dat_file(file_path, id_column='Object', weight_column='Vgfb'):
    """
    Load the dataset from a .dat file and extract star parameters.

    Parameters:
    file_path: Path to the .dat file
    id_column: Name of the column containing star IDs (default 'Object')
    weight_column: Name of the column for weighting (default 'Vgfb')

    Returns:
    A DataFrame containing the star data or None if loading fails
    """
    try:
        # Initialize dictionary to store parameters
        params = {}
        
        with open(file_path, 'r') as f:
            content = f.readlines()
        
        # Extract parameters from the header comments
        for line in content:
            line = line.strip()
            if line.startswith('#'):
                # Remove the '#' and split by '='
                line = line.lstrip('#').strip()
                # Look for entries like "Parameter = Value"
                match = re.search(r'(\w+\.?\w*)\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(.+)?', line)
                if match:
                    param_name = match.group(1).strip()
                    param_value = match.group(2).strip()
                    unit = match.group(3).strip() if match.group(3) else ""
                    
                    # Convert to appropriate type (float or string)
                    try:
                        # Try to convert to float
                        param_value = float(param_value)
                    except ValueError:
                        # Keep as string if conversion fails
                        pass
                    
                    # Store parameter
                    params[param_name] = param_value
                    
                # Special case for Object ID (it might not follow the "=" format)
                obj_match = re.search(r'Object\s*=?\s*(\w+)', line)
                if obj_match:
                    params['Object'] = obj_match.group(1).strip()
        
        # Create a DataFrame from the parameters
        df = pd.DataFrame([params])
        
        # Ensure the required columns exist
        if id_column not in df.columns:
            if 'Object' in df.columns:
                # Use 'Object' as fallback
                df[id_column] = df['Object']
            else:
                # Extract filename as object ID
                df[id_column] = os.path.basename(file_path).split('.')[0]
        
        # Convert ID to string
        df[id_column] = df[id_column].astype(str)
        
        # Ensure weight columns (Vgfb, Vgf, Chi2) exist with default values
        weight_options = ['Vgfb', 'Vgf', 'Chi2']
        for weight_opt in weight_options:
            if weight_opt not in df.columns:
                if weight_opt.lower() in [col.lower() for col in df.columns]:
                    # Case-insensitive match
                    matching_col = [col for col in df.columns if col.lower() == weight_opt.lower()][0]
                    df[weight_opt] = df[matching_col]
                else:
                    print(f"Warning: '{weight_opt}' not found in file {file_path}. Using default value of 1.0")
                    df[weight_opt] = 1.0
        
        # Ensure other common parameters are present
        for param in ['logg', 'Teff', 'Lbol']:
            if param not in df.columns and f'{param}_K' in df.columns:
                df[param] = df[f'{param}_K']
            elif param not in df.columns and f'{param}_Lsun' in df.columns:
                df[param] = df[f'{param}_Lsun']
            elif param not in df.columns:
                # Use placeholder values for missing parameters
                if param == 'logg':
                    df[param] = 0.0
                elif param == 'Teff':
                    df[param] = 5000.0
                elif param == 'Lbol':
                    df[param] = 1.0
        
        # Rename Teff column if needed
        if 'Teff' in df.columns and 'Teff_K' not in df.columns:
            df['Teff_K'] = df['Teff']
        
        # Rename Lbol column if needed
        if 'Lbol' in df.columns and 'Lbol_Lsun' not in df.columns:
            df['Lbol_Lsun'] = df['Lbol']
            
        print(f"File {file_path} loaded successfully.")
        return df
    
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while loading the file {file_path}: {e}")
        return None

def load_dataset_from_directory(directory_path, id_column='Object', weight_column='Vgfb'):
    """
    Load all .dat files from a directory and combine them into a single DataFrame.

    Parameters:
    directory_path: Path to the directory containing .dat files
    id_column: Name of the column containing star IDs (default 'Object')
    weight_column: Name of the column for weighting (default 'Vgfb')

    Returns:
    A combined DataFrame or None if loading fails
    """
    try:
        # Find all .dat files in the directory and subdirectories
        dat_files = glob.glob(os.path.join(directory_path, "**/*.dat"), recursive=True)
        
        if not dat_files:
            print(f"No .dat files found in {directory_path}")
            return None
        
        print(f"Found {len(dat_files)} .dat files in {directory_path}")
        
        # Load each file and combine
        dfs = []
        for file_path in dat_files:
            df = load_dat_file(file_path, id_column=id_column, weight_column=weight_column)
            if df is not None:
                dfs.append(df)
        
        if not dfs:
            print("No valid data files could be loaded.")
            return None
        
        # Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        
        print(f"Combined dataset created with {len(combined_df)} stars.")
        return combined_df
    
    except Exception as e:
        print(f"An error occurred while loading the dataset: {e}")
        return None

def find_star_by_id(df, object_id, id_column='Object'):
    """
    Find a star in the dataset by its ID (string-based matching).

    Parameters:
    df: The DataFrame containing the dataset
    object_id: The ID of the star to find (as a string)
    id_column: The column containing the star IDs (default 'Object')

    Returns:
    A DataFrame row containing the star data, or None if not found
    """
    if id_column not in df.columns:
        print(f"Error: Column '{id_column}' not found in the dataset.")
        return None
    
    # Filter the dataset for the exact match of the ID (string comparison)
    matches = df[df[id_column] == str(object_id)]  # Ensure both sides are strings
    
    if len(matches) == 0:
        print(f"No stars found with ID '{object_id}'.")
        return None
    elif len(matches) > 1:
        print(f"Multiple stars found with ID '{object_id}'. Using the first match.")
    
    return matches.iloc[0]

def filter_stars_by_query(df, query):
    """
    Filter stars based on a query (e.g., 'logg=5', 'logg>3', 'Teff_K<6000') and export matching IDs to a text file.

    Parameters:
    df: The DataFrame containing the dataset
    query: A string query (e.g., 'logg=5', 'logg>3', 'Teff_K<6000')
    """
    try:
        # Strip whitespace from the query
        query = query.strip()
        
        # Return None if the query is empty
        if not query:
            print("Please enter a valid query.")
            return
        
        # Parse the query into column, operator, and value
        if '=' in query:
            column, value = query.split('=')
            operator = '=='
        elif '>' in query:
            column, value = query.split('>')
            operator = '>'
        elif '<' in query:
            column, value = query.split('<')
            operator = '<'
        else:
            print(f"Error: Invalid query format. Use 'column=5', 'column>5', or 'column<5'.")
            return
        
        # Clean the column name and check for aliases
        column = column.strip().lower()
        
        # Handle aliases (case insensitive)
        column_aliases = {
            'temperature': 'Teff_K',
            'temp': 'Teff_K',
            't': 'Teff_K',
            'teff': 'Teff_K',
            'metallicity': 'Meta'
        }
        
        if column in column_aliases:
            actual_column = column_aliases[column]
        else:
            # For non-aliased columns, find case-insensitive match
            matching_columns = [col for col in df.columns if col.lower() == column]
            if matching_columns:
                actual_column = matching_columns[0]
            else:
                print(f"Error: Column '{column}' not found in the dataset.")
                return
        
        # Convert value to numeric if possible
        try:
            value = float(value.strip())
        except ValueError:
            print(f"Error: Invalid value '{value}' in query. Queries must involve numeric comparisons.")
            return
        
        # Apply the query
        if operator == '==':
            matches = df[df[actual_column] == value]
        elif operator == '>':
            matches = df[df[actual_column] > value]
        elif operator == '<':
            matches = df[df[actual_column] < value]
        else:
            print(f"Error: Unsupported operator '{operator}'.")
            return
        
        if len(matches) == 0:
            print(f"No stars found matching the query '{query}'.")
            return
        
        # Export matching IDs to a text file
        output_filename = f"stars_{query.replace('>', '_gt_').replace('<', '_lt_').replace('=', '_eq_')}.txt"
        with open(output_filename, "w") as f:
            f.write(f"List of stars with {query}\n")
            id_column = 'Object'  # Default ID column
            if id_column not in matches.columns and 'Object_ID' in matches.columns:
                id_column = 'Object_ID'
                
            for obj_id in matches[id_column]:
                f.write(f"{obj_id}\n")
        
        print(f"Exported list of stars matching the query '{query}' to {output_filename}.")
    except Exception as e:
        print(f"Error: Invalid query format. Use 'column=5', 'column>5', or 'column<5'.")
        print(e)

def create_weighted_histogram(df, param_name, weight_name='Vgfb', bins=20, object_id=None, id_column='Object'):
    """
    Create a histogram with weight-based coloring and optional star highlighting.

    Parameters:
    df: The DataFrame containing the dataset
    param_name: The parameter to plot (e.g., 'logg', 'Teff_K')
    weight_name: The column with weight values (default 'Vgfb')
    bins: Number of bins for the histogram (default 20)
    object_id: Optional Object ID to highlight in the histogram
    id_column: The column containing the star IDs (default 'Object')
    """
    if param_name not in df.columns:
        print(f"Error: Column '{param_name}' not found in the dataset.")
        return
    
    if weight_name not in df.columns:
        print(f"Error: Column '{weight_name}' not found in the dataset. Using placeholder values.")
        df[weight_name] = 1.0  # Default value if weight column is missing
    
    # Extract the relevant columns
    param_values = df[param_name]
    weight_values = df[weight_name]
    
    # Define custom bin edges based on parameter
    if param_name == 'Lbol_Lsun':  # Logarithmic scale for Lbol
        min_val = np.min(param_values[param_values > 0])  # Avoid log(0)
        max_val = np.max(param_values)
        bin_edges = np.logspace(np.log10(min_val), np.log10(max_val), bins + 1)
    elif param_name == 'Teff_K':  # Finer bins for Teff_K below 10,000
        lower_bins = np.linspace(0, 10000, int(bins * 0.8))  # 80% of bins for values <= 10,000
        upper_bins = np.linspace(10000, np.max(param_values), int(bins * 0.2) + 1)  # 20% for values > 10,000
        bin_edges = np.concatenate([lower_bins, upper_bins[1:]])
    else:  # Default linear bins for other parameters
        bin_edges = np.linspace(np.min(param_values), np.max(param_values), bins + 1)
    
    hist, _ = np.histogram(param_values, bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Calculate average weight for each bin
    weight_per_bin = []
    for i in range(len(bin_edges) - 1):
        mask = (param_values >= bin_edges[i]) & (param_values < bin_edges[i+1])
        weight_per_bin.append(np.mean(weight_values[mask]) if np.sum(mask) > 0 else 0)
    
    # Normalize weight values for color mapping
    min_weight = min(weight_per_bin)
    max_weight = max(weight_per_bin)
    
    # Create a high-contrast colormap
    colors = ["darkblue", "royalblue", "lightblue", "lightyellow", "orange", "red", "darkred"]
    cmap = mcolors.LinearSegmentedColormap.from_list('high_contrast', colors)
    
    # Create the figure and axis explicitly
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot the histogram with colors based on weight values
    for i in range(len(bin_centers)):
        norm_value = (weight_per_bin[i] - min_weight) / (max_weight - min_weight) if max_weight > min_weight else 0.5
        ax.bar(
            bin_centers[i], hist[i], width=(bin_edges[i+1]-bin_edges[i]), 
            color=cmap(norm_value), alpha=0.8, edgecolor='black', linewidth=0.5
        )
    
    # Highlight the entire bin if an Object ID is provided
    if object_id is not None:
        star_data = find_star_by_id(df, str(object_id), id_column=id_column)
        if star_data is not None:
            star_param_value = star_data[param_name]
            
            # Highlight the bin containing the star
            for i in range(len(bin_edges) - 1):
                if bin_edges[i] <= star_param_value < bin_edges[i+1]:
                    ax.bar(
                        bin_centers[i], hist[i], width=(bin_edges[i+1]-bin_edges[i]), 
                        color='red', alpha=0.7, edgecolor='black', linewidth=1.5, hatch='/',
                        label=f"Star Bin ({object_id})"
                    )
                    
                    # Add a dashed vertical line connecting the bin to the legend
                    ax.axvline(x=bin_centers[i], color='black', linestyle='--', linewidth=1.5, alpha=0.7, label="Star Position")
                    
                    # Add star info to the legend
                    star_info = (
                        f"Star: {object_id}\n"
                        f"{param_name}: {star_param_value:.4g}"
                    )
                    
                    # Place the legend at the bottom with only star info
                    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), fontsize=10, frameon=True, shadow=True, title=star_info)
                    break
    
    # Add a colorbar
    norm = mcolors.Normalize(min_weight, max_weight)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label=f'{weight_name} Value')
    
    # Add labels and title
    ax.set_xlabel(param_name)
    ax.set_ylabel('Frequency')
    ax.set_title(f'Histogram of {param_name} values colored by {weight_name}')
    ax.grid(alpha=0.3)
    
    # Adjust x-axis scale for Lbol and Teff_K
    if param_name == 'Lbol_Lsun':  # Logarithmic scale for Lbol
        ax.set_xscale('log')
        ax.set_xlabel(f"{param_name} (Log Scale)")
    elif param_name == 'Teff_K':  # Linear scale with finer bins for Teff_K
        ax.set_xlim(0, np.max(param_values))  # Ensure the x-axis spans the full range
    
    # Calculate and display statistics
    mean_param = np.mean(param_values)
    median_param = np.median(param_values)
    std_param = np.std(param_values)
    min_param = np.min(param_values)
    max_param = np.max(param_values)
    
    mean_weight = np.mean(weight_values)
    median_weight = np.median(weight_values)
    min_weight_val = np.min(weight_values)
    max_weight_val = np.max(weight_values)
    
    # Create statistics text
    stats_text = (
        f"{param_name} Statistics:\n"
        f"Mean: {mean_param:.4g}\n"
        f"Median: {median_param:.4g}\n"
        f"Min: {min_param:.4g}\n"
        f"Max: {max_param:.4g}\n"
        f"Std Dev: {std_param:.4g}\n\n"
        f"{weight_name} Statistics:\n"
        f"Mean: {mean_weight:.4g}\n"
        f"Median: {median_weight:.4g}\n"
        f"Min: {min_weight_val:.4g}\n"
        f"Max: {max_weight_val:.4g}"
    )
    
    # Add statistics textbox at the bottom
    plt.figtext(0.02, -0.15, stats_text, fontsize=9, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Adjust layout to make space for the legend and statistics
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.4)  # Increase bottom margin for statistics and legend
    
    # Display the plot
    plt.show()

def setup_google_drive():
    """
    Mount Google Drive for Colab usage
    """
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("Google Drive mounted successfully.")
        return True
    except ImportError:
        print("Not running in Google Colab or drive module not available.")
        return False
    except Exception as e:
        print(f"Error mounting Google Drive: {e}")
        return False

def interactive_search(data_path, id_column='Object', weight_column='Vgfb', is_directory=True):
    """
    Create an interactive search interface for finding stars in the dataset.

    Parameters:
    data_path: Path to the directory containing .dat files or a single .dat file
    id_column: Name of the column containing star IDs (default 'Object')
    weight_column: Name of the column containing weight values (default 'Vgfb')
    is_directory: Whether data_path is a directory (True) or a single file (False)
    """
    # Load the dataset
    if is_directory:
        df = load_dataset_from_directory(data_path, id_column=id_column, weight_column=weight_column)
    else:
        df = load_dat_file(data_path, id_column=id_column, weight_column=weight_column)
    
    if df is None:
        return
    
    # Create an input box for the star ID
    object_id_input = widgets.Text(
        value='',  # Start with an empty string
        description='Star ID:',
        placeholder='Enter star ID (e.g., 4190636669164572928)',
        style={'description_width': 'initial'}
    )

    # Create a query input box
    query_input = widgets.Text(
        value='',
        description='Query:',
        placeholder='Enter query (e.g., logg=5, logg>3, Teff_K<6000)',
        style={'description_width': 'initial'}
    )

    # Create weight selection dropdown
    weight_dropdown = widgets.Dropdown(
        options=['Vgfb', 'Vgf', 'Chi2'],
        value=weight_column,
        description='Weight by:',
        style={'description_width': 'initial'}
    )

    # Create buttons for actions
    find_star_button = widgets.Button(
        description="Find Star",
        button_style='success',
        tooltip='Click to find star by ID',
        icon='star'
    )
    
    run_query_button = widgets.Button(
        description="Run Query",
        button_style='info',
        tooltip='Click to execute the query',
        icon='search'
    )
    
    rerun_button = widgets.Button(
        description="Rerun with Selected Weight",
        button_style='warning',
        tooltip='Click to rerun with the selected weight option',
        icon='refresh'
    )

    # Update the current weight
    current_weight = [weight_column]  # List to allow modification inside functions

    # Define the update function for initial plots
    def update_plot(weight_option=weight_column, object_id=None):
        clear_output(wait=True)  # Clear previous output
        
        # Update the current weight
        current_weight[0] = weight_option
        
        # Control layout
        controls_top = widgets.HBox([find_star_button, object_id_input], layout=widgets.Layout(justify_content='flex-end'))
        controls_bottom = widgets.HBox([run_query_button, query_input], layout=widgets.Layout(justify_content='flex-end'))
        rerun_controls = widgets.HBox([weight_dropdown, rerun_button])
        
        # Display the widgets in the desired order
        display(rerun_controls)
        display(controls_top)
        display(controls_bottom)
        
        # Generate all histograms with the selected weight option
        for param in ['logg', 'Teff_K', 'Lbol_Lsun']:
            if param in df.columns:
                create_weighted_histogram(df, param, weight_name=weight_option, object_id=object_id, id_column=id_column)

    # Define the query execution function
    def run_query(button_click=None):
        query = query_input.value.strip()
        if query:
            filter_stars_by_query(df, query)  # Export matching IDs to a text file
        else:
            print("Please enter a valid query.")

    # Define the find star function
    def find_star(button_click=None):
        object_id = object_id_input.value.strip()
        if object_id:
            # Update the plot with the new star ID
            update_plot(weight_option=current_weight[0], object_id=object_id)
        else:
            print("Please enter a valid star ID.")

    # Define the rerun function with the new weight option
    def rerun_with_weight(button_click=None):
        # Get the selected weight option
        new_weight = weight_dropdown.value
        
        # Get the current star ID (if any)
        object_id = object_id_input.value.strip() if object_id_input.value.strip() else None
        
        # Update plots with the new weight option
        update_plot(weight_option=new_weight, object_id=object_id)

    # Link the buttons to their respective functions
    run_query_button.on_click(run_query)
    find_star_button.on_click(find_star)
    rerun_button.on_click(rerun_with_weight)

    # Initial display - first show the weight selection
    weight_selection = widgets.VBox([
        widgets.HTML("<h3>Select Weight Option</h3>"),
        widgets.Label("Choose which parameter to use for weighting the histograms:"),
        widgets.RadioButtons(
            options=['Vgfb', 'Vgf', 'Chi2'],
            value=weight_column,
            description='Weight by:',
            style={'description_width': 'initial'}
        ),
        widgets.Button(
            description="Start Analysis",
            button_style='primary',
            tooltip='Click to begin analysis with the selected weight option',
            icon='play'
        )
    ])
    
    def start_analysis(button_click=None):
        # Get the selected weight option
        selected_weight = weight_selection.children[2].value
        
        # Update the dropdown default
        weight_dropdown.value = selected_weight
        
        # Start the analysis with the selected weight
        update_plot(weight_option=selected_weight)
    
    # Link the start button to the start_analysis function
    weight_selection.children[3].on_click(start_analysis)
    
    # Display the initial weight selection screen
    display(weight_selection)

def get_data_source():
    """
    Provide options for the user to select the data source (upload, Google Drive, or local)
    """
    # Check if running in Colab
    try:
        import google.colab
        is_colab = True
    except:
        is_colab = False
    
    if is_colab:
        # Create options for Colab
        data_source = widgets.RadioButtons(
            options=['Upload Files', 'Google Drive'],
            description='Data Source:',
            disabled=False
        )
        
        # Create path input for Google Drive
        drive_path = widgets.Text(
            value='',
            placeholder='Enter path to folder in Drive (e.g., /content/drive/MyDrive/star_data)',
            description='Drive Path:',
            disabled=True,
            style={'description_width': 'initial'}
        )
        
        # Toggle drive_path based on data_source selection
        def toggle_drive_path(change):
            if change['new'] == 'Google Drive':
                drive_path.disabled = False
            else:
                drive_path.disabled = True
        
        data_source.observe(toggle_drive_path, names='value')
        
        # Create a button to proceed
        proceed_button = widgets.Button(
            description='Proceed',
            button_style='success',
            tooltip='Click to proceed with the selected data source'
        )
        
        # Define what happens when the proceed button is clicked
        def on_proceed_click(b):
            clear_output(wait=True)
            
            if data_source.value == 'Upload Files':
                from google.colab import files
                print("Please upload your .dat files:")
                uploaded = files.upload()
                
                # Save uploaded files to a temporary directory
                import tempfile
                temp_dir = tempfile.mkdtemp()
                
                for filename, content in uploaded.items():
                    with open(f"{temp_dir}/{filename}", "wb") as f:
                        f.write(content)
                
                print(f"Files uploaded to {temp_dir}")
                interactive_search(temp_dir, id_column='Object', weight_column='Vgfb', is_directory=True)
                
            elif data_source.value == 'Google Drive':
                # Mount Google Drive
                mounted = setup_google_drive()
                if mounted:
                    path = drive_path.value.strip()
                    if not path:
                        print("Please enter a valid path to your data in Google Drive.")
                        return
                    
                    if os.path.isdir(path):
                        interactive_search(path, id_column='Object', weight_column='Vgfb', is_directory=True)
                    elif os.path.isfile(path):
                        interactive_search(path, id_column='Object', weight_column='Vgfb', is_directory=False)
                    else:
                        print(f"Path not found: {path}")
        
        proceed_button.on_click(on_proceed_click)
        
        # Display widgets
        display(data_source, drive_path, proceed_button)
    
    else:
        # If not in Colab, prompt for local path
        file_path = input("Enter the path to your .dat file or directory containing .dat files: ").strip()
        if not file_path:
            print("Error: Path cannot be empty.")
        elif os.path.isdir(file_path):
            interactive_search(file_path, id_column='Object', weight_column='Vgfb', is_directory=True)
        elif os.path.isfile(file_path):
            interactive_search(file_path, id_column='Object', weight_column='Vgfb', is_directory=False)
        else:
            print(f"Error: Path not found: {file_path}")

if __name__ == "__main__":
    get_data_source()
