import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import ipywidgets as widgets
from IPython.display import display, clear_output
import os
import re
import glob


# --- Data Loading Functions ---
def load_dat_file(file_path, id_column='Object_ID', weight_columns=['Vgfb', 'Vgf', 'Chi2']):
    """
    Load parameters from a .dat file.

    Parameters:
    file_path: Path to the .dat file.
    id_column: Name of the column containing star IDs (default 'Object_ID').
    weight_columns: List of columns for weighting (default ['Vgfb', 'Vgf', 'Chi2']).

    Returns:
    A DataFrame containing the star data or None if loading fails.
    """
    try:
        params = {}
        with open(file_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            if line.startswith('#'):
                line = line.lstrip('#').strip()
                match = re.search(r'(\w+\.?\w*)\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(.+)?', line)
                if match:
                    param_name = match.group(1).strip()
                    param_value = match.group(2).strip()
                    unit = match.group(3).strip() if match.group(3) else ""

                    try:
                        param_value = float(param_value)
                    except ValueError:
                        pass

                    params[param_name] = param_value

                obj_match = re.search(r'Object\s*=?\s*(\w+)', line)
                if obj_match:
                    params['Object_ID'] = obj_match.group(1).strip()

        df = pd.DataFrame([params])

        if id_column not in df.columns:
            if 'Object_ID' in df.columns:
                df[id_column] = df['Object_ID']
            else:
                df[id_column] = os.path.basename(file_path).split('.')[0]

        df[id_column] = df[id_column].astype(str)

        # Ensure weight columns exist with default values
        for weight_opt in weight_columns:
            if weight_opt not in df.columns:
                if weight_opt.lower() in [col.lower() for col in df.columns]:
                    matching_col = [col for col in df.columns if col.lower() == weight_opt.lower()][0]
                    df[weight_opt] = df[matching_col]
                else:
                    print(f"Warning: '{weight_opt}' not found in file {file_path}. Using default value of 1.0")
                    df[weight_opt] = 1.0

        # Ensure other common parameters are present
        for param in ['logg', 'Teff', 'Meta.', 'Lbol']:
            if param not in df.columns:
                if f'{param}_K' in df.columns:
                    df[param] = df[f'{param}_K']
                elif f'{param}_Lsun' in df.columns:
                    df[param] = df[f'{param}_Lsun']
                else:
                    if param == 'logg':
                        df[param] = 0.0
                    elif param == 'Teff':
                        df[param] = 5000.0
                    elif param == 'Meta.':
                        df[param] = 0.0
                    elif param == 'Lbol':
                        df[param] = 1.0

        if 'Teff' in df.columns and 'Teff_K' not in df.columns:
            df['Teff_K'] = df['Teff']

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


def load_dataset_from_directory(directory_path, id_column='Object_ID', weight_columns=['Vgfb', 'Vgf', 'Chi2']):
    """
    Load all .dat files from a directory and combine them into a single DataFrame.

    Parameters:
    directory_path: Path to the directory containing .dat files.
    id_column: Name of the column containing star IDs (default 'Object_ID').
    weight_columns: List of columns for weighting (default ['Vgfb', 'Vgf', 'Chi2']).

    Returns:
    A combined DataFrame or None if no valid files are found.
    """
    try:
        dat_files = glob.glob(os.path.join(directory_path, "**/*.dat"), recursive=True)
        if not dat_files:
            print(f"No .dat files found in {directory_path}")
            return None

        print(f"Found {len(dat_files)} .dat files in {directory_path}")
        dfs = []
        for file_path in dat_files:
            df = load_dat_file(file_path, id_column=id_column, weight_columns=weight_columns)
            if df is not None:
                dfs.append(df)

        if not dfs:
            print("No valid data files could be loaded.")
            return None

        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"Combined dataset created with {len(combined_df)} stars.")
        return combined_df

    except Exception as e:
        print(f"An error occurred while loading the dataset: {e}")
        return None


# --- Query Functions ---
def filter_stars_by_query(df, query):
    """
    Filter stars based on a query (e.g., 'logg=5', 'logg>3', 'Teff_K<6000', 'Chi2<2') and export matching IDs to a text file.

    Parameters:
    df: The DataFrame containing the dataset.
    query: A string query (e.g., 'logg=5', 'logg>3', 'Teff_K<6000', 'Chi2<2').
    
    Returns:
    A DataFrame containing the filtered results.
    """
    try:
        query = query.strip()
        if not query:
            print("Please enter a valid query.")
            return None

        # Parse the query into column, operator, and value
        if '=' in query and not '>=' in query and not '<=' in query:
            column, value = query.split('=')
            operator = '=='
        elif '>' in query and not '>=' in query:
            column, value = query.split('>')
            operator = '>'
        elif '<' in query and not '<=' in query:
            column, value = query.split('<')
            operator = '<'
        elif '>=' in query:
            column, value = query.split('>=')
            operator = '>='
        elif '<=' in query:
            column, value = query.split('<=')
            operator = '<='
        else:
            print(f"Error: Invalid query format. Use 'column=5', 'column>5', 'column<5', 'column>=5', or 'column<=5'.")
            return None

        column = column.strip().lower()
        value = float(value.strip())

        # Expanded aliases to include Chi2, Vgfb, Vgf
        column_aliases = {
            'temperature': 'Teff_K',
            'temp': 'Teff_K',
            't': 'Teff_K',
            'teff': 'Teff_K',
            'metallicity': 'Meta.',
            'met': 'Meta.',
            'chi2': 'Chi2',
            'vgfb': 'Vgfb', 
            'vgf': 'Vgf'
        }
        
        actual_column = column_aliases.get(column, next((c for c in df.columns if c.lower() == column), None))
        if actual_column is None:
            print(f"Error: Column '{column}' not found in the dataset.")
            return None

        # Apply the query
        if operator == '==':
            matches = df[df[actual_column] == value]
        elif operator == '>':
            matches = df[df[actual_column] > value]
        elif operator == '<':
            matches = df[df[actual_column] < value]
        elif operator == '>=':
            matches = df[df[actual_column] >= value]
        elif operator == '<=':
            matches = df[df[actual_column] <= value]

        # Calculate and display statistics
        percentage = len(matches) / len(df) * 100 if len(df) > 0 else 0
        print(f"Query: {query}")
        print(f"Result: {len(matches)} stars ({percentage:.2f}% of total sample size)")

        if len(matches) == 0:
            print(f"No stars found matching the query '{query}'.")
        else:
            # Export matching stars to a text file
            output_filename = f"stars_{query.replace('>', '_gt_').replace('<', '_lt_').replace('=', '_eq_')}.txt"
            with open(output_filename, "w") as f:
                f.write(f"List of stars with {query}\n")
                f.write(f"Total stars: {len(df)}, Matching stars: {len(matches)} ({percentage:.2f}%)\n\n")
                f.write("Star ID\tTeff_K\tlogg\tMeta.\tLbol_Lsun\tChi2\tVgfb\tVgf\n")
                for _, row in matches.iterrows():
                    f.write(f"{row['Object_ID']}\t{row.get('Teff_K', '')}\t{row.get('logg', '')}\t{row.get('Meta.', '')}\t{row.get('Lbol_Lsun', '')}\t{row.get('Chi2', '')}\t{row.get('Vgfb', '')}\t{row.get('Vgf', '')}\n")

            print(f"Exported list of stars matching the query '{query}' to {output_filename}.")
            
        return matches

    except Exception as e:
        print(f"Error processing query: {e}")
        print("Format should be 'column=5', 'column>5', 'column<5', 'column>=5', or 'column<=5'.")
        print("Valid columns include: 'Teff_K', 'logg', 'Meta.', 'Lbol_Lsun', 'Chi2', 'Vgfb', 'Vgf'")
        return None


def find_star_by_id(df, object_id, id_column='Object_ID'):
    """
    Find a star by its ID.

    Parameters:
    df: The DataFrame containing the dataset.
    object_id: The ID of the star to find.
    id_column: The column containing the star IDs.

    Returns:
    A Series containing the star data or None if not found.
    """
    if df is None or df.empty:
        print("No data available to search for stars.")
        return None

    # Convert object_id to string for comparison
    object_id = str(object_id)
    
    # Find the star by ID
    star = df[df[id_column].astype(str) == object_id]
    
    if len(star) == 0:
        print(f"Star with ID '{object_id}' not found in the dataset.")
        return None
    
    return star.iloc[0]


# --- Plotting Functions ---
def create_weighted_histogram(df, param_name, weight_name='Vgfb', bins=20, object_id=None, id_column='Object_ID'):
    """
    Create a histogram with weight-based coloring and optional star highlighting.

    Parameters:
    df: The DataFrame containing the dataset.
    param_name: The parameter to plot (e.g., 'logg', 'Teff_K').
    weight_name: The column with weight values (default 'Vgfb').
    bins: Number of bins for the histogram (default 20).
    object_id: Optional Object ID to highlight in the histogram.
    id_column: The column containing the star IDs (default 'Object_ID').
    """
    if df is None or df.empty:
        print("No data available to plot histograms.")
        return

    if param_name not in df.columns:
        print(f"Error: Column '{param_name}' not found in the dataset.")
        return

    if weight_name not in df.columns:
        print(f"Error: Column '{weight_name}' not found in the dataset. Using placeholder values.")
        df[weight_name] = 1.0  # Default value if weight column is missing

    param_values = df[param_name].values
    weight_values = df[weight_name].values

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
        mask = (param_values >= bin_edges[i]) & (param_values < bin_edges[i + 1])
        weight_per_bin.append(np.mean(weight_values[mask]) if np.sum(mask) > 0 else 0)

    # Normalize weight values for color mapping
    min_weight = min(weight_per_bin) if weight_per_bin and any(weight_per_bin) else 0
    max_weight = max(weight_per_bin) if weight_per_bin and any(weight_per_bin) else 1

    # Create a high-contrast colormap
    colors = ["darkblue", "royalblue", "lightblue", "lightyellow", "orange", "red", "darkred"]
    cmap = mcolors.LinearSegmentedColormap.from_list('high_contrast', colors)

    # Create the figure and axis explicitly
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the histogram with colors based on weight values
    for i in range(len(bin_centers)):
        if max_weight > min_weight:
            norm_value = (weight_per_bin[i] - min_weight) / (max_weight - min_weight)
        else:
            norm_value = 0.5
        ax.bar(
            bin_centers[i], hist[i], width=(bin_edges[i + 1] - bin_edges[i]),
            color=cmap(norm_value), alpha=0.8, edgecolor='black', linewidth=0.5
        )

    # Highlight the entire bin if an Object_ID is provided
    if object_id is not None:
        star_data = find_star_by_id(df, str(object_id), id_column=id_column)
        if star_data is not None:
            star_param_value = star_data[param_name]

            # Highlight the bin containing the star
            for i in range(len(bin_edges) - 1):
                if bin_edges[i] <= star_param_value < bin_edges[i + 1]:
                    ax.bar(
                        bin_centers[i], hist[i], width=(bin_edges[i + 1] - bin_edges[i]),
                        color='red', alpha=0.7, edgecolor='black', linewidth=1.5, hatch='/',
                        label=f"Star Bin({object_id})"
                    )
                    ax.axvline(x=star_param_value, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label="Star Position")
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

    # Add legend if a star is highlighted
    if object_id is not None:
        ax.legend()

    # Adjust x-axis scale for Lbol and Teff_K
    if param_name == 'Lbol_Lsun':  # Logarithmic scale for Lbol
        ax.set_xscale('log')
        ax.set_xlabel(f"{param_name} (Log Scale)")
    elif param_name == 'Teff_K':  # Linear scale with finer bins for Teff_K
        ax.set_xlim(0, np.max(param_values))

    # Display the plot
    plt.tight_layout()
    plt.show()


# --- Interactive Widgets ---
def interactive_search(data_path, id_column='Object_ID', weight_columns=['Vgfb', 'Vgf', 'Chi2'], is_directory=True):
    """
    Provide an interactive interface for analyzing stars.

    Parameters:
    data_path: Path to the directory containing .dat files or a single .dat file.
    id_column: Name of the column containing star IDs (default 'Object_ID').
    weight_columns: List of columns for weighting (default ['Vgfb', 'Vgf', 'Chi2']).
    is_directory: Whether data_path is a directory (True) or a single file (False).
    """
    if is_directory:
        df = load_dataset_from_directory(data_path, id_column=id_column, weight_columns=weight_columns)
    else:
        df = load_dat_file(data_path, id_column=id_column, weight_columns=weight_columns)

    if df is None or df.empty:
        print("No valid data loaded.")
        return

    # Create an input box for the star ID
    object_id_input = widgets.Text(
        value='', 
        description='Star ID:', 
        placeholder='Enter star ID (e.g., 1976077657917533952)',
        style={'description_width': 'initial'}
    )

    # Create a query input box with hint about filtering by Chi2/Vgfb/Vgf
    query_input = widgets.Text(
        value='', 
        description='Query:', 
        placeholder='e.g., Teff_K>5000, logg<3, Chi2<10, Vgfb<30',
        style={'description_width': 'initial'}
    )

    # Create weight selection dropdown
    weight_dropdown = widgets.Dropdown(
        options=weight_columns, 
        value=weight_columns[0], 
        description='Weight by:',
        style={'description_width': 'initial'}
    )

    # Create filter sliders for Chi2, Vgfb, and Vgf
    chi2_filter = widgets.FloatSlider(
        value=100, min=0, max=100000, step=5,
        description='Filter Chi2 <=', 
        continuous_update=False,
        style={'description_width': 'initial'}
    )
    
    vgfb_filter = widgets.FloatSlider(
        value=30, min=0, max=1000, step=1,
        description='Filter Vgfb <=', 
        continuous_update=False,
        style={'description_width': 'initial'}
    )
    
    vgf_filter = widgets.FloatSlider(
        value=600, min=0, max=10000, step=10,
        description='Filter Vgf <=', 
        continuous_update=False,
        style={'description_width': 'initial'}
    )

    # Create filter activation checkboxes
    use_chi2_filter = widgets.Checkbox(
        value=False, 
        description="Apply Chi2 filter", 
        indent=False
    )
    
    use_vgfb_filter = widgets.Checkbox(
        value=False, 
        description="Apply Vgfb filter", 
        indent=False
    )
    
    use_vgf_filter = widgets.Checkbox(
        value=False, 
        description="Apply Vgf filter", 
        indent=False
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
    
    apply_filters_button = widgets.Button(
        description="Apply Filters", 
        button_style='warning', 
        tooltip='Click to apply selected filters', 
        icon='filter'
    )
    
    reset_button = widgets.Button(
        description="Reset", 
        button_style='danger', 
        tooltip='Click to reset all filters and show full dataset', 
        icon='refresh'
    )

    # Store current state
    state = {
        'filtered_df': df.copy(),
        'current_weight': weight_columns[0],
        'applied_filters': [],
        'current_query': None,
        'current_star_id': None
    }

    def apply_all_filters():
        """Apply all active filters to the dataset and return the filtered dataframe"""
        filtered_df = df.copy()
        state['applied_filters'] = []
        
        # Apply Chi2 filter if enabled
        if use_chi2_filter.value:
            filtered_df = filtered_df[filtered_df['Chi2'] <= chi2_filter.value]
            state['applied_filters'].append(f"Chi2 <= {chi2_filter.value}")
        
        # Apply Vgfb filter if enabled
        if use_vgfb_filter.value:
            filtered_df = filtered_df[filtered_df['Vgfb'] <= vgfb_filter.value]
            state['applied_filters'].append(f"Vgfb <= {vgfb_filter.value}")
        
        # Apply Vgf filter if enabled
        if use_vgf_filter.value:
            filtered_df = filtered_df[filtered_df['Vgf'] <= vgf_filter.value]
            state['applied_filters'].append(f"Vgf <= {vgf_filter.value}")
        
        # Apply any current query if it exists
        if state['current_query'] is not None:
            query_result = filter_stars_by_query(filtered_df, state['current_query'])
            if query_result is not None and not query_result.empty:
                filtered_df = query_result
                state['applied_filters'].append(f"Query: {state['current_query']}")
        
        # Update the state with the filtered dataframe
        state['filtered_df'] = filtered_df
        
        # Display statistics
        percentage = len(filtered_df) / len(df) * 100 if len(df) > 0 else 0
        print(f"Applied filters: {', '.join(state['applied_filters']) if state['applied_filters'] else 'None'}")
        print(f"Result: {len(filtered_df)} stars ({percentage:.2f}% of total sample size)")
        
        return filtered_df

    def update_display():
        """Update the display with controls and histograms based on current state"""
        clear_output(wait=True)  # Clear previous output
        
        # Control layout
        filter_controls = widgets.VBox([
            widgets.HTML("<h3>Filter Controls</h3>"),
            widgets.HBox([use_chi2_filter, chi2_filter]),
            widgets.HBox([use_vgfb_filter, vgfb_filter]),
            widgets.HBox([use_vgf_filter, vgf_filter]),
            widgets.HBox([apply_filters_button, reset_button])
        ])
        
        query_controls = widgets.VBox([
            widgets.HTML("<h3>Query and Star Search</h3>"),
            widgets.HTML("<p>Example queries: 'Teff_K>5000', 'logg<3', 'Chi2<10', 'Vgfb<=20', 'Meta.=-1.5'</p>"),
            widgets.HBox([run_query_button, query_input]),
            widgets.HBox([find_star_button, object_id_input])
        ])
        
        weight_controls = widgets.VBox([
            widgets.HTML("<h3>Display Controls</h3>"),
            widgets.HBox([weight_dropdown])
        ])
        
        # Display all controls
        display(widgets.VBox([
            filter_controls,
            query_controls,
            weight_controls
        ]))
        
        # Display stats about current filters
        if state['applied_filters']:
            print(f"Current filters applied: {', '.join(state['applied_filters'])}")
            percentage = len(state['filtered_df']) / len(df) * 100 if len(df) > 0 else 0
            print(f"Showing {len(state['filtered_df'])} stars ({percentage:.2f}% of total sample size)")
        
        # Generate all histograms with the current weight option and filtered data
        for param in ['logg', 'Teff_K', 'Meta.', 'Lbol_Lsun']:
            if param in state['filtered_df'].columns:
                create_weighted_histogram(
                    state['filtered_df'],  # This ensures we're using the filtered data
                    param, 
                    weight_name=state['current_weight'],
                    object_id=state['current_star_id'],
                    id_column=id_column
                )

    def on_weight_change(change):
        """Handle weight dropdown change"""
        state['current_weight'] = change['new']
        update_display()

    def on_reset_click(b):
        """Reset all filters and show the full dataset"""
        # Reset filter checkboxes
        use_chi2_filter.value = False
        use_vgfb_filter.value = False
        use_vgf_filter.value = False
        
        # Reset query and star ID
        query_input.value = ''
        object_id_input.value = ''
        state['current_query'] = None
        state['current_star_id'] = None
        
        # Reset to full dataset
        state['filtered_df'] = df.copy()
        state['applied_filters'] = []
        
        update_display()

    def on_apply_filters_click(b):
        """Apply selected filters to the dataset"""
        apply_all_filters()
        update_display()

    def on_run_query_click(b):
        """Run the current query and update the display"""
        query = query_input.value.strip()
        if query:
            # Store the query in state
            state['current_query'] = query
            
            # Apply the query to the dataset
            query_result = filter_stars_by_query(df, query)
            
            if query_result is not None and not query_result.empty:
                # Update the filtered dataset
                state['filtered_df'] = query_result
                state['applied_filters'] = [f"Query: {query}"]
                
                # Display statistics (these will also be shown in update_display)
                percentage = len(query_result) / len(df) * 100 if len(df) > 0 else 0
                print(f"Showing {len(query_result)} stars ({percentage:.2f}% of total sample size)")
                
                # Update the display with the filtered dataset
                update_display()
            else:
                print("Query returned no results.")
        else:
            print("Please enter a valid query.")

    def on_find_star_click(b):
        """Find a star by ID and highlight it in the histograms"""
        object_id = object_id_input.value.strip()
        if object_id:
            star_data = find_star_by_id(df, object_id, id_column=id_column)
            if star_data is not None:
                state['current_star_id'] = object_id
                # Print star details
                print(f"Star details for ID {object_id}:")
                for param in ['Teff_K', 'logg', 'Meta.', 'Lbol_Lsun', 'Chi2', 'Vgfb', 'Vgf']:
                    if param in star_data:
                        print(f"  {param}: {star_data[param]}")
                
                update_display()
        else:
            print("Please enter a valid star ID.")

    # Link the widgets to their respective functions
    weight_dropdown.observe(on_weight_change, names='value')
    find_star_button.on_click(on_find_star_click)
    run_query_button.on_click(on_run_query_click)
    apply_filters_button.on_click(on_apply_filters_click)
    reset_button.on_click(on_reset_click)

    # Initial display
    update_display()


def setup_google_drive():
    """
    Mount Google Drive for Colab usage.

    Returns:
    True if mounted successfully, False otherwise.
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
def get_data_source():
    """
    Provide options for the user to select the data source (upload, Google Drive, or local path).
    """
    try:
        import google.colab
        is_colab = True
    except ImportError:
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
            drive_path.disabled = change['new'] != 'Google Drive'

        data_source.observe(toggle_drive_path, names='value')

        # Create a button to proceed
        proceed_button = widgets.Button(
            description='Proceed',
            button_style='success',
            tooltip='Click to proceed with the selected data source'
        )

        def on_proceed_click(b):
            clear_output(wait=True)

            if data_source.value == 'Upload Files':
                from google.colab import files
                print("Please upload your .dat files:")
                uploaded = files.upload()

                if uploaded:
                    import tempfile
                    temp_dir = tempfile.mkdtemp()

                    for filename, content in uploaded.items():
                        with open(os.path.join(temp_dir, filename), "wb") as f:
                            f.write(content)

                    interactive_search(temp_dir, id_column='Object_ID', weight_columns=['Vgfb', 'Vgf', 'Chi2'], is_directory=True)
                else:
                    print("No files uploaded.")

            elif data_source.value == 'Google Drive':
                mounted = setup_google_drive()
                if mounted:
                    path = drive_path.value.strip()
                    if not path:
                        print("Please enter a valid path to your data in Google Drive.")
                        return

                    if os.path.isdir(path):
                        interactive_search(path, id_column='Object_ID', weight_columns=['Vgfb', 'Vgf', 'Chi2'], is_directory=True)
                    elif os.path.isfile(path):
                        interactive_search(path, id_column='Object_ID', weight_columns=['Vgfb', 'Vgf', 'Chi2'], is_directory=False)
                    else:
                        print(f"Path not found: {path}")

        proceed_button.on_click(on_proceed_click)

        # Display widgets
        display(data_source, drive_path, proceed_button)
    else:
        file_path = input("Enter the path to your .dat file or directory containing .dat files: ").strip()
        if not file_path:
            print("Error: Path cannot be empty.")
        elif os.path.isdir(file_path):
            interactive_search(file_path, id_column='Object_ID', weight_columns=['Vgfb', 'Vgf', 'Chi2'], is_directory=True)
        elif os.path.isfile(file_path):
            interactive_search(file_path, id_column='Object_ID', weight_columns=['Vgfb', 'Vgf', 'Chi2'], is_directory=False)
        else:
            print(f"Error: Path not found: {file_path}")


if __name__ == "__main__":
    get_data_source()
