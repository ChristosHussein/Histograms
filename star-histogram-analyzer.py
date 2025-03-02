# main_script.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import ipywidgets as widgets
from IPython.display import display, clear_output

def load_dataset(file_path, id_column='Object_ID', chi2_column='Chi2'):
    """
    Load the dataset from an Excel file and clean the Object_ID column.

    Parameters:
    file_path: Path to the Excel file
    id_column: Name of the column containing star IDs (default 'Object_ID')
    chi2_column: Name of the column containing Chi2 values (default 'Chi2')

    Returns:
    A cleaned DataFrame or None if loading fails
    """
    try:
        df = pd.read_excel(file_path)  # Load the Excel file
        
        # Clean the ID column: Convert to string to handle any format (numeric or alphanumeric)
        if id_column in df.columns:
            df[id_column] = df[id_column].astype(str)  # Treat Object_ID as a string
        
        print("File loaded successfully.")
        return df
    except FileNotFoundError:
        print("Error: File not found. Please check the file path and name.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return None

def find_star_by_id(df, object_id, id_column='Object_ID'):
    """
    Find a star in the dataset by its Object_ID (string-based matching).

    Parameters:
    df: The DataFrame containing the dataset
    object_id: The ID of the star to find (as a string)
    id_column: The column containing the star IDs (default 'Object_ID')

    Returns:
    A DataFrame row containing the star data, or None if not found
    """
    if id_column not in df.columns:
        print(f"Error: Column '{id_column}' not found in the dataset.")
        return None
    
    # Filter the dataset for the exact match of the Object_ID (string comparison)
    matches = df[df[id_column] == str(object_id)]  # Ensure both sides are strings
    
    if len(matches) == 0:
        print(f"No stars found with Object_ID '{object_id}'.")
        return None
    elif len(matches) > 1:
        print(f"Multiple stars found with Object_ID '{object_id}'. Using the first match.")
    
    return matches.iloc[0]

def filter_stars_by_query(df, query):
    """
    Filter stars based on a query (e.g., 'logg=5', 'logg>3', 'Teff_K<6000') and export matching Object_IDs to a text file.

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
        
        # Ensure the column exists
        if column.strip() not in df.columns:
            print(f"Error: Column '{column.strip()}' not found in the dataset.")
            return
        
        # Convert value to numeric if possible (only for queries, not Object_ID)
        try:
            value = float(value.strip())
        except ValueError:
            print(f"Error: Invalid value '{value}' in query. Queries must involve numeric comparisons.")
            return
        
        # Apply the query
        if operator == '==':
            matches = df[df[column.strip()] == value]
        elif operator == '>':
            matches = df[df[column.strip()] > value]
        elif operator == '<':
            matches = df[df[column.strip()] < value]
        else:
            print(f"Error: Unsupported operator '{operator}'.")
            return
        
        if len(matches) == 0:
            print(f"No stars found matching the query '{query}'.")
            return
        
        # Export matching Object_IDs to a text file
        output_filename = f"stars_{query.replace('>', '_gt_').replace('<', '_lt_').replace('=', '_eq_')}.txt"
        with open(output_filename, "w") as f:
            f.write(f"List of stars with {query}\n")
            for obj_id in matches.get('Object_ID', []):
                f.write(f"{obj_id}\n")
        
        print(f"Exported list of stars matching the query '{query}' to {output_filename}.")
    except Exception as e:
        print(f"Error: Invalid query format. Use 'column=5', 'column>5', or 'column<5'.")
        print(e)

def create_chi2_weighted_histogram(df, param_name, chi2_name='Chi2', bins=20, object_id=None, id_column='Object_ID'):
    """
    Create a histogram with chi2-based coloring and optional star highlighting.

    Parameters:
    df: The DataFrame containing the dataset
    param_name: The parameter to plot (e.g., 'logg', 'Teff_K')
    chi2_name: The column with Chi2 values (default 'Chi2')
    bins: Number of bins for the histogram (default 20)
    object_id: Optional Object_ID to highlight in the histogram
    id_column: The column containing the star IDs (default 'Object_ID')
    """
    if param_name not in df.columns or chi2_name not in df.columns:
        print(f"Error: Columns '{param_name}' or '{chi2_name}' not found in the dataset.")
        return
    
    # Extract the relevant columns
    param_values = df[param_name]
    chi2_values = df[chi2_name]
    
    # Create a figure and axis explicitly
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create bins for the histogram
    bin_edges = np.linspace(min(param_values), max(param_values), bins + 1)
    hist, _ = np.histogram(param_values, bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Calculate average chi2 for each bin
    chi2_per_bin = []
    for i in range(len(bin_edges) - 1):
        mask = (param_values >= bin_edges[i]) & (param_values < bin_edges[i+1])
        chi2_per_bin.append(np.mean(chi2_values[mask]) if np.sum(mask) > 0 else 0)
    
    # Normalize chi2 values for color mapping
    min_chi2 = min(chi2_per_bin)
    max_chi2 = max(chi2_per_bin)
    
    # Create a high-contrast colormap
    colors = ["darkblue", "royalblue", "lightblue", "lightyellow", "orange", "red", "darkred"]
    cmap = mcolors.LinearSegmentedColormap.from_list('high_contrast', colors)
    
    # Plot the histogram with colors based on chi2 values
    for i in range(len(bin_centers)):
        norm_value = (chi2_per_bin[i] - min_chi2) / (max_chi2 - min_chi2) if max_chi2 > min_chi2 else 0.5
        ax.bar(
            bin_centers[i], hist[i], width=(bin_edges[1]-bin_edges[0]), 
            color=cmap(norm_value), alpha=0.8, edgecolor='black', linewidth=0.5
        )
    
    # Highlight the entire bin if an Object_ID is provided
    if object_id is not None:
        star_data = find_star_by_id(df, str(object_id), id_column=id_column)
        if star_data is not None:
            star_param_value = star_data[param_name]
            
            # Highlight the bin containing the star
            for i in range(len(bin_edges) - 1):
                if bin_edges[i] <= star_param_value < bin_edges[i+1]:
                    ax.bar(
                        bin_centers[i], hist[i], width=(bin_edges[1]-bin_edges[0]), 
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
    norm = mcolors.Normalize(min_chi2, max_chi2)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label=f'{chi2_name} Value')
    
    # Add labels and title
    ax.set_xlabel(param_name)
    ax.set_ylabel('Frequency')
    ax.set_title(f'Histogram of {param_name} values colored by {chi2_name}')
    ax.grid(alpha=0.3)
    
    # Calculate and display statistics
    mean_param = np.mean(param_values)
    median_param = np.median(param_values)
    std_param = np.std(param_values)
    min_param = np.min(param_values)
    max_param = np.max(param_values)
    
    mean_chi2 = np.mean(chi2_values)
    median_chi2 = np.median(chi2_values)
    min_chi2_val = np.min(chi2_values)
    max_chi2_val = np.max(chi2_values)
    
    # Create statistics text
    stats_text = (
        f"{param_name} Statistics:\n"
        f"Mean: {mean_param:.4g}\n"
        f"Median: {median_param:.4g}\n"
        f"Min: {min_param:.4g}\n"
        f"Max: {max_param:.4g}\n"
        f"Std Dev: {std_param:.4g}\n\n"
        f"{chi2_name} Statistics:\n"
        f"Mean: {mean_chi2:.4g}\n"
        f"Median: {median_chi2:.4g}\n"
        f"Min: {min_chi2_val:.4g}\n"
        f"Max: {max_chi2_val:.4g}"
    )
    
    # Add statistics textbox at the bottom
    plt.figtext(0.02, -0.15, stats_text, fontsize=9, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Adjust layout to make space for the legend and statistics
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.4)  # Increase bottom margin for statistics and legend
    
    # Display the plot
    plt.show()

def interactive_search(file_path, id_column='Object_ID', chi2_column='Chi2'):
    """
    Create an interactive search interface for finding stars in the dataset.

    Parameters:
    file_path: Path to the Excel file
    id_column: Name of the column containing star IDs (default 'Object_ID')
    chi2_column: Name of the column containing Chi2 values (default 'Chi2')
    """
    # Load the dataset
    df = load_dataset(file_path, id_column=id_column, chi2_column=chi2_column)
    if df is None:
        return
    
    # Create an input box for the Object_ID
    object_id_input = widgets.Text(
        value='',  # Start with an empty string
        description='Object_ID:',
        placeholder='Enter Object_ID (e.g., AFGFSD1223453)',
        style={'description_width': 'initial'}
    )

    # Create a query input box
    query_input = widgets.Text(
        value='',
        description='Query:',
        placeholder='Enter query (e.g., logg=5, logg>3, Teff_K<6000)',
        style={'description_width': 'initial'}
    )

    # Create buttons for actions
    run_query_button = widgets.Button(
        description="Run Query",
        button_style='info',
        tooltip='Click to execute the query',
        icon='search'
    )

    find_star_button = widgets.Button(
        description="Find Star",
        button_style='success',
        tooltip='Click to find star by Object_ID',
        icon='star'
    )

    # Define the update function for Object_ID search
    def update_plot(change=None):
        clear_output(wait=True)  # Clear previous output
        
        # Generate all histograms initially without highlighting any star
        for param in ['logg', 'Teff_K', 'Lbol_Lsun']:
            if param in df.columns:
                create_chi2_weighted_histogram(df, param, chi2_name=chi2_column, id_column=id_column)

        # Re-display the widgets
        display(widgets.HBox([object_id_input, query_input, run_query_button, find_star_button]))

    # Define the query execution function
    def run_query(button_click=None):
        query = query_input.value.strip()
        if query:
            filter_stars_by_query(df, query)  # Export matching Object_IDs to a text file
        else:
            print("Please enter a valid query.")

    # Define the find star function
    def find_star(button_click=None):
        object_id = object_id_input.value.strip()
        if object_id:
            # Update the plot with the new Object_ID
            for param in ['logg', 'Teff_K', 'Lbol_Lsun']:
                if param in df.columns:
                    create_chi2_weighted_histogram(df, param, chi2_name=chi2_column, object_id=object_id, id_column=id_column)
        else:
            print("Please enter a valid Object_ID.")

    # Link the "Run Query" button to the query execution function
    run_query_button.on_click(run_query)

    # Link the "Find Star" button to the find star function
    find_star_button.on_click(find_star)

    # Initial plots (no star highlighted)
    display(widgets.HBox([object_id_input, query_input, run_query_button, find_star_button]))
    for param in ['logg', 'Teff_K', 'Lbol_Lsun']:
        if param in df.columns:
            create_chi2_weighted_histogram(df, param, chi2_name=chi2_column, id_column=id_column)

if __name__ == "__main__":
    # Prompt the user for the file path
    file_path = input("Enter the path to your Excel file: ").strip()
    if not file_path:
        print("Error: File path cannot be empty.")
    else:
        interactive_search(file_path)
