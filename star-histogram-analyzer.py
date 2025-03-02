"""
Star Histogram Analyzer

An interactive tool for visualizing astrophysical data with Chi-squared weighted histograms
and searchable star ID functionality.

Usage:
    python main.py path/to/your/data.xlsx
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.widgets import TextBox
import matplotlib.patches as patches
import argparse
import os
import sys

class StarHistogramAnalyzer:
    """
    A class for creating interactive histograms of star properties with chi-squared weighting
    and star ID search capabilities.
    """
    
    def __init__(self, data_path, id_column='ID', chi2_column='Chi2'):
        """
        Initialize the analyzer with data.
        
        Parameters:
        -----------
        data_path : str
            Path to the Excel file containing star data
        id_column : str
            Name of the column containing star IDs
        chi2_column : str
            Name of the column containing chi-squared values
        """
        self.id_column = id_column
        self.chi2_column = chi2_column
        self.load_data(data_path)
        self.figures = []
    
    def load_data(self, data_path):
        """
        Load data from Excel file.
        
        Parameters:
        -----------
        data_path : str
            Path to the Excel file
        """
        try:
            self.df = pd.read_excel(data_path)
            print(f"Successfully loaded data with {len(self.df)} entries.")
            print(f"Available columns: {', '.join(self.df.columns)}")
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            sys.exit(1)
    
    def create_interactive_histogram(self, param_name, bins=20):
        """
        Create an interactive histogram with search functionality.
        
        Parameters:
        -----------
        param_name : str
            The column name in the dataframe for the parameter to plot
        bins : int
            Number of bins for the histogram
            
        Returns:
        --------
        fig, ax : The figure and axis objects
        """
        # Extract the relevant columns
        param_values = self.df[param_name]
        chi2_values = self.df[self.chi2_column]
        
        # Create a figure and axis explicitly
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Create bins for the histogram
        bin_edges = np.linspace(min(param_values), max(param_values), bins+1)
        hist, _ = np.histogram(param_values, bins=bin_edges)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Calculate average chi2 for each bin
        chi2_per_bin = []
        for i in range(len(bin_edges) - 1):
            # Find values in this bin
            mask = (param_values >= bin_edges[i]) & (param_values < bin_edges[i+1])
            # Calculate average chi2 for this bin
            if np.sum(mask) > 0:
                chi2_per_bin.append(np.mean(chi2_values[mask]))
            else:
                chi2_per_bin.append(0)
        
        # Normalize chi2 values for color mapping
        min_chi2 = min(chi2_per_bin) if chi2_per_bin else 0
        max_chi2 = max(chi2_per_bin) if chi2_per_bin else 1
        
        # Create a high-contrast colormap (dark blue to dark red)
        colors = ["darkblue", "royalblue", "lightblue", "lightyellow", "orange", "red", "darkred"]
        cmap = mcolors.LinearSegmentedColormap.from_list('high_contrast', colors)
        
        # Plot the histogram with colors based on chi2 values
        for i in range(len(bin_centers)):
            # Normalize the chi2 value
            norm_value = (chi2_per_bin[i] - min_chi2) / (max_chi2 - min_chi2) if max_chi2 > min_chi2 else 0.5
            ax.bar(bin_centers[i], hist[i], width=(bin_edges[1]-bin_edges[0]), 
                   color=cmap(norm_value), alpha=0.8, 
                   edgecolor='black', linewidth=0.5)
        
        # Add a colorbar
        norm = mcolors.Normalize(min_chi2, max_chi2)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, label=f'{self.chi2_column} Value')
        
        # Add labels and title
        ax.set_xlabel(param_name)
        ax.set_ylabel('Frequency')
        ax.set_title(f'Histogram of {param_name} values colored by {self.chi2_column}')
        ax.grid(alpha=0.3)
        
        # Calculate and display statistics
        mean_param = np.mean(param_values)
        median_param = np.median(param_values)
        std_param = np.std(param_values)
        min_param = np.min(param_values)
        max_param = np.max(param_values)
        
        mean_chi2 = np.mean(chi2_values)
        median_chi2 = np.median(chi2_values)
        min_chi2 = np.min(chi2_values)
        max_chi2 = np.max(chi2_values)
        
        # Create statistics text
        stats_text = (
            f"{param_name} Statistics:\n"
            f"Mean: {mean_param:.4g}\n"
            f"Median: {median_param:.4g}\n"
            f"Min: {min_param:.4g}\n"
            f"Max: {max_param:.4g}\n"
            f"Std Dev: {std_param:.4g}\n\n"
            f"{self.chi2_column} Statistics:\n"
            f"Mean: {mean_chi2:.4g}\n"
            f"Median: {median_chi2:.4g}\n"
            f"Min: {min_chi2:.4g}\n"
            f"Max: {max_chi2:.4g}"
        )
        
        # Add statistics textbox
        plt.figtext(0.02, 0.02, stats_text, fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        # Create a text box for searching star IDs
        axbox = plt.axes([0.15, 0.9, 0.3, 0.05])
        text_box = TextBox(axbox, 'Search Star ID:', initial="")
        
        # Create a placeholder for star info
        star_info_text = plt.figtext(0.5, 0.9, "", fontsize=10,
                              bbox=dict(facecolor='lightyellow', alpha=0.9, boxstyle='round,pad=0.5'))
        
        # Store references for search elements
        search_marker = None
        search_highlight = None
        
        # Create the search function
        def search(text):
            nonlocal search_marker, search_highlight
            
            # Clear previous marker and highlight if they exist
            if search_marker:
                search_marker.remove()
                search_marker = None
            if search_highlight:
                search_highlight.remove()
                search_highlight = None
            
            # Search for the star ID in the dataframe
            try:
                # Convert input to the same type as in your dataframe
                search_id = text.strip()
                
                # Find the star in the dataframe
                star_found = self.df[self.df[self.id_column].astype(str) == search_id]
                
                if not star_found.empty:
                    star_row = star_found.iloc[0]
                    param_value = star_row[param_name]
                    chi2_value = star_row[self.chi2_column]
                    
                    # Find which bin it belongs to
                    bin_idx = np.digitize(param_value, bin_edges) - 1
                    if bin_idx >= len(bin_centers) or bin_idx < 0:
                        bin_idx = min(max(0, bin_idx), len(bin_centers) - 1)
                    
                    # Get bin height for marker placement
                    bin_height = hist[bin_idx]
                    
                    # Add marker at star's position
                    search_marker = ax.scatter(param_value, bin_height * 1.05, 
                                              color='black', marker='v', s=150, zorder=5)
                    
                    # Highlight the bin containing the star
                    bin_left = bin_edges[bin_idx]
                    bin_right = bin_edges[bin_idx + 1] if bin_idx < len(bin_edges) - 1 else bin_edges[bin_idx]
                    bin_width = bin_right - bin_left
                    
                    # Create a transparent rectangle to highlight the bin
                    search_highlight = ax.add_patch(
                        patches.Rectangle((bin_left, 0), bin_width, bin_height,
                                          facecolor='yellow', alpha=0.3, edgecolor='black', 
                                          linestyle='--', linewidth=2, zorder=4)
                    )
                    
                    # Calculate percentile of the star within the distribution
                    percentile = (self.df[param_name] <= param_value).mean() * 100
                    
                    # Update star info text
                    star_info = (
                        f"Star ID: {search_id}\n"
                        f"{param_name}: {param_value:.4g}\n"
                        f"{self.chi2_column}: {chi2_value:.4g}\n"
                        f"Percentile: {percentile:.1f}%"
                    )
                    star_info_text.set_text(star_info)
                    star_info_text.set_bbox(dict(facecolor='lightyellow', alpha=0.9, boxstyle='round,pad=0.5'))
                else:
                    star_info_text.set_text(f"Star ID '{search_id}' not found")
                    star_info_text.set_bbox(dict(facecolor='lightpink', alpha=0.9, boxstyle='round,pad=0.5'))
                    
            except Exception as e:
                star_info_text.set_text(f"Error: {str(e)}")
                star_info_text.set_bbox(dict(facecolor='lightpink', alpha=0.9, boxstyle='round,pad=0.5'))
            
            # Redraw the figure
            fig.canvas.draw_idle()
        
        # Connect the search function to the text box
        text_box.on_submit(search)
        
        # Adjust layout
        plt.tight_layout()
        # Increase the top margin for the search box
        plt.subplots_adjust(top=0.85, bottom=0.25)
        
        # Print some statistics to console as well
        print(f"{param_name} range: {min_param:.4g} to {max_param:.4g}")
        print(f"{self.chi2_column} range: {min_chi2:.4g} to {max_chi2:.4g}")
        
        # Return the figure and axis for potential further customization
        return fig, ax
    
    def visualize_parameters(self, parameters, bins=20):
        """
        Create interactive histograms for specified parameters.
        
        Parameters:
        -----------
        parameters : list
            List of column names to create histograms for
        bins : int
            Number of bins for the histograms
        """
        # Make sure plots are interactive
        plt.ion()
        
        print(f"Creating {len(parameters)} interactive histograms...")
        
        # Create each histogram in turn
        for param in parameters:
            if param in self.df.columns:
                print(f"\nCreating {param} histogram...")
                fig, ax = self.create_interactive_histogram(param, bins=bins)
                self.figures.append(fig)
            else:
                print(f"Warning: Column '{param}' not found in data.")
        
        print("\nAll interactive histograms created.")
        print("Search for stars by entering their IDs in the search boxes.")
        
        # Block to keep figures alive
        plt.show(block=True)


def main():
    """Main function to parse arguments and run the analyzer."""
    parser = argparse.ArgumentParser(
        description='Star Histogram Analyzer: Interactive visualization of star properties with chi-squared weighting.'
    )
    
    parser.add_argument('data_file', help='Path to Excel file containing star data')
    parser.add_argument('--id-column', default='ID', help='Name of the column containing star IDs (default: ID)')
    parser.add_argument('--chi2-column', default='Chi2', help='Name of the column containing chi-squared values (default: Chi2)')
    parser.add_argument('--parameters', nargs='+', default=['logg', 'Teff_K', 'Lbol_Lsun'], 
                        help='Parameters to create histograms for (default: logg Teff_K Lbol_Lsun)')
    parser.add_argument('--bins', type=int, default=20, help='Number of bins for histograms (default: 20)')
    
    args = parser.parse_args()
    
    # Check if the file exists
    if not os.path.isfile(args.data_file):
        print(f"Error: File '{args.data_file}' not found.")
        sys.exit(1)
    
    # Create the analyzer and visualize data
    analyzer = StarHistogramAnalyzer(args.data_file, args.id_column, args.chi2_column)
    analyzer.visualize_parameters(args.parameters, args.bins)


if __name__ == "__main__":
    main()
