# Star Histogram Analyzer

An interactive tool for visualizing astrophysical data with Chi-squared weighted histograms and searchable star ID functionality.

![example-visualization](example.png)

## Features

- Create interactive histograms of star properties colored by Chi-squared values
- Search for specific stars by ID and highlight their position in the distribution
- View detailed statistics about each parameter and Chi-squared distribution
- Interactive search functionality to locate specific stars within the histograms
- Automatic percentile calculation to see where stars fall in the distribution

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib

## Installation

Clone this repository:

```bash
git clone https://github.com/yourusername/star-histogram-analyzer.git
cd star-histogram-analyzer
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the analyzer with your data file:

```bash
python main.py path/to/your/data.xlsx
```

### Command-line Options

- `--id-column`: Specify the column name containing star IDs (default: 'ID')
- `--chi2-column`: Specify the column name containing chi-squared values (default: 'Chi2')
- `--parameters`: List of parameters to create histograms for (default: 'logg Teff_K Lbol_Lsun')
- `--bins`: Number of bins for histograms (default: 20)

Example with all options:

```bash
python main.py my_stars.xlsx --id-column StarID --chi2-column ChiSquared --parameters mass radius luminosity --bins 30
```

## Interactive Search

1. Once the histograms are displayed, use the search box at the top of each window
2. Enter a star ID and press Enter
3. The star's position will be highlighted in the histogram
4. Information about the star will be displayed including:
   - Parameter value
   - Chi-squared value
   - Percentile position in the distribution

## Data Format

The tool expects an Excel file (.xlsx) with at least:
- A column for star IDs
- A column for Chi-squared values
- Columns for each parameter to be visualized

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
