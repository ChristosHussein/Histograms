# Star Histogram Analyzer

An interactive tool for visualizing astrophysical data with Chi-squared weighted histograms and searchable star ID functionality.




## Overview
This project analyzes star data using Python and generates interactive histograms. It allows users to search for stars by `Object_ID` and filter stars based on specific queries.

## Features
- Interactive histograms for parameters such as `logg`, `Teff_K`, and `Lbol_Lsun`.
- Search for stars by `Object_ID` and highlight their bins with a dashed vertical line connecting to the legend at the bottom.
- Export lists of stars matching specific queries (e.g., `teff>10`) to text files.
- Two buttons for distinct operations:
  - "Find Star": Highlights the bin of a star based on its `Object_ID` and displays its information in the legend.
  - "Run Query": Filters stars based on a condition and exports results to a text file.

## Requirements
- Python 3.x
- Required libraries:
  - pandas
  - matplotlib
  - numpy
  - openpyxl
  - ipywidgets

Install dependencies using:
```bash
pip install -r requirements.txt

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib

## Installation

Clone this repository:

```bash
git clone https://github.com/ChristosHussein/star-histogram-analyzer.git
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
Object_ID	          RA_deg	          DEC_deg	          Av	    D_pc	  Teff_K  logg	Chi2	      Lbol_Lsun
2056435602670418688	306.604843860052	35.32039589761002	2,73794	1705,2	3500	  0,5	  6001,208956	117,5143635

The tool expects an Excel file (.xlsx) with at least:
- A column for star IDs
- A column for Chi-squared values
- Columns for each parameter to be visualized

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
