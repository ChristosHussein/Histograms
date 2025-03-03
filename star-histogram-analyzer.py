import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import ipywidgets as widgets
from IPython.display import display, clear_output

def load_dataset(file_path, id_column='Object_ID', chi2_column='Chi2'):
    try:
        df = pd.read_excel(file_path)
        if id_column in df.columns:
            df[id_column] = df[id_column].astype(str)
        print("File loaded successfully.")
        return df
    except FileNotFoundError:
        print("Error: File not found. Please check the file path and name.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return None

def find_star_by_id(df, object_id, id_column='Object_ID'):
    if id_column not in df.columns:
        print(f"Error: Column '{id_column}' not found in the dataset.")
        return None
    matches = df[df[id_column] == str(object_id)]
    if len(matches) == 0:
        print(f"No stars found with Object_ID '{object_id}'.")
        return None
    elif len(matches) > 1:
        print(f"Multiple stars found with Object_ID '{object_id}'. Using the first match.")
    return matches.iloc[0]

def filter_stars_by_query(df, query):
    try:
        query = query.strip()
        if not query:
            print("Please enter a valid query.")
            return
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
        if column.strip() not in df.columns:
            print(f"Error: Column '{column.strip()}' not found in the dataset.")
            return
        try:
            value = float(value.strip())
        except ValueError:
            print(f"Error: Invalid value '{value}' in query. Queries must involve numeric comparisons.")
            return
        if operator == '==':
            matches = df[df[column.strip()] == value]
        elif operator == '>':
            matches = df[df[column.strip()] > value]
        elif operator == '<':
            matches = df[df[column.strip()] < value]
        if len(matches) == 0:
            print(f"No stars found matching the query '{query}'.")
            return
        output_filename = f"stars_{query.replace('>', '_gt_').replace('<', '_lt_').replace('=', '_eq_')}.txt"
        with open(output_filename, "w") as f:
            f.write(f"List of stars with {query}\n")
            for obj_id in matches.get('Object_ID', []):
                f.write(f"{obj_id}\n")
        print(f"Exported list of stars matching the query '{query}' to {output_filename}.")
    except Exception as e:
        print(f"Error: Invalid query format. Use 'column=5', 'column>5', or 'column<5'.")
        print(e)

def interactive_search(file_path, id_column='Object_ID', chi2_column='Chi2'):
    df = load_dataset(file_path, id_column=id_column, chi2_column=chi2_column)
    if df is None:
        return
    object_id_input = widgets.Text(value='', description='Object_ID:', placeholder='Enter Object_ID', style={'description_width': 'initial'})
    query_input = widgets.Text(value='', description='Query:', placeholder='Enter query', style={'description_width': 'initial'})
    run_query_button = widgets.Button(description="Run Query", button_style='info', tooltip='Click to execute the query', icon='search')
    find_star_button = widgets.Button(description="Find Star", button_style='success', tooltip='Click to find star by Object_ID', icon='star')
    
    def run_query(button_click=None):
        query = query_input.value.strip()
        if query:
            filter_stars_by_query(df, query)
        else:
            print("Please enter a valid query.")
    
    def find_star(button_click=None):
        object_id = object_id_input.value.strip()
        if object_id:
            find_star_by_id(df, object_id, id_column=id_column)
        else:
            print("Please enter a valid Object_ID.")
    
    run_query_button.on_click(run_query)
    find_star_button.on_click(find_star)
    display(widgets.HBox([object_id_input, query_input, run_query_button, find_star_button]))

if __name__ == "__main__":
    file_path = input("Enter the path to your Excel file: ").strip()
    if not file_path:
        print("Error: File path cannot be empty.")
    else:
        interactive_search(file_path)
