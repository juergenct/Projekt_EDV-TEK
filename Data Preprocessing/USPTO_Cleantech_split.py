import pandas as pd
import os
import multiprocessing as mp

# Create function that processes all json patent files
def extract_cleantech_patents(file):

    # Define the folder path
    folder_path = "/Users/juergenthiesen/Documents/Full Text Bulk Data USPTO 02_23 Preprocessed/JSON_Cleantech_marked"
    
    # Read in json file as Pandas dataframe
    df_USPTO = pd.read_json(os.path.join(folder_path, file))

    # Extract Cleantech patents with abstracts
    df_cleantech_patents = df_USPTO[(df_USPTO['Cleantech'] == True) & (df_USPTO['abstract'].notnull())]

    return df_cleantech_patents

if __name__ == '__main__':

    # Define the folder path
    folder_path = "/Users/juergenthiesen/Documents/Full Text Bulk Data USPTO 02_23 Preprocessed/JSON_Cleantech_marked"

    # Get a list of all USPTO files in the folder
    files = os.listdir(folder_path)

    # Filter the list to include only files with the ".json" extension
    json_files = [f for f in files if f.endswith('.json')]

    # Define multiprocessing pool
    num_CPUs = mp.cpu_count() - 2
    pool = mp.Pool(processes=num_CPUs)
    df_cleantech_patents = pool.map(extract_cleantech_patents, json_files)
    pool.close()

    # Concatenate all dataframes
    df_cleantech_patents = pd.concat(df_cleantech_patents)

    # Save the dataframe as a json file
    df_cleantech_patents.to_json("/Users/juergenthiesen/Documents/Full Text Bulk Data USPTO 02_23 Preprocessed/JSON_Cleantech_trainingdata/USPTO_Cleantech_2013_2023.json", orient='records')