import pandas as pd
import os
import multiprocessing as mp

# Create function that processes all json patent files
def extract_random_patents(file):

    # Define the folder path
    folder_path = "/Users/juergenthiesen/Documents/Full Text Bulk Data USPTO 02_23 Preprocessed/JSON_Cleantech_marked"

    # Read in json file as Pandas dataframe
    df_USPTO = pd.read_json(os.path.join(folder_path, file))

    # Sample Size
    sample_size = min(int(124213/10 + 0.5), len(df_USPTO[df_USPTO['Cleantech'] == False]))

    # Extract random patents that are not Cleantech and have an abstract
    df_random_patents = df_USPTO[(df_USPTO['Cleantech'] == False) & (df_USPTO['abstract'].notnull())].sample(n=sample_size, random_state=42)

    return df_random_patents

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
    df_random_patents = pool.map(extract_random_patents, json_files)
    pool.close()

    # Concatenate all dataframes
    df_random_patents = pd.concat(df_random_patents)

    # Save the dataframe as a json file
    df_random_patents.to_json("/Users/juergenthiesen/Documents/Full Text Bulk Data USPTO 02_23 Preprocessed/JSON_Cleantech_trainingdata/USPTO_Non_Cleantech_2013_2023.json", orient='records')