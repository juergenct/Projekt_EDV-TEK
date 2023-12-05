import pandas as pd
import os
import multiprocessing as mp

# Create function that processes all json patent files
def preprocess_cleantech_patent(file):

    # Define the folder path
    # folder_path = "/Users/juergenthiesen/Documents/Full Text Bulk Data USPTO 02_23 Preprocessed/JSON_test"
    # Read in json file as Pandas dataframe
    # df_USPTO_preprocessed = pd.read_json(os.path.join(folder_path, file))
    df_USPTO_preprocessed = pd.read_json('/Users/juergenthiesen/Documents/Full Text Bulk Data USPTO 02_23 Preprocessed/JSON/USPTO_2023.json')
    
    # Initialize the Cleantech column with False
    df_USPTO_preprocessed['Cleantech'] = False

    # Define set containing Cleantech Patents in CPC classification scheme
    cleantech_cpc_sections = {'Y02A', 'Y02B', 'Y02C', 'Y02D', 'Y02E', 'Y02P', 'Y02T', 'Y02W'}

    # Loop over rows of Pandas dataframe and mark the number of Cleantech patents
    for i, row in df_USPTO_preprocessed.iterrows():
        further_cpcs = row.get('further_cpcs')
        if further_cpcs is not None:
            # Check whether further_cpcs is list or dictionary
            if isinstance(further_cpcs, list):
                # Iterate over the list of dictionaries in the further_cpcs column
                for c in further_cpcs:
                    # Check if section is in dictionary further_cpcs
                    if 'section' in c:
                        # Check whether the section of the CPC is in the set of Cleantech CPC sections
                        section_code = c['section'] + c['class'] + c['subclass']
                        if section_code in cleantech_cpc_sections:
                            # Mark the patent as Cleantech patent
                            df_USPTO_preprocessed.loc[i, 'Cleantech'] = True
            elif isinstance(further_cpcs, dict):
                # Check if section is in dictionary further_cpcs
                if 'section' in further_cpcs:
                    # Check whether the section of the CPC is in the set of Cleantech CPC sections
                    section_code = further_cpcs['section'] + further_cpcs['class'] + further_cpcs['subclass']
                    if section_code in cleantech_cpc_sections:
                        # Mark the patent as Cleantech patent
                        df_USPTO_preprocessed.loc[i, 'Cleantech'] = True

    # Save the dataframe to a json
    # df_USPTO_preprocessed.to_json(os.path.join('/Users/juergenthiesen/Documents/Full Text Bulk Data USPTO 02_23 Preprocessed/JSON_Preprocessed', file), orient = 'records')
    df_USPTO_preprocessed.to_json('/Users/juergenthiesen/Documents/Full Text Bulk Data USPTO 02_23 Preprocessed/JSON_Preprocessed/Cleantech_marked_2023.json', orient = 'records')

    return df_USPTO_preprocessed

if __name__ == '__main__':

    # Define the folder path
    # folder_path = "/Users/juergenthiesen/Documents/Full Text Bulk Data USPTO 02_23 Preprocessed/JSON_test"

    # Get a list of all USPTO files in the folder
    files = '/Users/juergenthiesen/Documents/Full Text Bulk Data USPTO 02_23 Preprocessed/JSON/USPTO_2023.json'

    # Filter the list to include only files with the ".json" extension
    # json_files = [f for f in files if f.endswith('.json')]
    json_files = files

    # Define multiprocessing pool
    num_CPUs = mp.cpu_count() - 2
    pool = mp.Pool(processes=num_CPUs)
    pool.map(preprocess_cleantech_patent, json_files)
    pool.close()