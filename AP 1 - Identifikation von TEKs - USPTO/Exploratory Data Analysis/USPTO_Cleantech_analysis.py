import pandas as pd
import os
import multiprocessing as mp

# Create function that processes all json patent files
def extract_cleantech_patent(file):
    
    # Define the folder path
    folder_path = "/Users/juergenthiesen/Documents/Full Text Bulk Data USPTO 02_23 Preprocessed/JSON"

    # Read in json file as Pandas dataframe
    df_USPTO = pd.read_json(os.path.join(folder_path, file))

    # Define set containing Cleantech Patents in CPC classification scheme
    cleantech_cpc_sections = {'Y02A', 'Y02B', 'Y02C', 'Y02D', 'Y02E', 'Y02P', 'Y02T', 'Y02W'}

    # Initialize Pandas dataframe to store Cleantech patents
    df_cleantech = pd.DataFrame()

    # Loop over rows of Pandas dataframe and extract the Cleantech patents
    for row in (r for i, r in df_USPTO.iterrows()):
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
                            # Extract the Cleantech patent and store in Cleantech patent dataframe
                            df_cleantech = pd.concat([df_cleantech, row])
                        else:
                            continue
            elif isinstance(further_cpcs, dict):
                # Check if section is in dictionary further_cpcs
                if 'section' in further_cpcs:
                    # Check whether the section of the CPC is in the set of Cleantech CPC sections
                    section_code = further_cpcs['section'] + further_cpcs['class'] + further_cpcs['subclass']
                    if section_code in cleantech_cpc_sections:
                        # Extract the Cleantech patent and store in Cleantech patent dataframe
                        df_cleantech = pd.concat([df_cleantech, row])
                        continue
                else:
                    continue

    # # Remove duplicates
    # df_cleantech = df_cleantech.drop_duplicates()
    
    return df_cleantech

if __name__ == '__main__':

    # Define the folder path
    folder_path = "/Users/juergenthiesen/Documents/Full Text Bulk Data USPTO 02_23 Preprocessed/JSON"

    # Get a list of all USPTO files in the folder
    files = os.listdir(folder_path)

    # Filter the list to include only files with the ".json" extension
    json_files = [f for f in files if f.endswith('.json')]

    # Define multiprocessing pool
    num_CPUs = mp.cpu_count() - 2
    pool = mp.Pool(processes=num_CPUs)
    temp_cleantech_df = pool.map(extract_cleantech_patent, json_files)
    pool.close()

    # Concatenate the individual dataframes into one dataframe
    df_cleantech = pd.concat(temp_cleantech_df)

    # Remove duplicates
    df_cleantech = df_cleantech.drop_duplicates()

    # Write the dataframe to a json file
    df_cleantech.to_json('/Users/juergenthiesen/Documents/Full Text Bulk Data USPTO 02_23 Preprocessed/USPTO_Cleantech_2013_2023.json', orient = 'records') 