import pandas as pd
import os

# Read in json file as Pandas dataframe
df_USPTO_Cleantech = pd.read_json('/Users/juergenthiesen/Documents/Full Text Bulk Data USPTO 02_23 Preprocessed/JSON_Cleantech_trainingdata/USPTO_Cleantech_2013_2023_regex.json')
df_USPTO_Non_Cleantech = pd.read_json('/Users/juergenthiesen/Documents/Full Text Bulk Data USPTO 02_23 Preprocessed/JSON_Cleantech_trainingdata/USPTO_Non_Cleantech_2013_2023_regex.json')

# Drop all columns except 'abstract' and 'Cleantech'
df_USPTO_Cleantech = df_USPTO_Cleantech[['abstract', 'Cleantech']]
df_USPTO_Non_Cleantech = df_USPTO_Non_Cleantech[['abstract', 'Cleantech']]

# Merge the two dataframes randomly
df_USPTO = pd.concat([df_USPTO_Cleantech, df_USPTO_Non_Cleantech], ignore_index=True)

# Randomly shuffle the rows of the dataframe
df_USPTO = df_USPTO.sample(frac=1).reset_index(drop=True)

# Save the dataframe as a json file
df_USPTO.to_json('/Users/juergenthiesen/Documents/Full Text Bulk Data USPTO 02_23 Preprocessed/JSON_Cleantech_trainingdata/USPTO_Cleantech_Non_Cleantech_2013_2023_regex.json')