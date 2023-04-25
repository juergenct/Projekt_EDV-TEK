import pandas as pd
import os

# Import the json files
# df_cleantech_patents = pd.read_json("/Users/juergenthiesen/Documents/Full Text Bulk Data USPTO 02_23 Preprocessed/JSON_Cleantech_trainingdata/USPTO_Cleantech_2013_2023_regex.json")
# df_non_cleantech_patents = pd.read_json("/Users/juergenthiesen/Documents/Full Text Bulk Data USPTO 02_23 Preprocessed/JSON_Cleantech_trainingdata/USPTO_Non_Cleantech_2013_2023.json")
df_patents = pd.read_json('/Users/juergenthiesen/Documents/Full Text Bulk Data USPTO 02_23 Preprocessed/JSON_Cleantech_trainingdata/USPTO_Cleantech_Non_Cleantech_2013_2023_regex.json')

# Print the number of patents
# print("Number of Cleantech patents: ", len(df_cleantech_patents))
# print("Number of Non-Cleantech patents: ", len(df_non_cleantech_patents))
print("Number of patents: ", len(df_patents))

# Extract first 5000 Non Cleantech patents
# df_cleantech_patents = df_cleantech_patents.iloc[:5000]
# df_non_cleantech_patents = df_non_cleantech_patents.iloc[:5000]
df_patents = df_patents.iloc[:5000]

# Write the dataframes to json files
# df_cleantech_patents.to_json("/Users/juergenthiesen/Documents/Full Text Bulk Data USPTO 02_23 Preprocessed/JSON_Cleantech_trainingdata/USPTO_Cleantech_regex_2013_2023_test.json", orient='records')
# df_non_cleantech_patents.to_json("/Users/juergenthiesen/Documents/Full Text Bulk Data USPTO 02_23 Preprocessed/JSON_Cleantech_trainingdata/USPTO_Non_Cleantech_regex_2013_2023_test.json", orient='records')
df_patents.to_json('/Users/juergenthiesen/Documents/Full Text Bulk Data USPTO 02_23 Preprocessed/JSON_Cleantech_trainingdata/USPTO_Cleantech_Non_Cleantech_regex_2013_2023_test.json', orient = 'records')

# # Print the number of duplicates
# print("Number of duplicates: ", len(df_cleantech_patents[df_cleantech_patents.duplicated(subset=['doc_number'])]))
# print("Number of duplicates: ", len(df_non_cleantech_patents[df_non_cleantech_patents.duplicated(subset=['doc_number'])]))

# # Drop duplicates
# df_cleantech_patents = df_cleantech_patents.drop_duplicates(subset=['doc_number'])
# df_non_cleantech_patents = df_non_cleantech_patents.drop_duplicates(subset=['doc_number'])

# # Write the dataframes to json files
# df_cleantech_patents.to_json("/Users/juergenthiesen/Documents/Full Text Bulk Data USPTO 02_23 Preprocessed/JSON_Cleantech_trainingdata/USPTO_Cleantech_2013_2023.json", orient='records')
# df_non_cleantech_patents.to_json("/Users/juergenthiesen/Documents/Full Text Bulk Data USPTO 02_23 Preprocessed/JSON_Cleantech_trainingdata/USPTO_Non_Cleantech_2013_2023.json", orient='records')

# # Print the number of duplicates
# print("Number of duplicates: ", len(df_cleantech_patents[df_cleantech_patents.duplicated(subset=['doc_number'])]))
# print("Number of duplicates: ", len(df_non_cleantech_patents[df_non_cleantech_patents.duplicated(subset=['doc_number'])]))