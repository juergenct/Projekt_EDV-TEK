import pandas as pd
import numpy as np 
import os
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# # Import json files as Pandas dataframes
# df_2013 = pd.read_json('/Users/juergenthiesen/Documents/Full Text Bulk Data USPTO 02_23 Preprocessed/USPTO_2013.json')
# df_2014 = pd.read_json('/Users/juergenthiesen/Documents/Full Text Bulk Data USPTO 02_23 Preprocessed/USPTO_2014.json')
# df_2015 = pd.read_json('/Users/juergenthiesen/Documents/Full Text Bulk Data USPTO 02_23 Preprocessed/USPTO_2015.json')
# df_2016 = pd.read_json('/Users/juergenthiesen/Documents/Full Text Bulk Data USPTO 02_23 Preprocessed/USPTO_2016.json')
# df_2017 = pd.read_json('/Users/juergenthiesen/Documents/Full Text Bulk Data USPTO 02_23 Preprocessed/USPTO_2017.json')
# df_2018 = pd.read_json('/Users/juergenthiesen/Documents/Full Text Bulk Data USPTO 02_23 Preprocessed/USPTO_2018.json')
# df_2019 = pd.read_json('/Users/juergenthiesen/Documents/Full Text Bulk Data USPTO 02_23 Preprocessed/USPTO_2019.json')
# df_2020 = pd.read_json('/Users/juergenthiesen/Documents/Full Text Bulk Data USPTO 02_23 Preprocessed/USPTO_2020.json')
# df_2021 = pd.read_json('/Users/juergenthiesen/Documents/Full Text Bulk Data USPTO 02_23 Preprocessed/USPTO_2021.json')
# df_2022 = pd.read_json('/Users/juergenthiesen/Documents/Full Text Bulk Data USPTO 02_23 Preprocessed/USPTO_2022.json')
# df_2023 = pd.read_json('/Users/juergenthiesen/Documents/Full Text Bulk Data USPTO 02_23 Preprocessed/USPTO_2023.json')

# df_USPTO = pd.concat([df_2013, df_2014, df_2015, df_2016, df_2017, df_2018, df_2019, df_2020, df_2021, df_2022, df_2023], ignore_index=True)

# # Write the concatenated dataframe to a json file
# df_USPTO.to_json('/Users/juergenthiesen/Documents/Full Text Bulk Data USPTO 02_23 Preprocessed/USPTO_2013_2023.json')
# Use read_json with lines=True to read in JSON file
df_USPTO = pd.read_json('/Users/juergenthiesen/Documents/Full Text Bulk Data USPTO 02_23 Preprocessed/USPTO_2013_2023.json')

# Print the number of patents
print('Number of patents: ', len(df_USPTO))

# # Extract the Main CPC section from the main CPC classification
# # df_USPTO['main_cpc_section'] = df_USPTO['main_cpc'].apply(lambda x: x['section'] if x and 'section' in x else '')

# # # Plot CPC section distribution
# # df_USPTO['main_cpc_section'].value_counts().plot(kind='bar')
# # plt.legend('Main CPC section distribution')
# # plt.show()

# # Print the count of different CPC sections
# # print(df_USPTO['main_cpc_section'].value_counts())

# # Use a vectorized function to extract further CPC sections
# def extract_further_cpc_section(x):
#     if x is None:
#         return ''
#     else:
#         section_str = ''
#         for d in x:
#             if isinstance(d, dict) and 'section' in d:
#                 for letter in d['section']:
#                     if letter.isalpha() and len(letter) == 1:
#                         section_str += letter
#         return section_str

# df_USPTO['further_cpc_section'] = df_USPTO['further_cpcs'].apply(lambda x: extract_further_cpc_section(x))

# # Count the frequency of each letter
# letter_counts = pd.Series(list(''.join(df_USPTO['further_cpc_section']))).value_counts()

# # Print the count of different CPC sections
# print(letter_counts)

# # Use a vectorized operation to explode and count the sections
# df_sections = df_USPTO.explode('further_cpc_section')['further_cpc_section'].dropna()
# # letter_counts = df_sections.value_counts()

# # Print the count of different CPC sections
# print(df_USPTO['further_cpc_section'].value_counts())

# # Plot CPC section distribution
# ax = letter_counts.plot(kind='bar')
# ax.set_xlabel('Further CPC Section')
# ax.set_ylabel('Count')
# plt.legend(['Further CPC section distribution'])
# # Add text labels above each bar
# for i, v in enumerate(letter_counts):
#     ax.text(i, v + 10, str(v), ha='center')
# ax.set_title('Further CPC Section Distribution')
# plt.show()

# # Create a dictionary to store the counts for each section
# further_cpc_section_counts = {}

# # Extract the further CPC section from the main CPC classification and update the counts
# df_USPTO['further_cpc_section'] = df_USPTO['further_cpcs'].apply(lambda x: [] if x is None else [d['section'] for d in x if isinstance(d, dict) and 'section' in d])
# for sections in df_USPTO['further_cpc_section']:
#     for section in sections:
#         section_letter = section.split('.')[0]
#         further_cpc_section_counts[section_letter] = further_cpc_section_counts.get(section_letter, 0) + 1

# # Plot Further CPC section distribution
# plt.bar(further_cpc_section_counts.keys(), further_cpc_section_counts.values())
# plt.show()
