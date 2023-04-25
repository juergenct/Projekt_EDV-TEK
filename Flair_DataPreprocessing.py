import pandas as pd
import json
import os
import csv

''' Load Data and Save Data as CSV'''
# Load data
df_USPTO_test_val = pd.read_json('/Users/juergenthiesen/Documents/USPTO Data/Full Text Bulk Data USPTO 02_23 Preprocessed/JSON_Cleantech_trainingdata/USPTO_Cleantech_Non_Cleantech_2013_2023_regex_test.json')

# Split test data into test and validation data (80/20)
df_USPTO_val = df_USPTO_test_val[:int(0.5*len(df_USPTO_test_val))]
df_USPTO_test = df_USPTO_test_val[int(0.5*len(df_USPTO_test_val)):]
# Set keys to 0 to n for test and validation data
df_USPTO_val = df_USPTO_val.reset_index(drop=True)
df_USPTO_test = df_USPTO_test.reset_index(drop=True)

df_USPTO_train = pd.read_json('/Users/juergenthiesen/Documents/USPTO Data/Full Text Bulk Data USPTO 02_23 Preprocessed/JSON_Cleantech_trainingdata/USPTO_Cleantech_Non_Cleantech_2013_2023_regex_train.json')

# Rename columns from 'Cleantech' to 'label' and 'abstract' to 'text'
df_USPTO_val = df_USPTO_val.rename(columns={'abstract': 'text', 'Cleantech': 'label'})
df_USPTO_test = df_USPTO_test.rename(columns={'abstract': 'text', 'Cleantech': 'label'})
df_USPTO_train = df_USPTO_train.rename(columns={'abstract': 'text', 'Cleantech': 'label'})

# Reframe labels from False to Not Cleantech and True to Cleantech
df_USPTO_val['label'] = df_USPTO_val['label'].apply(lambda x: 'Not Cleantech' if x == False else 'Cleantech')
df_USPTO_test['label'] = df_USPTO_test['label'].apply(lambda x: 'Not Cleantech' if x == False else 'Cleantech')
# df_USPTO_train['label'] = df_USPTO_train['label'].apply(lambda x: 'Not Cleantech' if 'label' == {'Cleantech_Positive': False, 'Cleantech_Negative': True} else 'Cleantech')

# convert dictionary column to string
df_USPTO_train['label'] = df_USPTO_train['label'].apply(json.dumps)
df_USPTO_train['label'] = df_USPTO_train['label'].replace({"{\"Cleantech_Positive\": true, \"Cleantech_Negative\": false}": "Cleantech", "{\"Cleantech_Positive\": false, \"Cleantech_Negative\": true}": "Not Cleantech"})

df_USPTO_train.drop(df_USPTO_train[df_USPTO_train['label'] == ''].index, inplace=True)
df_USPTO_val.drop(df_USPTO_val[df_USPTO_val['label'] == ''].index, inplace=True)
df_USPTO_test.drop(df_USPTO_test[df_USPTO_test['label'] == ''].index, inplace=True)

df_USPTO_train.drop(df_USPTO_train[df_USPTO_train['text'] == ''].index, inplace=True)
df_USPTO_val.drop(df_USPTO_val[df_USPTO_val['text'] == ''].index, inplace=True)
df_USPTO_test.drop(df_USPTO_test[df_USPTO_test['text'] == ''].index, inplace=True)

# Drop rows where label is not 'Cleantech' or 'Not Cleantech'
df_USPTO_train = df_USPTO_train[df_USPTO_train['label'].isin(['Cleantech', 'Not Cleantech'])]
df_USPTO_val = df_USPTO_val[df_USPTO_val['label'].isin(['Cleantech', 'Not Cleantech'])]
df_USPTO_test = df_USPTO_test[df_USPTO_test['label'].isin(['Cleantech', 'Not Cleantech'])]

# Drop the separator '\t' from the text
# df_USPTO_train['text'] = df_USPTO_train['text'].str.replace('\t', ' ')
# df_USPTO_val['text'] = df_USPTO_val['text'].str.replace('\t', ' ')
# df_USPTO_test['text'] = df_USPTO_test['text'].str.replace('\t', ' ')

# # Print dimensions of dataframes
# print('Dimensions of training data: ', df_USPTO_train.shape)
# print('Dimensions of validation data: ', df_USPTO_val.shape)
# print('Dimensions of test data: ', df_USPTO_test.shape)

# Save the dataframes as csv files
df_USPTO_train.to_csv('/Users/juergenthiesen/Documents/USPTO Data/Full Text Bulk Data USPTO 02_23 Preprocessed/JSON_Cleantech_trainingdata/Flair Model/train.csv', index=False)
df_USPTO_val.to_csv('/Users/juergenthiesen/Documents/USPTO Data/Full Text Bulk Data USPTO 02_23 Preprocessed/JSON_Cleantech_trainingdata/Flair Model/dev.csv', index=False)
df_USPTO_test.to_csv('/Users/juergenthiesen/Documents/USPTO Data/Full Text Bulk Data USPTO 02_23 Preprocessed/JSON_Cleantech_trainingdata/Flair Model/test.csv', index=False)

# df_USPTO_train.to_csv('/Users/juergenthiesen/Documents/USPTO Data/Full Text Bulk Data USPTO 02_23 Preprocessed/JSON_Cleantech_trainingdata/Flair Model/train.csv', sep='|', index=False)
# df_USPTO_val.to_csv('/Users/juergenthiesen/Documents/USPTO Data/Full Text Bulk Data USPTO 02_23 Preprocessed/JSON_Cleantech_trainingdata/Flair Model/dev.csv', sep='|', index=False)
# df_USPTO_test.to_csv('/Users/juergenthiesen/Documents/USPTO Data/Full Text Bulk Data USPTO 02_23 Preprocessed/JSON_Cleantech_trainingdata/Flair Model/test.csv', sep='|', index=False)

# df_USPTO_train.to_csv('/Users/juergenthiesen/Documents/USPTO Data/Full Text Bulk Data USPTO 02_23 Preprocessed/JSON_Cleantech_trainingdata/Flair Model/train.csv', sep='\t', quoting=csv.QUOTE_ALL, index=False)
# df_USPTO_val.to_csv('/Users/juergenthiesen/Documents/USPTO Data/Full Text Bulk Data USPTO 02_23 Preprocessed/JSON_Cleantech_trainingdata/Flair Model/dev.csv', sep='\t', quoting=csv.QUOTE_ALL, index=False)
# df_USPTO_test.to_csv('/Users/juergenthiesen/Documents/USPTO Data/Full Text Bulk Data USPTO 02_23 Preprocessed/JSON_Cleantech_trainingdata/Flair Model/test.csv', sep='\t', quoting=csv.QUOTE_ALL, index=False)