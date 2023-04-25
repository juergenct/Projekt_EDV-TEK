import pandas as pd
import os
import re
import nltk

# # Define the list of English words to keep
# from nltk.corpus import words as nltk_words
# words = set(nltk_words.words())

# Read in json file as Pandas dataframe
df_USPTO_preprocessed = pd.read_json('/Users/juergenthiesen/Documents/Full Text Bulk Data USPTO 02_23 Preprocessed/JSON_Cleantech_trainingdata/USPTO_Non_Cleantech_2013_2023.json')

# Loop over rows of Pandas dataframe and clean abstract and Abstract
for row in (r for i, r in df_USPTO_preprocessed.iterrows()):
    # Clean the text
    row['abstract'] = re.sub(r'\s+', ' ', row['abstract'])

    # Remove every parenthesis, square bracket, and curly bracket
    row['abstract'] = re.sub(r'[\(\)\[\]\{\}]', '', row['abstract'])

    # Remove special characters
    row['abstract'] = re.sub(r'[^a-zA-Z0-9\s]', '', row['abstract'])

    # Remove all numbers from the text
    row['abstract'] = re.sub(r'\d+', '', row['abstract'])

    # Remove hyperlinks from the text
    row['abstract'] = re.sub(r'https?://\S+', '', row['abstract'])

    # Remove all tokens that have a backslash in them
    row['abstract'] = re.sub(r'\\', '', row['abstract'])

    # If a word has a dash followed by a space at the end, remove the dash and the space
    row['abstract'] = re.sub(r'-\s+', '', row['abstract'])

    # Remove unusual brackets
    row['abstract'] = re.sub(r'\[.*?\]', '', row['abstract'])
    row['abstract'] = re.sub(r'\{.*?\}', '', row['abstract'])

    # Remove all non-ASCII characters
    row['abstract'] = re.sub(r'[^\x00-\x7F]+', '', row['abstract'])

    # Remove single characters
    row['abstract'] = re.sub(r'\s+[a-zA-Z]\s+', '', row['abstract'])

    # Remove all equation characters like $, \, ^, _, =, etc.
    row['abstract'] = re.sub(r'\\[a-zA-Z0-9]+', '', row['abstract'])

    # Remove all spaces that are not followed by a number or a letter
    row['abstract'] = re.sub(r'\s+(?=[^a-zA-Z0-9])', '', row['abstract'])

    # Remove all spaces that are not preceded by a number or a letter
    row['abstract'] = re.sub(r'(?<=[^a-zA-Z0-9])\s+', '', row['abstract'])

    # Remove all spaces that are not preceded or followed by a number or a letter
    row['abstract'] = re.sub(r'(?<=[^a-zA-Z0-9])\s+(?=[^a-zA-Z0-9])', '', row['abstract'])

    # Remove repeated characters
    row['abstract'] = re.sub(r'([a-zA-Z0-9])\1{3,}', r'\1', row['abstract'])

    # Remove repeated punctuation marks
    row['abstract'] = re.sub(r'([^\s\w]|_)\1{3,}', r'\1', row['abstract'])

    # Insert space after every punctuation mark
    row['abstract'] = re.sub(r'([^\s\w]|_)+', r'\1 ', row['abstract'])

    # Delete equal signs
    row['abstract'] = re.sub(r'=', '', row['abstract'])

    # Insert space before ( and [ if it is not preceded by a space
    # row['abstract'] = re.sub(r'(?<!\s)(\(|\[)', r' \1', row['abstract'])

    # Insert space after ) and ] if it is directly followed by a number or a letter
    # row['abstract'] = re.sub(r'(\)|\])(?=[a-zA-Z0-9])', r'\1 ', row['abstract'])

    # Remove every word that has \ in it
    row['abstract'] = re.sub(r'\w*\\+\w*', '', row['abstract'])

    # Remove single characters that are followed by a ,
    row['abstract'] = re.sub(r'\s+[a-zA-Z]\s+,', '', row['abstract'])

    # # Remove all non-english words
    # row['abstract'] = " ".join(w for w in nltk.wordpunct_tokenize(row['abstract']) if w.lower() in words or not w.isalpha())

    # Remove all leading and trailing spaces from the row['abstract']
    row['abstract'] = row['abstract'].strip()

    # Set all characters to lowercase
    row['abstract'] = row['abstract'].lower()

    # Remove punctuation
    # translator = str.maketrans('', '', string.punctuation)
    # row['abstract'] = row['abstract'].translate(translator)

# Save the dataframe as a json file
df_USPTO_preprocessed.to_json('/Users/juergenthiesen/Documents/Full Text Bulk Data USPTO 02_23 Preprocessed/JSON_Cleantech_trainingdata/USPTO_Non_Cleantech_2013_2023_regex.json', orient = 'records')
