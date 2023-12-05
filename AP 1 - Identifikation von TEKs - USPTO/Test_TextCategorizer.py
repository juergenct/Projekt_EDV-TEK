import pandas as pd
import os
import flair
from termcolor import colored
from flair.models import TextClassifier
from flair.data import Sentence
# import streamlit as st

# Load the Flair model from folder path
model_path = '/Users/juergenthiesen/Documents/USPTO Data/Full Text Bulk Data USPTO 02_23 Preprocessed/JSON_Cleantech_trainingdata/Flair Model/CNN_TextCategorizer/best-model.pt'
model = TextClassifier.load(model_path)

''' Testing the model on test data '''
# Load the data
df_USPTO = pd.read_json('/Users/juergenthiesen/Documents/USPTO Data/Full Text Bulk Data USPTO 02_23 Preprocessed/JSON_Cleantech_trainingdata/Flair Model/data/USPTO_Cleantech_Non_Cleantech_2013_2023_regex_test.json')
df_USPTO = df_USPTO[['abstract', 'Cleantech']].dropna()

test_data = df_USPTO.values.tolist()

''' Testing '''
print(colored("Testing the model on {} examples ...".format(len(test_data)), 'green'))
true_positives = 0
false_positives = 0
false_negatives = 0
null_values = 0

# # Streamlit
# st.title("ClimateBERT Text Categorizer")
# with st.form():
#     text = st.text_area("Enter text here:")
#     submit_button = st.form_submit_button(label='Submit')

# if submit_button:
#     sentence = Sentence(text)
#     model.predict(sentence)
#     cat = (sentence._printout_labels(add_score= False))
#     cat = cat[1:]
#     st.write(cat)

for text, label in test_data:
    
    # Read in text
    sentence = Sentence(text)
    model.predict(sentence)
    cat = (sentence._printout_labels(add_score= False))
    cat = cat[1:]

    # Check if the predicted category is correct
    if cat == 'Cleantech' and label == True:
        true_positives += 1
    elif cat == 'Not Cleantech' and label == False:
        true_positives += 1
    elif cat == 'Cleantech' and label == False:
        false_positives += 1
    elif cat == 'Not Cleantech' and label == True:
        false_negatives += 1

print(colored("True Positives: {}\tFalse Positives: {}\tFalse Negatives: {}\tNull Values: {}".format(true_positives, false_positives, false_negatives, null_values), 'red'))

# Calculate the precision, recall and f1 score
accuracy = true_positives / (true_positives + false_positives + false_negatives)
precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1 = 2 * (precision * recall) / (precision + recall)

print(colored("Accuracy: {:.3f}\tPrecision: {:.3f}\tRecall: {:.3f}\tF1-Score: {:.3f}".format(accuracy, precision, recall, f1), 'red'))