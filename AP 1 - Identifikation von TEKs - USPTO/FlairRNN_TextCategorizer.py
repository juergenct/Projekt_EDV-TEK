import os
from flair.data import Corpus
from flair.datasets import CSVClassificationCorpus

from flair.embeddings import WordEmbeddings
from flair.embeddings import DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer

# initialize corpus
data_folder_path = '/home/thiesen/Projects/Flair NLP/HuggingFace_Patent_Language_Models/data'
column_name_map = {0: "text", 1: "label"}
label_type = 'string'
corpus = CSVClassificationCorpus(data_folder_path, 
                                 column_name_map,
                                 label_type,
                                 train_file='train.csv', dev_file='dev.csv', test_file='test.csv')

# initialize Word embeddings
glove_embedding = WordEmbeddings('glove')

# initialize RNN embeddings
embedding = DocumentRNNEmbeddings([glove_embedding], hidden_size=512, reproject_words=True, reproject_words_dimension=256, rnn_type = 'RNN')

# initialize text classifier
label_dict = corpus.make_label_dictionary(label_type = label_type)
classifier = TextClassifier(embeddings = embedding,
                            label_dictionary = label_dict,
                            label_type = label_type)

# initialize trainer
trainer = ModelTrainer(classifier, corpus)

# start training
trainer.train('/home/thiesen/Projects/Flair NLP/RNN_TextCategorizer', 
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=10)