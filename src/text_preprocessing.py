import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import logging

logging.basicConfig(level=logging.INFO)

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

class TextPreprocessor:
    def __init__(self, remove_stopwords=True, lemmatize=True):
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.update(['company', 'business', 'service', 'services', 'product', 'products'])

    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = ' '.join(text.split())
        return text

    def remove_stop_words(self, tokens):
        return [token for token in tokens if token not in self.stop_words]

    def lemmatize_text(self, tokens):
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def preprocess_text(self, text):
        text = self.clean_text(text)
        tokens = word_tokenize(text)
        if self.remove_stopwords:
            tokens = self.remove_stop_words(tokens)
        if self.lemmatize:
            tokens = self.lemmatize_text(tokens)
        return ' '.join(tokens)

    def process_dataframe(self, df):
        df_processed = df.copy()
        df_processed.fillna('', inplace=True)
        df_processed['combined_description'] = df_processed['Description'] + df_processed['Sourcscrub Description'] + df_processed['Description.1']
        df_processed['combined_description'] = df_processed['combined_description'].apply(self.preprocess_text)
        df_processed = df_processed.drop_duplicates(subset=['Organization Id'])

        return df_processed

