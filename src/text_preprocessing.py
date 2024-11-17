import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

class TextPreprocessor:
    def __init__(self, remove_stopwords=True, lemmatize=True, min_word_length=2):
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.min_word_length = min_word_length
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.update(['company', 'business', 'service', 'services', 'product', 'products'])

    def combine_descriptions(self, row):
        descriptions = []
        desc_columns = ['Description', 'Sourcscrub Description', 'Description.1']
        for col in desc_columns:
            if col in row and pd.notna(row[col]):
                descriptions.append(str(row[col]))
        return ' '.join(descriptions)

    def clean_text(self, text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        text = ' '.join(text.split())
        return text

    def remove_stop_words(self, tokens):
        return [token for token in tokens if token not in self.stop_words]

    def lemmatize_text(self, tokens):
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def preprocess_text(self, text):
        text = self.clean_text(text)
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if len(token) > self.min_word_length]
        if self.remove_stopwords:
            tokens = self.remove_stop_words(tokens)
        if self.lemmatize:
            tokens = self.lemmatize_text(tokens)
        return ' '.join(tokens)

    def process_dataframe(self, df):
        df_processed = df.copy()
        df_processed['combined_description'] = df_processed.apply(self.combine_descriptions, axis=1)
        df_processed['processed_description'] = df_processed['combined_description'].apply(self.preprocess_text)
        df_processed = df_processed.drop_duplicates(subset=['Organization Id'])
        return df_processed

