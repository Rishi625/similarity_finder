import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import spacy
import contractions

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

class TextPreprocessor:
    def __init__(self, remove_stopwords=True, lemmatize=True, min_word_length=2):
        """
        Initialize the text preprocessor with configurable options
        
        Parameters:
        - remove_stopwords: Whether to remove stopwords
        - lemmatize: Whether to perform lemmatization
        - min_word_length: Minimum length of words to keep
        """
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.min_word_length = min_word_length
        
        # Initialize tools
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        # Add custom stopwords relevant to company descriptions
        self.stop_words.update(['company', 'business', 'service', 'services', 'product', 'products'])
        
    def combine_descriptions(self, row):
        """Combine multiple description fields into one"""
        descriptions = []
        
        # List of possible description columns
        desc_columns = ['Description', 'Sourcscrub Description', 'Description.1']
        
        for col in desc_columns:
            if col in row and pd.notna(row[col]):
                descriptions.append(str(row[col]))
                
        return ' '.join(descriptions)
    
    def expand_contractions(self, text):
        """Expand contractions like don't to do not"""
        return contractions.fix(text)
    
    def clean_text(self, text):
        """Basic text cleaning"""
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Expand contractions
        text = self.expand_contractions(text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def remove_stop_words(self, tokens):
        """Remove stop words from tokenized text"""
        return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize_text(self, tokens):
        """Lemmatize tokens to their root form"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess_text(self, text):
        """Main preprocessing pipeline"""
        # Basic cleaning
        text = self.clean_text(text)
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Remove short words
        tokens = [token for token in tokens if len(token) > self.min_word_length]
        
        # Remove stop words if enabled
        if self.remove_stopwords:
            tokens = self.remove_stop_words(tokens)
        
        # Lemmatization if enabled
        if self.lemmatize:
            tokens = self.lemmatize_text(tokens)
        
        return ' '.join(tokens)
    
    def process_dataframe(self, df):
        """Process entire dataframe"""
        # Create copy to avoid modifying original
        df_processed = df.copy()
        
        # Combine descriptions
        df_processed['combined_description'] = df_processed.apply(self.combine_descriptions, axis=1)
        
        # Preprocess combined description
        df_processed['processed_description'] = df_processed['combined_description'].apply(self.preprocess_text)
        
        # Remove duplicates based on Organization Id
        df_processed = df_processed.drop_duplicates(subset=['Organization Id'])
        
        return df_processed

