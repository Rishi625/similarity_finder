import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

class TfidfExtractor:
    def __init__(self, max_features=5000, min_df=2, max_df=0.95):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=(1, 2),
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True
        )
        self.feature_names = None
        self.tfidf_matrix = None
        
    def extract_features(self, texts):
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        return self.tfidf_matrix
    
    def transform(self, texts):
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted yet. Call extract_features first.")
        return self.vectorizer.transform(texts)
    