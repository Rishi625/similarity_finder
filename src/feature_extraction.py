from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union, Tuple
import torch
from tqdm import tqdm

class FeatureExtractor:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the feature extractor with multiple methods
        
        Parameters:
        - model_name: Name of the BERT model to use for embeddings
        """
        self.tfidf_vectorizer = None
        self.bm25 = None
        self.bert_model = None
        self.model_name = model_name
        
    def fit_tfidf(self, texts: List[str]) -> None:
        """
        Fit TF-IDF vectorizer on the corpus
        
        Parameters:
        - texts: List of preprocessed text documents
        """
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            strip_accents='unicode',
            token_pattern=r'\b\w+\b'
        )
        self.tfidf_vectorizer.fit(texts)
        
    def transform_tfidf(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to TF-IDF vectors
        
        Parameters:
        - texts: List of preprocessed text documents
        
        Returns:
        - Dense numpy array of TF-IDF features
        """
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer not fitted. Call fit_tfidf first.")
        return self.tfidf_vectorizer.transform(texts).toarray()
    
    def fit_transform_tfidf(self, texts: List[str]) -> np.ndarray:
        """
        Fit and transform texts to TF-IDF vectors
        
        Parameters:
        - texts: List of preprocessed text documents
        
        Returns:
        - Dense numpy array of TF-IDF features
        """
        self.fit_tfidf(texts)
        return self.transform_tfidf(texts)
    
    def fit_bm25(self, texts: List[str]) -> None:
        """
        Fit BM25 on the corpus
        
        Parameters:
        - texts: List of preprocessed text documents
        """
        # Tokenize texts for BM25
        tokenized_corpus = [text.split() for text in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
    def transform_bm25(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to BM25 vectors
        
        Parameters:
        - texts: List of preprocessed text documents
        
        Returns:
        - Numpy array of BM25 scores
        """
        if self.bm25 is None:
            raise ValueError("BM25 not fitted. Call fit_bm25 first.")
            
        # Initialize matrix to store BM25 scores
        n_docs = len(texts)
        scores_matrix = np.zeros((n_docs, n_docs))
        
        # Calculate BM25 scores for each document pair
        for i, query in enumerate(tqdm(texts, desc="Calculating BM25 scores")):
            query_tokens = query.split()
            scores = self.bm25.get_scores(query_tokens)
            scores_matrix[i] = scores
            
        return scores_matrix
    
    def fit_transform_bm25(self, texts: List[str]) -> np.ndarray:
        """
        Fit and transform texts to BM25 vectors
        
        Parameters:
        - texts: List of preprocessed text documents
        
        Returns:
        - Numpy array of BM25 scores
        """
        self.fit_bm25(texts)
        return self.transform_bm25(texts)
    
    def initialize_bert(self) -> None:
        """Initialize the BERT model"""
        if self.bert_model is None:
            self.bert_model = SentenceTransformer(self.model_name)
            
    def get_bert_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Get BERT embeddings for texts
        
        Parameters:
        - texts: List of preprocessed text documents
        - batch_size: Batch size for BERT encoding
        
        Returns:
        - Numpy array of BERT embeddings
        """
        self.initialize_bert()
        
        # Process texts in batches to handle memory efficiently
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Calculating BERT embeddings"):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.bert_model.encode(
                batch,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            embeddings.append(batch_embeddings)
            
        return np.vstack(embeddings)
    
    def extract_all_features(self, texts: List[str], methods: List[str] = ['tfidf', 'bm25', 'bert']) -> dict:
        """
        Extract features using multiple methods
        
        Parameters:
        - texts: List of preprocessed text documents
        - methods: List of methods to use ['tfidf', 'bm25', 'bert']
        
        Returns:
        - Dictionary containing features for each method
        """
        features = {}
        
        if 'tfidf' in methods:
            features['tfidf'] = self.fit_transform_tfidf(texts)
            
        if 'bm25' in methods:
            features['bm25'] = self.fit_transform_bm25(texts)
            
        if 'bert' in methods:
            features['bert'] = self.get_bert_embeddings(texts)
            
        return features


