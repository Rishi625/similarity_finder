
from rank_bm25 import BM25Okapi
import numpy as np

class BM25Index:
    def __init__(self, corpus):
        tokenized_corpus = [doc.split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def tokenize_query(self, query):
        return query.split()

    def get_top_k_indices(self, query, k=10):
        tokenized_query = self.tokenize_query(query)
        scores = self.bm25.get_scores(tokenized_query)
        indexes = np.argsort(scores)[-k-1:-1][::-1]
        scores = scores[indexes]
        return indexes, scores
    