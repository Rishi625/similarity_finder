import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from joblib import Parallel, delayed

class BM25Extractor:
    def __init__(self):
        self.bm25 = None
        self.corpus = None

    def fit(self, texts):
        self.corpus = [text.split() for text in texts]  # Tokenize texts
        self.bm25 = BM25Okapi(self.corpus)
        print("BM25 model fitted on corpus")

    def _compute_batch_scores(self, batch_queries):
        """Helper function to compute scores for a batch of queries."""
        return np.array([self.bm25.get_scores(query) for query in batch_queries])

    def extract_features(self, texts, batch_size=32, n_jobs=-1):
        if self.bm25 is None:
            raise ValueError("BM25 model is not fitted")

        # Convert texts to tokens if they're not already tokenized
        queries = [text.split() if isinstance(text, str) else text for text in texts]
        
        # Divide queries into batches
        batches = [queries[i:i+batch_size] for i in range(0, len(queries), batch_size)]

        # Process batches in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(self._compute_batch_scores)(batch) for batch in tqdm(batches)
        )

        # Combine all results into a single array
        return np.vstack(results)
