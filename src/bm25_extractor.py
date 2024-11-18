import os
import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from joblib import Parallel, delayed
from src.utils import save_np_array_to_file


class BM25Extractor:
    def __init__(self, feature_type):
        self.bm25 = None
        self.corpus = None
        self.feature_type = feature_type

        self.feature_dir = f"features_{self.feature_type}"
        os.makedirs(self.feature_dir, exist_ok=True)
        

    def fit(self, texts):
        self.corpus = [text.split() for text in texts]  # Tokenize texts
        print(len(self.corpus))
        self.bm25 = BM25Okapi(self.corpus)
        print("BM25 model fitted on corpus")

    def _compute_batch_scores(self, batch_queries):
        """Helper function to compute scores for a batch of queries."""
        return np.array([self.bm25.get_scores(query) for query in batch_queries])

    def extract_features_and_save(self, texts, batch_size=4):
        print(texts)
        if self.bm25 is None:
            raise ValueError("BM25 model is not fitted")

        # Convert texts to tokens if they're not already tokenized
        queries = [text.split() if isinstance(text, str) else text for text in texts]
        print(queries[0])
        
        # Divide queries into batches
        batches = [queries[i:i+batch_size] for i in range(0, len(queries), batch_size)]
        print(batches[0])

        # Process batches and save results
        for i, batch in enumerate(tqdm(batches)):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(queries))
            if not os.path.exists(f"{self.feature_dir}/similarity_matrix_{start}_{end}.npy"):
                similarity_matrix = np.array(self._compute_batch_scores(batch))
                save_np_array_to_file(similarity_matrix, f"{self.feature_dir}/similarity_matrix_{start}_{end}.npy")
                print(f"Saved similarity matrix for {start} to {end} documents to file")
            else:
                print(f"File similarity_matrix_{start}_{end}.npy already exists, skipping...")

        print("All batches processed and saved.")
   