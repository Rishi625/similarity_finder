import faiss
import numpy as np

class Indexer:
    def __init__(self, embedding_size):
        self.index = faiss.IndexFlatIP(embedding_size)
    

    def populate_index(self, embeddings):
        self.index.add(embeddings)
    
    def get_top_k_indices(self, query_embedding, k=5):
        distances, indices = self.index.search(np.array(query_embedding).reshape(1, -1), k)
        return indices[0][1:], distances[0][1:]