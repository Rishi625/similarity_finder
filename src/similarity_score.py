from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Tuple
from src.utils import save_np_array_to_file
import os
from src.tfidf_extractor import TfidfExtractor
from src.bert_extractor import BertExtractor
from src.bm25_extractor import BM25Extractor
import logging

logger = logging.getLogger(__name__)

class SimilarityCalculator:
    def __init__(self, feature_type):
        self.feature_type = feature_type
        self.feature_extractor = None
        if feature_type == "bert":
            self.feature_extractor = BertExtractor()
        elif feature_type == "tfidf":
            self.feature_extractor = TfidfExtractor()
        elif feature_type == "bm25":
            self.feature_extractor = BM25Extractor(feature_type="bm25")
        else:
            raise "Invalid feature type"
        
        self.feature_dir = f"features_{self.feature_type}"
        os.makedirs(self.feature_dir, exist_ok=True)

    
    def extract_features(self, texts):
        if self.feature_type == "bm25":
            self.feature_extractor.fit(texts)
            self.feature_extractor.extract_features_and_save(texts)
        else:
            features = self.feature_extractor.extract_features(texts)
            return features

    def calculate_similarity_matrix(self, features, batch_size=500):
        n = features.shape[0]

        print(f"Calculating similarity matrix for {n} documents")
        for i , start in enumerate(range(0, n, batch_size)):
            end = min(start + batch_size, n)
            similarity_matrix = cosine_similarity(features[start:end], features)
            if not os.path.exists(f"{self.feature_dir}/similarity_matrix_{start}_{end}.npy"):
                save_np_array_to_file(similarity_matrix, f"{self.feature_dir}/similarity_matrix_{start}_{end}.npy")
                print(f"Saved similarity matrix for {start} to {end} documents to file")        

    def get_similar_documents(self, company_idx,num_docs):
        similarity_files = os.listdir(self.feature_dir)
        for file_ in similarity_files:
            if file_.startswith("similarity_matrix"):
                start, end = file_.split("_")[-2], file_.split("_")[-1].split(".")[0]
                if int(start) <= company_idx <= int(end):
                    similarity_matrix = np.load(f"{self.feature_dir}/{file_}")
                    similar_docs = []
                    for idx, score in zip(np.argsort(similarity_matrix[company_idx])[::-1][1:num_docs+1], similarity_matrix[company_idx][np.argsort(similarity_matrix[company_idx])[::-1][1:num_docs+1]]):
                        similar_docs.append((idx, score))
                    return similar_docs

    
