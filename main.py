import pandas as pd
import logging
from src.text_preprocessing import TextPreprocessor
from src.semantic_index import Indexer
from src.encoder import Encoder
from src.tfidf_index import TFIDFIndex
from src.bm25_index import BM25Index
import os
from fastapi import FastAPI
import numpy as np


logging.basicConfig(level=logging.INFO)

app = FastAPI()

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

df = pd.read_csv("data/innovius_case_study_data.csv")
encoder_model = Encoder(model_name=MODEL_NAME)


def preprocess_data(df, feature_type="semantic"):
    if feature_type == "semantic":
        preprocessor = TextPreprocessor(remove_stopwords=False, lemmatize=False)
    else:
        preprocessor = TextPreprocessor()
    df_processed = preprocessor.process_dataframe(df)
    return df_processed

semantic_preprocessed_data = preprocess_data(df, feature_type="semantic")
non_semantic_preprocessed_data = preprocess_data(df, feature_type="non_semantic")
print(semantic_preprocessed_data.head())
print(non_semantic_preprocessed_data.head())


def create_semantic_index(preprocessed_data, encoder_model):
    text_descriptions = preprocessed_data['combined_description'].tolist()
    if not os.path.exists(f'embeddings.npy'):
        embeddings = encoder_model.generate_embeddings(text_descriptions)
        np.save('embeddings.npy', embeddings)
    else:
        embeddings = np.load('embeddings.npy')

    semantic_index = Indexer(embedding_size=encoder_model.model.config.hidden_size)
    semantic_index.populate_index(embeddings)
    return semantic_index


def create_tfidf_index(preprocessed_data):
    text_descriptions = preprocessed_data['combined_description'].tolist()
    tfidf_index = TFIDFIndex()
    tfidf_index.fit_vectorizer(text_descriptions)
    tfidf_index.get_tfidf_vectors(text_descriptions)
    return tfidf_index

def create_bm25_index(preprocessed_data):
    text_descriptions = preprocessed_data['combined_description'].tolist()
    bm25_index = BM25Index(text_descriptions)
    return bm25_index

logging.info("Creating semantic index")
semantic_index = create_semantic_index(semantic_preprocessed_data, encoder_model)
logging.info("Creating tfidf index")
tfidf_index = create_tfidf_index(non_semantic_preprocessed_data)
logging.info("Creating bm25 index")
bm25_index = create_bm25_index(non_semantic_preprocessed_data)



@app.post('/get_top_k')
def get_top_k_companies(request: dict):
    org_id = int(request['org_id'])
    k = request['k']
    feature_type = request['feature_type']
    logging.info(f"org_id: {org_id}, k: {k}, feature_type: {feature_type}")


    if feature_type == "semantic":
        query = semantic_preprocessed_data[semantic_preprocessed_data['Organization Id'].astype(int) == org_id]['combined_description'].values[0]
        logging.info(f"Query: {query}")
        query_embedding = encoder_model.generate_embeddings([query]).flatten()
        indices, scores = semantic_index.get_top_k_indices(query_embedding, k)
        logging.info(f"Indices: {indices}, Scores: {scores}")
    elif feature_type == "tfidf":
        query = non_semantic_preprocessed_data[non_semantic_preprocessed_data['Organization Id'].astype(int) == org_id]['combined_description'].values[0]
        logging.info(f"Query: {non_semantic_preprocessed_data[non_semantic_preprocessed_data['Organization Id'] == org_id]['combined_description']}")
        logging.info(f"Query: {query}")
        indices, scores = tfidf_index.get_top_k_indices(query, k)
        logging.info(f"Indices: {indices}, Scores: {scores}")
    elif feature_type == "bm25":
        query = non_semantic_preprocessed_data[non_semantic_preprocessed_data['Organization Id'].astype(int) == org_id]['combined_description'].values[0]
        logging.info(f"Query: {query}")
        indices, scores = bm25_index.get_top_k_indices(query, k)
        logging.info(f"Indices: {indices}, Scores: {scores}")
    else:
        indices, scores = None, None
    return {
            "indices": [int(i) for i in indices] if indices is not None else None,
            "scores": [float(s) for s in scores] if scores is not None else None,
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)