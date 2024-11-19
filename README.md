# Document Similarity Analysis System

A comprehensive system for analyzing and comparing document similarity using multiple approaches including TF-IDF, BM25, and BERT embeddings. The system includes both a FastAPI backend service and a Streamlit web interface.

## Project Structure
```
├── app.py                 # Streamlit web interface
├── main.py               # FastAPI backend service
└── src/                  # Core functionality
    ├── bm25_extractor.py    # BM25 feature extraction
    ├── bm25_index.py        # BM25 indexing and search
    ├── encoder.py           # BERT model encoding
    ├── semantic_index.py    # FAISS-based semantic search
    ├── similarity_score.py  # Main similarity calculation
    ├── text_preprocessing.py # Text preprocessing utilities
    ├── tfidf_extractor.py   # TF-IDF feature extraction
    ├── tfidf_index.py       # TF-IDF indexing and search
    └── utils.py             # Helper functions
```

## System Architecture

The project consists of two main applications:
- A FastAPI backend service (`main.py`)
- A Streamlit web interface (`app.py`)

## Component Details

1. Text Preprocessing (`text_preprocessing.py`)
The `TextPreprocessor` class handles all text cleaning and normalization tasks:
* **Key Features**:
   * Combines multiple description fields into a single text
   * Removes URLs, email addresses, special characters
   * Performs tokenization, stop word removal, and lemmatization
   * Custom stop words for business-specific terms
* **Main Methods**:
   * `preprocess_text()`: Applies full preprocessing pipeline
   * `clean_text()`: Performs basic text cleaning
   * `remove_stop_words()`: Filters out stop words
   * `lemmatize_text()`: Applies lemmatization
   * `process_dataframe()`: Processes entire dataframe of documents

2. Semantic Encoding (`encoder.py`)
The `Encoder` class handles BERT-based semantic embedding generation:
* **Key Features**:
   * Uses pre-trained BERT model (sentence-transformers/all-MiniLM-L6-v2)
   * Supports batch processing for efficiency
   * GPU acceleration when available
* **Main Methods**:
   * `generate_embeddings()`: Creates embeddings for input texts
   * `_mean_pooling()`: Pools token embeddings into sentence embeddings

3. Semantic Index (`semantic_index.py`)
The `Indexer` class manages FAISS-based similarity search:
* **Key Features**:
   * Uses FAISS for efficient similarity search
   * Inner product similarity metric
* **Main Methods**:
   * `populate_index()`: Adds embeddings to the index
   * `get_top_k_indices()`: Retrieves most similar documents

4. TF-IDF Components
   
a. TF-IDF Extractor (`tfidf_extractor.py`)
* **Key Features**:
   * Configurable feature extraction parameters
   * Support for n-grams
   * IDF smoothing and sublinear term frequency
* **Main Methods**:
   * `extract_features()`: Creates TF-IDF matrix
   * `transform()`: Transforms new texts using fitted vectorizer

b. TF-IDF Index (`tfidf_index.py`)
* **Key Features**:
   * Cosine similarity-based document comparison
* **Main Methods**:
   * `fit_vectorizer()`: Trains the TF-IDF vectorizer
   * `get_similarity_scores()`: Calculates similarity scores
   * `get_top_k_indices()`: Finds most similar documents

5. BM25 Components

a. BM25 Extractor (`bm25_extractor.py`)
* **Key Features**:
   * Parallel processing support
   * Batch processing for large datasets
   * File-based feature storage
* **Main Methods**:
   * `fit()`: Trains BM25 model on corpus
   * `extract_features_and_save()`: Generates and stores features

b. BM25 Index (`bm25_index.py`)
* **Key Features**:
   * Implementation of Okapi BM25 algorithm
* **Main Methods**:
   * `tokenize_query()`: Prepares queries for processing
   * `get_top_k_indices()`: Retrieves similar documents

6. Similarity Calculator (`similarity_score.py`)
The main orchestrator for similarity calculations:
* **Key Features**:
   * Supports multiple similarity methods (BERT, TF-IDF, BM25)
   * Batch processing for large-scale comparisons
* **Main Methods**:
   * `extract_features()`: Generates features using selected method
   * `calculate_similarity_matrix()`: Computes similarity scores
   * `get_similar_documents()`: Retrieves similar documents

7. Web Interface (`app.py`)
Streamlit-based user interface:
* **Key Features**:
   * Interactive company selection
   * Configurable similarity thresholds
   * Visualization of similarity results
* **Main Components**:
   * Similarity score calculation
   * Results visualization
   * Company selection interface

8. API Server (`main.py`)
FastAPI-based backend service:
* **Key Features**:
   * REST API endpoints for similarity search
   * Support for multiple similarity methods
   * Efficient data preprocessing and caching
* **Main Endpoints**:
   * `/get_top_k`: Retrieves similar companies

## Setup and Usage
1. Install Dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
2. Run the FastAPI Server:
```bash
python main.py
```

```
# Optionally use docker
docker build .
docker run -p 8000:8000 IMAGENAME
```
3. Launch the Streamlit Interface:
```bash
streamlit run app.py
```
## Dependencies
- streamlit
- pandas
- numpy
- scikit-learn
- transformers
- torch
- rank_bm25
- nltk
- joblib

## Data Requirements
Input CSV should contain columns:
- Organization Id
- Name
- Description
- Sourcscrub Description
- Description.1


## Comparison and Use Cases

### When to Use Each Method

1. **TF-IDF**
   - Quick similarity comparisons
   - Limited computational resources
   - Well-defined vocabulary domain
   - Need for interpretable results

2. **BERT**
   - Need for semantic understanding
   - Complex language patterns
   - Different ways of expressing same concept
   - High accuracy requirements

3. **BM25**
   - Information retrieval tasks
   - Document ranking
   - Need for length normalization
   - Balance between complexity and performance

### Performance Characteristics

| Method  | Semantic Understanding | Computational Cost | Memory Usage | Scalability |
|---------|----------------------|-------------------|--------------|-------------|
| TF-IDF  | Low                  | Low               | Medium       | High        |
| BERT    | High                 | High              | High         | Low         |
| BM25    | Low                  | Medium            | Medium       | Medium      |

### Implementation Considerations

1. **Resource Requirements**
   - TF-IDF: Minimal CPU and memory
   - BERT: GPU recommended, high memory
   - BM25: Moderate CPU and memory

2. **Preprocessing Importance**
   - TF-IDF: Critical for performance
   - BERT: Less critical (handles variations)
   - BM25: Moderate importance

3. **Batch Processing**
   - TF-IDF: Simple batching
   - BERT: Required for memory management
   - BM25: Beneficial for large datasets
---

# Scalable Document Similarity System with Milvus

## Overview

This project implements a **scalable document similarity system** that uses **Milvus**, an open-source vector database, for storing and querying vector embeddings. The system processes large datasets (such as company descriptions), extracts vector features, and enables efficient similarity search. It is built to scale horizontally, handle high data throughput, and provide low-latency similarity search.

## Table of Contents

1. [System Architecture Overview](#system-architecture-overview)
2. [Implementation Components](#implementation-components)
3. [Scalability Considerations](#scalability-considerations)
4. [Performance Optimization](#performance-optimization)
5. [Deployment Configuration](#deployment-configuration)
6. [System Requirements](#system-requirements)

## 1. System Architecture Overview

The system is designed with several layers for modularity and scalability. Data is ingested from external sources, preprocessed and transformed into vector embeddings, then indexed and stored in Milvus. The similarity search service allows querying of similar documents based on these embeddings.

### Key Components:
- **Data Ingestion Layer**: Collects and processes raw company data.
- **Text Preprocessing**: Cleans and tokenizes text data.
- **Feature Extraction**: Uses machine learning models (e.g., BERT) to extract vector embeddings from the text.
- **Milvus Vector Storage**: Stores embeddings for fast retrieval and similarity comparison.
- **Similarity Search Service**: Facilitates searching for similar documents based on vector similarity.
- **API Layer**: Exposes the system for querying via RESTful APIs.

## 2. Implementation Components

### 2.1 Data Ingestion
The Data Ingestion Service is responsible for fetching new company data, storing raw documents, and preparing them for vectorization. It batches data to improve throughput and efficiency, then processes and stores the vectors in Milvus.

### 2.2 Feature Extraction
Text data is processed using pre-trained models like **BERT**. The Feature Extraction component converts text data into vector embeddings, representing each document as a high-dimensional vector. This step is critical for enabling accurate and fast similarity searches.

### 2.3 Milvus Integration
Milvus is used for storing, indexing, and retrieving vector embeddings. It is configured to create collections and manage indexes optimized for efficient similarity search. The system supports different indexing strategies to optimize performance, like **IVF_FLAT**.

### 2.4 Similarity Search
The search service allows users to query for documents similar to a given input. It uses the vector embeddings stored in Milvus to find the most similar documents using vector similarity metrics like **Cosine Similarity** or **Euclidean Distance**.

## 3. Scalability Considerations

### 3.1 Horizontal Scaling
The system is designed to scale horizontally. As data volume increases, new processing nodes can be added to handle the load. The architecture includes components like **Kafka** for message queuing and **Redis** for caching, which allow the system to scale effectively across multiple workers.

### 3.2 Caching Strategy
Frequent similarity search results are cached in **Redis** to minimize repeated computation and reduce query latency. This improves the response time for users and reduces strain on the database.

## 4. Performance Optimization

### 4.1 Batch Processing
The system processes data in batches to improve throughput. By grouping similar tasks, the system avoids redundant computations and optimizes the time taken to process large datasets.

### 4.2 Index Optimization
Milvus indexes are optimized based on the dataset's characteristics, such as its size and distribution. Optimizing the indexing strategy ensures faster query response times and reduced storage requirements.

## 5. Deployment Configuration

The system uses **Docker Compose** for containerized deployments. It configures the Milvus, MongoDB, Redis, and Kafka services to work together. This allows for easy setup, scaling, and maintenance of the system on different environments.

## 6. System Requirements

- **Python 3.8+**
- **Milvus** (installed and running)
- **MongoDB** for raw data storage
- **Redis** for caching
- **Kafka** for event-driven processing (optional)
- **Torch** and **transformers** for feature extraction models


This provides a high-level overview of the architecture and components of the scalable document similarity system. The system is designed to be extensible, with considerations for handling large volumes of data and optimizing for high-performance similarity searches.
