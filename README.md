# Document Similarity Finder

## Overview
This project implements a document similarity system that supports multiple feature extraction methods (TF-IDF, BERT, and BM25) to find similar documents based on their textual content. It includes a Streamlit web interface for interactive document comparison and a batch processing system for large-scale similarity calculations.

## Project Structure
```
├── app.py                 # Streamlit web application
├── main.py               # Batch processing script
└── src/                  # Source code directory
    ├── __init__.py
    ├── bert_extractor.py
    ├── bm25_extractor.py
    ├── similarity_score.py
    ├── text_preprocessing.py
    ├── tfidf_extractor.py
    └── utils.py
```

## Component Details

### 1. Text Preprocessing (`text_preprocessing.py`)
The `TextPreprocessor` class handles all text cleaning and normalization tasks:
- **Key Features**:
  - Combines multiple description fields into a single text
  - Removes URLs, email addresses, special characters, and numbers
  - Performs tokenization, stop word removal, and lemmatization
  - Custom stop words for business-specific terms
- **Main Methods**:
  - `combine_descriptions()`: Merges multiple description fields
  - `clean_text()`: Performs basic text cleaning
  - `preprocess_text()`: Applies full preprocessing pipeline
  - `process_dataframe()`: Processes entire dataframe of documents

### 2. Feature Extractors

#### 2.1 TF-IDF Extractor (`tfidf_extractor.py`)
- **Class**: `TfidfExtractor`
- **Features**:
  - Uses scikit-learn's TfidfVectorizer
  - Configurable parameters for max_features, min_df, and max_df
  - Supports unigrams and bigrams
  - Sublinear term frequency scaling
- **Methods**:
  - `extract_features()`: Fits and transforms documents to TF-IDF vectors
  - `transform()`: Transforms new documents using fitted vectorizer

#### 2.2 BERT Extractor (`bert_extractor.py`)
- **Class**: `BertExtractor`
- **Features**:
  - Uses Hugging Face's BERT model
  - GPU support when available
  - Batch processing for memory efficiency
- **Methods**:
  - `extract_features()`: Extracts BERT embeddings with batching

#### 2.3 BM25 Extractor (`bm25_extractor.py`)
- **Class**: `BM25Extractor`
- **Features**:
  - Implements Okapi BM25 ranking algorithm
  - Parallel processing support
  - Batch processing with disk storage
- **Methods**:
  - `fit()`: Prepares BM25 model with corpus
  - `extract_features_and_save()`: Processes and saves similarity scores

### 3. Similarity Calculator (`similarity_score.py`)
- **Class**: `SimilarityCalculator`
- **Features**:
  - Unified interface for all feature extractors
  - Batch processing for large datasets
  - Disk-based storage for similarity matrices
- **Methods**:
  - `extract_features()`: Extracts features using selected method
  - `calculate_similarity_matrix()`: Computes cosine similarities
  - `get_similar_documents()`: Retrieves similar documents

### 4. Utilities (`utils.py`)
Helper functions for file operations:
- `save_np_array_to_file()`: Saves NumPy arrays to disk
- `load_np_array_from_file()`: Loads NumPy arrays from disk
- `save_json_to_file()`: Saves JSON data to disk
- `load_json_from_file()`: Loads JSON data from disk

### 5. Web Interface (`app.py`)
Streamlit-based web application featuring:
- Feature type selection (TF-IDF, BERT, BM25)
- Similarity threshold adjustment
- Number of similar documents selection
- Interactive company selection
- Results display with similarity scores

### 6. Batch Processing (`main.py`)
Script for processing large datasets:
- Loads and preprocesses document data
- Maintains organization ID mappings
- Processes features using selected extractors
- Handles error logging and reporting

## Usage

### Web Interface
1. Run the Streamlit app:
```bash
streamlit run app.py
```
2. Select feature type, threshold, and number of documents
3. Choose a company and click "Process"
4. View similar companies and their similarity scores

### Batch Processing
1. Place your data in CSV format in the data directory
2. Run the main script:
```bash
python main.py
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

## Performance Considerations
- BERT processing is GPU-accelerated when available
- BM25 uses parallel processing for large datasets
- Similarity matrices are saved to disk to handle large datasets
- Batch processing is implemented for memory efficiency
- 

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

Here’s an updated version of your README without the code, focusing more on the explanation of each component:

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
6. [Monitoring and Maintenance](#monitoring-and-maintenance)
7. [Usage Example](#usage-example)
8. [System Requirements](#system-requirements)
9. [Performance Metrics](#performance-metrics)

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

## 6. Monitoring and Maintenance

### 6.1 Health Checks
A health check service monitors the status of the system’s components. It ensures that the data ingestion, feature extraction, and similarity search processes are functioning properly and alerts administrators if there is any issue.

### 6.2 Data Validation
A validation service checks the integrity and format of incoming data, ensuring that it complies with predefined schemas before it is processed and stored.

## 7. Usage Example

Once the system is deployed, users can input company descriptions or other relevant text, and the similarity search service will return the most similar documents from the database. The API exposes endpoints that allow users to perform these searches programmatically.

## 8. System Requirements

- **Python 3.8+**
- **Milvus** (installed and running)
- **MongoDB** for raw data storage
- **Redis** for caching
- **Kafka** for event-driven processing (optional)
- **Torch** and **transformers** for feature extraction models

## 9. Performance Metrics

Performance is measured based on several factors:
- **Indexing Speed**: How quickly new data is indexed and made available for searches.
- **Search Speed**: Time taken to perform similarity searches and return results.
- **Query Latency**: Latency in searching for similar documents.
- **Cache Hit Rate**: Efficiency of the caching layer in serving frequent queries.

---

This README provides a high-level overview of the architecture and components of the scalable document similarity system. The system is designed to be extensible, with considerations for handling large volumes of data and optimizing for high-performance similarity searches.
