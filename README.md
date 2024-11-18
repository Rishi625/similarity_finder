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
