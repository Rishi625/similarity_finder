# Document Similarity System

## Overview
This system provides a scalable solution for computing and managing document similarities across different companies using multiple feature extraction methods (BERT, TF-IDF, and BM25).

## Architecture

### Components
1. **Feature Extractors**
   - BERT Extractor: Deep learning-based contextual embeddings
   - BM25 Extractor: Probabilistic ranking function
   - TF-IDF Extractor: Statistical feature extraction
   
2. **Text Preprocessing Pipeline**
   - Text cleaning and normalization
   - Stop word removal
   - Lemmatization
   - Business-specific term handling

3. **Similarity Calculator**
   - Batch processing for large-scale computations
   - Disk-based storage for similarity matrices
   - Multiple similarity computation methods

4. **Web Interface**
   - Streamlit-based user interface
   - Interactive company selection
   - Real-time similarity computation

## Setup and Installation

### Prerequisites
```bash
- Python 3.8+
- CUDA-capable GPU (optional, for BERT)
```

### Environment Setup
```bash
pip install -r requirements.txt
```

### Running the System
1. Process initial data:
```bash
python main.py
```

2. Start the web interface:
```bash
streamlit run app.py
```

## Technical Details

### Data Flow
1. Raw company data → Text Preprocessing
2. Processed text → Feature Extraction
3. Features → Similarity Computation
4. Similarity Matrices → Disk Storage
5. User Query → Real-time Retrieval

### Performance Considerations
- Batch processing for memory efficiency
- Parallel processing in BM25
- Disk-based storage for large matrices
- Incremental updates support

## API Reference

### TextPreprocessor
```python
preprocessor = TextPreprocessor(
    remove_stopwords=True,
    lemmatize=True,
    min_word_length=2
)
```

### SimilarityCalculator
```python
calculator = SimilarityCalculator(
    feature_type="tfidf"  # Options: "bert", "bm25", "tfidf"
)
```
