import numpy as np
from typing import List, Union
from sklearn.metrics.pairwise import cosine_similarity
import warnings

def calculate_similarity(features: np.ndarray) -> np.ndarray:
    """
    Calculate cosine similarity matrix from feature vectors
    
    Parameters:
    - features: Feature matrix (n_samples x n_features)
    
    Returns:
    - similarity_matrix: n_samples x n_samples cosine similarity matrix
    
    Raises:
    - ValueError: If input is invalid
    """
    # Input validation
    if not isinstance(features, np.ndarray):
        raise ValueError("Features must be a numpy array")
    
    if features.ndim != 2:
        raise ValueError("Features must be a 2D array")
        
    if np.isnan(features).any():
        raise ValueError("Features contain NaN values")
        
    if np.isinf(features).any():
        raise ValueError("Features contain infinite values")
    
    try:
        # Handle sparse matrices
        if hasattr(features, 'toarray'):
            features = features.toarray()
            
        # Calculate cosine similarity using scikit-learn's optimized implementation
        similarity_matrix = cosine_similarity(features)
        
        # Ensure the matrix is symmetric and has proper self-similarities
        np.fill_diagonal(similarity_matrix, 1.0)
        
        return similarity_matrix
        
    except Exception as e:
        raise RuntimeError(f"Error calculating cosine similarity: {str(e)}")

def get_top_similar(similarity_matrix: np.ndarray, idx: int, n: int = 5) -> List[tuple]:
    """
    Get top N most similar items for a given index
    
    Parameters:
    - similarity_matrix: Cosine similarity matrix
    - idx: Index of the item to find similarities for
    - n: Number of similar items to return
    
    Returns:
    - List of tuples (index, similarity_score) sorted by similarity
    """
    if idx >= len(similarity_matrix):
        raise ValueError("Index out of bounds")
        
    # Get similarities for the given index
    similarities = similarity_matrix[idx]
    
    # Get indices excluding the query index
    indices = np.arange(len(similarities))
    mask = indices != idx
    
    # Sort similarities (excluding self-similarity)
    similarities = similarities[mask]
    indices = indices[mask]
    
    # Get top N
    top_indices = np.argsort(similarities)[::-1][:n]
    
    return [(indices[i], similarities[i]) for i in top_indices]

# Example usage
