import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List
import nltk

from src.text_preprocessing import TextPreprocessor
from src.feature_extraction import FeatureExtractor
from src.similarity_score import calculate_similarity, get_top_similar

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)

def download_nltk_resources():
    """
    Download required NLTK resources
    """
    resources = [
        'punkt',
        'stopwords',
        'wordnet',
        'averaged_perceptron_tagger',
        'omw-1.4'  # Open Multilingual Wordnet
    ]
    
    for resource in resources:
        try:
            logging.info(f"Downloading NLTK resource: {resource}")
            nltk.download(resource, quiet=True)
        except Exception as e:
            logging.error(f"Error downloading {resource}: {str(e)}")
            raise

def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Load and preprocess the data
    """
    logging.info("Loading data from %s", file_path)
    df = pd.read_csv(file_path)
    
    logging.info("Initializing text preprocessor")
    preprocessor = TextPreprocessor(
        remove_stopwords=True,
        lemmatize=True,
        min_word_length=2
    )
    
    logging.info("Processing text data")
    df_processed = preprocessor.process_dataframe(df)
    
    logging.info("Preprocessing complete")
    logging.info(f"Original records: {len(df)}")
    logging.info(f"Processed records: {len(df_processed)}")
    
    return df_processed

def extract_features(texts: List[str], methods: List[str] = ['tfidf', 'bm25', 'bert']) -> Dict[str, np.ndarray]:
    """
    Extract features using multiple methods
    """
    logging.info("Initializing feature extractor")
    extractor = FeatureExtractor(model_name='all-MiniLM-L6-v2')
    
    logging.info("Extracting features using methods: %s", methods)
    features = extractor.extract_all_features(texts, methods=methods)
    
    for method, feature_matrix in features.items():
        logging.info(f"{method.upper()} features shape: {feature_matrix.shape}")
    
    return features

def calculate_similarities(features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Calculate similarity matrices for each feature type
    """
    logging.info("Calculating similarity matrices")
    similarities = {}
    
    for method, feature_matrix in features.items():
        logging.info(f"Calculating similarities for {method}")
        similarities[method] = calculate_similarity(feature_matrix)
        
    return similarities

def analyze_similarities(similarities: Dict[str, np.ndarray], df: pd.DataFrame, n_similar: int = 5):
    """
    Analyze and log similarity results
    """
    logging.info("Analyzing similarities")
    
    for method, sim_matrix in similarities.items():
        logging.info(f"\nAnalysis for {method.upper()}:")
        
        # Sample a few random companies for demonstration
        sample_indices = np.random.choice(len(df), 3, replace=False)
        
        for idx in sample_indices:
            company_name = df.iloc[idx].get('Organization Name', f'Company {idx}')
            logging.info(f"\nTop {n_similar} similar companies to {company_name}:")
            
            top_similar = get_top_similar(sim_matrix, idx, n=n_similar)
            for similar_idx, score in top_similar:
                similar_company = df.iloc[similar_idx].get('Organization Name', f'Company {similar_idx}')
                logging.info(f"- {similar_company}: {score:.3f}")

def save_results(df: pd.DataFrame, similarities: Dict[str, np.ndarray], output_dir: str):
    """
    Save processed data and similarity matrices
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    logging.info("Saving results to %s", output_path)
    
    # Save processed DataFrame
    df.to_csv(output_path / 'processed_data.csv', index=False)
    
    # Save similarity matrices
    for method, sim_matrix in similarities.items():
        np.save(output_path / f'similarities_{method}.npy', sim_matrix)

def main():
    """
    Main function to run the entire pipeline
    """
    try:
        # Download required NLTK resources
        logging.info("Downloading required NLTK resources...")
        download_nltk_resources()
        
        # Configuration
        input_file = 'data/innovius_case_study_data.csv'
        output_dir = 'output'
        feature_methods = ['tfidf', 'bm25', 'bert']
        n_similar = 5
        
        # Load and preprocess data
        df_processed = load_and_preprocess_data(input_file)
        
        # Extract features
        features = extract_features(
            texts=df_processed['processed_description'].tolist(),
            methods=feature_methods
        )
        
        # Calculate similarities
        similarities = calculate_similarities(features)
        
        # Analyze results
        analyze_similarities(similarities, df_processed, n_similar)
        
        # Save results
        save_results(df_processed, similarities, output_dir)
        
        logging.info("Pipeline completed successfully")
        
    except FileNotFoundError:
        logging.error("Input file not found. Please check the file path.")
    except pd.errors.EmptyDataError:
        logging.error("The input file is empty.")
    except Exception as e:
        logging.error("An error occurred: %s", str(e), exc_info=True)

if __name__ == "__main__":
    main()