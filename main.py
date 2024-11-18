import pandas as pd
import logging
from src.text_preprocessing import TextPreprocessor
from src.similarity_score import SimilarityCalculator
from src.utils import load_json_from_file, save_json_to_file
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(file_path):
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} companies from {file_path}")

        preprocessor = TextPreprocessor()
        df_processed = preprocessor.process_dataframe(df)
        if not os.path.exists('org_id_key_map.json'):
            org_id_key_map = {ord_id: idx for idx, ord_id in enumerate(df_processed['Organization Id'].to_list())}
            save_json_to_file(org_id_key_map, 'org_id_key_map.json')
        else:
            org_id_key_map = load_json_from_file('org_id_key_map.json')
        key_org_id_map = {v:k for k, v in org_id_key_map.items()}
        

        texts = df_processed['processed_description'].tolist()
        print(texts[0])
        feature_types = ["bm25"]
        for feature_type in feature_types:
            logger.info(f"Processing {feature_type} features...")
            similarity_calculator = SimilarityCalculator(feature_type)
            logger.info("this is done calling")
            if feature_type == "bm25":
                # For BM25, features are extracted and similarity matrices are saved directly
                logger.info("this is done calling 2")
                similarity_calculator.extract_features(texts)
                
                logger.info(f"{feature_type} features extracted and similarity matrices saved.")
            else:
                # For other feature types, extract features and calculate similarity matrix
                feature_matrix = similarity_calculator.extract_features(texts)
                similarity_calculator.calculate_similarity_matrix(feature_matrix)
                logger.info(f"Similarity matrix calculated and saved for {feature_type}.")

            logger.info(f"Completed processing {feature_type} features.")

        logger.info("Processing completed.")

    except Exception as e:
        logger.error(f"Error in processing: {e}", exc_info=True)

if __name__ == "__main__":
    main("data/innovius_case_study_data.csv")
