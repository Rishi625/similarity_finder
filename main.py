import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import spacy
import contractions
from src.text_preprocessing import TextPreprocessor
def main():
    try:
        # Load data
        df = pd.read_csv('/mnt/c/Users/rushi/OneDrive/Desktop/Innovious/innovius_case_study_data.csv')

        # Initialize preprocessor
        preprocessor = TextPreprocessor(
            remove_stopwords=True,
            lemmatize=True,
            min_word_length=2
        )

        # Process data
        df_processed = preprocessor.process_dataframe(df)

        # Print sample results
        print("\nSample of processed descriptions:")
        for i, row in df_processed.head(2).iterrows():
            print("\nOriginal:", row['combined_description'][:200], "...")
            print("Processed:", row['processed_description'][:200], "...")

        # Basic statistics
        print("\nPreprocessing Statistics:")
        print(f"Original number of companies: {len(df)}")
        print(f"After removing duplicates: {len(df_processed)}")
        print(f"Average word count before processing: {df_processed['combined_description'].str.split().str.len().mean():.1f}")
        print(f"Average word count after processing: {df_processed['processed_description'].str.split().str.len().mean():.1f}")
    except FileNotFoundError:
        print("File not found. Please check the file path.")
    except pd.errors.EmptyDataError:
        print("The data is empty. Please check the file path.")
    except pd.errors.ParserError:
        print("Error parsing the data. Please check the file format.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()