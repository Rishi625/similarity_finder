import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from src.similarity_score import SimilarityCalculator
from src.utils import load_json_from_file

# Set page configuration
st.set_page_config(
    page_title="Similarity Score Analyzer", 
    page_icon=":mag_right:", 
    layout="wide"
)

# Load data
df = pd.read_csv('data/innovius_case_study_data.csv').drop_duplicates()

# Sidebar inputs
with st.sidebar:
    # Feature selection
    feature_type = st.radio(
        "Select feature type",
        ("tfidf", "bert", "bm25"),
        horizontal=True
    )
    
    if feature_type == "bm25":
        similarity_threshold = st.number_input(
            "Select similarity threshold (0 to 100)", 
            min_value=0.0, 
            max_value=100.0, 
            value=50.0,  # Default value for bm25
            step=1.0
        )
    else:
        similarity_threshold = st.number_input(
            "Select similarity threshold (0 to 1)", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.5, 
            step=0.01
        )
    
    # Number of similar documents to display
    num_docs = st.number_input(
        "Enter the number of similar documents to display:", 
        min_value=1, 
        max_value=1000, 
        value=10, 
        step=1
    )


# Prepare company selection
org_id_key_map = load_json_from_file('org_id_key_map.json')
key_org_id_map = {v: k for k, v in org_id_key_map.items()}
df['Organization Id'] = df['Organization Id'].astype(str)
company_list = df['Name'] + '_' + df['Organization Id']

# Dropdown for company selection
selected_company = st.selectbox("Select company", company_list)

# Button to trigger similarity score calculation
if st.button("Calculate Similarity Scores"):
    if selected_company:
        # Initialize similarity calculator
        similarity_calculator = SimilarityCalculator(feature_type)
        
        # Extract selected company details
        org_id = selected_company.split('_')[-1]
        company_idx = org_id_key_map[org_id]
        
        # Calculate similarity scores
        similar_docs = similarity_calculator.get_similar_documents(company_idx,num_docs=num_docs)
        
        # Map results to a dataframe
        doc_idxs = [key_org_id_map[doc_idx] for doc_idx, _ in similar_docs]
        scores = [score for _, score in similar_docs]
        
        # Create a filtered dataframe with similarity scores
        df_similarity = df[df['Organization Id'].isin(doc_idxs)].copy()
        df_similarity['Similarity Score'] = [
            scores[doc_idxs.index(str(org_id))] if str(org_id) in doc_idxs else 0 
            for org_id in df_similarity['Organization Id']
        ]
        
        # Sort dataframe by similarity score
        df_similarity = df_similarity.sort_values(by='Similarity Score', ascending=False)
        
        # Display base company
        base_company = df.iloc[company_idx]['Name']
        st.write(f"### Base Company: {base_company}")
        
        # Categorize and visualize similarity scores
        above_threshold = df_similarity[df_similarity['Similarity Score'] >= similarity_threshold]
        below_threshold = df_similarity[df_similarity['Similarity Score'] < similarity_threshold]
        
        chart_data = pd.DataFrame({
            "Category": ["Above Threshold", "Below Threshold"],
            "Count": [len(above_threshold), len(below_threshold)]
        })

        bar_chart = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X("Category", title="Similarity Category"),
            y=alt.Y("Count", title="Number of Documents"),
            color=alt.Color("Category", legend=None)
        )

        st.altair_chart(bar_chart, use_container_width=True)
        
        
        # Display top results
        st.write(f"### Top {num_docs} Similar Documents:")
        st.dataframe(df_similarity.head(num_docs))
