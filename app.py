import streamlit as st
import pandas as pd
from src.utils import load_json_from_file
from src.similarity_score import SimilarityCalculator


st.title("demo")

with st.sidebar:
    radio = st.radio(
        "Select feature type",
        ("tfidf", "bert","bm25"),
        horizontal=True
    )


similarity_calculator = SimilarityCalculator(radio)

df = pd.read_csv('data/innovius_case_study_data.csv').drop_duplicates()

org_id_key_map = load_json_from_file('org_id_key_map.json')
key_org_id_map = {v:k for k, v in org_id_key_map.items()}

df['Organization Id'] = df['Organization Id'].apply(lambda x: str(x))
company_list = df['Name'] + '_' + df['Organization Id']

selected_company = st.selectbox("Select company", company_list)
num_docs = st.number_input ("Enter the number of similar documents to display:", 
    min_value=1, 
    max_value=100, 
    value=5, 
    step=1)

button = st.button("Process")

if button:
    if selected_company:
        org_id = selected_company.split('_')[-1]
        company_idx = org_id_key_map[org_id]
        st.write(f"Selected company: {df.iloc[company_idx]['Name']}")
        st.write(f"Description1: {df.iloc[company_idx]['Description']}")
        
        similar_docs = similarity_calculator.get_similar_documents(company_idx,num_docs)
        doc_idxs = [key_org_id_map[doc_idx] for doc_idx, score in similar_docs]
        scores = [score for doc_idx, score in similar_docs]
        df_selected = df[df['Organization Id'].isin(doc_idxs)].reset_index(drop=True)
        df_selected['Score'] = scores
        st.dataframe(df_selected)
