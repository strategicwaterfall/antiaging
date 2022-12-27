import faiss
import pickle
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
import vector_search


@st.cache
def read_data(data="data/publications/post2018_agingcomapnies_papers.csv"):
    """Read the data from local."""
    return pd.read_csv(data)


@st.cache(allow_output_mutation=True)
def load_bert_model():
    """Instantiate a sentence-level DistilBERT model."""
    return model = SentenceTransformer('sentence-transformers/allenai-specter')


@st.cache(allow_output_mutation=True)
def load_faiss_index(path_to_faiss="data/models/faiss_index_1000abstracts.pickle"):
    """Load and deserialize the Faiss index."""
    with open(path_to_faiss, "rb") as h:
        data = pickle.load(h)
    return faiss.deserialize_index(data)