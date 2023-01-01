import faiss
import pickle
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
import vector_search
import numpy as np


@st.cache
def read_data(data="data/publications/post2018_agingcomapnies_papers.csv"):
    """Read the data from local."""
    data = pd.read_csv(data)
    data['publication_date'] = pd.to_datetime(data['publication_date'])
    return data


@st.cache(allow_output_mutation=True)
def load_bert_model():
    """Instantiate a sentence-level allenai-specer model."""
    return SentenceTransformer('sentence-transformers/allenai-specter')

@st.cache(allow_output_mutation=True)
def load_embeddings(path_to_embeddings="data/models/embeddings_allabstracts.pickle"):
    """Load and deserialize the Faiss index."""
    with open(path_to_embeddings, "rb") as h:
        embeddings = pickle.load(h)
    return embeddings 


@st.cache(allow_output_mutation=True)
def load_faiss_index(path_to_faiss="data/models/faiss_index_allabstracts.pickle"):
    """Load and deserialize the Faiss index."""
    with open(path_to_faiss, "rb") as h:
        data = pickle.load(h)
    return faiss.deserialize_index(data)

#convert publication date column to date time DONE
#give option to search from 100 companies listed in thedatabase DONe
#combine authors DONE



#display the analysis as per the notebook for top authors, top comapnies within the search 
#clean keywords and display top keywords
#limit search, limit abstracts, info about authors to subscriber or member area
#give ability to search by keywords by direct regex match - or elastic match 
#authors are stored in CSV and you need ID as key of the paper to get the authors

def main():
    
    # Load data and models
    data = read_data()
    model = load_bert_model()
    faiss_index = load_faiss_index()
    embeddings = load_embeddings()
    # important columns - company_name, article_id, title, keywords, publication_date, abstract, journal, doi, authors
    # variables - user_input, filter_company, num_results
    comapny_list = list(set(data['company_name'].tolist()))

    st.title("AI Search for scientific database curated within anti-aging space")

    # User search
    user_input = st.text_area("Search box", "stem cell research")

    # Filters
    st.sidebar.markdown("**Filters**")
    #filter by keywords, company and seed terms (stem cell, aging, etc) within th abstract and title
    #display the number of results, authors, companies, journals, keywords
    filter_company = st.sidebar.multiselect('Select Company or Companies (optional)',comapny_list) #get dropdown of companies
    num_results = st.sidebar.slider("Number of search results", 10, 50, 10)

    # Fetch results
    if user_input:
        # Get paper IDs
        D, I = vector_search([user_input], model, faiss_index, num_results)
        # Slice data on comapny name
        frame = data[
          (data.company_names == filter_company) #see if this works if you have multiple companies
        ]
        # Get individual results
        for id_ in I.flatten().tolist():
            if id_ in set(frame.article_id):
                f = frame[(frame.article_id == id_)]
            else:
                continue

            st.write(
                f"""**{f.iloc[0].title}**  
            **Journal**: {f.iloc[0].journal}  
            **Publication Date**: {f.iloc[0].publication_date}  
            **Abstract**
            {f.iloc[0].abstract}
            """
            )


if __name__ == "__main__":
    main()
    
    
    
