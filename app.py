import faiss
import pickle
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np


@st.cache
def read_data(data="data/publications/post2018_agingcomapnies_papers.csv"):
    """Read the data from local."""
    data = pd.read_csv(data)
    # datetime conversation for display
    data['publication_date'] = pd.to_datetime(data['publication_date'])
    data['publication_date'] = data['publication_date'].dt.date
    # to capitalize each row in the company_name column.
    data['company_name'] = data['company_name'].str.capitalize()
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

@st.cache(allow_output_mutation=True)
def vector_search(query, model, index, num_results=10):
    """Tranforms query to vector using #https://huggingface.co/sentence-transformers/allenai-specter
model = SentenceTransformer('sentence-transformers/allenai-specter')
and finds similar vectors using FAISS.
    
    Args:
        query (str): User query that should be more than a sentence long.
        model (sentence_transformers.SentenceTransformer.SentenceTransformer)
        index (`numpy.ndarray`): FAISS index that needs to be deserialized.
        num_results (int): Number of results to return.
    
    Returns:
        D (:obj:`numpy.array` of `float`): Distance between results and query.
        I (:obj:`numpy.array` of `int`): Paper ID of the results.
    
    """
    vector = model.encode(list(query))
    D, I = index.search(np.array(vector).astype("float32"), k=num_results)
    return D, I

@st.cache(allow_output_mutation=True)
def id2details(df, I, column):
    """Returns the paper titles based on the paper index."""
    return [list(df[df.id == idx][column]) for idx in I[0]]




def main():
    try:
    
        # Load data and models
        data = read_data()
        model = load_bert_model()
        faiss_index = load_faiss_index()
        embeddings = load_embeddings()
        # important columns - company_name, article_id, title, keywords, publication_date, abstract, journal, doi, authors
        # variables - user_input, filter_company, num_results
        comapny_list = list(set(data['company_name'].tolist()))

        st.title("Vectorized Search over 🧬 PubMed abstracts database curated within :blue[anti-aging] industry and research👨‍🔬⚗️🧪💊*")
        st.caption("""There are over _3000_ abstracts in the database taken from PubMed. All publications are from _2018_ onwards.
                   These are compiled from 100 operating companies in the :blue[anti-aging] space, inlcuding Altos Labs, Unity Biotechnology, Insilico Medicine, and many more.""")
        st.caption("""The data about the companies is taken from [this](https://agingbiotech.info/companies) maintained by investor and creator Karl Pfleger.""")

        # User search
        user_input = st.text_area("Search box-Experimental", "stem cell research")

        # Filters
        st.sidebar.markdown("**Filters**")
        #filter by keywords, company and seed terms (stem cell, aging, etc) within th abstract and title
        #display the number of results, authors, companies, journals, keywords
        filter_company = st.sidebar.multiselect('Select Company or Companies (optional)',comapny_list, 'Altos labs') #get dropdown of companies
        num_results = st.sidebar.slider("Number of search results", 10, 50, 10)

        # Fetch results
        if user_input:
            # Get paper IDs
            D, I = vector_search([user_input], model, faiss_index, num_results)
            # Slice data on comapny name
            try:
                if filter_company:
                    frame = data[data['company_name'].isin(filter_company)]
                else:
                    frame = data
            except:
                pass #see if this works if you have multiple companies
            # Get individual results
            for id_ in I.flatten().tolist():
                if id_ in set(frame.article_id):
                    f = frame[(frame.article_id == id_)]
                else:
                    continue
                newline= '\n'
                st.write(
                        f"""**{f.iloc[0].title}**  
                    **Affiliate Anti-Aging Company Name**: {f.iloc[0].company_name.capitalize()}
                    **Journal**: {f.iloc[0].journal}  
                    **Publication Date**: {f.iloc[0].publication_date}  
                    {newline}**Keywords**: *{f.iloc[0].keywords}*
                    {newline}**DOI**: *{f.iloc[0].doi.split(newline)[0]}*
                    {newline}**Abstract**: {f.iloc[0].abstract}
                    """
                    )
        else:
            try:
                if filter_company:
                    frame = data[data['company_name'].isin(filter_company)]
                else:
                    frame = data
            
                for id_ in set(frame.article_id):
                    f = frame[(frame.article_id == id_)]

                    newline= '\n'

                    st.write(
                        f"""**{f.iloc[0].title}**  
                    **Affiliate Anti-Aging Company Name**: {f.iloc[0].company_name.capitalize()}
                    **Journal**: {f.iloc[0].journal}  
                    **Publication Date**: {f.iloc[0].publication_date}  
                    {newline}**Keywords**: *{f.iloc[0].keywords}*
                    {newline}**DOI**: *{f.iloc[0].doi.split(newline)[0]}*
                    {newline}**Abstract**: {f.iloc[0].abstract}
                    """
                    )
            except:
                pass #see if this works if you have multiple companies
        st.text("*Search results may not reflect all information available in the PubMed database, please search the title or DOI to get more information.")
    except Exception as e:
        st.write(e)


if __name__ == "__main__":
    main()
    
    
    
