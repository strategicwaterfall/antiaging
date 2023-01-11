import faiss
import pickle
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import regex as re
import random

st.markdown("""
<style>
.big-font {
    font-size:30px !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache(allow_output_mutation=True)
def find_indexes_of_matching_keywords(list_of_keywords,df,column):
    """" function that takes in a input list of strings, a dataframe and a column containing 
    lists as input and returns indexes of rows that match elements in the input list with 
    elements in lists in the column"""
    """match a string.
    :parameter
        :param column: string - name of column containing lists of text to match
        :param list_of_keywords: list - list of keywords to match
        :param df: dataframe - dataframe containing column
    :return
        index list"""

    indexes = []
    matched_words = []
    df_copy = df.copy(deep=True)
    df_copy[column] = df_copy[column].apply(lambda x: [item.lower() for item in x])
    list_of_keywords = [item.lower() for item in list_of_keywords]
    for i in range(len(df_copy[column])):
        for p in list_of_keywords:
            for j in range(len(df_copy[column][i])):
                    if p in df[column][i][j]:
                         indexes.append(i)
                         matched_words.append((p,df[column][i][j]))
    return indexes, matched_words


@st.cache(allow_output_mutation=True)
def display_dataframe_withindex(df,indexes):
    """Display dataframe."""
    df_display = df.iloc[indexes]
    return df_display

@st.cache(allow_output_mutation=True)
def random_list(list,n):
    """ function that creates a random list of n elements from a long list"""
    random_list = []
    for i in range(n):
        random_list.append(random.choice(list))
    return random_list

@st.cache(allow_output_mutation=True)
def combine_list_column(df,column):
    """ function that combines a column of lists in dataframe to a large list """
    df_copy = df.copy(deep=True)
    list_of_lists = df_copy[column].tolist()
    list_of_lists = [item for sublist in list_of_lists for item in sublist]
    return list(set(list_of_lists)) #drops duplicates


@st.cache(allow_output_mutation=True)
def convert_list_of_strings_to_list(list_words:list)->list:
    """ function that converts a list of strings sperated by comma into a list each strings"""
    list_output = []
    for text in list_words:
        text = text.split(',')
        for word in text:
            word = word.strip()
            list_output.append(word)
    return list_output



#usage of above functions examples 
#list_combined_keywords = combine_list_column(df_final_abstracts_clean,'keywords')
#list_combined_keywords_final =(convert_list_of_strings_to_list(list_combined_keywords))
#find_indexes_of_matching_keywords(test_list[2:4],df_final_abstracts_clean,'keywords')
#display_dataframe_withindex(df_final_abstracts_clean,find_indexes_of_matching_keywords(test_list,df_final_abstracts_clean,'keywords')[0])

@st.cache(allow_output_mutation=True)
def display_dataframe(df):
    titile = df.iloc[0].title
    return st.markdown('<p class="big-font">{0}</p>'.format(titile), unsafe_allow_html=True)


@st.cache(allow_output_mutation=True)
def read_data(data="data/publications/post2018_agingcomapnies_papers.csv"):
    return data


@st.cache(allow_output_mutation=True)
def load_bert_model():
    """Instantiate a sentence-level allenai-specer model."""
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

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
    """Tranforms query to vector using #https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
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
        list_combined_keywords = combine_list_column(data,'keywords')
        list_combined_keywords_final =(convert_list_of_strings_to_list(list_combined_keywords))

        st.title("🧬 Semantic Search over PubMed abstracts curated within :blue[anti-aging] industry and research👨‍🔬⚗️🧪💊*")
        st.caption("""There are over _3000_ abstracts in the database taken from PubMed. All publications are from _2018_ onwards.
                   These are compiled from 97 operating companies in the :blue[anti-aging] space, inlcuding Altos Labs, Unity Biotechnology, Insilico Medicine, and many more.""")
        st.caption("""The data about the companies is taken from [this](https://agingbiotech.info/companies) maintained by investor and creator Karl Pfleger.""")

        # User search
        user_input = st.text_area("Search box-Experimental", "stem cell research")
        # Keyword search
        keyword_list = st.multi_select('Select Keywords (Preffered)',list_combined_keywords, 'stem cell') #get dropdown of keywords

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
                display_dataframe(f)
                st.markdown(f"**Keywords**:")
                for i in ((f.iloc[0].keywords).strip('"')).split(','):
                    # clean string from all non alphanumeric characters
                    i = re.sub(r'\W+', '', i)
                    st.markdown(f"{i}")
                
                st.write(
                        f"""
                    {newline}**Affiliate Anti-Aging Company Name**: {f.iloc[0].company_name.capitalize()}
                    {newline}**Journal**: {f.iloc[0].journal}  
                    {newline}**Publication Date**: {f.iloc[0].publication_date}  
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
                    display_dataframe(f)
                    st.markdown(f"**Keywords**:")
                    keywords = []
                    for i in ((f.iloc[0].keywords).strip('"')).split(','):
                        i = re.sub(r'\W+', '', i)
                        keywords.append(i)

                    st.write(
                        f"""
                    {newline}**Affiliate Anti-Aging Company Name**: {f.iloc[0].company_name.capitalize()}
                    {newline}**Journal**: {f.iloc[0].journal}  
                    {newline}**Publication Date**: {f.iloc[0].publication_date}  
                    {newline}**Keywords**: *{keywords}*
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
    
    
    
