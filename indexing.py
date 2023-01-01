# Used to import data from S3.
import pandas as pd
import s3fs
import boto3

# Used to create the dense document vectors.
import torch
from sentence_transformers import SentenceTransformer

# Used to create and store the Faiss index.
import faiss
import numpy as np
import pickle

# Used to do vector searches and display the results.
from vector_engine.utils import vector_search, id2details

#df = df.fillna('') # remove NaNs
# Use pandas to read files from S3 buckets!
df = pd.read_csv('post2018_agingcomapnies_papers.csv', index_col=0)
"""" function that sepearates empty rows in a specific column in a dataframe and returns two dataframes one wiht non-empty rows and other with empty rows
input is a column and a dataframe
outputs are two dataframes"""

def remove_empty_rows(df, column):
    df_empty = df[df[column].isna()]
    df_nonempty = df[df[column].notna()]
    return df_empty, df_nonempty

df_noindex, df_report_nonempty_abs = remove_empty_rows(df, 'abstract')

#https://huggingface.co/sentence-transformers/allenai-specter
model = SentenceTransformer('sentence-transformers/allenai-specter')

# to get the embeddings for the first 1000 abstracts - given the load on the machine 
df_report_nonempty_abs = df_report_nonempty_abs.reset_index(drop=True)
abstracts = df_report_nonempty_abs.abstract.to_list()

# Convert abstracts to vectors
embeddings = model.encode(abstracts, show_progress_bar=True) #TLDR: np.nan objects are of type float and cannot be encoded by the model.

# Step 1: Change data type
embeddings = np.array([embedding for embedding in embeddings]).astype("float32")

# Step 2: Instantiate the index
index = faiss.IndexFlatL2(embeddings.shape[1])

# Step 3: Pass the index to IndexIDMap
index = faiss.IndexIDMap(index)
df.columns


# Step 4: Add vectors and their IDs
index.add_with_ids(embeddings, df_report_nonempty_abs.article_id.values)
 # type: ignore
D, I = index.search(np.array([embeddings[541]]), k=10)
 # type: ignore
 
#test it 
print(f'L2 distance: {D.flatten().tolist()}\n\n PubMed paper IDs: {I.flatten().tolist()}')
df.iloc[541, :]
df[df['article_id'] == 31455183]

#pickle the index for first 1000 abstracts 
with open("/Users/paritoshmacmini/Documents/Waveflow/pdf_scanner/models_final/faiss_index_allabstracts.pickle", "wb") as h:
    pickle.dump(faiss.serialize_index(index), h)
    
    
#pickle the embeddings for first 1000 abstracts 
with open("/Users/paritoshmacmini/Documents/Waveflow/pdf_scanner/models_final/embeddings_allabstracts.pickle", "wb") as h:
    pickle.dump(embeddings, h)
    
    
#test pickle objects \\