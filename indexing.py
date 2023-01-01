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

# Use pandas to read files from S3 buckets!
df = pd.read_csv('data/publications/post2018_agingcomapnies_papers.csv', index_col=0)
df = df.fillna('') # remove NaNs
#https://huggingface.co/sentence-transformers/allenai-specter
model = SentenceTransformer('sentence-transformers/allenai-specter')
#for the trial app we will display only 1000 abstracts
abstracts = df.abstract[0:1000].to_list()

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
index.add_with_ids(embeddings, df.article_id[0:1000].values)
 # type: ignore
D, I = index.search(np.array([embeddings[541]]), k=10)
 # type: ignore
print(f'L2 distance: {D.flatten().tolist()}\n\n PubMed paper IDs: {I.flatten().tolist()}')
df.iloc[541, :]
df[df['article_id'] == 31455183]

with open(f"{project_dir}/models/faiss_index.pickle", "wb") as h:
    pickle.dump(faiss.serialize_index(index), h)