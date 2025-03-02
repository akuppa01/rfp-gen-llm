import faiss
import numpy as np
import pickle

def create_faiss_index(embeddings):
    dim = embeddings.shape[1]  # dimension of the embeddings
    index = faiss.IndexFlatL2(dim)  # L2 distance for similarity
    index.add(np.array(embeddings))  # Add embeddings to the index
    return index

def save_faiss_index(index, filename='faiss_index.index'):
    faiss.write_index(index, filename)

def load_faiss_index(filename='faiss_index.index'):
    return faiss.read_index(filename)
