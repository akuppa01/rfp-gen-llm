from sentence_transformers import SentenceTransformer
import os
import pickle

def create_embeddings(texts, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, convert_to_tensor=True)
    return embeddings

def save_embeddings(embeddings, filename='embeddings.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(embeddings, f)
