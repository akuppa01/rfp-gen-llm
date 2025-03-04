from transformers import pipeline
import os
import fitz  # PyMuPDF
import pinecone
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import requests


# Load environment variables from .env file
load_dotenv()

# Debug: Print environment variables
# print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))
# print("PINECONE_API_KEY:", os.getenv("PINECONE_API_KEY"))

# Check if OPENAI_API_KEY is set
# openai_api_key = os.getenv("OPENAI_API_KEY")
# if not openai_api_key:
#     raise ValueError("OPENAI_API_KEY environment variable is not set. Please check your .env file.")

# # Initialize the OpenAI client
# client = OpenAI(api_key=openai_api_key)

# Initialize Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY environment variable is not set. Please check your .env file.")

pc = Pinecone(api_key=pinecone_api_key)

# Define the index name
index_name = "rfp-index"

# Create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # Update based on your embedding model
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# Connect to Pinecone index
index = pc.Index(index_name)

# Initialize embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to extract text from PDFs
def extract_text_from_pdfs(data_folder="data"):
    pdf_texts = []
    for filename in os.listdir(data_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(data_folder, filename)
            with fitz.open(pdf_path) as doc:
                text = ""
                for page in doc:
                    text += page.get_text()
                chunks = [chunk.strip() for chunk in text.split(". ") if chunk.strip()]
                pdf_texts.extend(chunks)
    return pdf_texts

# Function to store embeddings in Pinecone
def store_embeddings_in_pinecone(data_folder="data"):
    pdf_texts = extract_text_from_pdfs(data_folder)
    for i, text in enumerate(pdf_texts):
        truncated_text = text[:500]
        embedding = embedding_model.encode(truncated_text)
        index.upsert([(f"doc{i}_chunk{i}", embedding.tolist(), {"text": truncated_text})])

# Function to retrieve relevant chunks from Pinecone
def retrieve_relevant_chunks(query, top_k=5):
    query_embedding = embedding_model.encode(query).tolist()
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    relevant_chunks = [match.metadata["text"] for match in results.matches]
    return relevant_chunks


# Initialize the OpenAI client
client = OpenAI(api_key="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")

def generate_rfp(query):
    # Retrieve relevant chunks (same as before)
    relevant_chunks = retrieve_relevant_chunks(query, top_k=5)
    context = "\n\n".join(relevant_chunks)

    # Define the prompt
    prompt = f"""
    You are an expert in writing Request for Proposals (RFPs). Based on the following context, generate a detailed RFP document with the following sections:
    1. Introduction
    2. Scope of Work
    3. Requirements
    4. Evaluation Criteria
    5. Submission Guidelines
    6. Timeline

    Context:
    {context}

    Query:
    {query}

    Ensure the RFP is 3-4 pages long and follows a professional tone.
    """

    # Generate RFP using OpenAI GPT-3.5
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates professional RFPs."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2000,
        temperature=0.7
    )

    return response.choices[0].message.content