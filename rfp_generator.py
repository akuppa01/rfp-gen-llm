from transformers import pipeline
import os
import fitz  # PyMuPDF
import pinecone
import numpy as np
import openai
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()

# Initialize APIs
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)
# pc.delete_index("my-index")

# Define the index name
index_name = "my-index"

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

# Load LLM for RFP generation
llm = pipeline("text-generation", model="gpt2", max_length=1024)


# Function to extract text from PDFs
def extract_text_from_pdfs(data_folder="data"):
    pdf_texts = []
    for filename in os.listdir(data_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(data_folder, filename)
            doc = fitz.open(pdf_path)
            text = ""
            for page_num in range(doc.page_count):
                text += doc.load_page(page_num).get_text()
            pdf_texts.append((filename, text))
    return pdf_texts


# ✅ Function to split text into smaller chunks
def split_text_into_chunks(text, chunk_size=200):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


# Function to generate embeddings and store in Pinecone
def store_embeddings_in_pinecone():
    pdf_texts = extract_text_from_pdfs()
    for doc_name, text in pdf_texts:
        chunks = split_text_into_chunks(text)  # Split text into chunks before embedding
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) > 10:  # Avoid tiny chunks
                vector = embedding_model.encode(chunk).tolist()
                # Store minimal metadata (doc_id and chunk index)
                metadata = {"doc_id": doc_name, "chunk_index": i}
                index.upsert(vectors=[(f"{doc_name}-{i}", vector, metadata)])


# Function to retrieve relevant text from Pinecone
def retrieve_relevant_data(query, top_k=5):
    query_vector = embedding_model.encode(query).tolist()
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    
    # Debugging: print the results to check the structure
    print("Pinecone query results:", results)
    
    retrieved_texts = []
    
    # Ensure we're checking if 'text' is in metadata
    for match in results["matches"]:
        if "text" in match["metadata"]:
            retrieved_texts.append(match["metadata"]["text"])
        else:
            # Handle the case where 'text' is missing in the metadata
            retrieved_texts.append("No relevant text found in this document.")
    
    return "\n\n".join(retrieved_texts)

# Generate RFP from query
def generate_rfp_from_query(query):
    retrieved_text = retrieve_relevant_data(query, top_k=10)  # Retrieve more relevant text
    input_text = f"Create a comprehensive RFP for a cloud computing solution provider for an enterprise with 500 employees. The solution should include secure data storage, disaster recovery, and scalability. The vendor should provide detailed pricing, implementation timeline, and support options. Please include the following sections: Executive Summary, Technical Requirements, Pricing and Cost Structure, Implementation Timeline, and Support Options."
    response = llm(input_text, max_new_tokens=2048)  # Generate longer text
    generated_rfp = response[0]['generated_text']
    file_path = save_rfp_to_file(generated_rfp)
    return generated_rfp, file_path


# Save the generated RFP to a file
def save_rfp_to_file(rfp, output_dir="./outputs"):
    os.makedirs(output_dir, exist_ok=True)
    file_path = f"{output_dir}/generated_rfp.txt"
    
    if isinstance(rfp, list) or isinstance(rfp, tuple):
        rfp = "\n".join(str(item) for item in rfp)
    
    with open(file_path, "w") as f:
        f.write(rfp)
    
    return file_path


# Main function
def generate_rfp(query):
    generated_rfp, file_path = generate_rfp_from_query(query)
    print(f"Generated RFP:\n{generated_rfp}")
    print(f"RFP saved to: {file_path}")
    return generated_rfp, file_path
