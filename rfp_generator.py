from transformers import pipeline
import os
import fitz
import pinecone
import numpy as np
import openai
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# Initialize OpenAI API and Pinecone (Make sure API keys are set in .env)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone with your API key
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Create index if it doesn't exist (or you can modify the dimension and other details based on your needs)
if 'my-index' not in pc.list_indexes().names():
    pc.create_index(
        name='my-index', 
        dimension=1536,  # Update dimension based on your embeddings
        metric='cosine',  # Or 'cosine' based on your use case
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

# Initialize the LLM for RFP generation (You can switch to a more powerful model if needed)
llm = pipeline("text-generation", model="gpt2", max_length=1024)  # Set max_length to a larger value

# Load your RFP training data (PDFs in the /data folder)
def extract_text_from_pdfs(data_folder="data"):
    text = ""
    for filename in os.listdir(data_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(data_folder, filename)
            doc = fitz.open(pdf_path)
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                text += page.get_text()
    return text

# Function to handle long input by splitting it into chunks (if needed)
def handle_long_input(input_text, tokenizer, max_length=512):
    tokens = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_length)
    return tokens

# Simple retrieval mechanism (can be replaced with real retrieval system)
def retrieve_relevant_data(query, training_data):
    # Placeholder function for retrieving relevant data from Pinecone or a search system
    return "Relevant RFP data based on query"

# Generate RFP from the query
def generate_rfp_from_query(query):
    # Extract and preprocess text data from training PDFs
    training_text = extract_text_from_pdfs()

    # Retrieve relevant data for the query (could be replaced with Pinecone search)
    retrieved_data = retrieve_relevant_data(query, training_text)

    # Combine the query and retrieved data for input to the model
    input_text = f"Create an RFP based on this query: {query}\nTraining Data:\n{retrieved_data}"

    # Handle long input and ensure it fits within the model's token limit
    inputs = handle_long_input(input_text, llm.tokenizer)

    # Generate the RFP using the LLM with max_new_tokens to control how much text to generate
    generated_rfp = llm(f"Create an RFP based on this query: {query}\nTraining Data:\n{retrieved_data}", max_new_tokens=150)  # Set max_new_tokens instead of max_length

    # Save the generated RFP to a file
    file_path = save_rfp_to_file(generated_rfp)

    return generated_rfp, file_path

def save_rfp_to_file(rfp, output_dir="./outputs"):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the file path
    file_path = f"{output_dir}/generated_rfp.txt"
    
    # If the generated RFP is a list, join the items into a string
    if isinstance(rfp, list) or isinstance(rfp, tuple):
        rfp = "\n".join(str(item) for item in rfp)  # Join list items into a string
    
    print( '/n','/n','/n',type(rfp),'/n','/n','/n')
    # Now write the RFP (which is a string) to the file
    with open(file_path, "w") as f:
        f.write(rfp)
    
    return file_path

# Main function to generate an RFP based on the user's query
def generate_rfp(query):
    generated_rfp, file_path = generate_rfp_from_query(query)
    print(f"Generated RFP:\n{generated_rfp}")
    print(f"RFP saved to: {file_path}")
    return generated_rfp, file_path
