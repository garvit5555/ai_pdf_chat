import streamlit as st
import fitz  # PyMuPDF for text extraction
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai  # Google Gemini API

# Configure Gemini Pro API
GEMINI_API_KEY = your_api_key  # Replace with actual API key
genai.configure(api_key=GEMINI_API_KEY)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return text

# Chunk text dynamically
def chunk_text(text, max_length=512):
    sentences = text.split(". ")
    chunks, chunk = [], ""
    for sentence in sentences:
        if len(chunk) + len(sentence) < max_length:
            chunk += sentence + ". "
        else:
            chunks.append(chunk.strip())
            chunk = sentence + ". "
    if chunk:
        chunks.append(chunk.strip())
    return chunks

# Generate embeddings
def generate_embeddings(chunks):
    return model.encode(chunks, convert_to_numpy=True)

# FAISS Vector Store class
class FAISSVectorStore:
    def __init__(self, embedding_dim):
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.text_data = []

    def store_vectors(self, chunks, embeddings):
        self.text_data = chunks
        self.index.add(embeddings)

    def search_query(self, query, top_k=3):
        query_embedding = model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)
        results = [(self.text_data[idx], distances[0][i]) for i, idx in enumerate(indices[0]) if idx < len(self.text_data)]
        return results

# Generate response using Gemini Pro
def generate_natural_response(query, results):
    top_chunks = "\n".join([f"- {text}" for text, _ in results])
    prompt = f"""
    You are an AI assistant. Answer the user's query based on the retrieved information.
    User Query: {query}
    
    Retrieved Information:
    {top_chunks}
    
    Provide a well-structured and natural response.
    """
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    return response.text

# Streamlit UI
st.title("PDF AI Assistant")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    pdf_path = "uploaded.pdf"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    embeddings = generate_embeddings(chunks)

    # Initialize FAISS vector store
    vector_store = FAISSVectorStore(embedding_dim=embeddings.shape[1])
    vector_store.store_vectors(chunks, embeddings)
    
    query = st.text_input("Enter your query")
    if query:
        search_results = vector_store.search_query(query)
        response = generate_natural_response(query, search_results)
        st.subheader("AI Response:")
        st.write(response)
