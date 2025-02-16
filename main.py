import fitz  # PyMuPDF for text extraction
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return text


def chunk_text(text, max_length=512):
    sentences = text.split(". ")  # Simple sentence-based splitting
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


model = SentenceTransformer("all-MiniLM-L6-v2")

def generate_embeddings(chunks):
    return model.encode(chunks, convert_to_numpy=True)


class FAISSVectorStore:
    def __init__(self, embedding_dim):
        self.index = faiss.IndexFlatL2(embedding_dim)  # FAISS index for vector search
        self.text_data = []

    def store_vectors(self, chunks, embeddings):
        self.text_data = chunks  # Store text data for retrieval
        self.index.add(embeddings)  # Add vectors to FAISS index
        print(f"Successfully stored {len(chunks)} vectors in FAISS.")

    def search_query(self, query, top_k=3):
        query_embedding = model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.text_data):  # Ensure valid index
                results.append((self.text_data[idx], distances[0][i]))

        return results

# Rerank results using cosine similarity
def rerank_results(query, results):
    query_embedding = model.encode([query], convert_to_numpy=True).reshape(1, -1)  # Ensure 2D
    reranked = sorted(
        results,
        key=lambda x: cosine_similarity(
            query_embedding, model.encode([x[0]], convert_to_numpy=True).reshape(1, -1)
        )[0, 0],  # Extract scalar value
        reverse=True,
    )
    return reranked


if __name__ == "__main__":
    pdf_path = "a.pdf"  # Change to actual PDF path
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    embeddings = generate_embeddings(chunks)

    # Initialize FAISS vector store
    vector_store = FAISSVectorStore(embedding_dim=embeddings.shape[1])
    vector_store.store_vectors(chunks, embeddings)

    query = "What is an AI agent?"
    search_results = vector_store.search_query(query)
    reranked_results = rerank_results(query, search_results)

    print("Top results:", reranked_results)
