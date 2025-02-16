# ai_pdf_chat




https://github.com/user-attachments/assets/0ed40f40-b500-4797-b4d2-5928f9ca1705


# PDF AI Assistant

This project is a **PDF AI Assistant** that allows users to extract text from a PDF, store it in a FAISS vector database, perform efficient similarity searches, and generate responses using Google Gemini Pro.

## Project Structure

- **`app.py`**: A Streamlit-based UI for uploading PDFs, querying the extracted text, and generating responses using Gemini Pro.
- **`main.py`**: A command-line version that extracts text, stores embeddings in FAISS, performs a basic reranking of retrieved results, but does not generate an LLM response.

## Features

### `main.py` Features:
- Extracts text from a PDF using PyMuPDF.
- Splits text into manageable chunks.
- Generates embeddings using `all-MiniLM-L6-v2` from SentenceTransformers.
- Stores and searches embeddings using FAISS.
- Reranks retrieved results using cosine similarity.

### `app.py` Features:
- Streamlit-based frontend for ease of use.
- Uploads a PDF and processes its text.
- Searches and retrieves relevant information using FAISS.
- Generates a natural response using Gemini Pro.

## Installation

Install the required dependencies using:

```bash
pip install streamlit pymupdf sentence-transformers faiss-cpu google-generativeai numpy scikit-learn
```

## Running main.py

```bash
python main.py
```
## Running app.py

```bash
streamlit run app.py
```
## API Key Setup

```bash
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
```

