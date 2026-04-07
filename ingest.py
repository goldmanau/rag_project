import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.pdf import extract_text_from_pdf
from utils.chunking import chunk_text
from utils.embeddings import EmbeddingStore

def ingest_pdf(pdf_path: str):
    """Full pipeline: extract → chunk → embed → store."""
    print(f"Processing: {pdf_path}")

    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)

    store = EmbeddingStore()
    store.add_chunks(chunks)

    print(f"Ingested {len(chunks)} chunks.")
    return True

if __name__ == "__main__":
    pdf_folder = "data/pdfs"
    for file in os.listdir(pdf_folder):
        if file.lower().endswith(".pdf"):
            ingest_pdf(os.path.join(pdf_folder, file))
