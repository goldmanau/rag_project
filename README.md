# rag_project

Retrieval‑Augmented Generation (RAG) AI project

## description

Upload PDFs → ask questions → provide answers.

Learn: chunking, embeddings, vector DBs, RAG Pipeline.

## Operation Instruction

1) > python ingest.py   # upload + Text Extraction + chunking + embeddings + vector DB (Chroma)

2) > python app.py   # User Question + Retrieve Relevant Chunks + LLM Generates Answer (RAG) - using ctransformers and light weight llm -> mistral-7b-instruct-v0.2.Q4_K_M.gguf (4.2G)

4) Open a browser with http://127.0.0.1:7860  and ask question ...

## Detail Description
A fully offline Retrieval‑Augmented Generation (RAG) system that lets you upload PDFs, convert them into vector embeddings, store them in a local Chroma database, and query them using a local GGUF large language model. No API keys, no cloud dependencies — everything runs locally on your machine.

## Features
Extracts text from PDF documents

Splits text into clean, semantic chunks

Generates embeddings using SentenceTransformers (MiniLM-L6-v2)

Stores vectors in a local ChromaDB

Loads a local GGUF model using llama.cpp

Answers questions using retrieved context + LLM

Runs fully offline

## Simple Gradio web interface

Project Structure

Code
rag_project/
│── app.py                # Main Gradio app for Q&A
│── ingest.py             # PDF ingestion → chunking → embeddings → Chroma
│── utils/
│     ├── pdf.py          # PDF text extraction
│     ├── chunking.py     # Text splitting logic
│     └── embeddings.py   # Embedding + ChromaDB wrapper
│── data/
│     └── pdfs/           # Your PDF files
│── db/
│     └── chroma/         # Local vector database
│── models/
│     └── model.gguf      # Your local GGUF LLM
│── rag_env/              # Virtual environment (optional to commit)

## Add your PDF files
Place them in:

Code
data/pdfs/

## Ingest your documents
Run:

Code
python ingest.py

This will:

Read all PDFs

Extract text

Chunk the text

Generate embeddings

Store them in ChromaDB

You should see output like:

Code
Processing: Recursion_Chapter1.pdf
Ingested 85 chunks.

## Run the local RAG assistant
Start the app:

Code
python app.py
Gradio will open in your browser:

Code
http://127.0.0.1:7860
Ask questions such as:

“What is recursion?”

“Summarise the main idea from Chapter 1.”

“Explain the example on page 3.”

## Choosing a GGUF model
Recommended CPU‑friendly models:

Mistral‑7B-Instruct‑v0.2.Q4_K_M.gguf

Llama‑2‑7B‑Chat.Q4_K_M.gguf

Phi‑2.Q4_K_M.gguf

Place your model in:

Code
models/
Then update the path in app.py.

## Notes
Llama‑3 and Phi‑3 GGUF models are not yet fully compatible with some backends.

Larger context windows require more RAM.

CPU‑only inference is slower; quantized models (Q4_K_M) are recommended.
