# rag_project

RAG AI project

Upload PDFs → ask questions → provide answers.

Learn: chunking, embeddings, vector DBs, RAG Pipeline.

Instruction

1) > cmd  ->  cd C:\Users\user\rag_project   -> rag_env\Scripts\activate     # with python 3.11

2)  (rag_env) C:\Users\user\rag_project> python ingest.py   # upload + Text Extraction + chunking + embeddings + vector DB (Chroma)

3) (rag_env) C:\Users\user\rag_project> python app.py   # User Question + Retrieve Relevant Chunks + LLM Generates Answer (RAG) - using ctransformers and light weight llm -> mistral-7b-instruct-v0.2.Q4_K_M.gguf (4.2G)

4) Open a browser with http://127.0.0.1:7860  and ask question:
