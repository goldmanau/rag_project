from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

class EmbeddingStore:
    def __init__(self, collection_name="documents", persist_dir="db/chroma"):
        # Load embedding model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # New Chroma client API
        self.client = PersistentClient(path=persist_dir)

        # Create or load collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks):
        """Embed and store chunks in ChromaDB."""
        for i, chunk in enumerate(chunks):
            embedding = self.model.encode(chunk).tolist()
            self.collection.add(
                documents=[chunk],
                embeddings=[embedding],
                ids=[str(i)]
            )

    def query(self, question: str, n_results: int = 5):
        """Retrieve the most relevant chunks."""
        q_emb = self.model.encode(question).tolist()
        results = self.collection.query(
            query_embeddings=[q_emb],
            n_results=n_results
        )
        return results["documents"][0]
