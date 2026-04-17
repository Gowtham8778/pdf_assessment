# app/vectorstore/chroma_store.py

import chromadb
from app.logger import get_logger

logger = get_logger("ChromaStore")

class ChromaStore:
    def __init__(self, persist_dir="./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)

        self.collection = self.client.get_or_create_collection(
            name="financial_rag"
        )

    def upsert(self, ids, documents, embeddings, metadatas, batch_size=50):
        total = len(ids)
        logger.info(f"Upserting {total} chunks in batches...")

        for i in range(0, total, batch_size):
            batch_ids = ids[i:i+batch_size]
            batch_docs = documents[i:i+batch_size]
            batch_embs = embeddings[i:i+batch_size]
            batch_meta = metadatas[i:i+batch_size]

            self.collection.upsert(
                ids=batch_ids,
                documents=batch_docs,
                embeddings=batch_embs,   # already list of lists
                metadatas=batch_meta,
            )

            logger.info(f"Upserted {i + len(batch_ids)}/{total}")

        logger.info("Upsert completed.")

    def query(self, query_embedding, top_k=10):
        # ✅ Ensure correct format
        if not isinstance(query_embedding, list):
            raise ValueError("Query embedding must be a list")

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        return results["documents"][0]