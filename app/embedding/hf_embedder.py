# app/embedding/hf_embedder.py

from sentence_transformers import SentenceTransformer
from app.logger import get_logger

logger = get_logger("Embedder")

class HFEmbedder:
    def __init__(self):
        self.model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    def embed(self, texts, batch_size=8):
        logger.info(f"Embedding {len(texts)} chunks...")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        # ✅ Convert each embedding to list
        return [emb.tolist() for emb in embeddings]

    def embed_query(self, query: str):
        embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )[0]

        # ✅ Always return list
        return embedding.tolist()