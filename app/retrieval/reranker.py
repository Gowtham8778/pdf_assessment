# app/retrieval/reranker.py

from sentence_transformers import CrossEncoder
from app.logger import get_logger

logger = get_logger("Reranker")

class Reranker:
    def __init__(self):
        self.model = CrossEncoder("BAAI/bge-reranker-base")

    def rerank(self, query, docs, top_k=4):
        logger.info("Reranking documents...")

        pairs = [(query, d) for d in docs]
        scores = self.model.predict(pairs)

        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]]