# app/retrieval/retriever.py

class Retriever:
    def __init__(self, embedder, vectorstore):
        self.embedder = embedder
        self.vectorstore = vectorstore

    def retrieve(self, query):
        query_emb = self.embedder.embed_query(query)  # already list
        docs = self.vectorstore.query(query_emb, top_k=10)
        return docs