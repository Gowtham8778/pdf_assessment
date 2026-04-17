# test/rag_chain_test.py

from app.embedding.hf_embedder import HFEmbedder
from app.vectorstore.chroma_store import ChromaStore
from app.retrieval.retriever import Retriever
from app.retrieval.reranker import Reranker
from app.generation.llm import GroqLLM
from app.generation.rag_chain import RAGChain

embedder = HFEmbedder()
store = ChromaStore()

retriever = Retriever(embedder, store)
reranker = Reranker()
llm = GroqLLM()

rag = RAGChain(retriever, reranker, llm)

while True:
    query = input("Query: ")
    print(rag.run(query))