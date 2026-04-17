# main.py

from app.embedding.hf_embedder import HFEmbedder
from app.vectorstore.chroma_store import ChromaStore
from app.retrieval.retriever import Retriever
from app.retrieval.reranker import Reranker
from app.generation.llm import GroqLLM
from app.generation.rag_chain import RAGChain
from app.pipeline.ingestion_pipeline import ingest_folder

def main():
    embedder = HFEmbedder()
    store = ChromaStore()

    retriever = Retriever(embedder, store)
    reranker = Reranker()
    llm = GroqLLM()

    rag = RAGChain(retriever, reranker, llm)

    print("Type 'ingest' to load PDFs or ask questions:")

    while True:
        query = input("\n>> ")

        if query == "exit":
            break

        if query == "ingest":
            ingest_folder("./data", embedder, store)
            continue

        answer = rag.run(query)
        print("\n", answer)


if __name__ == "__main__":
    main()