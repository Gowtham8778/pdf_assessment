# app/generation/rag_chain.py

from app.logger import get_logger

logger = get_logger("RAG")

PROMPT = """You are a financial data extraction assistant.

Task:
Find the capital expenditure for the given mine.

Instructions:
- If a table exists → return FULL table (all rows, columns)
- Preserve exact structure
- If multiple tables → pick the correct mine
- If no table → extract values from text clearly
- Do NOT hallucinate
- Do NOT summarize
- If not found → say "Not found in context"

Context:
{context}

Question:
{query}

Answer:
"""

class RAGChain:
    def __init__(self, retriever, reranker, llm):
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm

    def run(self, query):
        docs = self.retriever.retrieve(query)
        reranked = self.reranker.rerank(query, docs)

        context = "\n\n".join(reranked)

        prompt = PROMPT.format(context=context, query=query)

        return self.llm.generate(prompt)