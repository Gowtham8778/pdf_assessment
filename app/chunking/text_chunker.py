# app/chunking/text_chunker.py

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from app.logger import get_logger

logger = get_logger("Chunker")

def is_table(chunk: str):
    return "|" in chunk and "---" in chunk

def chunk_text(markdown_text: str):
    logger.info("Starting chunking...")

    headers = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]

    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
    md_chunks = md_splitter.split_text(markdown_text)

    final_chunks = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    for chunk in md_chunks:
        content = chunk.page_content

        if is_table(content):
            final_chunks.append((content, "table"))
        else:
            splits = splitter.split_text(content)
            for s in splits:
                final_chunks.append((s, "text"))

    logger.info(f"Total chunks: {len(final_chunks)}")
    return final_chunks