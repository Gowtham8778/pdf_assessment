# app/pipeline/ingestion_pipeline.py

import os
import hashlib
from app.parsers.pdf_parser import parse_pdf
from app.chunking.text_chunker import chunk_text
from app.logger import get_logger

logger = get_logger("Ingestion")

def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def ingest_folder(folder_path, embedder, store):
    seen_hashes = set()

    for file in os.listdir(folder_path):
        if not file.endswith(".pdf"):
            continue

        full_path = os.path.join(folder_path, file)
        f_hash = file_hash(full_path)

        if f_hash in seen_hashes:
            logger.info(f"Skipping duplicate: {file}")
            continue

        seen_hashes.add(f_hash)

        md = parse_pdf(full_path)
        chunks = chunk_text(md)

        texts = [c[0] for c in chunks]
        types = [c[1] for c in chunks]

        embeddings = embedder.embed(texts)

        ids = [f"{file}_{i}" for i in range(len(texts))]
        metadata = [
            {
                "source": file,
                "chunk_id": i,
                "type": types[i],
            }
            for i in range(len(texts))
        ]

        store.upsert(ids, texts, embeddings, metadata)