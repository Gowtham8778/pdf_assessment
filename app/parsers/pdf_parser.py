# app/parsers/pdf_parser.py

import pymupdf4llm
from app.logger import get_logger

logger = get_logger("PDFParser")

def parse_pdf(file_path: str) -> str:
    logger.info(f"Parsing PDF: {file_path}")
    md_text = pymupdf4llm.to_markdown(file_path)
    return md_text