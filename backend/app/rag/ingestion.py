"""Manual ingestion: PDF/DOCX/TXT -> chunk -> embed -> เก็บใน ChromaDB.

W1: skeleton only. W3: implement จริง.
"""

from pathlib import Path


def load_manual(path: Path) -> str:
    raise NotImplementedError("W3: load PDF/DOCX/TXT into raw text")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    raise NotImplementedError("W3: implement chunking strategy")


def ingest_manual(path: Path):
    raise NotImplementedError("W3: load -> chunk -> embed -> add to collection")
