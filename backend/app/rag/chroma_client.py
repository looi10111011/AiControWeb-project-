"""ChromaDB connection.

W1: skeleton — เชื่อมต่อและสร้าง/เปิด collection ได้.
W2: ออกแบบ schema จริงของ metadata (source, section, page, ฯลฯ).
"""

import chromadb
from chromadb import Collection

from backend.app.config import settings


def get_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=settings.chroma_persist_dir)


def get_collection() -> Collection:
    client = get_client()
    return client.get_or_create_collection(name=settings.chroma_collection_name)
