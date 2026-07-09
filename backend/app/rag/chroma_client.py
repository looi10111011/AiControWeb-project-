"""ChromaDB connection.

W1: skeleton — เชื่อมต่อและสร้าง/เปิด collection ได้.
W2: ออกแบบ schema จริงของ metadata (source, section, page, ฯลฯ).
W3: เปลี่ยนมาใช้ local embedding (all-MiniLM-L6-v2 ผ่าน ChromaDB DefaultEmbeddingFunction)
    แทน Gemini API — รันได้ offline ไม่ต้องมี API key, โหลดโมเดลครั้งแรก ~90MB
    (เก็บ cache ไว้ในเครื่อง ครั้งต่อไปไม่โหลดซ้ำ)
"""

import chromadb
from chromadb import Collection
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from backend.app.config import settings

# ทุก collection ต้องใช้ embedding function เดียวกันเสมอ (ingest กับ query ต้องมิติตรงกัน)
_embedding_function = DefaultEmbeddingFunction()


def get_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=settings.chroma_persist_dir)


def get_collection() -> Collection:
    client = get_client()
    return client.get_or_create_collection(
        name=settings.chroma_collection_name,
        embedding_function=_embedding_function,
    )
