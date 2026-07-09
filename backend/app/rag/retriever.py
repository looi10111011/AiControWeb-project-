"""Query คู่มือทุกครั้งที่ Planner วางแผน (เชื่อมกับ orchestrator ใน W6).

W1: skeleton only.
"""

from backend.app.rag.chroma_client import get_collection


def query_manual(query: str, n_results: int = 5) -> list[str]:
    collection = get_collection()
    results = collection.query(query_texts=[query], n_results=n_results)
    return results.get("documents", [[]])[0]
