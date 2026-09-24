"""
Task 5 — Semantic search.

Embed query bằng chính hàm của Task 4, query ChromaDB và đổi cosine distance
thành similarity. Output phải theo SearchResult, sort giảm dần và không quá top_k.
"""

from .task4_chunking_indexing import embed_texts, get_collection, restore_metadata_defaults


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về dense SearchResult theo score giảm dần.

    Dùng CHUNG embed_texts() với Task 4 (không tự viết lại 1 hàm embed khác) —
    query và document phải được embed bằng đúng 1 model, nếu không vector của
    2 bên sẽ không nằm trong cùng không gian và so cosine sẽ vô nghĩa.
    """
    query_vector = embed_texts([query])[0]
    response = get_collection().query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    results = []
    for item_id, content, metadata, distance in zip(
        response["ids"][0],
        response["documents"][0],
        response["metadatas"][0],
        response["distances"][0],
    ):
        results.append(
            {
                "id": item_id,
                "content": content,
                # Collection dùng cosine distance (Task 4: hnsw:space="cosine"),
                # với distance = 1 - cosine_similarity. Đổi lại thành similarity
                # (0..1, cao hơn = liên quan hơn) để cùng thang đo với BM25/RRF
                # score khi so sánh sau này. max(0.0, ...) chặn giá trị âm hiếm
                # gặp do sai số dấu phẩy động khi 2 vector gần như đối nhau.
                "score": max(0.0, 1.0 - distance),
                "metadata": restore_metadata_defaults(metadata),
                "retrieval_method": "dense",
            }
        )

    # ChromaDB thường đã trả kết quả theo distance tăng dần (tức score giảm
    # dần), nhưng sort lại ở đây để KHÔNG phụ thuộc vào việc DB có đảm bảo thứ
    # tự đó hay không — an toàn hơn khi đổi backend hoặc đổi cấu hình index.
    return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]


if __name__ == "__main__":
    for result in semantic_search("test query", top_k=3):
        print(result)
