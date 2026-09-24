"""
Task 7 — Reciprocal Rank Fusion.

RRF gộp nhiều bảng xếp hạng mà không cộng trực tiếp cosine score với BM25
score. Công thức: RRF(d) = sum(1 / (k + rank)), rank bắt đầu từ 1.

Lưu ý: RRF score chỉ phản ánh thứ hạng, không dùng để quyết định fallback.

-> Dùng Jina hoặc self host hoặc bất cứ công cụ nào bạn quen
"""


def rerank_rrf(
    ranked_lists: list[list[dict]],
    top_k: int = 5,
    k: int = 60,
) -> list[dict]:
    """Fuse nhiều ranked lists (ví dụ dense + BM25) thành 1 danh sách hybrid.

    Ý tưởng RRF: không cộng trực tiếp cosine score với BM25 score (2 thang đo
    khác nhau, cộng thẳng vô nghĩa) — chỉ dùng RANK (vị trí 1, 2, 3...) của mỗi
    item trong từng danh sách, quy về cùng 1 thang qua 1/(k+rank). Hằng số k=60
    (giá trị chuẩn trong paper RRF gốc) làm giảm chênh lệch giữa rank #1 và
    rank #2 — tránh 1 danh sách "đè" hoàn toàn danh sách còn lại chỉ vì rank 1
    luôn thắng tuyệt đối.

    1 item xuất hiện ở NHIỀU danh sách sẽ được CỘNG DỒN điểm từ mỗi lần xuất
    hiện (scores.get(item_id, 0.0) + ...) — đây chính là cách RRF "đồng thuận":
    item được cả dense và BM25 đều xếp hạng cao sẽ vượt lên item chỉ được 1
    bên đánh giá cao.
    """
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            item_id = item["id"]
            scores[item_id] = scores.get(item_id, 0.0) + 1 / (k + rank)
            items[item_id] = item

    ranked_ids = sorted(scores, key=scores.get, reverse=True)

    results = []
    for item_id in ranked_ids[:top_k]:
        fused_item = items[item_id].copy()
        fused_item["score"] = scores[item_id]
        fused_item["retrieval_method"] = "hybrid"
        results.append(fused_item)
    return results


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """(Bonus) Rerank lại candidates bằng cross-encoder BAAI/bge-reranker-v2-m3.

    Khác với dense search (embed query và document RIÊNG rồi so cosine),
    cross-encoder nhận (query, document) làm 1 cặp input DUY NHẤT và cho điểm
    liên quan trực tiếp — thường chính xác hơn vì mô hình "nhìn" được cả 2 văn
    bản cùng lúc, nhưng đổi lại chậm hơn nhiều (phải chạy model 1 lần cho MỖI
    cặp query-candidate, không tận dụng được index đã embed sẵn).

    HẠN CHẾ ĐÃ QUAN SÁT THỰC TẾ: cross-encoder tối ưu cho độ liên quan CHỦ ĐỀ
    (topical similarity), không phải "câu này có chứa đúng số liệu trả lời
    không". Test trên corpus PTIT cho thấy nó có thể xếp 1 đoạn brochure mô tả
    chung về ngành (nhiều từ liên quan, không có số liệu) cao hơn đoạn bảng
    điểm chuẩn thật (ít từ mô tả, nhiều số) — nên KHÔNG dùng cross-encoder làm
    bước lọc cuối cùng cho câu hỏi cần số liệu chính xác; rerank_rrf ở trên vẫn
    là lựa chọn an toàn hơn cho use case này.
    """
    from sentence_transformers import CrossEncoder

    if not candidates:
        return []

    model = CrossEncoder("BAAI/bge-reranker-v2-m3")
    pairs = [(query, item["content"]) for item in candidates]
    scores = model.predict(pairs)

    reranked = []
    for item, score in zip(candidates, scores):
        new_item = item.copy()
        new_item["score"] = float(score)
        new_item["retrieval_method"] = "hybrid"
        reranked.append(new_item)

    reranked.sort(key=lambda item: item["score"], reverse=True)
    return reranked[:top_k]


if __name__ == "__main__":
    print("Implement rerank_rrf, then run contract tests.")
