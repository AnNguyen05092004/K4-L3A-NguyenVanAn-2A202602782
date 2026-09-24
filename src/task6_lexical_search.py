"""
Task 6 — Lexical search bằng BM25.

Dùng cùng corpus chunks với Task 5. BM25 phù hợp với từ khóa chính xác, mã tài
liệu và tên riêng. Output phải theo SearchResult và sort score giảm dần.
"""

from .task4_chunking_indexing import get_collection, restore_metadata_defaults


def _load_corpus() -> list[dict]:
    """Lấy toàn bộ chunk đã index ở Task 4 để BM25 dùng đúng cùng corpus.

    BM25 và dense search (Task 5) PHẢI xếp hạng trên cùng 1 tập chunk — nếu
    không, id trả về từ 2 nhánh sẽ không khớp nhau và RRF (Task 7) không thể
    gộp thứ hạng đúng cách. Lấy trực tiếp từ ChromaDB collection (thay vì tự
    load lại Markdown và chunk lần 2) để đảm bảo tuyệt đối cùng 1 nguồn.
    """
    response = get_collection().get(include=["documents", "metadatas"])
    return [
        {
            "id": item_id,
            "content": content,
            "metadata": restore_metadata_defaults(item_metadata),
        }
        for item_id, content, item_metadata in zip(
            response["ids"], response["documents"], response["metadatas"]
        )
    ]


# Load corpus 1 lần duy nhất khi import module — BM25 cần build lại toàn bộ
# index mỗi khi corpus thay đổi (khác dense search, vốn đã có index sẵn trong
# ChromaDB), nên tránh phải query lại ChromaDB mỗi lần gọi lexical_search().
CORPUS: list[dict] = _load_corpus()


def build_bm25_index(corpus: list[dict]):
    """Tạo BM25 index từ cùng corpus chunks của Task 4.

    Dùng BM25L (variant "L" = length-normalized) từ rank_bm25, KHÔNG dùng
    BM25Okapi mặc định. Lý do: công thức IDF gốc của BM25Okapi —
    ln(N - n + 0.5) - ln(n + 0.5) — có thể ra ĐÚNG 0 khi 1 từ xuất hiện ở đúng
    N/2 tài liệu (N = tổng số tài liệu). Với corpus nhỏ (test fixture chỉ 2
    tài liệu), điều này khiến MỌI SO SÁNH có từ đó bị điểm 0 — kể cả tài liệu
    thật sự liên quan — nên lexical_search() không phân biệt được tài liệu
    đúng với tài liệu không liên quan. BM25L đổi công thức IDF để tránh suy
    biến về 0 kiểu này, ổn định hơn cho cả corpus nhỏ và lớn.
    """
    from rank_bm25 import BM25L

    tokenized = [item["content"].lower().split() for item in corpus]
    return BM25L(tokenized)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về BM25 SearchResult theo score giảm dần.

    So với dense search: BM25 chấm điểm theo trùng khớp TỪ (term overlap có
    trọng số theo độ hiếm của từ), không hiểu ngữ nghĩa — nên mạnh với từ khóa
    chính xác, mã số, tên riêng (những thứ dense/embedding dễ bỏ sót vì không
    có "gần nghĩa" nào thay thế được), nhưng yếu với câu hỏi diễn đạt khác từ
    so với văn bản gốc.
    """
    import numpy as np

    bm25 = build_bm25_index(CORPUS)
    scores = bm25.get_scores(query.lower().split())
    # argsort() mặc định tăng dần -> đảo ngược ([::-1]) để lấy điểm cao nhất
    # trước, rồi mới cắt lấy top_k.
    indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for index in indices:
        # score=0 nghĩa là câu query không có từ nào trùng với chunk này —
        # loại hẳn ra thay vì trả về 1 "kết quả" không thực sự liên quan.
        if scores[index] <= 0:
            continue
        item = CORPUS[index]
        results.append(
            {
                "id": item["id"],
                "content": item["content"],
                "score": float(scores[index]),
                "metadata": item["metadata"],
                "retrieval_method": "bm25",
            }
        )
    return results


if __name__ == "__main__":
    for result in lexical_search("test query", top_k=3):
        print(result)
