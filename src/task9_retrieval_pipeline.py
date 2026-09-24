"""
Task 9 — Retrieval pipeline hoàn chỉnh.

Luồng xử lý:
    1. Chạy semantic_search, lexical_search VÀ exact_major_match.
    2. Fuse cả 3 danh sách bằng RRF đúng một lần.
    3. Lấy best cosine score gốc từ dense results.
    4. Nếu score dưới threshold, thử PageIndex fallback.
    5. Nếu fallback lỗi, trả hybrid results thay vì crash.

Không so sánh threshold với RRF score vì hai thang đo khác nhau.
"""

import os
import re

from dotenv import load_dotenv

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank_rrf
from .task8_pageindex_vectorless import pageindex_search

load_dotenv()

# Ngưỡng cosine similarity (0..1) dưới mức này coi là dense "không tự tin" và
# thử fallback sang PageIndex. Không có giá trị đúng chung cho mọi corpus —
# phải hiệu chỉnh bằng cách thử vài query rõ-trong-domain (score phải cao) và
# vài query rõ-ngoài-domain (score phải thấp) trên chính dữ liệu của nhóm.
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD") or 0.3)
DEFAULT_TOP_K = 5


def _load_major_index() -> tuple[list[dict], dict[str, list[int]]]:
    """Cache toàn bộ chunk (như CORPUS ở Task 6) + map lowercase(major) -> vị
    trí trong cache, để exact_major_match() so khớp mà không phải quét lại
    ChromaDB mỗi lần gọi retrieve().
    """
    from .task4_chunking_indexing import get_collection, restore_metadata_defaults

    response = get_collection().get(include=["documents", "metadatas"])
    corpus = [
        {
            "id": item_id,
            "content": content,
            "metadata": restore_metadata_defaults(metadata),
        }
        for item_id, content, metadata in zip(
            response["ids"], response["documents"], response["metadatas"]
        )
    ]
    by_major: dict[str, list[int]] = {}
    for position, item in enumerate(corpus):
        major = item["metadata"].get("major")
        if major:
            by_major.setdefault(major.lower(), []).append(position)
    return corpus, by_major


_CORPUS, _MAJOR_INDEX = _load_major_index()

_WANTS_BAC = re.compile(r"(phía|miền|cơ sở)\s*bắc")
_WANTS_NAM = re.compile(r"(phía|miền|cơ sở)\s*nam")


def exact_major_match(query: str, top_k: int) -> list[dict]:
    """Lọc CHÍNH XÁC theo metadata["major"] khi tên ngành xuất hiện literal
    trong câu hỏi.

    Bổ sung cho dense/BM25 (Task 5/6): đã verify thực tế trên corpus PTIT —
    dense/BM25 xếp hạng rất thấp (rank 10-80/974) 1 dòng ngành cụ thể giữa
    hàng chục dòng bảng điểm có cấu trúc số liệu gần giống hệt nhau, vì
    embedding/BM25 không phân biệt tốt "tên ngành" giữa nhiều dòng cùng dạng
    (xem docs/RAG_PIPELINE_NOTES.md mục 7). Field "major" được trích sẵn ở
    Task 4 (dò đúng cột theo header, không hard-code vị trí cột) nên so khớp
    ở đây là so khớp CHÍNH XÁC theo tên ngành thật, không phải similarity.

    Nếu câu hỏi có nhắc "phía/miền/cơ sở bắc" hoặc "...nam", lọc thêm theo
    metadata["campus"] — tránh trộn lẫn dữ liệu 2 cơ sở khi người dùng đã hỏi
    rõ (KHÔNG dùng "bắc"/"nam" trần vì "nam" là substring của rất nhiều từ
    không liên quan, VD "Việt Nam").
    """
    query_lower = query.lower()

    matched_positions: list[int] = []
    seen: set[int] = set()
    for major_lower, positions in _MAJOR_INDEX.items():
        # 2 chiều: tên ngành nằm trong câu hỏi (trường hợp chính), hoặc câu
        # hỏi chỉ vỏn vẹn là tên ngành (hiếm, nhưng vẫn nên bắt được).
        if major_lower in query_lower or query_lower in major_lower:
            for position in positions:
                if position not in seen:
                    seen.add(position)
                    matched_positions.append(position)

    if not matched_positions:
        return []

    wants_bac = bool(_WANTS_BAC.search(query_lower))
    wants_nam = bool(_WANTS_NAM.search(query_lower))
    if wants_bac != wants_nam:  # người dùng chỉ định đúng 1 cơ sở, không mơ hồ
        target_campus = "bắc" if wants_bac else "nam"
        filtered = [
            position
            for position in matched_positions
            if _CORPUS[position]["metadata"].get("campus") == target_campus
        ]
        # Chỉ áp dụng lọc campus nếu còn kết quả — an toàn hơn là trả rỗng
        # khi (hiếm khi) không xác định được campus cho đúng chunk cần tìm.
        if filtered:
            matched_positions = filtered

    # 1 ngành có thể xuất hiện ở NHIỀU LOẠI bảng khác nhau (bảng điểm chuẩn,
    # bảng chỉ tiêu tuyển sinh...) — chỉ khớp theo major/campus thôi CHƯA đủ
    # để biết người dùng đang hỏi về bảng nào (bug thật đã gặp: query hỏi
    # "điểm chuẩn" nhưng bị chunk bảng "chỉ tiêu" của cùng ngành/cơ sở lấn át
    # trong top_k). Sắp theo số từ trong query (trừ từ quá ngắn) xuất hiện
    # trong CHÍNH NỘI DUNG chunk — chunk thuộc đúng loại bảng người dùng hỏi
    # (VD chứa sẵn chữ "điểm chuẩn") sẽ có độ khớp cao hơn hẳn. sort() ổn định
    # nên các vị trí đồng điểm vẫn giữ nguyên thứ tự match ban đầu.
    query_words = [word for word in query_lower.split() if len(word) > 2]

    def _content_overlap(position: int) -> int:
        content_lower = _CORPUS[position]["content"].lower()
        return sum(1 for word in query_words if word in content_lower)

    matched_positions.sort(key=_content_overlap, reverse=True)

    results = []
    for position in matched_positions[:top_k]:
        item = _CORPUS[position]
        results.append(
            {
                "id": item["id"],
                "content": item["content"],
                # Không có similarity thật (đây là exact filter, không phải
                # xếp hạng) — điểm cố định chỉ để hợp lệ với SearchResult
                # contract; RRF (Task 7) chỉ dùng RANK trong list này, không
                # dùng giá trị score, nên số 1.0 không ảnh hưởng kết quả fuse.
                "score": 1.0,
                "metadata": item["metadata"],
                "retrieval_method": "dense",
            }
        )
    return results


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """Trả về hybrid SearchResult (dense+BM25 qua RRF), hoặc pageindex nếu dense yếu.

    Lấy top_k*2 từ mỗi nhánh (dense/sparse) TRƯỚC khi fuse — nới rộng candidate
    pool để RRF có nhiều lựa chọn hơn là chỉ top_k, vì 1 chunk có thể xếp hạng
    thấp ở dense nhưng cao ở BM25 (hoặc ngược lại); nếu chỉ lấy top_k mỗi bên,
    những chunk kiểu đó dễ bị loại trước khi RRF có cơ hội cộng điểm cho chúng.
    """
    dense = semantic_search(query, top_k=top_k * 2)
    sparse = lexical_search(query, top_k=top_k * 2)
    # exact luôn đứng RANK 1 trong danh sách riêng của nó -> khi fuse (RRF),
    # nó gần như chắc chắn lọt top_k dù dense/sparse có xếp nó rất thấp.
    exact = exact_major_match(query, top_k=top_k)
    ranked_lists = [dense, sparse, exact] if exact else [dense, sparse]
    hybrid = (
        rerank_rrf(ranked_lists, top_k=top_k) if use_reranking else dense[:top_k]
    )

    # Fallback PHẢI xét trên cosine score GỐC của dense (dense[0]["score"]),
    # không phải RRF score của hybrid — RRF score chỉ phản ánh THỨ HẠNG (dùng
    # công thức 1/(k+rank)) nên không còn cùng thang đo với SCORE_THRESHOLD
    # (được hiệu chỉnh theo thang cosine 0..1). So sai thang đo sẽ làm
    # threshold vô nghĩa (không bao giờ hoặc luôn luôn kích hoạt fallback).
    best_dense_score = dense[0]["score"] if dense else 0.0
    if best_dense_score < score_threshold:
        try:
            fallback = pageindex_search(query, top_k=top_k)
            if fallback:
                return fallback
        except Exception:
            # PageIndex là dịch vụ ngoài (network, quota, timeout...) — lỗi ở
            # đây không được làm sập cả pipeline, cứ dùng tạm kết quả hybrid.
            pass

    return hybrid[:top_k]


if __name__ == "__main__":
    for result in retrieve("test query", top_k=3):
        print(result)
