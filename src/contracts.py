# Đây là "hợp đồng" schema/invariant chung mà mọi task (5-10) phải tuân theo,
# để các module ghép được với nhau (VD: RRF ở Task 7 cần SearchResult của cả
# Task 5 và Task 6 có đúng CÙNG shape). Không tự đổi field/kiểu dữ liệu ở đây
# — nếu cần đổi, phải sửa đồng bộ ở mọi nơi dùng các type/hàm validate này.

from typing import Literal, TypedDict


# RetrievalMethod: gắn trên MỖI SearchResult, cho biết nó ra từ nhánh nào.
# RetrievalSource: gắn trên GenerationResult cuối cùng (Task 10) — "none" khi
# an toàn từ chối trả lời (không dùng được nguồn nào).
RetrievalMethod = Literal["dense", "bm25", "hybrid", "pageindex"]
RetrievalSource = Literal["hybrid", "pageindex", "none"]


class DocumentMetadata(TypedDict):
    source: str
    title: str
    doc_type: str
    url: str | None


class ChunkMetadata(DocumentMetadata):
    chunk_index: int


class Document(TypedDict):
    id: str
    content: str
    metadata: DocumentMetadata


class Chunk(TypedDict):
    id: str
    content: str
    metadata: ChunkMetadata


class EmbeddedChunk(Chunk):
    embedding: list[float]


class SearchResult(TypedDict):
    id: str
    content: str
    score: float
    metadata: ChunkMetadata
    retrieval_method: RetrievalMethod


class GenerationResult(TypedDict):
    answer: str
    sources: list[SearchResult]
    retrieval_source: RetrievalSource


def validate_document(item: object, *, require_chunk: bool = False) -> None:
    """Raise ``ValueError`` when an item violates the document contract."""
    if not isinstance(item, dict):
        raise ValueError("document must be a dict")
    if not isinstance(item.get("id"), str) or not item["id"].strip():
        raise ValueError("document.id must be a non-empty string")
    if not isinstance(item.get("content"), str) or not item["content"].strip():
        raise ValueError("document.content must be a non-empty string")

    metadata = item.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("document.metadata must be a dict")
    for key in ("source", "title", "doc_type"):
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise ValueError(f"metadata.{key} must be a non-empty string")
    if "url" not in metadata or not (
        metadata["url"] is None or isinstance(metadata["url"], str)
    ):
        raise ValueError("metadata.url must be a string or None")
    if require_chunk and (
        not isinstance(metadata.get("chunk_index"), int)
        or metadata["chunk_index"] < 0
    ):
        raise ValueError("metadata.chunk_index must be a non-negative integer")


def validate_search_results(
    results: object,
    *,
    top_k: int | None = None,
    expected_method: RetrievalMethod | None = None,
) -> None:
    """Validate schema, ordering, uniqueness and result count."""
    if not isinstance(results, list):
        raise ValueError("search results must be a list")
    if top_k is not None and len(results) > max(top_k, 0):
        raise ValueError("search results exceed top_k")

    ids: list[str] = []
    scores: list[float] = []
    valid_methods = {"dense", "bm25", "hybrid", "pageindex"}
    for item in results:
        validate_document(item, require_chunk=True)
        score = item.get("score")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise ValueError("result.score must be numeric")
        method = item.get("retrieval_method")
        if method not in valid_methods:
            raise ValueError("result.retrieval_method is invalid")
        if expected_method is not None and method != expected_method:
            raise ValueError(f"expected retrieval_method={expected_method}")
        ids.append(item["id"])
        scores.append(float(score))

    if len(ids) != len(set(ids)):
        raise ValueError("search result IDs must be unique")
    # LUÔN phải sort giảm dần theo score — ngay cả sau khi qua reorder_for_llm
    # (Task 10, chống lost-in-the-middle). Nếu 1 hàm nào đó trả về list đã bị
    # xáo thứ tự (không sort theo score), phải sort/relabel LẠI trước khi trả
    # ra public API, không trả thẳng list đã xáo — xem
    # task10_generation._relabel_citations() cho ví dụ cách xử lý đúng.
    if scores != sorted(scores, reverse=True):
        raise ValueError("search results must be sorted by score descending")


def validate_generation_result(result: object) -> None:
    """Validate the public output of ``generate_with_citation``."""
    if not isinstance(result, dict):
        raise ValueError("generation result must be a dict")
    if not isinstance(result.get("answer"), str) or not result["answer"].strip():
        raise ValueError("generation answer must be a non-empty string")
    validate_search_results(result.get("sources"))
    if result.get("retrieval_source") not in {"hybrid", "pageindex", "none"}:
        raise ValueError("generation retrieval_source is invalid")
