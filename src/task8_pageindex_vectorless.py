"""
Task 8 — PageIndex vectorless fallback.

Hướng dẫn:
    1. Đọc PAGEINDEX_API_KEY từ .env.
    2. Upload tài liệu ở định dạng PageIndex hỗ trợ.
    3. Cache document IDs để không upload lại.
    4. Parse kết quả thành SearchResult có method pageindex.

PageIndex là dịch vụ ngoài: cần timeout và xử lý lỗi để pipeline không crash.
"""

import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from pageindex import PageIndexClient

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
LEGAL_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
DOC_ID_CACHE = Path(__file__).parent.parent / "data" / "pageindex_doc_ids.json"

POLL_INTERVAL_SECONDS = 5
# Chỉ dùng khi upload/dựng cây cấu trúc (upload_documents) — chat_completions()
# ở pageindex_search() là API đồng bộ (trả ngay), không cần poll/timeout riêng.
INDEXING_TIMEOUT_SECONDS = 300


def _load_doc_id_cache() -> dict[str, str]:
    if DOC_ID_CACHE.exists():
        return json.loads(DOC_ID_CACHE.read_text(encoding="utf-8"))
    return {}


def _save_doc_id_cache(cache: dict[str, str]) -> None:
    DOC_ID_CACHE.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def upload_documents() -> None:
    """Upload tài liệu lên PageIndex và lưu lại doc_id để tái sử dụng.

    PageIndex không embed từng chunk như ChromaDB — nó dựng 1 cây cấu trúc
    (mục lục/section) cho toàn bộ tài liệu, nên cần thời gian xử lý (polling)
    trước khi có thể query. Cache doc_id vào JSON để chạy lại script không
    upload trùng (upload lại vừa tốn phí vừa mất thời gian chờ dựng cây).
    """
    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    cache = _load_doc_id_cache()

    for path in sorted(LEGAL_DIR.glob("*.pdf")):
        if path.name in cache:
            print(f"Skip (đã upload trước đó): {path.name}")
            continue

        submission = client.submit_document(str(path))
        doc_id = submission["doc_id"]
        print(f"Uploaded {path.name} -> doc_id={doc_id}, đang chờ dựng cây cấu trúc...")

        deadline = time.monotonic() + INDEXING_TIMEOUT_SECONDS
        while not client.is_retrieval_ready(doc_id):
            if time.monotonic() > deadline:
                raise TimeoutError(f"PageIndex xử lý quá lâu cho {path.name}")
            time.sleep(POLL_INTERVAL_SECONDS)

        cache[path.name] = doc_id
        _save_doc_id_cache(cache)
        print(f"Sẵn sàng: {path.name}")


# LLM chèn tag dạng "<doc=...>" ngay sau đoạn văn nó vừa trích dẫn — dùng để
# tách answer thành từng đoạn tương ứng với 1 citation.
CITATION_TAG = re.compile(r"<doc=[^>]+>")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Trả về pageindex SearchResult bằng cách hỏi trực tiếp qua chat_completions.

    LƯU Ý: PageIndex SDK có API cũ dạng submit_query()/get_retrieval() (kiểu
    "vector search" — trả top-k chunk theo query) nhưng API đó đã bị
    deprecated, gọi vào sẽ nhận response có field "deprecation" báo dùng
    chat_completions thay thế. chat_completions() KHÔNG trả sẵn các đoạn text
    liên quan (retrieved chunks) như vector search thông thường — nó để LLM
    tự đọc cây cấu trúc tài liệu và trả lời trực tiếp, kèm mảng "citations"
    (document/page/block_id, không có text gốc). Vì vậy phải tự suy ra
    "content" của mỗi citation bằng cách cắt đoạn văn NGAY TRƯỚC mỗi tag
    <doc=...> trong chính câu trả lời của LLM (xem vòng lặp dưới) — nghĩa là
    "content" ở đây là văn xuôi do LLM viết lại, không phải câu chữ gốc trong
    PDF như dense/BM25 search.
    """
    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    cache = _load_doc_id_cache()
    if not cache:
        raise RuntimeError("Chưa có doc_id nào — chạy upload_documents() trước.")

    response = client.chat_completions(
        messages=[{"role": "user", "content": query}],
        doc_id=list(cache.values()),
        enable_citations=True,
    )
    answer = response["choices"][0]["message"]["content"]
    citations = response.get("citations", [])
    tag_positions = list(CITATION_TAG.finditer(answer))

    results = []
    cursor = 0
    # zip(tag_positions, citations): PageIndex trả citations theo đúng thứ tự
    # các tag <doc=...> xuất hiện trong answer, nên ghép theo vị trí là đủ.
    for rank, (tag_match, citation) in enumerate(zip(tag_positions, citations), 1):
        content = answer[cursor : tag_match.start()].strip()
        cursor = tag_match.end()
        if not content:
            continue
        results.append(
            {
                "id": f"{citation['document']}::{citation['block_id']}::{rank}",
                "content": content,
                # Không có cosine score thật (không phải vector search) — dùng
                # 1/rank làm proxy: citation xuất hiện càng sớm trong câu trả
                # lời của LLM thì coi là càng liên quan.
                "score": 1.0 / rank,
                "metadata": {
                    "source": citation["document"],
                    "title": citation["document"],
                    "doc_type": "legal",
                    "url": None,
                },
                "retrieval_method": "pageindex",
            }
        )

    return results[:top_k]


if __name__ == "__main__":
    upload_documents()
