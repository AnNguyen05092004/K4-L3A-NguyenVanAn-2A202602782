"""
Streamlit UI — trực quan hóa RAG Pipeline (Task 5-10) và chạy test nhanh.

Chạy: streamlit run app.py
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent

st.set_page_config(
    page_title="RAG Pipeline — PTIT Admissions",
    page_icon="🎓",
    layout="wide",
)

METHOD_COLORS = {
    "dense": "blue",
    "bm25": "orange",
    "hybrid": "green",
    "pageindex": "violet",
}


# st.cache_resource: Streamlit chạy lại TOÀN BỘ script mỗi khi người dùng
# tương tác (gõ câu hỏi, bấm nút...) — không cache thì mỗi lần rerun sẽ query
# lại ChromaDB chỉ để hiện thống kê ở sidebar. Cache theo tiến trình (process),
# chỉ mất tác dụng khi bấm nút "Refresh" (gọi .clear()) hoặc restart app.
@st.cache_resource(show_spinner=False)
def _get_collection_stats():
    from src.task4_chunking_indexing import get_collection

    collection = get_collection()
    data = collection.get(include=["metadatas"])
    doc_types: dict[str, int] = {}
    titles: set[str] = set()
    for metadata in data["metadatas"]:
        doc_types[metadata["doc_type"]] = doc_types.get(metadata["doc_type"], 0) + 1
        titles.add(metadata["title"])
    return {
        "total_chunks": len(data["ids"]),
        "by_doc_type": doc_types,
        "unique_titles": len(titles),
    }


def render_source_card(rank: int, source: dict) -> None:
    metadata = source["metadata"]
    with st.container(border=True):
        top = st.columns([5, 2, 2])
        top[0].markdown(f"**#{rank} · {metadata['title']}**")
        with top[1]:
            st.badge(source["retrieval_method"], color=METHOD_COLORS.get(source["retrieval_method"], "gray"))
        top[2].caption(f"score = {source['score']:.4f}")

        caption = f"source: {metadata['source']}"
        if metadata.get("url"):
            caption += f" · {metadata['url']}"
        st.caption(caption)

        preview = source["content"][:400]
        st.text(preview + ("..." if len(source["content"]) > 400 else ""))


with st.sidebar:
    st.title("🎓 RAG Pipeline")
    st.caption("PTIT Admissions — Day 8 Lab (Task 5-10)")

    st.divider()
    top_k = st.slider("Số chunks (top_k)", 3, 10, 5)

    st.divider()
    st.subheader("⚙️ Cấu hình hiện tại")
    st.text(f"Embedding: {os.getenv('EMBEDDING_PROVIDER', 'sentence_transformers')}")
    st.text(f"LLM: {os.getenv('LLM_PROVIDER', 'openai')}")
    st.text(f"Score threshold: {os.getenv('SCORE_THRESHOLD', '0.3')}")

    st.divider()
    st.subheader("📊 Corpus")
    if st.button("🔄 Refresh"):
        _get_collection_stats.clear()

    try:
        stats = _get_collection_stats()
        st.metric("Tổng chunks", stats["total_chunks"])
        st.metric("Tài liệu (unique)", stats["unique_titles"])
        for doc_type, count in stats["by_doc_type"].items():
            st.text(f"  {doc_type}: {count} chunks")
    except Exception as exc:
        st.warning(f"Chưa index dữ liệu: {exc}")

st.title("RAG Chatbot — Tra cứu thông tin tuyển sinh PTIT")

with st.expander("🧩 Kiến trúc pipeline (Task 5-10)"):
    st.markdown(
        """
        1. **Dense search** (Task 5) — embedding cosine similarity trên ChromaDB.
        2. **Lexical search** (Task 6) — BM25L, khớp từ khóa/mã ngành chính xác.
        3. **RRF fusion** (Task 7) — gộp 2 ranked list theo `1 / (k + rank)`.
        4. **PageIndex fallback** (Task 8) — khi dense score thấp, hỏi trực tiếp cây cấu trúc PDF.
        5. **Retrieval pipeline** (Task 9) — `retrieve()` tự quyết định hybrid hay fallback.
        6. **Generation + citation** (Task 10) — LLM trả lời kèm `[Document N]`, có validate citation trước khi trả về.
        """
    )

tab_chat, tab_debug, tab_tests = st.tabs(["💬 Chat", "🔍 Retrieval Debug", "🧪 Tests & Health"])

# ---------------------------------------------------------------------------
# TAB 1 — CHAT
# ---------------------------------------------------------------------------
with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                label = f"📚 {len(message['sources'])} nguồn · retrieval = {message.get('retrieval_source')}"
                with st.expander(label):
                    for rank, source in enumerate(message["sources"], 1):
                        render_source_card(rank, source)

    query = st.chat_input("Nhập câu hỏi về tuyển sinh PTIT...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Đang truy xuất và sinh câu trả lời..."):
                from src.task10_generation import generate_with_citation

                start = time.monotonic()
                result = generate_with_citation(query, top_k=top_k)
                elapsed = time.monotonic() - start

            st.markdown(result["answer"])
            st.caption(f"⏱ {elapsed:.1f}s · retrieval_source = {result['retrieval_source']}")

            if result["sources"]:
                with st.expander(f"📚 {len(result['sources'])} nguồn trích dẫn"):
                    for rank, source in enumerate(result["sources"], 1):
                        render_source_card(rank, source)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
                "retrieval_source": result["retrieval_source"],
            }
        )

# ---------------------------------------------------------------------------
# TAB 2 — RETRIEVAL DEBUG (so sánh dense / BM25 / hybrid)
# ---------------------------------------------------------------------------
with tab_debug:
    st.caption(
        "So sánh 3 chiến lược retrieval trên cùng 1 câu hỏi — để thấy vì sao hybrid (RRF) "
        "thường tốt hơn từng cái riêng lẻ."
    )
    debug_query = st.text_input("Câu hỏi để debug", key="debug_query")
    debug_k = st.slider("top_k cho debug", 3, 10, 5, key="debug_k")

    if st.button("So sánh retrieval", type="primary") and debug_query:
        from src.task5_semantic_search import semantic_search
        from src.task6_lexical_search import lexical_search
        from src.task7_reranking import rerank_rrf

        with st.spinner("Đang chạy dense + BM25 + RRF..."):
            dense = semantic_search(debug_query, top_k=debug_k)
            sparse = lexical_search(debug_query, top_k=debug_k)
            hybrid = rerank_rrf([dense, sparse], top_k=debug_k)

        col_dense, col_bm25, col_hybrid = st.columns(3)
        for col, label, results in (
            (col_dense, "🔵 Dense (cosine)", dense),
            (col_bm25, "🟠 BM25 (lexical)", sparse),
            (col_hybrid, "🟢 Hybrid (RRF)", hybrid),
        ):
            with col:
                st.subheader(label)
                if not results:
                    st.info("Không có kết quả.")
                for rank, item in enumerate(results, 1):
                    render_source_card(rank, item)

# ---------------------------------------------------------------------------
# TAB 3 — TESTS & HEALTH
# ---------------------------------------------------------------------------
with tab_tests:
    st.caption("Chạy pytest ngay trong UI để kiểm tra contract mà không cần rời sang terminal.")

    col_a, col_b = st.columns(2)
    with col_a:
        run_contracts = st.button("▶️ Chạy tests/test_contracts.py", type="primary")
    with col_b:
        run_all = st.button("▶️ Chạy toàn bộ tests/")

    target = None
    if run_contracts:
        target = "tests/test_contracts.py"
    elif run_all:
        target = "tests"

    if target:
        with st.spinner(f"pytest {target} -q ..."):
            # sys.executable (không phải "python" trần) để chắc chắn dùng
            # đúng interpreter của .venv đang chạy Streamlit — tránh trường
            # hợp máy có nhiều Python và "python"/"pytest" trên PATH trỏ đến
            # bản khác. Không dùng shell=True -> không có rủi ro command
            # injection, vì `target` chỉ nhận 1 trong 2 giá trị cố định ở trên,
            # không phải input tự do từ người dùng.
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", target, "-q"],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
                timeout=180,
            )
        output = proc.stdout + proc.stderr
        last_line = next(
            (line for line in output.strip().splitlines()[::-1] if line.strip()), ""
        )

        if proc.returncode == 0:
            st.success(f"✅ PASS — {last_line}")
        else:
            st.error(f"❌ FAIL — {last_line}")

        with st.expander("Xem output đầy đủ", expanded=proc.returncode != 0):
            st.code(output, language="text")
