"""
Task 10 — Generation có citation.

Luồng xử lý:
    1. Retrieve top-k chunks (Task 9: dense + BM25 + RRF, có fallback PageIndex).
    2. Reorder chunks để giảm hiệu ứng "lost-in-the-middle" khi nhét vào prompt.
    3. Format context kèm title/source, đánh số [Document N] cho LLM trích dẫn.
    4. Gọi LLM provider được cấu hình trong .env.
    5. Đối chiếu lại citation LLM viết ra có thực sự khớp với chunk được trích không.
    6. Trả answer, sources và retrieval_source đúng theo GenerationResult contract.

Nếu context không đủ, LLM lỗi, hoặc citation không khớp nguồn: trả safe refusal
cố định — không được để LLM tự bịa số liệu khi không chắc.
"""

import os
import re

from dotenv import load_dotenv

from .task9_retrieval_pipeline import retrieve

load_dotenv()

# top_k mặc định khi gọi generate_with_citation() mà không truyền tham số riêng.
# 8 (thay vì 5) vì nhiều câu hỏi cần nhìn thấy >1 hàng dữ liệu liên quan (ví dụ
# 1 ngành có nhiều mức điểm chuẩn theo chương trình/cơ sở) mới đủ ngữ cảnh để
# trả lời đầy đủ mà không bỏ sót biến thể nào.
TOP_K = 8
TOP_P = 0.9
TEMPERATURE = 0.3

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "")

# Câu trả lời an toàn cố định — dùng ở MỌI nhánh không chắc chắn (context rỗng,
# provider lỗi, citation không khớp nguồn). Để cố định 1 câu duy nhất giúp dễ so
# sánh output trong lúc evaluate (so sánh string) và tránh LLM tự diễn giải lại
# câu từ chối theo nhiều cách khác nhau.
REFUSAL_ANSWER = "Tôi không thể xác minh thông tin này từ nguồn hiện có."

SYSTEM_PROMPT = """Trả lời chỉ dựa vào context được cung cấp bên dưới, không dùng kiến thức ngoài.
Mỗi khẳng định trong câu trả lời phải kèm trích dẫn dạng [Document N] tương ứng với
số thứ tự tài liệu trong context — chỉ trích dẫn Document nào THỰC SỰ chứa đúng số liệu/thông tin đó,
không trích dẫn Document chỉ vì nó cùng chủ đề.
Nếu một ngành có nhiều mức điểm chuẩn khác nhau (ví dụ chương trình đại trà và chất lượng cao),
hãy liệt kê rõ từng mức kèm chương trình tương ứng, không chỉ chọn 1 giá trị.
Nếu context không đủ thông tin để trả lời, hãy trả lời đúng câu:
"Tôi không thể xác minh thông tin này từ nguồn hiện có." — không suy đoán, không bịa."""

# Bắt "[Document 3]", "[Document 12]"... để tách số thứ tự LLM trích dẫn.
CITATION_PATTERN = re.compile(r"\[Document (\d+)\]")
# Bắt số có phần thập phân (23.00) hoặc số nguyên — dùng để so khớp số liệu
# trong câu trả lời với số liệu thật trong chunk khi kiểm tra grounding.
NUMBER_PATTERN = re.compile(r"\d+[.,]\d+|\d+")


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Đưa 2 chunk quan trọng nhất (rank 1 và 2) về đầu và cuối context.

    LLM có xu hướng đọc kỹ phần đầu/cuối của context và dễ bỏ sót phần giữa
    ("lost in the middle"). `chunks` đưa vào đây đã được sort theo score giảm
    dần (từ Task 9), nên rank 1 luôn rơi vào vị trí đầu (front[0]) và rank 2
    luôn rơi vào vị trí cuối (back[-1] sau khi đảo) — 2 vị trí LLM chú ý nhất.
    """
    if len(chunks) <= 2:
        return list(chunks)
    front = chunks[::2]  # rank 1, 3, 5, ... giữ nguyên thứ tự, đứng đầu
    back = chunks[1::2]  # rank 2, 4, 6, ...
    return front + back[::-1]  # đảo `back` để rank 2 rơi vào vị trí cuối cùng


def format_context(chunks: list[dict]) -> str:
    """Ghép các chunk thành 1 khối text đưa vào prompt, mỗi chunk có nhãn [Document N].

    Số N ở đây là vị trí trong THAM SỐ `chunks` truyền vào (tức thứ tự đã
    reorder ở trên, không phải thứ tự score gốc) — đây là thứ tự LLM nhìn thấy
    và sẽ dùng để viết "[Document N]" trong câu trả lời.
    """
    parts = []
    for index, chunk in enumerate(chunks, 1):
        metadata = chunk["metadata"]
        parts.append(
            f"[Document {index} | Title: {metadata['title']} | "
            f"Source: {metadata['source']}]\n{chunk['content']}"
        )
    return "\n\n---\n\n".join(parts)


def call_llm(system_prompt: str, user_message: str) -> str:
    """Gọi đúng 1 trong 3 provider theo LLM_PROVIDER trong .env.

    Import SDK provider ngay trong từng nhánh (không import ở đầu file) để
    máy không cần cài cả 3 SDK cùng lúc — chỉ cần cài đúng provider đang dùng.
    """
    if LLM_PROVIDER == "openai":
        from openai import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model=LLM_MODEL or "gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        return response.choices[0].message.content

    if LLM_PROVIDER == "gemini":
        from google import genai

        client = genai.Client()
        response = client.models.generate_content(
            model=LLM_MODEL or "gemini-2.0-flash",
            contents=f"{system_prompt}\n\n{user_message}",
        )
        return response.text

    if LLM_PROVIDER == "anthropic":
        import anthropic

        client = anthropic.Anthropic()
        response = client.messages.create(
            model=LLM_MODEL or "claude-sonnet-5",
            max_tokens=1024,
            temperature=TEMPERATURE,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text

    raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")


def _relabel_citations(
    answer: str, reordered: list[dict], sorted_chunks: list[dict]
) -> str:
    """Đổi số [Document N] trong answer từ thứ tự "đã reorder" sang thứ tự "đã sort".

    LLM viết "[Document N]" dựa theo `reordered` (thứ tự nó thấy trong prompt,
    xem format_context()). Nhưng contract yêu cầu `sources` trả về phải sort
    giảm dần theo score (validate_search_results) — mà `reordered` thì KHÔNG
    sort (cố ý xáo để chống lost-in-the-middle). Nếu trả "sources": reordered
    thì vi phạm contract; nếu trả "sources": sorted_chunks mà giữ nguyên số cũ
    trong answer thì [Document N] sẽ trỏ sai chunk.

    => Giải pháp: giữ answer đúng theo LLM viết, chỉ đổi lại SỐ trong
    "[Document N]" để nó trỏ đúng vị trí của cùng 1 chunk trong danh sách đã
    sort — answer vẫn đúng nội dung, sources vẫn đúng contract, và số trích
    dẫn vẫn khớp đúng nguồn.
    """
    position_in_sorted = {
        chunk["id"]: position + 1 for position, chunk in enumerate(sorted_chunks)
    }
    relabel = {
        reordered_position + 1: position_in_sorted[chunk["id"]]
        for reordered_position, chunk in enumerate(reordered)
    }

    def _replace(match: re.Match) -> str:
        old_label = int(match.group(1))
        new_label = relabel.get(old_label)
        # Số không nằm trong relabel nghĩa là LLM trích dẫn 1 Document không hề
        # tồn tại (hallucination) — giữ nguyên số cũ để _citations_are_grounded()
        # bên dưới chắc chắn phát hiện và từ chối trả lời.
        return f"[Document {new_label if new_label is not None else old_label}]"

    return CITATION_PATTERN.sub(_replace, answer)


def _citations_are_grounded(answer: str, chunks: list[dict]) -> bool:
    """Câu trả lời phải có ít nhất 1 citation, và số liệu ngay trước mỗi
    [Document N] phải thực sự xuất hiện trong content của đúng chunk N.

    Đây là lưới an toàn cuối: dù prompt đã dặn LLM chỉ trích dẫn Document nào
    THỰC SỰ chứa số liệu đó, LLM vẫn có thể trích dẫn sai (hallucination). Hàm
    này không hiểu ngữ nghĩa câu văn — chỉ so khớp số liệu ở mức ký tự, nên vẫn
    có thể bỏ sót số liệu sai gán nhầm sang chương trình/cơ sở khác trong CÙNG
    1 chunk (chunk chứa nhiều số nên chỉ cần 1 số đúng là qua được check).
    """
    if answer.strip() == REFUSAL_ANSWER:
        return True

    citations = list(CITATION_PATTERN.finditer(answer))
    if not citations:
        return False

    for match in citations:
        doc_index = int(match.group(1)) - 1
        if doc_index < 0 or doc_index >= len(chunks):
            return False

        # Lấy phần câu ngay trước tag citation (từ dấu "." hoặc xuống dòng gần
        # nhất) — đó là phần khẳng định mà citation này đang bảo vệ.
        preceding = answer[: match.start()]
        sentence_start = max(preceding.rfind("."), preceding.rfind("\n")) + 1
        claim = preceding[sentence_start:]

        source_content = chunks[doc_index]["content"]
        for number in NUMBER_PATTERN.findall(claim):
            if number not in source_content:
                return False
    return True


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Sinh câu trả lời có trích dẫn cho `query`. Trả về GenerationResult.

    3 nhánh trả safe refusal (context rỗng / LLM lỗi / citation không khớp)
    đều trả cùng REFUSAL_ANSWER với sources=[] — để mọi nơi gọi hàm này chỉ
    cần so sánh answer == REFUSAL_ANSWER là biết chắc không có nguồn đáng tin.
    """
    chunks = retrieve(query, top_k=top_k)  # đã sort giảm dần theo score (Task 9)
    if not chunks:
        return {"answer": REFUSAL_ANSWER, "sources": [], "retrieval_source": "none"}

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    try:
        answer = call_llm(SYSTEM_PROMPT, user_message)
    except Exception:
        return {"answer": REFUSAL_ANSWER, "sources": [], "retrieval_source": "none"}

    # Đổi số citation từ thứ tự "đã reorder" (LLM thấy) sang thứ tự "đã sort"
    # (chunks) NGAY TRƯỚC khi validate — để phần kiểm tra và phần trả về dùng
    # cùng 1 hệ đánh số duy nhất.
    answer = _relabel_citations(answer, reordered, chunks)

    if not _citations_are_grounded(answer, chunks):
        return {"answer": REFUSAL_ANSWER, "sources": [], "retrieval_source": "none"}

    return {
        "answer": answer,
        "sources": chunks,  # sort giảm dần theo score -> đúng contract
        "retrieval_source": chunks[0]["retrieval_method"],
    }


if __name__ == "__main__":
    print(generate_with_citation("test query"))
