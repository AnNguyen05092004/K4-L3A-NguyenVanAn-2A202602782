"""
Task 4 — Chunking, embedding và indexing.

Hướng dẫn:
    1. Đọc toàn bộ Markdown trong data/standardized/.
    2. Chia văn bản bằng strategy đã chọn.
    3. Embed chunks bằng một provider duy nhất.
    4. Upsert vào ChromaDB với cosine distance.

Mỗi document/chunk phải theo docs/MODULE_CONTRACTS.md. ID cần ổn định để
chạy lại pipeline không tạo dữ liệu trùng. Task 5 phải dùng chung embed_texts().
"""

import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# Giải thích lựa chọn tham số trong báo cáo nhóm.
# CHUNK_SIZE áp dụng cho văn bản PROSE (RecursiveCharacterTextSplitter). Chunk
# BẢNG (_chunk_table_lines) dùng CÙNG con số này làm ngưỡng cắt, nhưng 1 dòng
# bảng điểm (nhiều cột điểm theo phương thức xét tuyển) cộng với phần "nhãn
# ngữ cảnh" bắt buộc (title + cơ sở + section + header, xem _chunk_table_lines)
# có thể tự nó đã vượt CHUNK_SIZE — đây là trade-off CHỦ ĐỘNG: ưu tiên giữ đủ
# ngữ cảnh để không tra nhầm dữ liệu, hơn là ép cứng theo 1 giới hạn ký tự.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

COLLECTION_NAME = "rag_documents"


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed 1 danh sách text thành vector, dùng đúng 1 provider theo .env.

    Đây là hàm DUY NHẤT tạo embedding trong toàn bộ project — Task 5
    (semantic_search) phải import và gọi lại chính hàm này cho query, để query
    và document luôn nằm trong cùng 1 không gian vector (đổi provider chỉ cần
    sửa .env, không phải sửa nhiều nơi).
    """
    provider = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers")

    if provider == "sentence_transformers":
        # Import bên trong nhánh (không import ở đầu file) để máy chỉ cần cài
        # đúng 1 SDK provider đang dùng, không phải cài cả sentence-transformers
        # + openai + google-genai cùng lúc.
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(EMBEDDING_MODEL)
        # normalize_embeddings=True: đưa vector về độ dài 1 (unit vector) — cần
        # thiết vì ChromaDB collection dùng cosine distance (xem get_collection).
        return model.encode(texts, normalize_embeddings=True).tolist()

    if provider == "openai":
        from openai import OpenAI

        client = OpenAI()
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        return [item.embedding for item in response.data]

    if provider == "gemini":
        from google import genai

        client = genai.Client()
        response = client.models.embed_content(model=EMBEDDING_MODEL, contents=texts)
        return [embedding.values for embedding in response.embeddings]

    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider}")


def get_collection():
    """Mở (hoặc tạo mới) Chroma collection dùng cosine distance.

    get_or_create_collection nên KHÔNG bao giờ tạo trùng collection khi chạy
    lại script nhiều lần — cùng COLLECTION_NAME sẽ trả về đúng collection cũ.
    """
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# ChromaDB ÂM THẦM BỎ QUA (không lưu) bất kỳ metadata key nào có giá trị
# None khi upsert — đã verify trực tiếp bằng 1 collection test riêng. Ảnh
# hưởng thật: mọi chunk legal (url=None) hoặc chunk không phải bảng điểm
# (major=None, campus=None) sẽ THIẾU HẲN các key đó khi đọc lại từ Chroma,
# dù lúc upsert đã truyền đủ — làm validate_document() báo lỗi
# "metadata.url must be a string or None" (thiếu key, không phải sai kiểu).
# Mọi nơi ĐỌC metadata từ Chroma (semantic_search, lexical_search corpus,
# exact_major_match index...) phải gọi hàm này ngay sau khi lấy kết quả về,
# KHÔNG được giả định key nào cũng có mặt.
_METADATA_NONE_DEFAULTS = {"url": None, "major": None, "campus": None}


def restore_metadata_defaults(metadata: dict) -> dict:
    """Khôi phục các metadata key có thể bị Chroma bỏ qua vì giá trị gốc là None."""
    return {**_METADATA_NONE_DEFAULTS, **metadata}


def _extract_title(content: str, fallback: str) -> str:
    """Lấy dòng heading Markdown cấp 1 đầu tiên ("# ...") làm title."""
    for line in content.splitlines():
        if line.startswith("# "):
            return line.removeprefix("# ").strip()
    return fallback


def _extract_source_url(content: str) -> str | None:
    """Lấy URL gốc từ dòng "**Source:** ..." do Task 3 chèn vào đầu file news."""
    for line in content.splitlines():
        if line.startswith("**Source:**"):
            return line.removeprefix("**Source:**").strip()
    return None


def load_documents() -> list[dict]:
    """Đọc toàn bộ Markdown trong data/standardized/ và trả về Document theo contract."""
    documents = []
    for path in STANDARDIZED_DIR.rglob("*.md"):
        content = path.read_text(encoding="utf-8")
        # doc_type suy ra từ tên thư mục cha (standardized/legal/... hoặc
        # standardized/news/...) — không cần lưu riêng trong metadata gốc.
        doc_type = "legal" if "legal" in path.parts else "news"
        documents.append(
            {
                # Dùng path tương đối (VD "news/article_03.md") làm id — ổn
                # định qua nhiều lần chạy, và là tiền tố cho id của chunk
                # (xem chunk_documents) để dễ truy ngược chunk về đúng file.
                "id": path.relative_to(STANDARDIZED_DIR).as_posix(),
                "content": content,
                "metadata": {
                    "source": path.name,
                    "title": _extract_title(content, fallback=path.stem),
                    "doc_type": doc_type,
                    # url chỉ có ý nghĩa với news (crawl từ web); legal đọc từ
                    # PDF/DOCX nội bộ, không có URL nguồn.
                    "url": _extract_source_url(content) if doc_type == "news" else None,
                },
            }
        )
    return documents


# Nhận diện 1 dòng bảng Markdown: bắt đầu và kết thúc bằng "|".
TABLE_LINE = re.compile(r"^\s*\|.*\|\s*$")


def _is_separator_line(line: str) -> bool:
    """Nhận diện dòng phân cách bảng Markdown kiểu "| --- | --- |".

    Sau khi bỏ hết |, :, -, khoảng trắng mà dòng còn lại rỗng (chỉ có 4 ký tự
    đó) VÀ có ít nhất 1 dấu "-" thì chắc chắn là dòng phân cách, không phải
    data row thật (data row luôn còn lại chữ/số sau khi bỏ các ký tự đó).
    """
    stripped = line.replace("|", "").replace(":", "").replace("-", "").replace(" ", "")
    return stripped == "" and "-" in line


def _section_label(line: str) -> str | None:
    """Nhận diện dòng label riêng trong bảng, ví dụ "| **CHƯƠNG TRÌNH CHẤT LƯỢNG CAO** |".

    Các bảng điểm chuẩn PTIT chèn 1 dòng chỉ có 1 cell để đánh dấu nhóm con dữ
    liệu (VD: NGÀNH ĐẠI TRÀ vs CHƯƠNG TRÌNH CHẤT LƯỢNG CAO) — không có dòng
    phân cách riêng nên nếu không xử lý, dòng này bị hiểu lầm là 1 data row
    bình thường và mất luôn tác dụng phân nhóm khi bị cắt sang chunk khác.

    Heuristic: 1 dòng bảng bình thường (ngành, mã ngành, điểm, ...) luôn có
    NHIỀU cell không rỗng; dòng label kiểu này chỉ có ĐÚNG 1 cell không rỗng.
    Không dựa vào in đậm (**) để nhận diện vì dữ liệu nguồn không nhất quán —
    có label thiếu markdown bold (đã gặp thực tế, xem "CÁC CHƯƠNG TRÌNH LIÊN
    KẾT QUỐC TẾ" trong article_03.md).
    """
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    non_empty = [cell for cell in cells if cell]
    if len(non_empty) == 1:
        return non_empty[0].strip("*").strip()
    return None


def _major_column_index(header: str) -> int | None:
    """Tìm vị trí cột chứa tên ngành trong header bảng, dựa vào chữ "ngành"
    xuất hiện trong tên cột — KHÔNG hard-code cột 0 hay cột 1, vì đã xác nhận
    thực tế các bảng khác nhau đặt cột "Tên ngành" ở vị trí khác nhau (bảng
    điểm chuẩn: cột 0 "Tên ngành, chương trình"; bảng chỉ tiêu tuyển sinh:
    cột 1 "Tên ngành/chương trình xét tuyển").
    """
    cells = [cell.strip().strip("*").strip() for cell in header.strip().strip("|").split("|")]
    for index, cell in enumerate(cells):
        if "ngành" in cell.lower():
            return index
    return None


def _row_major(row: str, major_column: int | None) -> str | None:
    """Lấy giá trị cột tên ngành của 1 data row, nếu xác định được cột đó.

    Giá trị này lưu vào metadata["major"] của chunk (xem chunk_documents) để
    Task 9 lọc CHÍNH XÁC theo tên ngành thay vì chỉ trông chờ dense/BM25 —
    2 kỹ thuật đó xếp hạng rất thấp 1 dòng ngành cụ thể giữa hàng chục dòng có
    cấu trúc số liệu giống hệt nhau (đã verify thực tế, xem
    docs/RAG_PIPELINE_NOTES.md mục 7).
    """
    if major_column is None:
        return None
    cells = [cell.strip().strip("*").strip() for cell in row.strip().strip("|").split("|")]
    if major_column >= len(cells):
        return None
    return cells[major_column] or None


def _detect_campus(context: str | None) -> str | None:
    """Suy ra cơ sở đào tạo (bắc/nam) từ dòng heading ngay trước bảng.

    VD context = "1. **CƠ SỞ ĐÀO TẠO PHÍA BẮC (mã BVH)**" -> "bắc". Cần thiết
    vì cùng 1 ngành có thể có điểm khác nhau giữa 2 cơ sở (đã gặp thật: cùng
    tên ngành + mã ngành, chỉ khác điểm) — nếu câu hỏi có nói rõ cơ sở, Task 9
    dùng field này lọc bớt, tránh trộn lẫn dữ liệu 2 cơ sở khi trả lời.
    """
    if not context:
        return None
    normalized = context.lower()
    if "phía bắc" in normalized or "mã bvh" in normalized:
        return "bắc"
    if "phía nam" in normalized or "mã bvs" in normalized:
        return "nam"
    return None


def _split_prose_and_tables(content: str) -> list[tuple[str, list[str]]]:
    """Tách nội dung thành từng đoạn liên tiếp: ("prose", dòng) hoặc ("table", dòng).

    Cần tách riêng vì 2 loại nội dung phải chunk theo 2 cách khác nhau: prose
    dùng RecursiveCharacterTextSplitter (cắt theo câu/đoạn văn), bảng dùng
    _chunk_table_lines (cắt theo TỪNG DÒNG, không cắt giữa dòng — cắt giữa 1
    dòng bảng sẽ phá vỡ cấu trúc cột và mất hẳn nghĩa của số liệu).
    """
    segments: list[tuple[str, list[str]]] = []
    current_type = None
    current_lines: list[str] = []

    for line in content.splitlines():
        line_type = "table" if TABLE_LINE.match(line) else "prose"
        if current_type is not None and line_type != current_type:
            segments.append((current_type, current_lines))
            current_lines = []
        current_type = line_type
        current_lines.append(line)

    if current_lines:
        segments.append((current_type, current_lines))
    return segments


def _chunk_table_lines(
    title: str,
    lines: list[str],
    chunk_size: int,
    context: str | None = None,
    campus: str | None = None,
) -> list[tuple[str, str | None, str | None]]:
    """Chunk 1 bảng Markdown, lặp lại title + context + section label + header ở mỗi chunk.

    `context` là dòng heading gần nhất TRƯỚC bảng (VD "CƠ SỞ ĐÀO TẠO PHÍA
    BẮC") — cần thiết khi 1 tài liệu có nhiều bảng con giống nhau (theo cơ sở,
    theo năm...) chỉ khác nhau ở heading phía trên. Heading đó là prose, nằm
    NGOÀI bảng (xem _split_prose_and_tables), nên nếu không truyền vào đây thì
    chunk của bảng mất hoàn toàn ngữ cảnh đó — không thể phân biệt được hàng
    dữ liệu này thuộc cơ sở/năm nào (bug thật đã gặp: 2 bảng của 2 cơ sở Bắc/
    Nam có cùng tên ngành + mã ngành, chỉ khác điểm chuẩn, nếu mất context thì
    không thể biết dòng nào ứng với cơ sở nào).

    `campus` do CALLER tính sẵn (chunk_documents quét TOÀN BỘ đoạn prose gần
    nhất, không chỉ dòng cuối) — KHÔNG tự suy ra từ `context` ở đây, vì có
    tài liệu chèn 1 dòng khác (VD địa chỉ cơ sở) giữa heading "CƠ SỞ ĐÀO TẠO
    PHÍA BẮC" và bảng, khiến dòng cuối cùng trước bảng (context) không phải
    là dòng chứa tên cơ sở — bug thật đã gặp (data/standardized/news/
    article_01.md), phải quét rộng hơn 1 dòng mới bắt đúng.

    Mỗi chunk luôn được ghép: [title] + [context] (nếu có) + [section label
    hiện tại] (nếu có) + header + separator + các data row — để dù retrieval
    chỉ lấy đúng 1 chunk lẻ, LLM đọc vào vẫn hiểu đủ "đây là bảng gì, cơ sở
    nào, thuộc nhóm chương trình nào" mà không cần đọc thêm chunk khác.

    Trả về (text, major, campus) thay vì chỉ text — major/campus suy ra được
    cho từng chunk (xem _row_major) sẽ lưu vào metadata để Task 9 lọc CHÍNH
    XÁC theo tên ngành/cơ sở, bổ sung cho dense/BM25.
    """
    if len(lines) < 3:
        return [("\n".join(lines), None, None)]

    header = lines[0]
    has_separator = _is_separator_line(lines[1])
    data_rows = lines[2:] if has_separator else lines[1:]
    major_column = _major_column_index(header)

    prefix_head = [f"[{title}]"]
    if context:
        prefix_head.append(f"[{context}]")

    def build_prefix(section: str | None) -> str:
        parts = list(prefix_head)
        if section:
            parts.append(f"[{section}]")
        parts.append(header)
        if has_separator:
            parts.append(lines[1])
        return "\n".join(parts)

    chunks: list[tuple[str, str | None, str | None]] = []
    current_rows: list[str] = []
    current_section: str | None = None
    # Section áp dụng cho CHUNK hiện tại (không đổi giữa chunk, dù section có
    # đổi ngay trước dòng cuối) — chốt lại ngay khi chunk mới bắt đầu tích
    # dòng đầu tiên, để mọi dòng trong 1 chunk luôn đồng nhất 1 label.
    chunk_start_section: str | None = None
    # Major của DÒNG DATA THẬT gần nhất đã thêm vào chunk hiện tại (dòng label
    # không có major riêng) — trong thực tế mỗi chunk hầu như luôn đúng 1 dòng
    # data (do prefix + 1 dòng bảng điểm đã gần lấp đầy chunk_size), nên giá
    # trị này đại diện đúng cho chunk trong tuyệt đại đa số trường hợp.
    chunk_major: str | None = None

    for row in data_rows:
        label = _section_label(row)
        if label:
            current_section = label
            row_major = None
        else:
            row_major = _row_major(row, major_column)
        if chunk_start_section is None:
            chunk_start_section = current_section

        trial_rows = current_rows + [row]
        prefix = build_prefix(chunk_start_section)
        # Cắt chunk mới NGAY KHI thêm dòng tiếp theo sẽ vượt chunk_size — dòng
        # bảng là đơn vị không thể chia nhỏ hơn (khác prose, không thể cắt
        # giữa 1 dòng số liệu mà vẫn giữ được nghĩa).
        if current_rows and len(prefix) + len("\n".join(trial_rows)) > chunk_size:
            chunks.append((prefix + "\n" + "\n".join(current_rows), chunk_major, campus))
            current_rows = [row]
            chunk_start_section = current_section
            chunk_major = row_major
        else:
            current_rows = trial_rows
            if row_major:
                chunk_major = row_major

    if current_rows:
        chunks.append(
            (build_prefix(chunk_start_section) + "\n" + "\n".join(current_rows), chunk_major, campus)
        )
    return chunks


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chia Document thành chunks có id và chunk_index; xử lý riêng bảng Markdown."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Thứ tự ưu tiên cắt: giữa 2 đoạn văn ("\n\n") trước, rồi giữa dòng
        # ("\n"), rồi giữa câu (". "), rồi giữa từ (" "), cuối cùng mới cắt
        # bừa theo ký tự ("") — giữ chunk càng "toàn vẹn về ý" càng tốt.
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for document in documents:
        segments = _split_prose_and_tables(document["content"])
        # (text, major, campus) cho từng chunk — major/campus là None với
        # chunk prose (chỉ bảng mới xác định được, xem _chunk_table_lines).
        entries: list[tuple[str, str | None, str | None]] = []
        # Heading gần nhất TRƯỚC 1 đoạn bảng — dùng làm `context` truyền cho
        # _chunk_table_lines() khi gặp đoạn "table" tiếp theo. Cập nhật mỗi
        # lần đi qua 1 đoạn "prose", nên luôn là heading GẦN NHẤT, không phải
        # heading đầu tài liệu.
        last_heading: str | None = None
        # Cơ sở (bắc/nam) được phát hiện gần nhất — quét TOÀN BỘ đoạn prose
        # (không chỉ dòng cuối như last_heading), vì có tài liệu chèn dòng
        # khác (địa chỉ...) giữa heading "CƠ SỞ ĐÀO TẠO PHÍA BẮC" và bảng, nên
        # nếu chỉ nhìn dòng cuối cùng sẽ bỏ lỡ heading thật. Giữ giá trị cũ
        # (không reset về None) khi đoạn prose hiện tại không nhắc gì đến cơ
        # sở, để nó áp dụng cho các bảng con tiếp theo cho tới khi đổi cơ sở.
        campus_hint: str | None = None
        for segment_type, lines in segments:
            if segment_type == "table":
                entries.extend(
                    _chunk_table_lines(
                        document["metadata"]["title"],
                        lines,
                        CHUNK_SIZE,
                        context=last_heading,
                        campus=campus_hint,
                    )
                )
            else:
                text = "\n".join(lines).strip()
                if text:
                    entries.extend((piece, None, None) for piece in splitter.split_text(text))
                non_empty = [line.strip() for line in lines if line.strip()]
                if non_empty:
                    last_heading = non_empty[-1]
                detected_campus = _detect_campus(text)
                if detected_campus:
                    campus_hint = detected_campus

        for index, (text, major, campus) in enumerate(entries):
            chunks.append(
                {
                    # "<document_id>::chunk-<index>" — ổn định giữa các lần
                    # chạy lại (miễn nội dung file không đổi), nên upsert vào
                    # ChromaDB (index_to_vectorstore) không tạo bản trùng.
                    "id": f"{document['id']}::chunk-{index}",
                    "content": text,
                    "metadata": {
                        **document["metadata"],
                        "chunk_index": index,
                        # Field mở rộng ngoài ChunkMetadata contract (dùng cho
                        # exact-match filter ở Task 9) — None với chunk prose.
                        "major": major,
                        "campus": campus,
                    },
                }
            )
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Thêm embedding vào từng chunk (embed 1 lần theo batch, không lặp từng chunk)."""
    vectors = embed_texts([chunk["content"] for chunk in chunks])
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector
    return chunks


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Upsert chunks vào ChromaDB.

    upsert (không phải add/insert): chạy lại pipeline với cùng id sẽ CẬP NHẬT
    thay vì tạo bản trùng — quan trọng vì content của chunk có thể đổi (VD sau
    khi sửa lại logic chunking) mà id vẫn giữ nguyên.
    """
    collection = get_collection()
    collection.upsert(
        ids=[chunk["id"] for chunk in chunks],
        documents=[chunk["content"] for chunk in chunks],
        embeddings=[chunk["embedding"] for chunk in chunks],
        metadatas=[chunk["metadata"] for chunk in chunks],
    )


def run_pipeline() -> None:
    """Chạy toàn bộ: load Markdown -> chunk -> embed -> index vào ChromaDB."""
    documents = load_documents()
    chunks = chunk_documents(documents)
    embedded_chunks = embed_chunks(chunks)
    index_to_vectorstore(embedded_chunks)
    print(f"Indexed {len(embedded_chunks)} chunks")


if __name__ == "__main__":
    run_pipeline()
