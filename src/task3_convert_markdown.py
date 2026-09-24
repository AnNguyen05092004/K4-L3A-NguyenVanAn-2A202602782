"""
Task 3 — Chuẩn hóa dữ liệu sang Markdown.

Hướng dẫn:
    1. Dùng MarkItDown để convert PDF/DOCX.
    2. Đọc JSON và giữ metadata ở đầu file Markdown.
    3. Giữ cấu trúc thư mục legal/ và news/.
    4. Không tạo file rỗng hoặc file trùng khi chạy lại.

Cài đặt:
    Dependency MarkItDown đã được khai báo trong pyproject.toml.

-> Hoặc dùng công cụ nào bạn quen khác Markitdown
"""

import json
from pathlib import Path

from markitdown import MarkItDown
from pdf2image import convert_from_path, pdfinfo_from_path
import pytesseract

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"

# Dưới ngưỡng này coi là PDF scan (không có text layer thật, chỉ là ảnh chụp
# chữ) -> chuyển sang OCR. 300 ký tự/trang là ngưỡng thực nghiệm: PDF có text
# layer thật thường ra vài trăm đến vài nghìn ký tự/trang qua MarkItDown; PDF
# scan (MarkItDown chỉ đọc được vài dòng rác hoặc rỗng) sẽ rơi hẳn dưới mức này.
MIN_CHARS_PER_PAGE = 300
OCR_LANG = "vie"
OCR_DPI = 200


def _extract_via_ocr(path: Path) -> str:
    """OCR từng trang PDF scan bằng Tesseract (tiếng Việt).

    Chỉ gọi khi MarkItDown không đọc được text layer (xem convert_legal_docs)
    — OCR chậm hơn nhiều và có thể sai chính tả/số, nên chỉ dùng làm fallback,
    không dùng làm phương pháp mặc định cho mọi PDF.
    """
    pages = convert_from_path(str(path), dpi=OCR_DPI)
    texts = [pytesseract.image_to_string(page, lang=OCR_LANG) for page in pages]
    return "\n\n".join(text.strip() for text in texts if text.strip())


def convert_legal_docs() -> None:
    """Convert PDF/DOCX vào standardized/legal; OCR fallback cho file scan."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)
    converter = MarkItDown()

    for path in sorted(legal_dir.iterdir()):
        if path.suffix.lower() not in {".pdf", ".doc", ".docx"}:
            continue

        # Luôn thử extract "native" (text layer có sẵn trong PDF) trước —
        # nhanh và chính xác hơn OCR nhiều, chỉ OCR khi kết quả này quá ít chữ.
        text = converter.convert(str(path)).text_content.strip()
        method = "native"

        if path.suffix.lower() == ".pdf":
            page_count = pdfinfo_from_path(str(path)).get("Pages", 1)
            chars_per_page = len(text) / max(page_count, 1)
            if chars_per_page < MIN_CHARS_PER_PAGE:
                text = _extract_via_ocr(path)
                method = "ocr"

        if not text.strip():
            print(f"Skipped (no extractable text): {path.name}")
            continue

        (output_dir / f"{path.stem}.md").write_text(text, encoding="utf-8")
        print(f"Converted ({method}): {path.name}")


def convert_news_articles() -> None:
    """Convert JSON (crawl ở Task 2) vào standardized/news, giữ metadata ở đầu file.

    Metadata (title/source/crawled date) được ghi thành 1 đoạn header Markdown
    NGAY TRONG file .md (không lưu file .json/.meta riêng) — để Task 4
    (_extract_title/_extract_source_url) đọc lại được từ chính nội dung file,
    không cần giữ đồng bộ 2 file riêng biệt.
    """
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in sorted(news_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        header = (
            f"# {data['title']}\n\n"
            f"**Source:** {data['url']}\n\n"
            f"**Crawled:** {data['date_crawled']}\n\n---\n\n"
        )
        (output_dir / f"{path.stem}.md").write_text(
            header + data["content_markdown"], encoding="utf-8"
        )
        print(f"Converted: {path.name}")


def convert_all() -> None:
    """Convert toàn bộ dữ liệu landing."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    convert_legal_docs()
    convert_news_articles()
    print(f"Saved Markdown to: {OUTPUT_DIR}")


if __name__ == "__main__":
    convert_all()
