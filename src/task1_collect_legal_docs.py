"""
Task 1 — Thu thập tài liệu chính sách/quy định.

Hướng dẫn:
    1. Chọn chủ đề của nhóm.
    2. Tìm tối thiểu 3 tài liệu PDF/DOCX từ nguồn công khai.
    3. Lưu file gốc vào data/landing/legal/.
    4. Đặt tên không dấu và thể hiện đúng nội dung.

Ví dụ tài liệu: học phí, học bổng, ký túc xá, quy trình đăng ký.
Nếu website chặn crawler, hãy chọn nguồn công khai khác; không vượt WAF.
"""

from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"


def setup_directory() -> None:
    """Tạo thư mục lưu tài liệu gốc (chưa qua xử lý) nếu chưa tồn tại."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ready: {DATA_DIR}")


def download_documents() -> None:
    """Xác nhận đã có ít nhất 3 PDF/DOCX hợp lệ trong DATA_DIR.

    Hàm này KHÔNG tự tải file — nó chỉ kiểm tra điều kiện đủ (>=3 file) rồi
    báo lỗi rõ ràng nếu thiếu, để pipeline dừng sớm ở Task 1 thay vì chạy tiếp
    Task 3/4 với dữ liệu rỗng và lỗi khó hiểu ở bước sau.

    Tài liệu tuyển sinh PTIT được tải THỦ CÔNG từ trang tuyensinh.ptit.edu.vn
    và đặt vào DATA_DIR — trang này không có endpoint tải trực tiếp ổn định để
    dùng requests, và không nên cố vượt WAF/anti-bot của trang để tự động hoá.
    """
    files = sorted(
        path
        for path in DATA_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in {".pdf", ".doc", ".docx"}
    )
    if len(files) < 3:
        raise RuntimeError(
            f"Cần tối thiểu 3 PDF/DOCX trong {DATA_DIR}, hiện có {len(files)}. "
            "Tải thủ công từ nguồn công khai rồi chạy lại."
        )
    for path in files:
        size_kb = path.stat().st_size / 1024
        print(f"Found: {path.name} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    setup_directory()
    download_documents()
