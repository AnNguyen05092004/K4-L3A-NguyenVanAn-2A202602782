"""
Task 2 — Crawl bài viết/thông báo.

Hướng dẫn:
    1. Điền tối thiểu 5 URL công khai vào ARTICLE_URLS.
    2. Crawl từng URL bằng Crawl4AI.
    3. Lưu mỗi bài thành một JSON trong data/landing/news/.
    4. Giữ đủ url, title, date_crawled và content_markdown.

Cài browser trước khi chạy:
    python -m playwright install chromium
    
-> Dùng Firecrawl or bất cứ công cụ nào bạn quen    
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

# PruningContentFilter chấm điểm "mật độ nội dung" từng block HTML và loại bỏ
# block dưới ngưỡng threshold — dùng để tự động cắt menu/logo/breadcrumb/footer
# (thường ít chữ, nhiều thẻ HTML lồng nhau) trước khi crawl4ai chuyển sang
# Markdown, chỉ giữ lại phần nội dung chính của bài viết. threshold=0.48 là
# giá trị chỉnh bằng tay sau khi soi thử output — chỉnh cao hơn nếu Markdown ra
# vẫn còn rác, chỉnh thấp hơn nếu bị cắt mất nội dung thật.
RUN_CONFIG = CrawlerRunConfig(
    markdown_generator=DefaultMarkdownGenerator(
        content_filter=PruningContentFilter(threshold=0.48)
    )
)

ARTICLE_URLS = [
    "https://tuyensinh.ptit.edu.vn/de-an-tuyen-sinh/thong-tin-tuyen-sinh-dai-hoc-chinh-quy-nam-2026/",
    "http://tuyensinh.ptit.edu.vn/thong-bao-ve-viec-nhap-hoc-dai-hoc-chinh-quy-nam-2026-co-so-dao-tao-phia-bac-bvh/",
    "https://tuyensinh.ptit.edu.vn/thong-bao-diem-chuan-trung-tuyen-vao-dai-hoc-he-chinh-quy-nam-2026/",
    "https://tuyensinh.ptit.edu.vn/thong-bao-ket-qua-xet-tuyen-thang-va-utxt-vao-dai-hoc-chinh-quy-nam-2026/",
    "https://tuyensinh.ptit.edu.vn/thong-bao-tuyen-sinh-chuong-trinh-dao-tao-thac-si-tai-nang-nam-2026/",
]


async def crawl_article(url: str) -> dict:
    """Crawl một URL và trả về bài viết dạng Markdown.

    fit_markdown (thay vì .markdown thô) là bản đã được PruningContentFilter
    lọc bớt nội dung ít liên quan — fallback về str(result.markdown) cho
    trường hợp filter không tạo ra fit_markdown (trang quá đơn giản/lỗi filter).
    """
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url, config=RUN_CONFIG)
        if not result.success:
            raise RuntimeError(f"Crawl failed for {url}: {result.error_message}")
        title = (result.metadata or {}).get("title", "Unknown").strip()
        content = result.markdown.fit_markdown or str(result.markdown)
        return {
            "url": url,
            "title": title or "Unknown",
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": content,
        }


async def crawl_all() -> None:
    """Crawl và lưu từng bài thành một file JSON.

    Mỗi URL crawl ĐỘC LẬP trong try/except riêng — 1 URL lỗi (mạng, trang đổi
    cấu trúc, bị chặn...) chỉ in cảnh báo và tiếp tục crawl URL còn lại, không
    làm hỏng cả batch.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for index, url in enumerate(ARTICLE_URLS, 1):
        try:
            article = await crawl_article(url)
            output = DATA_DIR / f"article_{index:02d}.json"
            output.write_text(
                json.dumps(article, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Saved: {output}")
        except Exception as error:
            print(f"Failed: {url} — {error}")


if __name__ == "__main__":
    asyncio.run(crawl_all())
