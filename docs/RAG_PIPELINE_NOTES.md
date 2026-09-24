# RAG Pipeline — Tổng hợp kiến thức quan trọng (Lab 8)

Tài liệu này tổng hợp lại những khái niệm, quyết định thiết kế, bug thật đã
gặp và cách chẩn đoán trong quá trình xây pipeline RAG cho đề tài tuyển sinh
PTIT. Mục tiêu: đọc lại 1 lần là nhớ được TẠI SAO code viết như vậy, không chỉ
là code làm gì.

## 1. Tổng quan pipeline

```
Task 1  Thu thập tài liệu chính sách (PDF/DOCX)      -> data/landing/legal/
Task 2  Crawl bài viết/thông báo                     -> data/landing/news/
Task 3  Chuẩn hoá về Markdown (MarkItDown/OCR)       -> data/standardized/
Task 4  Chunk + embed + index vào ChromaDB           -> chroma_db/
Task 5  Dense search (embedding + cosine similarity)
Task 6  Lexical search (BM25L)
Task 7  RRF fusion (gộp thứ hạng dense + BM25)        [+ bonus: cross-encoder]
Task 8  PageIndex vectorless fallback (vượt ngưỡng)
Task 9  Retrieval pipeline hoàn chỉnh (gộp Task 5-8)
Task 10 Generation + citation + safe refusal
        UI: Streamlit app.py (chat + retrieval debug + test runner)
```

Nguyên tắc xuyên suốt: **mọi module ghép với nhau qua `src/contracts.py`** —
`Document` -> `Chunk` -> `SearchResult` -> `GenerationResult`. Không tự đổi
shape dữ liệu giữa các Task; đổi ở 1 nơi thì phải đổi contract trước.

## 2. Dense search vs BM25 — vì sao cần cả 2

| | Dense (Task 5) | BM25 (Task 6) |
|---|---|---|
| Cách chấm điểm | Cosine similarity giữa vector embedding | Trùng khớp từ, có trọng số theo độ hiếm của từ (IDF) |
| Mạnh khi | Câu hỏi diễn đạt khác từ so với văn bản gốc (hiểu ngữ nghĩa) | Từ khóa chính xác, mã số, tên riêng |
| Yếu khi | Từ hiếm/mã số cụ thể (embedding "làm mờ" nghĩa) | Câu hỏi paraphrase, không dùng đúng từ trong tài liệu |

Hai nhánh trả điểm trên **2 thang đo khác nhau** (cosine 0..1 vs BM25 score
không chặn trên) — không được cộng thẳng 2 điểm này với nhau.

**Bug thật đã gặp — BM25Okapi suy biến về 0:** công thức IDF gốc
`ln(N-n+0.5) - ln(n+0.5)` ra ĐÚNG 0 khi 1 từ xuất hiện ở đúng N/2 tài liệu. Với
corpus nhỏ (test fixture 2 tài liệu), điều này khiến CẢ tài liệu đúng và
tài liệu sai đều bị điểm 0, không phân biệt được. **Fix:** đổi sang `BM25L`
(cùng thư viện `rank_bm25`), công thức IDF ổn định hơn cho corpus nhỏ.
→ [task6_lexical_search.py](../src/task6_lexical_search.py)

## 3. RRF (Reciprocal Rank Fusion) — Task 7

```
RRF(chunk) = Σ 1 / (k + rank)   với mỗi danh sách chunk đó xuất hiện
```

- Chỉ dùng **RANK** (vị trí #1, #2, #3...), không dùng score gốc — vì dense
  score và BM25 score không cùng thang đo, cộng thẳng vô nghĩa.
- `k=60` (chuẩn trong paper gốc) làm giảm chênh lệch giữa rank #1 và #2.
- 1 chunk xuất hiện ở CẢ 2 danh sách được **cộng dồn** điểm — đây là cách RRF
  thể hiện "đồng thuận" giữa dense và BM25.
- RRF score chỉ phản ánh thứ hạng — **không so sánh RRF score với
  SCORE_THRESHOLD** (khác thang đo với cosine). Threshold ở Task 9 phải dùng
  cosine score GỐC từ dense, lấy trước khi qua RRF.

**Bonus — cross-encoder reranker:** nhận `(query, document)` làm 1 input duy
nhất (khác dense: embed riêng rồi so cosine) → chính xác hơn về mặt "liên
quan chủ đề" nhưng CHẬM hơn nhiều (chạy model cho mỗi cặp). **Hạn chế thực tế
đã quan sát:** cross-encoder tối ưu cho độ liên quan chủ đề, không phải "có
đúng số liệu trả lời hay không" — từng xếp 1 đoạn brochure mô tả ngành (nhiều
chữ, không số liệu) cao hơn đoạn bảng điểm chuẩn thật (ít chữ, nhiều số) →
không dùng làm bước lọc cuối cho câu hỏi cần số liệu chính xác.

## 4. PageIndex — vectorless RAG (Task 8)

Ý tưởng khác hẳn dense/BM25: không embed từng chunk, mà để LLM đọc trực tiếp
**cây cấu trúc** (mục lục/section) của tài liệu và tự suy luận câu trả lời.
Phù hợp cho tài liệu có cấu trúc rõ (văn bản pháp quy nhiều điều/khoản) hơn là
tra cứu bảng số liệu.

**Bài học quan trọng — API đổi, phải đọc kỹ response:** bản SDK có API cũ
`submit_query()`/`get_retrieval()` kiểu vector-search, nhưng đã bị
**deprecated** — gọi vào vẫn "chạy" (không lỗi cú pháp) nhưng response chứa
field `deprecation` báo chuyển sang `chat_completions()`. Nếu không đọc kỹ
raw response mà chỉ theo tài liệu SDK cũ/theo trực giác, dễ debug sai hướng
rất lâu (đã xảy ra thật trong lab này). `chat_completions()` là API đồng bộ,
trả `citations` (document/page/block_id — KHÔNG có text gốc), nên phải tự cắt
đoạn text ngay trước mỗi tag `<doc=...>` trong câu trả lời của LLM để suy ra
"content" — nội dung này là văn xuôi LLM viết lại, không phải câu chữ gốc
trong PDF (khác hẳn dense/BM25).
→ [task8_pageindex_vectorless.py](../src/task8_pageindex_vectorless.py)

## 5. Chunking bảng Markdown — vấn đề mất ngữ cảnh nhiều tầng

Bảng điểm chuẩn PTIT có **2 trục biến thể lồng nhau** trong CÙNG 1 tài liệu:
cơ sở (Bắc/Nam) × chương trình (đại trà/chất lượng cao/Việt-Nhật/ứng
dụng/liên kết quốc tế). Cùng 1 ngành ("Công nghệ thông tin") xuất hiện **6-7
lần** với điểm khác nhau, chỉ phân biệt được bằng 1 dòng heading nằm phía
trên bảng (cơ sở) và 1 dòng label nằm giữa bảng (chương trình) — cả 2 đều
KHÔNG tự động đi theo khi bảng bị cắt thành nhiều chunk.

**Cách chẩn đoán:** không đoán — đọc trực tiếp file Markdown gốc, xác định
chính xác dòng nào là heading/label, dòng nào là data row, rồi mới sửa logic
cắt chunk theo đúng cấu trúc thật (không phải cấu trúc mình tưởng).

**Fix:** mỗi chunk bảng giờ lặp lại `[title] [context cơ sở] [section
chương trình] [header] [separator]` trước data row của nó — `context` là
heading prose gần nhất trước bảng, `section` là dòng label 1-cell trong bảng
(dò bằng số cell không rỗng, KHÔNG dò theo in đậm `**` vì dữ liệu nguồn không
nhất quán — có label thiếu bold, đã gặp thật).
→ `_chunk_table_lines()`, `_section_label()` trong
[task4_chunking_indexing.py](../src/task4_chunking_indexing.py)

**Trade-off đã chấp nhận:** ~6% chunk bảng vượt ngưỡng kích thước
`CHUNK_SIZE * 1.1` (max 729 ký tự, ngưỡng 550) — vì 1 dòng bảng (nhiều cột
điểm theo phương thức xét tuyển) + phần ngữ cảnh bắt buộc tự nó đã vượt ngưỡng,
và 1 dòng bảng là đơn vị không thể chia nhỏ hơn mà không mất nghĩa. Ưu tiên
đúng ngữ cảnh hơn đúng giới hạn ký tự.

## 6. Generation + Citation (Task 10)

**Lost-in-the-middle:** LLM đọc kỹ đầu/cuối context, dễ bỏ sót phần giữa.
`reorder_for_llm()` dùng `chunks[::2] + chunks[1::2][::-1]` để đưa 2 chunk
quan trọng nhất (rank 1, rank 2) về đúng 2 vị trí đầu và cuối.

**Vấn đề đã phát hiện và sửa — citation numbering lệch với sources trả về:**
`format_context()` đánh số `[Document N]` theo thứ tự ĐÃ REORDER (thứ tự LLM
nhìn thấy trong prompt). Nhưng `validate_search_results()` (contract) yêu cầu
`sources` trả về phải **sort giảm dần theo score** — mà danh sách đã reorder
thì KHÔNG sort (cố ý xáo để chống lost-in-the-middle). Trả thẳng `reordered`
làm `sources` sẽ VI PHẠM contract; giữ `chunks` (đã sort) làm `sources` mà
không đổi lại số trong `answer` thì `[Document N]` sẽ trỏ SAI chunk.

**Cách sửa đúng — decouple label khỏi thứ tự hiển thị:** giữ `answer` đúng
như LLM viết, chỉ **đổi lại SỐ** trong mỗi `[Document N]` từ vị trí-trong-
`reordered` sang vị trí-trong-`chunks` (đã sort) bằng `_relabel_citations()` —
tra qua `id` của chunk (không tra theo nội dung, tránh nhầm khi 2 chunk có nội
dung giống nhau). Kết quả: `sources` vẫn đúng contract (sort giảm dần), citation
trong `answer` vẫn trỏ đúng nguồn. Luôn tự verify bằng
`contracts.validate_generation_result()` sau khi đổi logic ở đây — bug này
không bị test nào bắt (không test nào gọi validator trên nhánh trả lời thành
công), chỉ lộ ra khi tự trace tay qua ví dụ cụ thể.

**Citation grounding — lưới an toàn cuối:** `_citations_are_grounded()` so
khớp số liệu ngay trước mỗi `[Document N]` với content của đúng chunk đó ở
mức KÝ TỰ (không hiểu ngữ nghĩa). Không có citation nào (khi câu trả lời không
phải REFUSAL_ANSWER) → coi là chưa grounded, từ chối trả lời. Hạn chế: chỉ bắt
được số liệu HOÀN TOÀN không tồn tại trong chunk; không bắt được trường hợp
số liệu có tồn tại thật trong chunk nhưng LLM gán NHẦM nhãn (VD nhầm điểm
"chất lượng cao" thành "đại trà" khi cả 2 số đều nằm trong cùng 1 chunk).

**Safe refusal dùng 1 câu cố định (`REFUSAL_ANSWER`)** ở MỌI nhánh không chắc
chắn (context rỗng / LLM lỗi / citation không khớp) — so sánh string là đủ để
biết chắc không có nguồn đáng tin, không cần LLM tự diễn giải lại lý do từ
chối theo nhiều cách khác nhau.

## 7. Retrieval recall yếu cho tra cứu 1 ngành cụ thể — ĐÃ SỬA bằng metadata filter

**Vấn đề gốc (đã verify thật):** dense/BM25 xếp hạng rất thấp (rank 10-80
trong ~974 chunk) 1 dòng ngành cụ thể giữa hàng chục dòng bảng điểm có cấu
trúc số liệu gần giống hệt nhau — mọi dòng bảng đều "giống nhau" về cấu trúc
số/cột với embedding, và BM25 bị loãng vì nhiều tên ngành chia sẻ chung từ
("Công nghệ", "Kỹ thuật"...). Trước khi sửa, hệ thống xử lý AN TOÀN (từ chối
trả lời thay vì đoán nhầm) nhưng KHÔNG trả lời được những câu hỏi thực sự có
đủ dữ liệu để trả lời đúng.

**Fix — structured extraction + exact-match filter:**
1. Task 4 dò đúng CỘT chứa tên ngành trong header bảng bằng chữ "ngành" (không
   hard-code vị trí cột — đã verify 2 bảng khác nhau đặt cột này ở vị trí khác
   nhau), trích thành `metadata["major"]` cho mỗi chunk bảng. Tương tự dò
   `metadata["campus"]` (bắc/nam) từ heading TRƯỚC bảng.
   → `_major_column_index()`, `_row_major()`, `_detect_campus()` trong
   [task4_chunking_indexing.py](../src/task4_chunking_indexing.py)
2. Task 9 thêm `exact_major_match()`: khi tên ngành xuất hiện literal trong
   câu hỏi, lọc CHÍNH XÁC theo `metadata["major"]` (và `campus` nếu câu hỏi
   nói rõ cơ sở), đưa thẳng vào RRF như 1 ranked list thứ 3 (cùng dense +
   BM25) — luôn đứng rank 1 trong list riêng của nó nên gần như chắc chắn lọt
   top_k dù dense/BM25 xếp nó rất thấp.
   → [task9_retrieval_pipeline.py](../src/task9_retrieval_pipeline.py)

**Bài học khi sửa — heuristic đơn giản dễ vỡ ở edge case, phải TEST RỘNG chứ
không chỉ test đúng 1 câu hỏi đã biết lỗi:**
- Lấy "dòng cuối cùng trước bảng" làm context/campus không đủ — có tài liệu
  chèn 1 dòng khác (địa chỉ cơ sở) giữa heading "CƠ SỞ ĐÀO TẠO PHÍA BẮC" và
  bảng, khiến campus bị None dù heading rõ ràng có ở gần đó. Phải quét TOÀN
  BỘ đoạn prose liền trước, không chỉ dòng cuối.
- 1 ngành có thể xuất hiện ở NHIỀU LOẠI bảng khác nhau (bảng điểm chuẩn, bảng
  chỉ tiêu tuyển sinh...) trong cùng tài liệu — chỉ khớp theo major/campus
  không phân biệt được người dùng đang hỏi về bảng nào, có thể khiến chunk
  đúng loại bị chunk sai loại (cùng ngành, cùng cơ sở, khác bảng) lấn át
  trong top_k. Sửa bằng cách sắp lại các exact match theo số từ trong câu hỏi
  trùng với NỘI DUNG chunk (không chỉ trùng tên ngành).

**Hạn chế còn lại (chưa sửa — phát hiện mới, nhỏ hơn):** cách sắp xếp trên
dựa vào trùng CHỮ, không hiểu ngữ nghĩa — câu hỏi "chỉ tiêu tuyển sinh ngành
X" vẫn có thể không tìm đúng bảng chỉ tiêu nếu cột dữ liệu ghi "Số lượng dự
kiến" thay vì đúng chữ "chỉ tiêu" (đã verify: vẫn từ chối trả lời, an toàn
nhưng chưa trả lời được). Hướng sửa đúng nếu cần: thêm 1 field
`metadata["table_kind"]` ("điểm chuẩn"/"chỉ tiêu"/...) tương tự cách làm với
major/campus, dò theo tên cột header (VD cột có "Điểm chuẩn" → table_kind=
"điểm chuẩn"; cột có "Số lượng dự kiến" → table_kind="chỉ tiêu").

## 8. ChromaDB âm thầm bỏ qua metadata có giá trị None

**Bug thật, tồn tại từ đầu project, chỉ lộ ra khi retrieval trả về đúng 1
chunk legal (url=None) qua `validate_generation_result`:** Chroma KHÔNG lưu
metadata key nào có value là `None` khi `upsert()` — verify trực tiếp bằng 1
collection test độc lập. Hệ quả: mọi chunk có field nào đó = None lúc tạo
(VD `url=None` cho tài liệu legal, hay `major=None`/`campus=None` cho chunk
không phải bảng điểm) sẽ bị THIẾU HẲN key đó khi đọc lại — không phải sai
kiểu dữ liệu, mà là key biến mất — khiến `validate_document()` báo lỗi dễ
gây hiểu lầm ("metadata.url must be a string or None", trong khi bản chất là
key không tồn tại).

**Fix:** không sửa ở Task 4 (upsert vẫn truyền đủ, không lỗi của mình) — sửa
ở MỌI nơi ĐỌC metadata từ Chroma (Task 5 `semantic_search`, Task 6
`_load_corpus`, Task 9 `_load_major_index`): gọi
`restore_metadata_defaults()` ngay sau khi lấy kết quả về, khôi phục lại các
key có thể bị thiếu bằng giá trị None mặc định.
→ [task4_chunking_indexing.py](../src/task4_chunking_indexing.py)

**Bài học:** một thư viện có vẻ hoạt động đúng (không báo lỗi, không
exception) vẫn có thể âm thầm làm mất dữ liệu — chỉ phát hiện được bằng cách
tự verify trực tiếp (viết 1 test cô lập tối thiểu để tái hiện), không suy
đoán từ tài liệu hay từ việc "code chạy không lỗi".

## 9. Checklist trước khi push / nộp bài

- [ ] `pytest tests/test_contracts.py -q` — phải pass toàn bộ (15/15).
- [ ] `.env` KHÔNG được commit (đã có trong `.gitignore`); chỉ commit
      `.env.example`.
- [ ] `chroma_db/` là dữ liệu build ra được (`python -m
      src.task4_chunking_indexing`) — nên thêm vào `.gitignore`, không cần
      commit (đã phát hiện: hiện tại CHƯA có trong `.gitignore`).
- [ ] `data/pageindex_doc_ids.json` không commit (đã có trong `.gitignore`).
- [ ] `group_project/evaluation/golden_dataset.json` (≥15 câu, có
      `question`/`expected_answer`/`expected_context`) — CHƯA làm.
- [ ] `group_project/evaluation/RESULT.md` (không còn "TODO", đủ heading
      Overall scores / A-B comparison / Worst performers / Recommendations)
      — CHƯA làm.
- [ ] `group_project/ịndividual/INDIVIDUAL_REPORT.md` — CHƯA làm.
- [ ] Trong báo cáo, nhắc rõ các trade-off/hạn chế đã biết: (1) một số chunk
      bảng vượt giới hạn kích thước để giữ đủ ngữ cảnh (mục 5); (2) exact-match
      theo tên ngành/cơ sở đã tăng recall đáng kể nhưng chưa phân biệt được
      loại bảng (điểm chuẩn vs chỉ tiêu) khi cần (mục 7).
