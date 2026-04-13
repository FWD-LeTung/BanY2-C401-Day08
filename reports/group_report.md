# Báo Cáo Nhóm — Lab Day 08: Full RAG Pipeline

**Tên nhóm:** Y2 - C401  
**Thành viên:**  
- Lê Thanh Thưởng   
- Lê Văn Tùng 
- Nguyễn Đức Sĩ
- Đinh Thái Tuấn  
**Ngày nộp:** 13/04/2026  

---

## 1. Mục tiêu & bối cảnh

Nhóm xây dựng một hệ thống **RAG (Retrieval-Augmented Generation)** để trả lời các câu hỏi nội bộ cho **CS + IT Helpdesk** dựa trên tập tài liệu chính sách/SLA/SOP/FAQ . Mục tiêu khi thiết kế pipeline:

1. **Grounded answer**: câu trả lời chỉ dựa trên ngữ cảnh retrieve được, có **citation**.
2. **Abstain khi thiếu dữ liệu**: nếu tài liệu không có thông tin, hệ thống phải nói “không đủ dữ liệu / tôi không biết”, không bịa.
3. **Đo được bằng số liệu**: chạy evaluation và có scorecard baseline/variant.

---

## 2. Tổng quan kiến trúc (end-to-end)

```
[Raw Docs in data/docs]
        ↓
[index.py: preprocess → chunk → embed → store]
        ↓
[ChromaDB vector store (chroma_db/)]
        ↓
[rag_answer.py: query → retrieve (dense/hybrid) → (rerank) → generate]
        ↓
[Answer + sources/citation]
        ↓
[eval.py: scorecard + A/B comparison]
```

**Mapping theo file:**
- Indexing: `index.py`
- Retrieval + Generation: `rag_answer.py`
- Evaluation: `eval.py`
- Documentation: `docs/architecture.md`, `docs/tuning-log.md`

---

## 3. Sprint deliverables 

### Sprint 1 — Indexing + metadata 
- Preprocess tài liệu: parse metadata từ header (`Source`, `Department`, `Effective Date`, `Access`).
- Chunking ưu tiên ranh giới tự nhiên theo **heading**, nếu section dài thì pack theo **paragraph** và thêm **overlap** để không “đứt ý”.
- Mỗi chunk giữ metadata quan trọng phục vụ retrieval/citation.

### Sprint 2 — Baseline dense retrieval + grounded answer
- Dense retrieval qua ChromaDB (cosine similarity) với cấu hình top-k search/select rõ ràng.
- Prompt ép “answer only from context”, có citation dạng `[1]`.
- Với các câu thiếu ngữ cảnh (insufficient context), hệ thống **abstain** thay vì bịa.

### Sprint 3 — Variant (A/B rule: đổi tối thiểu)
- Implement **hybrid retrieval**: Dense + Sparse (BM25) và merge bằng **Reciprocal Rank Fusion (RRF)**.
- Implement **rerank** bằng cross-encoder để chọn top-k chunk thực sự trả lời câu hỏi trước khi đưa vào prompt.

### Sprint 4 — Evaluation + A/B comparison + docs
- Chạy scorecard cho baseline và variant, xuất:
  - `results/scorecard_baseline.md`
  - `results/scorecard_variant.md`
- Điền tài liệu:
  - `docs/architecture.md`
  - `docs/tuning-log.md`

---

## 4. Quyết định kỹ thuật quan trọng

### 4.1 Indexing / Chunking
**Cấu hình:**
- `CHUNK_SIZE = 400` tokens (ước lượng tokens ≈ ký tự/4)
- `CHUNK_OVERLAP = 80` tokens

**Chiến lược:**
- Split theo heading để giữ cấu trúc tài liệu.
- Nếu section quá dài: pack theo paragraph đến khi đủ size.
- Overlap lấy **1 câu cuối** (fallback: ~30% cuối) để giảm mất ngữ cảnh.

**Vì sao chọn vậy?**
- Các câu hỏi SLA/policy thường cần đủ 2 ý (ví dụ “phản hồi” + “resolution”). Nếu chunk boundary cắt giữa điều khoản hoặc select thiếu chunk, LLM dễ trả lời đúng một nửa → completeness giảm.

**Tài liệu & số chunk (từ pipeline chunking hiện tại):**
| File | Nguồn | Dept | #chunks |
|------|------|------|--------:|
| `access_control_sop.txt` | it/access-control-sop.md | IT Security | 7 |
| `policy_refund_v4.txt` | policy/refund-v4.pdf | CS | 6 |
| `it_helpdesk_faq.txt` | support/helpdesk-faq.md | IT | 6 |
| `sla_p1_2026.txt` | support/sla-p1-2026.pdf | IT | 5 |
| `hr_leave_policy.txt` | hr/leave-policy-2026.pdf | HR | 5 |

### 4.2 Retrieval baseline vs variant
**Baseline config:**
- `retrieval_mode = "dense"`
- `top_k_search = 10`
- `top_k_select = 3`
- `use_rerank = False`

**Variant config:**
- `retrieval_mode = "hybrid"`
- `top_k_search = 20`
- `top_k_select = 3`
- `use_rerank = True`

**Hybrid detail:**
- Sparse: BM25 (`rank_bm25`) với tokenizer đơn giản Việt/Anh.
- Merge: RRF với `dense_weight=0.75`, `sparse_weight=0.25`, `rrf_k=60`.

**Rerank detail:**
- Cross-encoder: `cross-encoder/ms-marco-MiniLM-L-6-v2`.

**Vì sao tune biến này?**
- Baseline đã đạt **Context Recall trung bình 5.00/5**, nhưng vẫn có vài câu **completeness thấp** (thiếu chi tiết).
- Kỳ vọng: hybrid giúp “không bỏ sót” khi query có keyword/alias; rerank giúp ưu tiên chunk đầy đủ ý trước khi đưa vào prompt.

---

## 5. Generation & Guardrails

**Prompt grounding:**
- “Answer only from retrieved context”
- “If insufficient, say you do not know”
- “Cite the source field in brackets like [1]”

**LLM call:**
- Model: `gpt-4o`
- `temperature = 0`
- `max_tokens = 512`

---

## 6. Kết quả đánh giá (scorecard) & A/B analysis

**Evaluation approach:**
- **Faithfulness / Relevance / Completeness**: chấm bằng **LLM-as-Judge (Qwen-turbo)**, ép output JSON và `temperature=0` để ổn định.
- **Context Recall**: kiểm tra xem các chunk đã dùng có chứa **expected_sources** không (match theo tên file nguồn).
- Baseline/Variant config được cố định trong `BASELINE_CONFIG` và `VARIANT_CONFIG` để A/B so sánh minh bạch.

### 6.1 Scorecard baseline
File: `results/scorecard_baseline.md`
- Faithfulness: **4.80/5**
- Relevance: **5.00/5**
- Context Recall: **5.00/5**
- Completeness: **4.20/5**

### 6.2 Scorecard variant
File: `results/scorecard_variant.md`
- Faithfulness: **4.50/5**
- Relevance: **4.60/5**
- Context Recall: **5.00/5**
- Completeness: **3.90/5**

### 6.3 A/B summary (có delta + giải thích)
| Metric | Baseline | Variant | Delta |
|--------|----------|---------|------:|
| Faithfulness | 4.80 | 4.50 | -0.30 |
| Relevance | 5.00 | 4.60 | -0.40 |
| Context Recall | 5.00 | 5.00 | +0.00 |
| Completeness | 4.20 | 3.90 | -0.30 |

**Cải thiện rõ:**
- `gq01` (SLA): baseline (F=4, C=3) → variant (F=5, C=4). Lý do hợp lý là hybrid + rerank ưu tiên chunk đủ ý hơn.

**Tệ hơn (root cause rõ):**
- `gq05` (Access Control): baseline đạt 5/5 toàn bộ; variant tụt xuống 1/5 (Faithful/Relevant/Complete) dù **Context Recall vẫn 5**.
  - Diễn giải: retrieve đã lấy đúng nguồn, nhưng rerank/select đã **đẩy sai chunk lên top-3** hoặc chọn chunk không chứa đáp án, khiến LLM abstain (“tôi không biết”) → điểm tụt.

**Kết luận chọn cấu hình chạy thật (để tối đa điểm):**
- Với score hiện tại, **baseline_dense** là cấu hình ổn định và có điểm trung bình cao hơn.
- Variant được giữ lại như một thí nghiệm tuning (đúng yêu cầu Sprint 3 + A/B comparison), nhưng nếu dùng cho grading thật thì cần kiểm soát rerank.

---

## 7. Failure modes & cách nhóm debug 
**Error tree nhóm dùng:** Index → Retrieval → Generation

### Ví dụ failure mode thực tế 1 (completeness thiếu ý)
- **Triệu chứng:** câu trả lời đúng chủ đề nhưng thiếu một mốc/ý quan trọng (thường gặp ở các câu SLA/FAQ có nhiều chi tiết).
- **Root cause:** top_k_select=3 + chunk boundary làm một phần thông tin nằm ở chunk #4-#5, không vào prompt.
- **Fix/mitigation:** tăng `top_k_search` để có pool tốt hơn; cân nhắc `top_k_select` theo loại câu hỏi; cải thiện chunking để “cùng điều khoản nằm cùng chunk”.

### Ví dụ failure mode thực tế 2 (rerank chọn sai chunk)
- **Triệu chứng:** Variant tụt mạnh ở `gq05` dù recall vẫn cao.
- **Root cause:** cross-encoder rerank không ổn định với tiếng Việt/thuật ngữ hoặc candidates quá nhiễu → chọn sai top-3.
- **Fix/mitigation:** thử reranker multilingual phù hợp hơn, thêm điều kiện bật/tắt rerank theo category, hoặc rerank top_k lớn hơn trước khi cắt top-3.

---

## 8. Hướng cải tiến
1. **Kiểm soát rerank để tránh gq05-type regression**: thử reranker multilingual/Việt hoá, hoặc chỉ bật rerank khi score gap nhỏ/nhiều noise.
2. **Adaptive top_k_select**: câu multi-detail (SLA/FAQ) có thể cần top-4/top-5 để tăng completeness; câu fact ngắn vẫn dùng top-3 để tiết kiệm token.
3. **Metadata-aware retrieval**: filter theo `department`/`source` khi query có tín hiệu (SLA vs Refund) để giảm noise candidates.

---

## 9. Checklist nộp bài (để không mất điểm vì thiếu file)

Theo `SCORING.md`, repo cần có:
- [x] `index.py`, `rag_answer.py`, `eval.py`
- [x] `data/docs/` đủ 5 tài liệu + `data/test_questions.json`
- [x] `results/scorecard_baseline.md`, `results/scorecard_variant.md`
- [] `docs/architecture.md`, `docs/tuning-log.md`
- [x] `reports/group_report.md`
- [x] `reports/individual/*.md`
- [x] `logs/grading_run.json` (**sẽ tạo khi `grading_questions.json` được public**) - nhóm em để ở trong results/grading_testing

---

