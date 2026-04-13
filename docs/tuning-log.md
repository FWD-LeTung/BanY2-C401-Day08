# Tuning Log — RAG Pipeline (Day 08 Lab)

> Template: Ghi lại mỗi thay đổi và kết quả quan sát được.
> A/B Rule: Chỉ đổi MỘT biến mỗi lần.

---

## Baseline (Sprint 2)

**Ngày:** 13/04/2026  
**Config:**
```
retrieval_mode = "dense"
chunk_size = 400 tokens
overlap = 80 tokens
top_k_search = 10
top_k_select = 3
use_rerank = False
llm_model = gpt-4o
```

**Scorecard Baseline:**
| Metric | Average Score |
|--------|--------------|
| Faithfulness | 4.80/5 |
| Answer Relevance | 5.00/5 |
| Context Recall | 5.00/5 |
| Completeness | 4.20/5 |

**Câu hỏi yếu nhất (điểm thấp):**
- **gq01 (SLA)**: Faithfulness = 4, Completeness = 3 → đúng ý chính nhưng thiếu/thiếu rõ một phần chi tiết.
- **gq09 (IT Helpdesk)**: Faithfulness = 4, Completeness = 3 → trả lời đúng hướng nhưng thiếu một mốc/thành phần chi tiết.

**Giả thuyết nguyên nhân (Error Tree):**
- [x] Indexing: Chunking cắt giữa điều khoản (một phần thông tin quan trọng nằm ở chunk khác)
- [ ] Indexing: Metadata thiếu effective_date
- [ ] Retrieval: Dense bỏ lỡ exact keyword / alias (Context Recall trung bình = 5.00/5 → ít khả năng)
- [x] Retrieval: Top-k quá ít → thiếu evidence (top_k_select=3 có thể bỏ lỡ phần “ý thứ hai” nếu nằm ở chunk #4-#5)
- [x] Generation: Prompt đủ grounding nhưng LLM có xu hướng tóm tắt làm rơi chi tiết
- [ ] Generation: Context quá dài → lost in the middle (context ngắn, ít khả năng)

---

## Variant 1 (Sprint 3)

**Ngày:** 13/04/2026  
**Biến thay đổi:** Hybrid retrieval + bật rerank (kèm tăng top_k_search để đủ candidates)  
**Lý do chọn biến này:**
Baseline đã rất tốt về recall, nhưng vẫn có vài câu bị thiếu detail (completeness thấp). Nhóm chọn hybrid (dense + BM25) để giảm rủi ro “bỏ sót” khi query có keyword/alias, đồng thời bật rerank để ưu tiên những chunk đủ ý thay vì chunk chỉ chứa keyword.

**Config thay đổi:**
```
retrieval_mode = "hybrid"
top_k_search = 20        # search rộng hơn để rerank chọn được chunk tốt
use_rerank = True        # cross-encoder rerank
# top_k_select = 3 giữ nguyên
```

**Scorecard Variant 1:**
| Metric | Baseline | Variant 1 | Delta |
|--------|----------|-----------|-------|
| Faithfulness | 4.80/5 | 4.50/5 | -0.30 |
| Answer Relevance | 5.00/5 | 4.60/5 | -0.40 |
| Context Recall | 5.00/5 | 5.00/5 | +0.00 |
| Completeness | 4.20/5 | 3.90/5 | -0.30 |

**Nhận xét:**
- **Cải thiện:** gq01 (SLA) tăng chất lượng (Faithfulness 4→5, Completeness 3→4), phù hợp với mục tiêu “đủ ý” nhờ hybrid + rerank.
- **Giảm mạnh:** gq05 (Access Control) bị tụt nghiêm trọng (Faithful/Relevant/Complete = 1). Đây là dấu hiệu rerank/hybrid đã chọn sai chunk hoặc cross-encoder không ổn định với ngữ cảnh tiếng Việt/thuật ngữ, dẫn tới answer không bám đúng evidence.
- Tổng thể, variant làm giảm điểm trung bình dù cải thiện được một số câu cụ thể.

**Kết luận:**
Variant 1 **không tốt hơn** baseline về tổng thể (các metric trung bình đều giảm, đặc biệt do gq05). Tuy nhiên, hybrid + rerank cho thấy lợi ích ở các câu cần đủ nhiều chi tiết (ví dụ gq01). Nếu tiếp tục tune, cần kiểm soát rerank (model phù hợp ngôn ngữ, giảm noise candidates, hoặc điều kiện bật/tắt theo loại câu hỏi).

---

## Variant 2 (nếu có thời gian)

**Biến thay đổi:** Không thực hiện (giới hạn thời gian, ưu tiên ổn định baseline)  
**Config:**
```
N/A
```

**Scorecard Variant 2:**
| Metric | Baseline | Variant 1 | Variant 2 | Best |
|--------|----------|-----------|-----------|------|
| Faithfulness | 4.80 | 4.50 | N/A | Baseline |
| Answer Relevance | 5.00 | 4.60 | N/A | Baseline |
| Context Recall | 5.00 | 5.00 | N/A | Tie |
| Completeness | 4.20 | 3.90 | N/A | Baseline |

---

## Tóm tắt học được

1. **Lỗi phổ biến nhất trong pipeline này là gì?**
   > “Thiếu ý” (completeness) do chunking/selection: đúng tài liệu nhưng không đưa đủ các chunk cần thiết vào prompt (top_k_select thấp) hoặc chunk boundary làm tách thông tin.

2. **Biến nào có tác động lớn nhất tới chất lượng?**
   > Rerank (cross-encoder) và cách chọn candidate pool (top_k_search). Rerank có thể cải thiện một số câu nhưng cũng có thể làm tụt mạnh nếu chọn sai chunk.

3. **Nếu có thêm 1 giờ, nhóm sẽ thử gì tiếp theo?**
   > (1) Tune hybrid weights (dense/sparse) và giảm noise bằng metadata filter theo category; (2) thử reranker multilingual hoặc điều kiện hoá rerank theo loại câu hỏi; (3) tăng top_k_select lên 4–5 cho các câu multi-detail và đánh đổi token/cost dựa trên scorecard.
