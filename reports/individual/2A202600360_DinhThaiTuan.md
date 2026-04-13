# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Đinh Thái Tuấn  
**MSSV:** 2A202600360  
**Vai trò trong nhóm:** Documentation Owner  
**Ngày nộp:** 13/04/2026  
**Nhóm Y2 - C401**

---

## 1. Tôi đã làm gì trong lab này?

Trong lab này, tôi đảm nhận vai trò Documentation Owner và tham gia trực tiếp vào phần kiểm tra chất lượng index ở Sprint 1. Cụ thể, tôi implement hàm `inspect_metadata_coverage()` trong `index.py` — hàm này kiểm tra phân phối metadata toàn bộ index sau khi build: thống kê source coverage, phân bố chunk theo department, kiểm tra chunk nào thiếu `effective_date`, phân bố theo section và access level. Đây là bước quan trọng để đảm bảo metadata đầy đủ trước khi chạy retrieval.

Song song đó, tôi chịu trách nhiệm viết và hoàn thiện hai tài liệu kỹ thuật chính của nhóm: `docs/architecture.md` (mô tả kiến trúc pipeline, quyết định chunking, retrieval config, failure mode checklist) và `docs/tuning-log.md` (ghi lại quá trình tuning A/B: config baseline vs variant, scorecard, nhận xét và kết luận). Công việc của tôi kết nối trực tiếp với tất cả các sprint: tôi cần hiểu rõ phần index (Sprint 1), retrieval + generation (Sprint 2–3), và eval (Sprint 4) để ghi chép chính xác.

---

## 2. Điều tôi hiểu rõ hơn sau lab này

Sau lab, tôi hiểu rõ hơn hai điều. Thứ nhất là tầm quan trọng của **metadata trong indexing**. Khi implement `inspect_metadata_coverage()`, tôi thấy rằng nếu metadata thiếu (ví dụ `effective_date = "unknown"` hoặc `department` sai), thì dù retrieval tìm đúng chunk, LLM vẫn không thể citation chính xác và người dùng không biết thông tin đến từ phiên bản nào của tài liệu. Metadata không chỉ để debug mà còn là nền tảng của trust trong RAG.

Thứ hai, qua việc tổng hợp `tuning-log.md`, tôi hiểu rằng **tuning RAG phải đổi từng biến một** và đánh giá bằng số liệu cụ thể. Nhóm đổi cùng lúc hybrid + rerank, khi variant bị tụt điểm (Faithfulness 4.80→4.50) thì rất khó biết nguyên nhân từ hybrid hay rerank. Bài học là luôn A/B test từng biến riêng lẻ.

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn

Điều ngạc nhiên nhất là **variant (hybrid + rerank) lại kém hơn baseline** ở hầu hết metrics. Ban đầu tôi kỳ vọng kết hợp nhiều kỹ thuật sẽ tốt hơn, nhưng thực tế scorecard cho thấy Faithfulness giảm từ 4.80 xuống 4.50, Relevance từ 5.00 xuống 4.60. Đặc biệt, câu gq05 (Access Control) bị tụt từ 5/5/5/5 xuống 1/1/5/1 — rerank đã đẩy chunk đúng ra khỏi top-3 dù Context Recall vẫn đạt 5.

Khó khăn khi viết documentation là phải hiểu sâu phần code của tất cả thành viên để mô tả chính xác. Tôi phải đọc lại từng hàm trong `index.py`, `rag_answer.py`, `eval.py` và chạy thử pipeline nhiều lần để ghi đúng config và failure mode vào `architecture.md`.

---

## 4. Phân tích một câu hỏi trong scorecard

**Câu hỏi:** gq01 — "SLA xử lý ticket P1 là bao lâu?"

**Phân tích:**

Đáp án kỳ vọng gồm 2 ý: phản hồi ban đầu **15 phút** và thời gian resolution **4 giờ**. Ở baseline (dense), câu này đạt Faithfulness = 4, Completeness = 3 — đúng ý chính nhưng thiếu một phần chi tiết. Qua kiểm tra, tôi nhận thấy lỗi nằm ở **retrieval + chunking**: thông tin "15 phút" và "4 giờ" nằm ở hai chunk khác nhau do boundary cắt giữa section, và top_k_select = 3 không đủ đưa cả hai chunk vào prompt.

Ở variant (hybrid + rerank), câu gq01 cải thiện rõ rệt: Faithfulness tăng lên 5, Completeness tăng lên 4. Hybrid retrieval giúp BM25 bắt được keyword "resolution" mà dense có thể miss, và rerank ưu tiên chunk chứa đủ cả hai mốc thời gian. Đây là một trong số ít câu variant thực sự tốt hơn baseline, cho thấy hybrid + rerank có hiệu quả khi câu hỏi cần tổng hợp nhiều chi tiết rải rác.

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì?

Tôi sẽ mở rộng `inspect_metadata_coverage()` thành một công cụ health-check tự động: chạy trước mỗi lần eval để cảnh báo nếu có chunk thiếu metadata hoặc phân bố department bất thường (ví dụ 1 department chiếm > 50% chunks). Ngoài ra, từ kết quả scorecard cho thấy variant làm tụt gq05, tôi muốn thử tách biến: chạy riêng config `hybrid + no rerank` và `dense + rerank` để xác định chính xác rerank hay hybrid gây tụt điểm, rồi ghi vào tuning-log cho nhóm ra quyết định cuối cùng.
