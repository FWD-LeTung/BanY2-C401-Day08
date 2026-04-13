# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Lê Thanh Thưởng  
**Vai trò trong nhóm:** Tech Lead  
**Ngày nộp:** 13/04/2026  
**Độ dài yêu cầu:** 500–800 từ

---

## 1. Tôi đã làm gì trong lab này? (100-150 từ)

> Mô tả cụ thể phần bạn đóng góp vào pipeline:
> - Sprint nào bạn chủ yếu làm?
> - Cụ thể bạn implement hoặc quyết định điều gì?
> - Công việc của bạn kết nối với phần của người khác như thế nào?

Trong lab này tôi đóng vai trò Tech Lead và trực tiếp tham gia 3 sprint chính để nối pipeline end-to-end. Ở Sprint 1, tôi làm file `index.py`, tập trung vào 2 hàm `build_index` và `list_chunks`: chuẩn hoá quy trình đọc tài liệu, chunking + gắn metadata, và đảm bảo sau khi index xong có thể kiểm tra nhanh các chunk đã tạo (phục vụ debug và đối chiếu nguồn). Ở Sprint 2, tôi implement `call_llm` trong `rag_answer.py` để chuẩn hoá cách gọi LLM theo prompt của nhóm, đảm bảo trả lời có cấu trúc và bám ngữ cảnh. Sang Sprint 3, tôi bổ sung query transform và rerank (cũng trong `rag_answer.py`) để cải thiện chất lượng retrieval trước khi đưa vào generation.

---

## 2. Điều tôi hiểu rõ hơn sau lab này (100-150 từ)

> Chọn 1-2 concept từ bài học mà bạn thực sự hiểu rõ hơn sau khi làm lab.
> Ví dụ: chunking, hybrid retrieval, grounded prompt, evaluation loop.
> Giải thích bằng ngôn ngữ của bạn — không copy từ slide.

Sau lab, tôi hiểu rõ hơn vì sao “chunking” gần như quyết định trần hiệu năng của RAG. Nếu chunk quá dài, retrieval dễ kéo về đoạn có nhiều nhiễu và model khó trích đúng ý; nếu chunk quá ngắn, thông tin bị “đứt mạch” (ví dụ SLA phản hồi và SLA resolution nằm ở hai dòng khác nhau) khiến câu trả lời thiếu ý quan trọng. Tôi cũng hiểu rõ hybrid retrieval không chỉ là “cộng 2 kiểu search” mà là cách tận dụng ưu điểm của sparse (khớp keyword/alias) và dense (bắt ngữ nghĩa). Khi kết hợp đúng trọng số và có rerank, hybrid giúp vừa tìm đúng tài liệu, vừa đưa lên top những chunk thực sự trả lời câu hỏi, giảm tình trạng trả lời lan man.

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn (100-150 từ)

> Điều gì xảy ra không đúng kỳ vọng?
> Lỗi nào mất nhiều thời gian debug nhất?
> Giả thuyết ban đầu của bạn là gì và thực tế ra sao?

Điều tôi ngạc nhiên là chỉ cần thay đổi nhỏ ở query transform hoặc chiến lược rerank cũng có thể làm score biến động mạnh giữa các câu. Khó khăn lớn nhất là debug “đúng tài liệu nhưng sai hành vi”: có lúc retrieval lấy đúng nguồn, nhưng LLM vẫn trả lời thiếu một phần (thường do context chứa nhiều chunk gần giống nhau, hoặc chunk bị thiếu câu then chốt). Ban đầu tôi nghĩ lỗi nằm ở LLM “ảo giác”, nhưng khi soi lại bằng `list_chunks` và log retrieval thì nguyên nhân chính thường là chunking/ordering: chunk chứa keyword đúng nhưng thiếu chi tiết (ví dụ chỉ có “15 phút” mà không có “4 giờ”), khiến model trả lời đúng một nửa. Việc thêm rerank giúp ưu tiên chunk đầy đủ ý hơn, nhưng cũng phải cẩn thận để không làm tụt các câu khác.

---

## 4. Phân tích một câu hỏi trong scorecard (150-200 từ)

> Chọn 1 câu hỏi trong test_questions.json mà nhóm bạn thấy thú vị.
> Phân tích:
> - Baseline trả lời đúng hay sai? Điểm như thế nào?
> - Lỗi nằm ở đâu: indexing / retrieval / generation?
> - Variant có cải thiện không? Tại sao có/không?

**Câu hỏi:** q01 — “SLA xử lý ticket P1 là bao lâu?”

**Phân tích:**

Với câu q01, đáp án kỳ vọng gồm đủ 2 ý: phản hồi ban đầu **15 phút** và thời gian xử lý (resolution) **4 giờ** (nguồn: tài liệu SLA P1). Ở baseline (dense), hệ thống trả lời đúng chủ đề và nhìn chung bám ngữ cảnh, nhưng completeness bị thấp hơn (trong scorecard gq01 completeness = 3). Tôi suy luận nguyên nhân là retrieval đã kéo được chunk có “P1/15 phút” nhưng phần “resolution 4 giờ” nằm ở chunk khác hoặc bị xếp thứ hạng thấp, nên prompt context đưa vào LLM không đủ để model trả lời trọn vẹn.

Ở variant (hybrid + query transform + rerank), điểm câu tương ứng gq01 tăng (faithfulness = 5, completeness = 4). Query transform giúp “chuẩn hoá” câu hỏi về dạng chứa các từ khoá quan trọng (P1, SLA, response/resolution), còn hybrid retrieval giúp không bỏ sót đoạn có keyword “resolution”. Rerank sau cùng ưu tiên những chunk có đủ cả hai mốc thời gian, nên context đầy đủ hơn và câu trả lời hoàn chỉnh hơn.

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì? (50-100 từ)

> 1-2 cải tiến cụ thể bạn muốn thử.
> Không phải "làm tốt hơn chung chung" mà phải là:
> "Tôi sẽ thử X vì kết quả eval cho thấy Y."

Nếu có thêm thời gian, tôi sẽ làm một vòng tuning có hệ thống cho hybrid: thử nhiều mức trọng số sparse/dense theo từng category (SLA, Refund, HR), rồi chốt config theo scorecard thay vì chọn cảm tính. Tôi cũng muốn cải thiện rerank bằng cách thêm feature “coverage” (chunk nào chứa đủ các slot quan trọng như response + resolution), và bổ sung logging rõ ràng hơn (top-k chunk + lý do rerank) để debug nhanh khi một câu bị tụt điểm.

---

*Lưu file này với tên: `reports/individual/[ten_ban].md`*
*Ví dụ: `reports/individual/nguyen_van_a.md`*
