# Bao Cao Ca Nhan - Lab Day 08: RAG Pipeline

**Ho va ten:** Nguyen Duc Si  
**Vai tro trong nhom:** Retrieval Owner  
**Ngay nop:** 2026-04-13

---

## 1. Toi da lam gi trong lab nay?

Trong bai lab nay, phan toi chiu trach nhiem chinh la retrieval va test pipeline. Cu the, toi implement ham `get_embedding` trong `index.py` de tao vector embedding cho chunk, co co che fallback provider (OpenAI va local) de tranh bi nghen khi thieu key hoac loi model. Nhung phan toi tap trung nhat la trong `rag_answer.py`: toi lam luong `dense` retrieval, `sparse` (BM25), va `hybrid` retrieval bang Reciprocal Rank Fusion. Ngoai ra toi cung ket noi rerank vao pipeline `rag_answer` de co the so sanh variant theo dung setup Sprint 3.

Sau khi implement xong retrieval, toi la nguoi chay kiem thu va danh gia A/B thong qua `eval.py`, voi baseline (`dense`) va variant (`hybrid + rerank`). Ket qua duoc ghi ra trong `results/scorecard_baseline.md`, `results/scorecard_variant.md`, va `results/ab_comparison.csv`. Cong viec cua toi gan truc tiep voi indexing va generation: neu retrieval sai chunk thi prompt du tot van de den answer thieu hoac abstain sai.

---

## 2. Dieu toi hieu ro hon sau lab nay

Dieu toi hieu ro nhat sau lab la: retrieval mode manh hon khong dong nghia voi ket qua cuoi cung se tot hon. Luc dau toi nghi hybrid + rerank chac chan se vuot dense baseline vi ket hop duoc semantic va keyword. Nhung du lieu scorecard cho thay nguoc lai: variant khong tang context recall (van 5.00/5) nhung lai giam faithfulness va relevance. Dieu nay lam toi hieu ro mot diem quan trong: context recall cao chi noi rang lay dung tai lieu, chua noi rang lay dung **chunk** can thiet cho cau hoi.

Toi cung hieu ro hon ve tac dong day chuyen trong RAG pipeline. `index.py` chunking + metadata la lop dau; retrieval xep hang la lop thu hai; prompt generation la lop cuoi. Chi can lop retrieval chon sai section trong cung mot source, LLM co xu huong abstain hoac tra loi thieu y, du khong hallucinate. Vi vay, toi thay tuning RAG khong nen chi nhin mot metric, ma phai nhin tong hop faithfulness, relevance, completeness va doi chieu theo tung cau hoi.

---

## 3. Dieu toi gap kho khan thuc te

Kho khan lon nhat la viec dung variant (hybrid, rerank) luc dau cho ket qua khong tot bang baseline khi duoc mot model LLM khac danh gia. Dieu nay trai ky vong ban dau cua toi, vi tren ly thuyet hybrid va rerank thuong giup retrieval on dinh hon. Qua trinh tim huong cai thien rat mat thoi gian: toi phai thu thay doi retrieval mode, top_k_search, va ket hop rerank, sau do chay lai eval de so sanh tung lan.

Sau nhieu vong dieu chinh, variant chi len duoc muc ngang baseline o mot so cau, va tong the van chua vuot baseline theo diem trung binh. Bai hoc thuc te toi rut ra la: khong nen tuning qua nhieu bien cung luc. Neu doi retrieval mode, top_k va rerank dong thoi thi rat kho xac dinh root cause. Cach lam hieu qua hon la A/B tung bien, ghi lai tuning log ro rang, va uu tien tinh on dinh cua pipeline thay vi theo duoi cau hinh "nghe co ve manh".

---

## 4. Phan tich mot cau hoi grading cu the

**Cau hoi chon phan tich: gq05**  
"Contractor tu ben ngoai cong ty co the duoc cap quyen Admin Access khong? Neu co, can bao nhieu ngay va co yeu cau dac biet gi?"

Day la cau phan tich ro nhat su khac biet giua baseline va variant. Trong `results/ab_comparison.csv`, baseline (`baseline_dense`) dat 5/5/5/5 va tra loi day du: co the cap quyen, can IT Manager + CISO phe duyet, 5 ngay lam viec, va training bat buoc. Nguoc lai, variant (`variant_hybrid_rerank`) cho ra cau tra loi "Toi khong biet", diem faithfulness/relevance/completeness chi con 1/1/1, trong khi context_recall van 5.

Theo toi, failure mode chinh nam o **retrieval ranking stage** (hybrid + rerank), khong phai generation. Ly do: cung prompt generation va cung model LLM, baseline tra loi dung, variant moi fail. Context recall = 5 cho thay source ky vong da duoc lay ve, nhung kha nang cao la chunk duoc chon sau rerank khong chua day du 3 y can thiet (thoi gian, phe duyet, training), hoac chunk dung bi day xuong rank thap nen khong vao top_k_select. Khi context vao prompt khong du thong tin cot loi, model chon abstain de tranh hallucination.

Fix cu the toi de xuat cho cau nay: giu hybrid nhung tat rerank de kiem tra lai chat luong top_k_select, hoac bo sung rule "uu tien chunk co tu khoa dieu kien + so lieu" truoc khi gui vao LLM. Nhu vay co the giu duoc uu diem anti-hallucination ma khong mat tinh day du.

---

## 5. De xuat cai tien dua tren evidence scorecard

1) **Tach bien khi tuning va uu tien retrieval on dinh truoc rerank.** Evidence: scorecard tong the cho thay variant giam so voi baseline (Faithfulness 4.50 vs 4.80, Relevance 4.60 vs 5.00, Completeness 3.90 vs 4.20) trong khi Context Recall cung 5.00. Dieu nay cho thay rerank trong cau hinh hien tai dang lam giam chat luong chunk selection. Vong tiep theo toi se chay them cau hinh `hybrid + no rerank` va `dense + rerank` de khoanh vung root cause.

2) **Tang completeness bang answer-template theo loai cau hoi.** Evidence: ca baseline va variant deu bi tru completeness o nhom cau can nhieu chi tiet nhu gq01, gq03, gq09. Toi de xuat them huong dan generation theo checklist (vi du cau temporal phai co "hien tai" va "phien ban truoc"; cau quy trinh phai co "dieu kien + thoi gian + hanh dong") de tranh tra loi dung y chinh nhung thieu y phu. Muc tieu la giu faithfulness cao nhung day completeness len gan 4.5+.
