"""
rag_answer.py — Sprint 2 + Sprint 3: Retrieval & Grounded Answer
================================================================
Sprint 2 (60 phút): Baseline RAG
  - Dense retrieval từ ChromaDB
  - Grounded answer function với prompt ép citation
  - Trả lời được ít nhất 3 câu hỏi mẫu, output có source

Sprint 3 (60 phút): Tuning tối thiểu
  - Thêm hybrid retrieval (dense + sparse/BM25)
  - Hoặc thêm rerank (cross-encoder)
  - Hoặc thử query transformation (expansion, decomposition, HyDE)
  - Tạo bảng so sánh baseline vs variant

Definition of Done Sprint 2:
  ✓ rag_answer("SLA ticket P1?") trả về câu trả lời có citation
  ✓ rag_answer("Câu hỏi không có trong docs") trả về "Không đủ dữ liệu"

Definition of Done Sprint 3:
  ✓ Có ít nhất 1 variant (hybrid / rerank / query transform) chạy được
  ✓ Giải thích được tại sao chọn biến đó để tune
"""

import os
import json
import re
import unicodedata
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# CẤU HÌNH
# =============================================================================

TOP_K_SEARCH = 10    # Số chunk lấy từ vector store trước rerank (search rộng)
TOP_K_SELECT = 3     # Số chunk gửi vào prompt sau rerank/select (top-3 sweet spot)

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")


# =============================================================================
# RETRIEVAL — DENSE (Vector Search)
# =============================================================================

def retrieve_dense(query: str, top_k: int = TOP_K_SEARCH) -> List[Dict[str, Any]]:
    """
    Dense retrieval: tìm kiếm theo embedding similarity trong ChromaDB.

    Args:
        query: Câu hỏi của người dùng
        top_k: Số chunk tối đa trả về

    Returns:
        List các dict, mỗi dict là một chunk với:
          - "text": nội dung chunk
          - "metadata": metadata (source, section, effective_date, ...)
          - "score": cosine similarity score

    TODO Sprint 2:
    1. Embed query bằng cùng model đã dùng khi index (xem index.py)
    2. Query ChromaDB với embedding đó
    3. Trả về kết quả kèm score

    Gợi ý:
        import chromadb
        from index import get_embedding, CHROMA_DB_DIR

        client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        collection = client.get_collection("rag_lab")

        query_embedding = get_embedding(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        # Lưu ý: distances trong ChromaDB cosine = 1 - similarity
        # Score = 1 - distance
    """
    import chromadb
    from index import get_embedding, CHROMA_DB_DIR
    if not query or not query.strip():
        return []
    
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    collection = client.get_collection("rag_lab")

    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include = ["documents", "metadatas", "distances"]
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    output = []

    for doc, meta, dist in zip(documents, metadatas, distances):

        score = 1 - dist
        output.append({
            "text": doc,
            "metadata": meta,
            "score": score
        })

    output.sort(key=lambda x: x["score"], reverse=True)
    return output[:top_k]


# =============================================================================
# RETRIEVAL — SPARSE / BM25 (Keyword Search)
# Dùng cho Sprint 3 Variant hoặc kết hợp Hybrid
# =============================================================================

def retrieve_sparse(query: str, top_k: int = TOP_K_SEARCH) -> List[Dict[str, Any]]:
    """
    Sparse retrieval: tìm kiếm theo keyword (BM25).

    Mạnh ở: exact term, mã lỗi, tên riêng (ví dụ: "ERR-403", "P1", "refund")
    Hay hụt: câu hỏi paraphrase, đồng nghĩa

    TODO Sprint 3 (nếu chọn hybrid):
    1. Cài rank_bm25: pip install rank-bm25
    2. Load tất cả chunks từ ChromaDB (hoặc rebuild từ docs)
    3. Tokenize và tạo BM25Index
    4. Query và trả về top_k kết quả

    Gợi ý:
        from rank_bm25 import BM25Okapi
        corpus = [chunk["text"] for chunk in all_chunks]
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    """
    # TODO Sprint 3: Implement BM25 search
    # Tạm thời return empty list
    print("[retrieve_sparse] Chưa implement — Sprint 3")
    return []


# =============================================================================
# RETRIEVAL — HYBRID (Dense + Sparse với Reciprocal Rank Fusion)
# =============================================================================

def retrieve_hybrid(
    query: str,
    top_k: int = TOP_K_SEARCH,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval: kết hợp dense và sparse bằng Reciprocal Rank Fusion (RRF).

    Mạnh ở: giữ được cả nghĩa (dense) lẫn keyword chính xác (sparse)
    Phù hợp khi: corpus lẫn lộn ngôn ngữ tự nhiên và tên riêng/mã lỗi/điều khoản

    Args:
        dense_weight: Trọng số cho dense score (0-1)
        sparse_weight: Trọng số cho sparse score (0-1)

    TODO Sprint 3 (nếu chọn hybrid):
    1. Chạy retrieve_dense() → dense_results
    2. Chạy retrieve_sparse() → sparse_results
    3. Merge bằng RRF:
       RRF_score(doc) = dense_weight * (1 / (60 + dense_rank)) +
                        sparse_weight * (1 / (60 + sparse_rank))
       60 là hằng số RRF tiêu chuẩn
    4. Sort theo RRF score giảm dần, trả về top_k

    Khi nào dùng hybrid (từ slide):
    - Corpus có cả câu tự nhiên VÀ tên riêng, mã lỗi, điều khoản
    - Query như "Approval Matrix" khi doc đổi tên thành "Access Control SOP"
    """
    # TODO Sprint 3: Implement hybrid RRF
    # Tạm thời fallback về dense
    print("[retrieve_hybrid] Chưa implement RRF — fallback về dense")
    return retrieve_dense(query, top_k)


# =============================================================================
# RERANK (Sprint 3 alternative)
# Cross-encoder để chấm lại relevance sau search rộng
# =============================================================================

_rerank_model = None

def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = TOP_K_SELECT,
) -> List[Dict[str, Any]]:
    """
    Rerank các candidate chunks bằng cross-encoder.

    Cross-encoder: chấm lại "chunk nào thực sự trả lời câu hỏi này?"
    MMR (Maximal Marginal Relevance): giữ relevance nhưng giảm trùng lặp

    Funnel logic (từ slide):
      Search rộng (top-20) → Rerank (top-6) → Select (top-3)

    TODO Sprint 3 (nếu chọn rerank):
    Option A — Cross-encoder:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [[query, chunk["text"]] for chunk in candidates]
        scores = model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in ranked[:top_k]]

    Option B — Rerank bằng LLM (đơn giản hơn nhưng tốn token):
        Gửi list chunks cho LLM, yêu cầu chọn top_k relevant nhất

    Khi nào dùng rerank:
    - Dense/hybrid trả về nhiều chunk nhưng có noise
    - Muốn chắc chắn chỉ 3-5 chunk tốt nhất vào prompt
    """
    # TODO Sprint 3: Implement rerank
    # Tạm thời trả về top_k đầu tiên (không rerank)
    global _rerank_model

    if not candidates:
        return []
    
    if _rerank_model is None:
        from sentence_transformers import CrossEncoder
        _rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    pairs = [[query, chunk["text"]] for chunk in candidates]
    scores = _rerank_model.predict(pairs)

    for i, score in enumerate(scores):
        candidates[i]["score"] = float(score)

    ranked_candidates = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )
    final_selection = [chunk for chunk, score in ranked_candidates[:top_k]]
    return final_selection


# =============================================================================
# QUERY TRANSFORMATION (Sprint 3 alternative)
# =============================================================================

def transform_query(query: str, strategy: str = "expansion") -> List[str]:
    """Biến đổi query để tăng recall (Sprint 3 — Query Transform variant).

    Trả về danh sách query/biến thể để đưa vào retrieval.

    Strategies:
      - "expansion": thêm alias/đồng nghĩa/tên cũ
      - "decomposition": tách câu hỏi phức tạp thành 2-3 sub-queries
      - "hyde": sinh "pseudo-document" (HyDE) để embed thay query

    Ghi chú:
    - Ưu tiên gọi LLM để sinh ra JSON array (ổn định cho eval).
    - Nếu không có API key/LLM lỗi → fallback heuristic, nhưng vẫn luôn trả về tối thiểu [query].
    """

    def _norm(s: str) -> str:
        s = unicodedata.normalize("NFKC", s)
        s = s.strip().lower()
        s = re.sub(r"\s+", " ", s)
        return s

    def _dedupe_keep_order(items: List[str], max_n: int) -> List[str]:
        out: List[str] = []
        seen = set()
        for it in items:
            if not it:
                continue
            it2 = it.strip()
            if not it2:
                continue
            key = _norm(it2)
            if key in seen:
                continue
            seen.add(key)
            out.append(it2)
            if len(out) >= max_n:
                break
        return out

    def _parse_json_string_list(text: str) -> List[str]:
        """Parse LLM output into list[str]. Accepts JSON array or {"queries": [...]}.
        Falls back to extracting first [...] block; then to line/bullet splitting.
        """
        if not text:
            return []
        t = text.strip()

        def _try_load(s: str) -> List[str]:
            obj = json.loads(s)
            if isinstance(obj, list):
                return [str(x) for x in obj if str(x).strip()]
            if isinstance(obj, dict):
                for k in ("queries", "items", "alternatives", "subqueries"):
                    if k in obj and isinstance(obj[k], list):
                        return [str(x) for x in obj[k] if str(x).strip()]
            return []

        try:
            return _try_load(t)
        except Exception:
            pass

        # Try to extract JSON array substring
        try:
            start = t.find("[")
            end = t.rfind("]")
            if 0 <= start < end:
                return _try_load(t[start : end + 1])
        except Exception:
            pass

        # Fallback: split lines/bullets
        lines = []
        for line in t.splitlines():
            line = re.sub(r"^[-*•\d.\)\s]+", "", line).strip()
            if line:
                lines.append(line)
        return lines

    q = (query or "").strip()
    if not q:
        return []

    strat = (strategy or "expansion").strip().lower()

    # Heuristic alias map (bổ sung nhẹ để bắt các tên cũ/alias thường gặp trong bộ docs)
    alias_map = {
        "approval matrix": [
            "Access Control SOP",
            "quy trình cấp quyền",
            "quy trình kiểm soát truy cập",
            "ma trận phê duyệt cấp quyền",
        ],
        "access control": ["Access Control SOP", "kiểm soát truy cập", "cấp quyền"],
        "sla": ["SLA", "thời gian xử lý ticket", "thời gian phản hồi"],
        "ticket p1": ["SLA ticket P1", "P1", "ưu tiên P1"],
        "refund": ["hoàn tiền", "chính sách hoàn tiền", "policy refund v4"],
        "hoàn tiền": ["refund", "chính sách hoàn tiền", "policy refund v4"],
        "helpdesk": ["IT Helpdesk FAQ", "support helpdesk faq"],
    }

    base_candidates: List[str] = [q]
    nq = _norm(q)
    for k, vals in alias_map.items():
        if k in nq:
            base_candidates.extend(vals)

    # LLM prompts: yêu cầu trả về JSON array để parse ổn định
    def _llm_generate_list(prompt: str) -> List[str]:
        try:
            raw = call_llm(prompt)
            return _parse_json_string_list(raw)
        except Exception:
            return []

    if strat == "expansion":
        prompt = (
            "You are a query rewriting assistant for RAG retrieval. "
            "Generate 2-3 alternative phrasings / related terms for the Vietnamese query below. "
            "Focus on synonyms, aliases, and document title variants. "
            "Return ONLY a JSON array of strings.\n\n"
            f"Query: {q}"
        )
        llm_alts = _llm_generate_list(prompt)
        return _dedupe_keep_order(base_candidates + llm_alts, max_n=4)

    if strat == "decomposition":
        prompt = (
            "Decompose the Vietnamese question below into 2-3 simpler sub-queries suitable for retrieval. "
            "Each sub-query should be self-contained and short. "
            "Return ONLY a JSON array of strings.\n\n"
            f"Question: {q}"
        )
        llm_sub = _llm_generate_list(prompt)
        if llm_sub:
            return _dedupe_keep_order([q] + llm_sub, max_n=4)

        # Heuristic decomposition if no LLM
        parts = re.split(r"\s+(?:và|&|\+|/|,|;|\.)\s+", q)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) <= 1:
            return [q]
        # Keep original + first 2-3 parts
        return _dedupe_keep_order([q] + parts, max_n=4)

    if strat == "hyde":
        # HyDE: sinh pseudo-document chứa keyword nhưng tránh bịa số liệu
        prompt = (
            "Write a short pseudo-document (3-6 sentences) that would help retrieve relevant policy/FAQ text for the question below. "
            "Include likely keywords and section titles, but DO NOT invent specific numbers, dates, or penalties. "
            "If a number/date would be needed, use placeholders like <NUMBER> or <DATE>. "
            "Return ONLY a JSON array with exactly 1 string (the pseudo-document).\n\n"
            f"Question: {q}"
        )
        hyde_list = _llm_generate_list(prompt)
        hyde_doc = hyde_list[0] if hyde_list else ""
        if hyde_doc.strip():
            return _dedupe_keep_order([hyde_doc, q] + base_candidates[1:], max_n=4)
        return _dedupe_keep_order(base_candidates, max_n=4)

    raise ValueError(f"Unknown transform strategy: {strategy}")


# =============================================================================
# GENERATION — GROUNDED ANSWER FUNCTION
# =============================================================================

def build_context_block(chunks: List[Dict[str, Any]]) -> str:
    """
    Đóng gói danh sách chunks thành context block để đưa vào prompt.

    Format: structured snippets với source, section, score (từ slide).
    Mỗi chunk có số thứ tự [1], [2], ... để model dễ trích dẫn.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("source", "unknown")
        section = meta.get("section", "")
        score = chunk.get("score", 0)
        text = chunk.get("text", "")

        # TODO: Tùy chỉnh format nếu muốn (thêm effective_date, department, ...)
        header = f"[{i}] {source}"
        if section:
            header += f" | {section}"
        if score > 0:
            header += f" | score={score:.2f}"

        context_parts.append(f"{header}\n{text}")

    return "\n\n".join(context_parts)


def build_grounded_prompt(query: str, context_block: str) -> str:
    """
    Xây dựng grounded prompt theo 4 quy tắc từ slide:
    1. Evidence-only: Chỉ trả lời từ retrieved context
    2. Abstain: Thiếu context thì nói không đủ dữ liệu
    3. Citation: Gắn source/section khi có thể
    4. Short, clear, stable: Output ngắn, rõ, nhất quán

    TODO Sprint 2:
    Đây là prompt baseline. Trong Sprint 3, bạn có thể:
    - Thêm hướng dẫn về format output (JSON, bullet points)
    - Thêm ngôn ngữ phản hồi (tiếng Việt vs tiếng Anh)
    - Điều chỉnh tone phù hợp với use case (CS helpdesk, IT support)
    """
    prompt = f"""Answer only from the retrieved context below.
If the context is insufficient to answer the question, say you do not know and do not make up information.
Cite the source field (in brackets like [1]) when possible.
Keep your answer short, clear, and factual.
Respond in the same language as the question.

Question: {query}

Context:
{context_block}

Answer:"""
    return prompt


def call_llm(prompt: str) -> str:
    """
    Gọi LLM để sinh câu trả lời.

    TODO Sprint 2:
    Chọn một trong hai:

    Option A — OpenAI (cần OPENAI_API_KEY):
    """
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model= "gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,     # temperature=0 để output ổn định, dễ đánh giá
        max_tokens=512,
    )
    return response.choices[0].message.content

def rag_answer(
    query: str,
    retrieval_mode: str = "dense",
    top_k_search: int = TOP_K_SEARCH,
    top_k_select: int = TOP_K_SELECT,
    use_rerank: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Pipeline RAG hoàn chỉnh: query → retrieve → (rerank) → generate.

    Args:
        query: Câu hỏi
        retrieval_mode: "dense" | "sparse" | "hybrid"
        top_k_search: Số chunk lấy từ vector store (search rộng)
        top_k_select: Số chunk đưa vào prompt (sau rerank/select)
        use_rerank: Có dùng cross-encoder rerank không
        verbose: In thêm thông tin debug

    Returns:
        Dict với:
          - "answer": câu trả lời grounded
          - "sources": list source names trích dẫn
          - "chunks_used": list chunks đã dùng
          - "query": query gốc
          - "config": cấu hình pipeline đã dùng

    TODO Sprint 2 — Implement pipeline cơ bản:
    1. Chọn retrieval function dựa theo retrieval_mode
    2. Gọi rerank() nếu use_rerank=True
    3. Truncate về top_k_select chunks
    4. Build context block và grounded prompt
    5. Gọi call_llm() để sinh câu trả lời
    6. Trả về kết quả kèm metadata

    TODO Sprint 3 — Thử các variant:
    - Variant A: đổi retrieval_mode="hybrid"
    - Variant B: bật use_rerank=True
    - Variant C: thêm query transformation trước khi retrieve
    """
    config = {
        "retrieval_mode": retrieval_mode,
        "top_k_search": top_k_search,
        "top_k_select": top_k_select,
        "use_rerank": use_rerank,
    }

    # --- Bước 1: Retrieve ---
    if retrieval_mode == "dense":
        candidates = retrieve_dense(query, top_k=top_k_search)
    elif retrieval_mode == "sparse":
        candidates = retrieve_sparse(query, top_k=top_k_search)
    elif retrieval_mode == "hybrid":
        candidates = retrieve_hybrid(query, top_k=top_k_search)
    else:
        raise ValueError(f"retrieval_mode không hợp lệ: {retrieval_mode}")

    if verbose:
        print(f"\n[RAG] Query: {query}")
        print(f"[RAG] Retrieved {len(candidates)} candidates (mode={retrieval_mode})")
        for i, c in enumerate(candidates[:3]):
            print(f"  [{i+1}] score={c.get('score', 0):.3f} | {c['metadata'].get('source', '?')}")

    # --- Bước 2: Rerank (optional) ---
    if use_rerank:
        candidates = rerank(query, candidates, top_k=top_k_select)
    else:
        candidates = candidates[:top_k_select]

    if verbose:
        print(f"[RAG] After select: {len(candidates)} chunks")

    # --- Bước 3: Build context và prompt ---
    context_block = build_context_block(candidates)
    prompt = build_grounded_prompt(query, context_block)

    if verbose:
        print(f"\n[RAG] Prompt:\n{prompt[:500]}...\n")

    # --- Bước 4: Generate ---
    answer = call_llm(prompt)

    # --- Bước 5: Extract sources ---
    sources = list({
        c["metadata"].get("source", "unknown")
        for c in candidates
    })

    return {
        "query": query,
        "answer": answer,
        "sources": sources,
        "chunks_used": candidates,
        "config": config,
    }


# =============================================================================
# SPRINT 3: SO SÁNH BASELINE VS VARIANT
# =============================================================================

def compare_retrieval_strategies(query: str) -> None:
    """
    So sánh các retrieval strategies với cùng một query.

    TODO Sprint 3:
    Chạy hàm này để thấy sự khác biệt giữa dense, sparse, hybrid.
    Dùng để justify tại sao chọn variant đó cho Sprint 3.

    A/B Rule (từ slide): Chỉ đổi MỘT biến mỗi lần.
    """
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print('='*60)

    strategies = ["dense", "hybrid"]  # Thêm "sparse" sau khi implement

    for strategy in strategies:
        print(f"\n--- Strategy: {strategy} ---")
        try:
            result = rag_answer(query, retrieval_mode=strategy, verbose=False)
            print(f"Answer: {result['answer']}")
            print(f"Sources: {result['sources']}")
        except NotImplementedError as e:
            print(f"Chưa implement: {e}")
        except Exception as e:
            print(f"Lỗi: {e}")



# =============================================================================
# MAIN — Demo và Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Sprint 2 + 3: RAG Answer Pipeline")
    print("=" * 60)

    # Test queries từ data/test_questions.json
    test_queries = [
        "SLA xử lý ticket P1 là bao lâu?",
        "Khách hàng có thể yêu cầu hoàn tiền trong bao nhiêu ngày?",
        "Ai phải phê duyệt để cấp quyền Level 3?",
        "ERR-403-AUTH là lỗi gì?",  # Query không có trong docs → kiểm tra abstain
    ]

    print("\n--- Sprint 2: Test Baseline (Dense) ---")
    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            result = rag_answer(query, retrieval_mode="dense", verbose=True)
            print(f"Answer: {result['answer']}")
            print(f"Sources: {result['sources']}")
        except NotImplementedError:
            print("Chưa implement — hoàn thành TODO trong retrieve_dense() và call_llm() trước.")
        except Exception as e:
            print(f"Lỗi: {e}")

    # Uncomment sau khi Sprint 3 hoàn thành:
    # print("\n--- Sprint 3: So sánh strategies ---")
    # compare_retrieval_strategies("Approval Matrix để cấp quyền là tài liệu nào?")
    # compare_retrieval_strategies("ERR-403-AUTH")

    print("\n\nViệc cần làm Sprint 2:")
    print("  1. Implement retrieve_dense() — query ChromaDB")
    print("  2. Implement call_llm() — gọi OpenAI hoặc Gemini")
    print("  3. Chạy rag_answer() với 3+ test queries")
    print("  4. Verify: output có citation không? Câu không có docs → abstain không?")

    print("\nViệc cần làm Sprint 3:")
    print("  1. Chọn 1 trong 3 variants: hybrid, rerank, hoặc query transformation")
    print("  2. Implement variant đó")
    print("  3. Chạy compare_retrieval_strategies() để thấy sự khác biệt")
    print("  4. Ghi lý do chọn biến đó vào docs/tuning-log.md")
