
import json
import os
import re
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm
from rag_answer import rag_answer

# =============================================================================
# CẤU HÌNH
# =============================================================================

TEST_QUESTIONS_PATH = Path(__file__).parent / "data" / "test_questions.json"
GRADING_TESTING_PATH = Path(__file__).parent / "data" / "grading_testing.json"
RESULTS_DIR = Path(__file__).parent / "results"

# Cấu hình baseline (Sprint 2)
BASELINE_CONFIG = {
    "retrieval_mode": "dense",
    "top_k_search": 10,
    "top_k_select": 3,
    "use_rerank": False,
    "label": "baseline_dense",
}

# TODO Sprint 4: Cập nhật VARIANT_CONFIG theo variant nhóm đã implement
VARIANT_CONFIG = {
    "retrieval_mode": "hybrid",   
    "top_k_search": 10,
    "top_k_select": 3,
    "use_rerank": True,           
    "label": "variant_hybrid_rerank",
}

load_dotenv()

# =============================================================================
# SCORING FUNCTIONS VỚI LLM-AS-JUDGE (GỘP CHUNG ĐỂ TỐI ƯU TỐC ĐỘ)
# =============================================================================

def call_qwen_judge(prompt: str) -> Dict[str, Any]:
    """Gọi API Qwen và trả về JSON."""
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("QWEN_API_KEY"),
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1" 
    )
    system_prompt = """Bạn là một giám khảo chuyên môn cao, đánh giá khắt khe hệ thống RAG. 
    BẠN PHẢI TRẢ VỀ ĐỊNH DẠNG JSON CHUẨN XÁC: {"score": <int>, "notes": "<string>"}
    Không trả về bất kỳ text nào khác ngoài JSON."""
    try:
        response = client.chat.completions.create(
            model="qwen-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        content = response.choices[0].message.content or ""
        content_cleaned = re.sub(r"```json\n|\n```|```", "", content).strip()
        return json.loads(content_cleaned)
    except Exception as e:
        print(f"\n[LLM Judge Error] Lỗi: {e}")
        return {}


def score_all_metrics(
    query: str, 
    answer: str, 
    expected_answer: str, 
    chunks_used: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Hàm gộp gọi Qwen 1 lần để chấm 3 tiêu chí"""
    context_text = "\n\n".join([f"- {c.get('text', '')}" for c in chunks_used])
    
    prompt = f"""Đánh giá câu trả lời RAG theo 3 tiêu chí: Faithfulness, Relevance, Completeness. Thang điểm 1-5.

    Ngữ cảnh: {context_text if context_text else "(Không có ngữ cảnh)"}
    Câu hỏi: {query}
    Đáp án mẫu: {expected_answer if expected_answer else "(Không có)"}
    Câu trả lời RAG: {answer}

    Hướng dẫn chấm:
    1. Faithfulness: 5 là bám sát ngữ cảnh hoàn toàn. 1 là bịa đặt. (Nếu trả lời 'không đủ dữ liệu' khi ngữ cảnh rỗng -> 5).
    2. Relevance: 5 là trả lời đúng trọng tâm. 1 là lạc đề.
    3. Completeness: 5 là bao phủ đủ ý của đáp án mẫu. 1 là thiếu sót quá nhiều. (Nếu không có đáp án mẫu -> 5).

    Trả về JSON ĐÚNG cấu trúc sau:
    {{
      "faithfulness": {{"score": <int 1-5>, "notes": "<lý do ngắn>"}},
      "relevance": {{"score": <int 1-5>, "notes": "<lý do ngắn>"}},
      "completeness": {{"score": <int 1-5>, "notes": "<lý do ngắn>"}}
    }}"""

    result = call_qwen_judge(prompt)
    return {
        "faithfulness": result.get("faithfulness", {"score": 1, "notes": "Lỗi API"}),
        "relevance": result.get("relevance", {"score": 1, "notes": "Lỗi API"}),
        "completeness": result.get("completeness", {"score": 1, "notes": "Lỗi API"})
    }


def score_context_recall(chunks_used: List[Dict[str, Any]], expected_sources: List[str]) -> Dict[str, Any]:
    if not expected_sources:
        return {"score": None, "recall": None, "notes": "No expected sources"}

    retrieved_sources = {c.get("metadata", {}).get("source", "") for c in chunks_used}
    found = 0
    missing = []
    
    for expected in expected_sources:
        expected_name = expected.split("/")[-1].replace(".pdf", "").replace(".md", "")
        matched = any(expected_name.lower() in r.lower() for r in retrieved_sources)
        if matched:
            found += 1
        else:
            missing.append(expected)

    recall = found / len(expected_sources) if expected_sources else 0
    return {
        "score": round(recall * 5),  
        "recall": recall,
        "found": found,
        "missing": missing,
        "notes": f"Retrieved: {found}/{len(expected_sources)} expected sources" +
                 (f". Missing: {missing}" if missing else ""),
    }

# =============================================================================
# SCORECARD RUNNER
# =============================================================================

def run_scorecard(
    config: Dict[str, Any],
    test_questions: Optional[List[Dict]] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    
    if test_questions is None:
        with open(TEST_QUESTIONS_PATH, "r", encoding="utf-8") as f:
            test_questions = json.load(f)

    results = []
    label = config.get("label", "unnamed")

    # GIỮ NGUYÊN FORMAT IN
    print(f"\n{'='*70}")
    print(f"Chạy scorecard: {label}")
    print(f"Config: {config}")
    print('='*70)

    # Vẫn dùng tqdm nhưng bọc bên ngoài
    for q in tqdm(test_questions, desc="Processing", disable=verbose):
        question_id = q["id"]
        query = q["question"]
        expected_answer = q.get("expected_answer", "")
        expected_sources = q.get("expected_sources", [])
        category = q.get("category", "")

        # GIỮ NGUYÊN FORMAT IN
        if verbose:
            print(f"\n[{question_id}] {query}")

        try:
            result = rag_answer(
                query=query,
                retrieval_mode=config.get("retrieval_mode", "dense"),
                top_k_search=config.get("top_k_search", 10),
                top_k_select=config.get("top_k_select", 3),
                use_rerank=config.get("use_rerank", False),
                verbose=False,
            )
            answer = result["answer"]
            chunks_used = result["chunks_used"]

        except NotImplementedError:
            answer = "PIPELINE_NOT_IMPLEMENTED"
            chunks_used = []
        except Exception as e:
            answer = f"ERROR: {e}"
            chunks_used = []

        # Gọi các hàm chấm điểm
        recall = score_context_recall(chunks_used, expected_sources)
        llm_scores = score_all_metrics(query, answer, expected_answer, chunks_used)
        
        faith_score = llm_scores.get("faithfulness", {}).get("score", 1)
        rel_score = llm_scores.get("relevance", {}).get("score", 1)
        comp_score = llm_scores.get("completeness", {}).get("score", 1)

        row = {
            "id": question_id,
            "category": category,
            "query": query,
            "answer": answer,
            "expected_answer": expected_answer,
            "faithfulness": faith_score,
            "faithfulness_notes": llm_scores.get("faithfulness", {}).get("notes", ""),
            "relevance": rel_score,
            "relevance_notes": llm_scores.get("relevance", {}).get("notes", ""),
            "context_recall": recall["score"],
            "context_recall_notes": recall["notes"],
            "completeness": comp_score,
            "completeness_notes": llm_scores.get("completeness", {}).get("notes", ""),
            "config_label": label,
        }
        results.append(row)

        # GIỮ NGUYÊN FORMAT IN
        if verbose:
            print(f"  Answer: {answer[:100]}...")
            print(f"  Faithful: {faith_score} | Relevant: {rel_score} | "
                  f"Recall: {recall['score']} | Complete: {comp_score}")

    # GIỮ NGUYÊN FORMAT IN AVERAGES
    for metric in ["faithfulness", "relevance", "context_recall", "completeness"]:
        scores = [r[metric] for r in results if r[metric] is not None]
        avg = sum(scores) / len(scores) if scores else None
        print(f"\nAverage {metric}: {avg:.2f}" if avg else f"\nAverage {metric}: N/A (chưa chấm)")

    return results


def load_questions(path: Path) -> List[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            questions = json.load(f)
        print(f"\nLoading test questions từ: {path}")
        print(f"Tìm thấy {len(questions)} câu hỏi")
        for q in questions[:3]:
            print(f"  [{q['id']}] {q['question']} ({q.get('category', '')})")
        print("  ...")
        return questions
    except FileNotFoundError:
        print(f"\nKhông tìm thấy file: {path}")
        return []


# =============================================================================
# A/B COMPARISON
# =============================================================================

def compare_ab(
    baseline_results: List[Dict],
    variant_results: List[Dict],
    output_csv: Optional[str] = None,
) -> None:
    metrics = ["faithfulness", "relevance", "context_recall", "completeness"]

    # GIỮ NGUYÊN FORMAT IN BẢNG A/B TỔNG THỂ
    print(f"\n{'='*70}")
    print("A/B Comparison: Baseline vs Variant")
    print('='*70)
    print(f"{'Metric':<20} {'Baseline':>10} {'Variant':>10} {'Delta':>8}")
    print("-" * 55)

    for metric in metrics:
        b_scores = [r[metric] for r in baseline_results if r[metric] is not None]
        v_scores = [r[metric] for r in variant_results if r[metric] is not None]

        b_avg = sum(b_scores) / len(b_scores) if b_scores else None
        v_avg = sum(v_scores) / len(v_scores) if v_scores else None
        delta = (v_avg - b_avg) if (b_avg is not None and v_avg is not None) else None

        b_str = f"{b_avg:.2f}" if b_avg is not None else "N/A"
        v_str = f"{v_avg:.2f}" if v_avg is not None else "N/A"
        d_str = f"{delta:+.2f}" if delta is not None else "N/A"

        print(f"{metric:<20} {b_str:>10} {v_str:>10} {d_str:>8}")

    # GIỮ NGUYÊN FORMAT IN SO SÁNH TỪNG CÂU
    print(f"\n{'Câu':<6} {'Baseline F/R/Rc/C':<22} {'Variant F/R/Rc/C':<22} {'Better?':<10}")
    print("-" * 65)

    b_by_id = {r["id"]: r for r in baseline_results}
    for v_row in variant_results:
        qid = v_row["id"]
        b_row = b_by_id.get(qid, {})

        b_scores_str = "/".join([str(b_row.get(m, "?")) for m in metrics])
        v_scores_str = "/".join([str(v_row.get(m, "?")) for m in metrics])

        b_total = sum(b_row.get(m, 0) or 0 for m in metrics)
        v_total = sum(v_row.get(m, 0) or 0 for m in metrics)
        better = "Variant" if v_total > b_total else ("Baseline" if b_total > v_total else "Tie")

        print(f"{qid:<6} {b_scores_str:<22} {v_scores_str:<22} {better:<10}")

    if output_csv:
        csv_path = RESULTS_DIR / output_csv
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        combined = baseline_results + variant_results
        if combined:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=combined[0].keys())
                writer.writeheader()
                writer.writerows(combined)
            print(f"\nKết quả đã lưu vào: {csv_path}")


# =============================================================================
# REPORT GENERATOR
# =============================================================================

def generate_scorecard_summary(results: List[Dict], label: str) -> str:
    """GIỮ NGUYÊN CHÍNH XÁC STRING FORMATTING GỐC"""
    metrics = ["faithfulness", "relevance", "context_recall", "completeness"]
    averages = {}
    for metric in metrics:
        scores = [r[metric] for r in results if r[metric] is not None]
        averages[metric] = sum(scores) / len(scores) if scores else None

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    md = f"""# Scorecard: {label}
Generated: {timestamp}

## Summary

| Metric | Average Score |
|--------|--------------|
"""
    for metric, avg in averages.items():
        avg_str = f"{avg:.2f}/5" if avg else "N/A"
        md += f"| {metric.replace('_', ' ').title()} | {avg_str} |\n"

    md += "\n## Per-Question Results\n\n"
    md += "| ID | Category | Faithful | Relevant | Recall | Complete | Notes |\n"
    md += "|----|----------|----------|----------|--------|----------|-------|\n"

    for r in results:
        md += (f"| {r['id']} | {r['category']} | {r.get('faithfulness', 'N/A')} | "
               f"{r.get('relevance', 'N/A')} | {r.get('context_recall', 'N/A')} | "
               f"{r.get('completeness', 'N/A')} | {str(r.get('faithfulness_notes', ''))[:50]} |\n")

    return md


# =============================================================================
# MAIN — Chạy evaluation
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Sprint 4: Evaluation & Scorecard")
    print("=" * 60)

    datasets = {
        "test_questions": TEST_QUESTIONS_PATH,
        "grading_testing": GRADING_TESTING_PATH,
    }

    for dataset_name, dataset_path in datasets.items():
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*70}")

        test_questions = load_questions(dataset_path)
        dataset_results_dir = RESULTS_DIR / dataset_name
        dataset_results_dir.mkdir(parents=True, exist_ok=True)

        print("\n--- Chạy Baseline ---")
        try:
            baseline_results = run_scorecard(config=BASELINE_CONFIG, test_questions=test_questions, verbose=True)
            baseline_md = generate_scorecard_summary(baseline_results, f"{dataset_name}_baseline_dense")
            baseline_path = dataset_results_dir / "scorecard_baseline.md"
            baseline_path.write_text(baseline_md, encoding="utf-8")
            print(f"\nScorecard lưu tại: {baseline_path}")
        except NotImplementedError:
            print("Pipeline chưa implement. Hoàn thành Sprint 2 trước.")
            baseline_results = []

        print("\n--- Chạy Variant ---")
        variant_results = run_scorecard(config=VARIANT_CONFIG, test_questions=test_questions, verbose=True)
        variant_md = generate_scorecard_summary(variant_results, f"{dataset_name}_{VARIANT_CONFIG['label']}")
        variant_path = dataset_results_dir / "scorecard_variant.md"
        variant_path.write_text(variant_md, encoding="utf-8")
        print(f"\nScorecard lưu tại: {variant_path}")

        if baseline_results and variant_results:
            compare_ab(
                baseline_results,
                variant_results,
                output_csv=f"{dataset_name}/ab_comparison.csv",
            )
