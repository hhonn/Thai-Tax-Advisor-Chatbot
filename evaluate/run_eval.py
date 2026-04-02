from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.rag_chain import answer_question

def tokenize_thai(text: str) -> List[str]:
    try:
        from pythainlp.tokenize import word_tokenize
        return word_tokenize(text, engine="newmm")
    except ImportError:
        return text.split()

def normalize_text(text: str) -> str:
    return " ".join(text.split()).strip()


def token_set(text: str) -> set[str]:
    return {t.lower() for t in tokenize_thai(normalize_text(text)) if t.strip()}


def safe_mean(values: List[float]) -> float:
    return float(mean(values)) if values else 0.0


def keyword_recall(answer: str, expected_keywords: List[str]) -> float:
    if not expected_keywords:
        return 0.0
    ans = answer.lower()
    hits = sum(1 for k in expected_keywords if k.lower() in ans)
    return hits / len(expected_keywords)


def faithfulness_proxy(answer: str, contexts: List[str]) -> float:
    if not answer.strip() or not contexts:
        return 0.0
    context_tokens = token_set(" ".join(contexts))
    if not context_tokens:
        return 0.0

    sentences = [s.strip() for s in answer.replace("\n", " ").split(".") if s.strip()]
    if not sentences:
        return 0.0

    grounded = 0
    for sent in sentences:
        sent_tokens = token_set(sent)
        if sent_tokens and len(sent_tokens.intersection(context_tokens)) > 0:
            grounded += 1
    return grounded / len(sentences)


def answer_relevancy_proxy(question: str, answer: str) -> float:
    q_tokens = token_set(question)
    a_tokens = token_set(answer)
    if not q_tokens or not a_tokens:
        return 0.0
    return len(q_tokens.intersection(a_tokens)) / len(q_tokens)


def compute_source_metrics(relevant_sources: List[str], retrieved_sources: List[str]) -> Dict[str, float]:
    rel = set(relevant_sources)
    got = set(retrieved_sources)

    if not rel:
        return {
            "source_hit": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
        }

    intersection = rel.intersection(got)
    source_hit = 1.0 if intersection else 0.0
    context_precision = len(intersection) / len(got) if got else 0.0
    context_recall = len(intersection) / len(rel) if rel else 0.0

    return {
        "source_hit": source_hit,
        "context_precision": context_precision,
        "context_recall": context_recall,
    }


def _to_float_or_zero(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list):
        nums = [float(v) for v in value if isinstance(v, (int, float))]
        return safe_mean(nums)
    try:
        return float(value)
    except Exception:
        return 0.0


def _extract_ragas_metric(result: Any, metric_name: str) -> float:
    try:
        value = result[metric_name]
        return _to_float_or_zero(value)
    except Exception:
        pass

    scores_attr = getattr(result, "scores", None)
    if isinstance(scores_attr, list):
        metric_values: List[float] = []
        for row in scores_attr:
            if isinstance(row, dict) and metric_name in row:
                metric_values.append(_to_float_or_zero(row[metric_name]))
        if metric_values:
            return safe_mean(metric_values)

    to_pandas = getattr(result, "to_pandas", None)
    if callable(to_pandas):
        try:
            df: Any = to_pandas()
            series = df[metric_name]
            values = [
                _to_float_or_zero(v)
                for v in series.dropna().tolist()
            ]
            return safe_mean(values)
        except Exception:
            pass

    return 0.0


def run_ragas_if_enabled(rows: List[dict], output_rows: List[dict], enable_ragas: bool) -> Dict[str, Any]:
    if not enable_ragas:
        return {"enabled": False}

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

        ds = Dataset.from_dict(
            {
                "question": [r["question"] for r in rows],
                "answer": [o["answer"] for o in output_rows],
                "contexts": [o["contexts"] for o in output_rows],
                "ground_truth": [r.get("reference_answer", "") for r in rows],
            }
        )

        result = evaluate(
            ds,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )

        return {
            "enabled": True,
            "scores": {
                "faithfulness": _extract_ragas_metric(result, "faithfulness"),
                "answer_relevancy": _extract_ragas_metric(result, "answer_relevancy"),
                "context_precision": _extract_ragas_metric(result, "context_precision"),
                "context_recall": _extract_ragas_metric(result, "context_recall"),
            },
        }
    except Exception as exc:
        return {
            "enabled": True,
            "error": str(exc),
            "note": "RAGAS run failed. Check API keys and ragas dependency compatibility.",
        }


def main() -> None:
    ragas_default = os.getenv("ENABLE_RAGAS", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }

    parser = argparse.ArgumentParser(description="Evaluate Thai Tax RAG backend")
    parser.add_argument("--testset", type=Path, default=Path("evaluate/testset.json"))
    parser.add_argument("--output", type=Path, default=Path("evaluate/results/eval_report.json"))
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument(
        "--enable-ragas",
        action="store_true",
        default=ragas_default,
        help="Enable RAGAS metrics. Default reads ENABLE_RAGAS env var.",
    )
    args = parser.parse_args()

    if not args.testset.exists():
        raise FileNotFoundError(f"Testset not found: {args.testset}")

    with args.testset.open("r", encoding="utf-8") as f:
        test_rows = json.load(f)

    per_case: List[dict] = []
    keyword_scores: List[float] = []
    faithfulness_scores: List[float] = []
    relevancy_scores: List[float] = []
    source_hit_scores: List[float] = []
    context_precision_scores: List[float] = []
    context_recall_scores: List[float] = []

    for row in test_rows:
        question = row["question"]
        expected_keywords = row.get("expected_keywords", [])
        relevant_sources = row.get("relevant_sources", [])

        # Updated RAG inference
        answer, context_str = answer_question(question)
        contexts = [context_str]

        # Extract mock sources blindly - in reality we could parse `context_str` for sources
        import re
        retrieved_sources = re.findall(r"\[อ้างอิง: (.*?)\]", context_str)

        kw = keyword_recall(answer, expected_keywords)
        faith = faithfulness_proxy(answer, contexts)
        rel = answer_relevancy_proxy(question, answer)
        source_metrics = compute_source_metrics(relevant_sources, retrieved_sources)

        keyword_scores.append(kw)
        faithfulness_scores.append(faith)
        relevancy_scores.append(rel)
        source_hit_scores.append(source_metrics["source_hit"])
        context_precision_scores.append(source_metrics["context_precision"])
        context_recall_scores.append(source_metrics["context_recall"])

        per_case.append(
            {
                "id": row.get("id", ""),
                "question": question,
                "answer": answer,
                "expected_keywords": expected_keywords,
                "relevant_sources": relevant_sources,
                "retrieved_sources": retrieved_sources,
                "metrics": {
                    "keyword_recall": kw,
                    "faithfulness_proxy": faith,
                    "answer_relevancy_proxy": rel,
                    **source_metrics,
                },
                "contexts": contexts,
            }
        )

    summary = {
        "cases": len(per_case),
        "keyword_recall_avg": round(safe_mean(keyword_scores), 4),
        "faithfulness_proxy_avg": round(safe_mean(faithfulness_scores), 4),
        "answer_relevancy_proxy_avg": round(safe_mean(relevancy_scores), 4),
        "source_hit_rate": round(safe_mean(source_hit_scores), 4),
        "context_precision_avg": round(safe_mean(context_precision_scores), 4),
        "context_recall_avg": round(safe_mean(context_recall_scores), 4),
    }

    ragas_result = run_ragas_if_enabled(test_rows, per_case, args.enable_ragas)

    report = {
        "summary": summary,
        "ragas": ragas_result,
        "details": per_case,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("Evaluation complete")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if ragas_result.get("enabled"):
        print("RAGAS:", json.dumps(ragas_result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
