"""
Information Retrieval Evaluation for Thai Tax Advisor RAG pipeline.

Metrics
-------
Recall@k      – proportion of queries where ≥1 relevant doc is in top-k
Precision@k   – fraction of top-k retrieved docs that are relevant
MRR           – Mean Reciprocal Rank (rank of first relevant doc)
nDCG@k        – normalized Discounted Cumulative Gain
mAP           – Mean Average Precision (summary metric across all queries)

Usage
-----
    python evaluate/eval_ir.py
    python evaluate/eval_ir.py --k-values 1 3 5 10 --top-k 10 --output evaluate/results/ir_report.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv

load_dotenv()

# ── Relevance Judgment ────────────────────────────────────────────────────────

def _is_relevant(
    page_content: str,
    metadata: dict,
    expected_keywords: List[str],
    relevant_sources: List[str],
) -> bool:
    """
    A retrieved chunk is considered relevant when:
      (a) ALL expected_keywords appear (case-insensitive) in the chunk text, OR
      (b) the chunk's source / tax_id metadata overlaps with relevant_sources.

    Using ALL-keyword matching avoids false positives from single common words.
    """
    text = page_content.lower()

    if expected_keywords:
        keyword_hit = all(kw.lower() in text for kw in expected_keywords)
    else:
        keyword_hit = False

    if relevant_sources:
        meta_values = {
            metadata.get("source", ""),
            metadata.get("tax_id", ""),
        }
        source_hit = bool(set(relevant_sources) & meta_values)
    else:
        source_hit = False

    return keyword_hit or source_hit


def _build_relevance_list(
    docs: list,
    expected_keywords: List[str],
    relevant_sources: List[str],
    k: int,
) -> List[int]:
    """Return a length-k binary relevance list (1 = relevant, 0 = not)."""
    return [
        int(_is_relevant(doc.page_content, doc.metadata, expected_keywords, relevant_sources))
        for doc in docs[:k]
    ]


# ── IR Metric Implementations ─────────────────────────────────────────────────

def recall_at_k(relevance: List[int]) -> float:
    """1.0 if at least one relevant document appears in the ranked list, else 0.0."""
    return 1.0 if any(r == 1 for r in relevance) else 0.0


def precision_at_k(relevance: List[int]) -> float:
    """Fraction of retrieved documents that are relevant."""
    if not relevance:
        return 0.0
    return sum(relevance) / len(relevance)


def reciprocal_rank(relevance: List[int]) -> float:
    """1 / rank of the first relevant document; 0.0 if none found."""
    for rank, r in enumerate(relevance, start=1):
        if r == 1:
            return 1.0 / rank
    return 0.0


def dcg_at_k(relevance: List[int]) -> float:
    """Discounted Cumulative Gain with log_2 discounting."""
    return sum(r / math.log2(i + 1) for i, r in enumerate(relevance, start=1))


def ndcg_at_k(relevance: List[int]) -> float:
    """
    Normalized DCG.  The ideal ranking puts all relevant documents first.
    Returns 0.0 when no relevant document exists in the list.
    """
    ideal = sorted(relevance, reverse=True)
    ideal_dcg = dcg_at_k(ideal)
    if ideal_dcg == 0.0:
        return 0.0
    return dcg_at_k(relevance) / ideal_dcg


def average_precision(relevance: List[int]) -> float:
    """
    Average Precision for a single query.

    AP = (1 / R) * sum_{i: rel(i)=1} Precision@i

    where R = number of relevant documents retrieved (used because the total
    number of relevant documents in the corpus is not known).
    Returns 0.0 when no relevant document was retrieved.
    """
    total_relevant = sum(relevance)
    if total_relevant == 0:
        return 0.0
    hits = 0
    sum_prec = 0.0
    for i, r in enumerate(relevance, start=1):
        if r == 1:
            hits += 1
            sum_prec += hits / i
    return sum_prec / total_relevant


# ── Per-query Evaluation ──────────────────────────────────────────────────────

def evaluate_query(
    docs: list,
    expected_keywords: List[str],
    relevant_sources: List[str],
    k_values: List[int],
) -> Dict:
    """
    Compute all IR metrics for a single query at every requested k.

    Parameters
    ----------
    docs            : ranked list of retrieved Document objects (length = max(k_values))
    expected_keywords / relevant_sources : relevance criteria from testset
    k_values        : list of cut-off depths, e.g. [1, 3, 5, 10]
    """
    max_k = max(k_values)
    full_relevance = _build_relevance_list(docs, expected_keywords, relevant_sources, max_k)

    per_k: Dict[int, Dict[str, float]] = {}
    for k in k_values:
        rel_k = full_relevance[:k]
        per_k[k] = {
            "recall": round(recall_at_k(rel_k), 4),
            "precision": round(precision_at_k(rel_k), 4),
            "ndcg": round(ndcg_at_k(rel_k), 4),
            "ap": round(average_precision(rel_k), 4),
        }

    return {
        "mrr": round(reciprocal_rank(full_relevance), 4),
        "relevance_flags": full_relevance,
        "per_k": per_k,
    }


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate(per_query_results: List[Dict], k_values: List[int]) -> Dict:
    """
    Aggregate per-query IR metrics into dataset-level averages.
    Returns mAP (at the largest k) as the primary summary metric.
    """
    mrr_scores = [r["mrr"] for r in per_query_results]
    summary: Dict = {
        "num_queries": len(per_query_results),
        "MRR": round(mean(mrr_scores), 4),
        "per_k": {},
    }

    for k in k_values:
        recalls = [r["per_k"][k]["recall"] for r in per_query_results]
        precisions = [r["per_k"][k]["precision"] for r in per_query_results]
        ndcgs = [r["per_k"][k]["ndcg"] for r in per_query_results]
        aps = [r["per_k"][k]["ap"] for r in per_query_results]
        summary["per_k"][f"@{k}"] = {
            "Recall": round(mean(recalls), 4),
            "Precision": round(mean(precisions), 4),
            "nDCG": round(mean(ndcgs), 4),
            "AP": round(mean(aps), 4),
        }

    max_k = max(k_values)
    summary["mAP"] = summary["per_k"][f"@{max_k}"]["AP"]
    return summary


# ── Pretty Printing ───────────────────────────────────────────────────────────

def _print_table(summary: Dict, k_values: List[int]) -> None:
    col_w = 12
    ks = [f"@{k}" for k in k_values]
    metrics = ["Recall", "Precision", "nDCG", "AP"]

    header = f"{'Metric':<14}" + "".join(f"{k:>{col_w}}" for k in ks)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for metric in metrics:
        row = f"{metric:<14}"
        for k_label in ks:
            row += f"{summary['per_k'][k_label][metric]:>{col_w}.4f}"
        print(row)
    print("-" * len(header))
    print(f"{'MRR':<14}{'':>{col_w * (len(ks) - 1)}}{summary['MRR']:>{col_w}.4f}")
    print("=" * len(header))
    print(f"\n  Summary metric  →  mAP@{max(k_values)} = {summary['mAP']:.4f}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="IR evaluation for Thai Tax RAG")
    parser.add_argument(
        "--testset", type=Path, default=Path("evaluate/testset.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("evaluate/results/ir_report.json")
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[1, 3, 5, 10],
        metavar="K",
        help="Cutoff depths to evaluate (default: 1 3 5 10)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Documents to retrieve per query. Defaults to max(k_values).",
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default="data/vector_store",
        help="ChromaDB persist directory (must match build_index output)",
    )
    parser.add_argument(
        "--ollama-base-url",
        type=str,
        default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    args = parser.parse_args()

    top_k = args.top_k or max(args.k_values)

    if not args.testset.exists():
        raise FileNotFoundError(f"Testset not found: {args.testset}")

    with args.testset.open("r", encoding="utf-8") as f:
        test_rows: List[dict] = json.load(f)

    print(f"Loaded {len(test_rows)} queries from {args.testset}")
    print(f"Evaluating at k ∈ {args.k_values}  (retrieving top-{top_k} per query)")

    # ── Init retriever ──────────────────────────────────────────────────────
    from langchain_ollama import OllamaEmbeddings
    from app.services.rag_service import RAGService

    embeddings = OllamaEmbeddings(
        model="all-minilm",
        base_url=args.ollama_base_url,
    )
    rag = RAGService(embeddings=embeddings, persist_dir=args.persist_dir)

    # ── Run per-query evaluation ────────────────────────────────────────────
    per_query: List[dict] = []

    for row in test_rows:
        qid = row.get("id", "?")
        question = row["question"]
        expected_keywords = row.get("expected_keywords", [])
        relevant_sources = row.get("relevant_sources", [])

        docs_with_scores = rag.similarity_search_with_scores(question, k=top_k)
        docs = [d for d, _ in docs_with_scores]
        scores = [s for _, s in docs_with_scores]

        result = evaluate_query(docs, expected_keywords, relevant_sources, args.k_values)

        per_query.append(
            {
                "id": qid,
                "question": question,
                "expected_keywords": expected_keywords,
                "relevant_sources": relevant_sources,
                "retrieved": [
                    {
                        "rank": i + 1,
                        "score": round(scores[i], 4) if i < len(scores) else None,
                        "relevant": bool(result["relevance_flags"][i]),
                        "tax_id": docs[i].metadata.get("tax_id", ""),
                        "source": docs[i].metadata.get("source", ""),
                        "snippet": docs[i].page_content[:120],
                    }
                    for i in range(len(docs))
                ],
                "metrics": {
                    "mrr": result["mrr"],
                    **{
                        f"recall@{k}": result["per_k"][k]["recall"]
                        for k in args.k_values
                    },
                    **{
                        f"precision@{k}": result["per_k"][k]["precision"]
                        for k in args.k_values
                    },
                    **{
                        f"ndcg@{k}": result["per_k"][k]["ndcg"]
                        for k in args.k_values
                    },
                    **{
                        f"ap@{k}": result["per_k"][k]["ap"]
                        for k in args.k_values
                    },
                },
            }
        )

        rel_flags = result["relevance_flags"][:top_k]
        hit = "✓" if any(rel_flags) else "✗"
        print(
            f"  [{hit}] {qid}: {question[:40]:<40}  "
            f"MRR={result['mrr']:.3f}  "
            f"nDCG@{args.k_values[-1]}={result['per_k'][args.k_values[-1]]['ndcg']:.3f}"
        )

    # ── Aggregate ───────────────────────────────────────────────────────────
    summary = aggregate(per_query, args.k_values)
    _print_table(summary, args.k_values)

    report = {
        "summary": summary,
        "details": per_query,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Full report saved to {args.output}")


if __name__ == "__main__":
    main()
