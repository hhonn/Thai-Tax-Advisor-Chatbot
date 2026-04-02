from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from tqdm import tqdm


def load_chunks(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Chroma index from chunked JSONL")
    parser.add_argument("--input", type=Path, default=Path("data/processed/tax_chunks.jsonl"))
    parser.add_argument("--persist-dir", type=Path, default=Path("data/vector_store"))
    parser.add_argument("--collection-name", type=str, default="tax_law_docs")
    parser.add_argument("--embedding-model", type=str, default="all-minilm")
    parser.add_argument("--ollama-base-url", type=str, default="http://localhost:11434")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    if args.reset and args.persist_dir.exists():
        shutil.rmtree(args.persist_dir)

    rows = load_chunks(args.input)
    if not rows:
        raise ValueError("No chunks found in input file")

    docs = [
        Document(page_content=row["text"], metadata=row.get("metadata", {}))
        for row in rows
        if row.get("text")
    ]

    embedding = OllamaEmbeddings(
        model=args.embedding_model,
        base_url=args.ollama_base_url,
    )

    db = None
    for start in tqdm(range(0, len(docs), args.batch_size), desc="Indexing"):
        batch = docs[start : start + args.batch_size]
        if db is None:
            db = Chroma.from_documents(
                documents=batch,
                embedding=embedding,
                persist_directory=str(args.persist_dir),
                collection_name=args.collection_name,
            )
        else:
            db.add_documents(batch)

    print(f"Done. Indexed documents: {len(docs)} | persist_dir: {args.persist_dir}")


if __name__ == "__main__":
    main()
