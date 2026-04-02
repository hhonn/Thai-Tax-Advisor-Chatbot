import json
import pickle
import argparse
from pathlib import Path
from rank_bm25 import BM25Okapi
from pythainlp.tokenize import word_tokenize

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    with args.input.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                
    print(f"Loaded {len(rows)} chunks for BM25 indexing")
    
    tokenized_corpus = []
    for row in rows:
        text = row.get("text", "")
        tokens = word_tokenize(text, engine="newmm")
        tokenized_corpus.append(tokens)
        
    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Use output filename for saving both instance AND mapped corpus 
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    with open(args.output, "wb") as f:
        pickle.dump({"bm25": bm25, "docs": rows}, f)
        
    print(f"BM25 index saved to {args.output}")

if __name__ == '__main__':
    main()
