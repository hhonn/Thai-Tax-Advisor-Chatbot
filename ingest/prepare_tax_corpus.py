import json
import argparse
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs='+', required=True)        
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )

    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    with out_path.open("w", encoding="utf-8") as f:
        for input_path in args.inputs:
            rows = json.loads(input_path.read_text(encoding="utf-8-sig"))       
            for row in rows:
                text = row.get("text", "")
                if not text: continue

                chunks = splitter.create_documents(
                    [text],
                    metadatas=[{
                        "tax_id": str(row.get('law', '')).strip() + " " + str(row.get('section', '')).strip(),
                        "url": row.get("url", ""),
                        "source": row.get("source", "Unknown")
                    }]
                )
                for i, chunk in enumerate(chunks):
                    import re
                    base_id = f"{row.get('law', 'doc')}-{row.get('section', i)}"
                    safe_id = re.sub(r'[\W_]+', '_', base_id)
                    record = {
                        "id": f"{input_path.stem}-{safe_id}-{i}",
                        "text": chunk.page_content,
                        "metadata": chunk.metadata
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_chunks += 1
            print(f"Processed {input_path}")
    print(f"Done processing to {out_path}, total {total_chunks} chunks.")

if __name__ == '__main__':
    main()
