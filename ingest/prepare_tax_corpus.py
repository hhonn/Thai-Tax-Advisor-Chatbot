import json
import re
import argparse
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Thai legal document separators (highest to lowest priority)
THAI_LEGAL_SEPARATORS = [
    "\nมาตรา ",       # Legal article / section
    "\nหมวด ",        # Chapter
    "\nส่วนที่ ",     # Part
    "\nข้อ ",          # Clause
    "\nบทที่ ",       # Chapter (alt)
    "\n\n",            # Paragraph break
    "\n",              # Line break
    " ",               # Space
    "",                # Character fallback
]

# Heuristic category detection from tax_id / law field
CATEGORY_KEYWORDS = {
    "ภาษีเงินได้บุคคลธรรมดา": "PIT",
    "บุคคลธรรมดา": "PIT",
    "ภาษีเงินได้นิติบุคคล": "CIT",
    "นิติบุคคล": "CIT",
    "ภาษีมูลค่าเพิ่ม": "VAT",
    "VAT": "VAT",
    "ภาษีธุรกิจเฉพาะ": "SBT",
    "อากรแสตมป์": "SD",
    "ภาษีหัก ณ ที่จ่าย": "WHT",
    "หัก ณ ที่จ่าย": "WHT",
    "ค่าลดหย่อน": "DED",
    "ประมวลรัษฎากร": "RC",
}

def detect_category(text: str, law: str = "", section: str = "") -> str:
    combined = f"{law} {section} {text[:200]}"
    for keyword, cat in CATEGORY_KEYWORDS.items():
        if keyword in combined:
            return cat
    return "GENERAL"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs='+', required=True)        
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=500,
                        help="Target chunk size in characters (default: 500)")
    parser.add_argument("--chunk-overlap", type=int, default=50,
                        help="Overlap between chunks in characters (default: 50)")
    args = parser.parse_args()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        separators=THAI_LEGAL_SEPARATORS,
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

                law = str(row.get('law', '')).strip()
                section = str(row.get('section', '')).strip()
                category = detect_category(text, law, section)

                chunks = splitter.create_documents(
                    [text],
                    metadatas=[{
                        "tax_id": f"{law} {section}".strip(),
                        "url": row.get("url", ""),
                        "source": row.get("source", "Unknown"),
                        "category": category,
                        "law": law,
                        "section": section,
                    }]
                )
                for i, chunk in enumerate(chunks):
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
