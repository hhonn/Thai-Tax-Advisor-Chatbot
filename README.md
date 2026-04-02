# Thai Tax Advisor Chatbot (Backend Only)

ระบบแชทบอทให้คำปรึกษาภาษีไทยด้วยเทคนิค Retrieval-Augmented Generation (RAG)

## ขอบเขตโปรเจค

- ภาษีเงินได้บุคคลธรรมดา (Personal Income Tax)
- ค่าลดหย่อนและสิทธิประโยชน์ ปีภาษี 2567-2568
- แบบยื่น ภ.ง.ด.90, ภ.ง.ด.91, ภ.ง.ด.94
- อัตราภาษีขั้นบันไดและการคำนวณพื้นฐาน
- มี fallback สำหรับคำถามนอกขอบเขต

ไม่ครอบคลุมภาษีนิติบุคคล, VAT และคำปรึกษากฎหมายเชิงวิชาชีพ 

## โมเดลและเทคโนโลยีหลัก

- LLM: Qwen 3.0 (ผ่าน OpenAI-compatible API)
- Embedding: BAAI/bge-m3
- Retrieval: Hybrid (Dense + BM25) + RRF
- Reranker (optional): BAAI/bge-reranker-v2-m3
- Vector DB: ChromaDB
- API: FastAPI
- Thai NLP: PyThaiNLP

## โครงสร้างไฟล์สำคัญ

- `app/api.py` FastAPI endpoints
- `app/rag_chain.py` retrieval + generation pipeline (Hybrid Search + Reranker) 
- `app/prompts.py` system prompt
- `ingest/prepare_tax_corpus.py` เตรียมข้อมูลดิบ เป็น chunk JSONL
- `ingest/build_bm25.py` สร้าง index sparse (BM25)
- `ingest/build_index.py` สร้าง Chroma index (Dense)
- `evaluate/run_eval.py` ประเมินผลแบบ proxy metrics + optional RAGAS

## เริ่มต้นใช้งาน

### 1) ติดตั้ง dependencies

```bash
pip install -r requirements.txt
```

### 2) ตั้งค่า environment

```bash
copy .env.example .env
```

จากนั้นแก้ค่า `QWEN_API_KEY`, `QWEN_BASE_URL`, `QWEN_MODEL`

### 3) เตรียมข้อมูลและดัชนีการค้นหา (Index)

```bash
python ingest/prepare_tax_corpus.py --input data/raw/master_tax_laws.json --output data/processed/tax_chunks.jsonl
python ingest/build_bm25.py --input data/processed/tax_chunks.jsonl --output data/processed/bm25_index.pkl
python ingest/build_index.py --input data/processed/tax_chunks.jsonl --persist-dir data/chroma_tax --reset
```

### 4) รัน API server

```bash
uvicorn app.api:app --reload --host 127.0.0.1 --port 8000
```

### 5) ทดสอบ API

```bash
curl -X GET http://127.0.0.1:8000/
```

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\":\"ลดหย่อนประกันชีวิตได้สูงสุดเท่าไหร่\"}"
```

## ประเมินผล

```bash
python evaluate/run_eval.py --testset evaluate/testset.json --output evaluate/results/eval_report.json
```

ถ้าต้องการเปิด RAGAS ให้ตั้ง `ENABLE_RAGAS=1` หรือส่งพารามิเตอร์ `--enable-ragas`  

## หมายเหตุสำคัญ

- ไม่ใช่คำปรึกษาภาษีหรือกฎหมายอย่างเป็นทางการ
- ควรตรวจสอบข้อมูลล่าสุดจากกรมสรรพากรหรือผู้เชี่ยวชาญก่อนตัดสินใจ