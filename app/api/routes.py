"""
FastAPI Routes for optimized RAG pipeline.

Pipeline: Query Rewriting → HyDE → Hybrid Search (Vector + BM25 → RRF) → Rerank → LLM Answer
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List
import json
import os
import logging
from dotenv import load_dotenv
from app.services.rag_service import RAGService
from app.services.llm_service import LLMService
from app.services.reranker_service import RerankerService
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

logger = logging.getLogger(__name__)

router = APIRouter(tags=["rag"])

TYPHOON_API_KEY = os.getenv("TYPHOON_API_KEY") or os.getenv("OPENAI_API_KEY")
TYPHOON_BASE_URL = os.getenv("TYPHOON_BASE_URL", "https://api.opentyphoon.ai/v1")
TYPHOON_MODEL = os.getenv("TYPHOON_MODEL", "typhoon-v2.1-12b-instruct")

# ── Service Initialization ────────────────────────────────────────────────

try:
    # BGE-M3: multilingual embedding with strong Thai support (1024-dim)
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        encode_kwargs={"normalize_embeddings": True},
    )
    rag_service = RAGService(embeddings=embeddings)

    # Cross-encoder reranker for post-retrieval scoring
    reranker = RerankerService()

    if not TYPHOON_API_KEY:
        raise ValueError("TYPHOON_API_KEY (or OPENAI_API_KEY) is not set in the environment.")
    _max_tokens = int(os.getenv("LLM_MAX_TOKENS", "4096"))
    llm_service = LLMService(model_name=TYPHOON_MODEL, api_key=TYPHOON_API_KEY, base_url=TYPHOON_BASE_URL, max_tokens=_max_tokens)
except Exception as e:
    embeddings = None
    rag_service = None
    reranker = None
    llm_service = None
    print(f"Warning: Could not initialize AI services: {e}")

class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        examples=["ลดหย่อนประกันชีวิตได้สูงสุดเท่าไหร่"],
        description="คำถามเกี่ยวกับภาษีเงินได้บุคคลธรรมดาของไทย",
    )

class AskResponse(BaseModel):
    answer: str = Field(..., description="คำตอบที่สร้างโดย LLM")
    context_sources: int = Field(..., description="จำนวน document chunks ที่ใช้เป็น context")

class IngestResponse(BaseModel):
    message: str = Field(..., description="สถานะการ ingest")
    chunks_ingested: int = Field(..., description="จำนวน chunks ที่ถูก ingest เข้า vector store")

class ChatMessage(BaseModel):
    role: str = Field(..., description="บทบาทของข้อความ ('user' หรือ 'assistant')")
    content: str = Field(..., description="เนื้อหาข้อความ")

class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        examples=["ลดหย่อนประกันชีวิตได้สูงสุดเท่าไหร่"],
        description="ข้อความใหม่จากผู้ใช้",
    )
    history: List[ChatMessage] = Field(
        default=[],
        description="ประวัติการสนทนา (เรียงจากเก่าไปใหม่)",
    )

@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest documents",
    response_description="Ingestion result with chunk count",
)
async def ingest_documents():
    """
    สแกนไดเรกทอรี `raw_documents` และแตก chunk เข้า ChromaDB

    - ต้องมีไฟล์เอกสารอยู่ใน `raw_documents/` ก่อนเรียกใช้
    - ใช้ Google Generative AI Embeddings (`models/embedding-001`)
    """
    if not rag_service:
        raise HTTPException(status_code=500, detail="RAG service not initialized. Verify GOOGLE_API_KEY is active.")
        
    try:
        count = rag_service.ingest_documents()
        return IngestResponse(
            message="Ingestion successful." if count > 0 else "No documents found to ingest.",
            chunks_ingested=count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a tax question",
    response_description="LLM-generated answer with source count",
)
async def ask_question(request: AskRequest):
    """
    รับคำถามภาษาไทยเกี่ยวกับภาษีเงินได้บุคคลธรรมดา แล้วประมวลผลผ่าน optimized RAG pipeline

    **Pipeline:**
    1. **Query Rewriting** — LLM ปรับปรุงคำถามให้ชัดเจน เพิ่มคำสำคัญ
    2. **HyDE** — LLM สร้างคำตอบสมมติเพื่อใช้ค้น vector ที่ตรงความหมายมากขึ้น
    3. **Hybrid Search** — ค้นหาจาก Vector (BGE-M3) + BM25 (keyword) แล้ว fuse ด้วย RRF
    4. **Re-ranking** — Cross-Encoder (BGE-Reranker-V2-M3) จัดอันดับใหม่ตามความเกี่ยวข้อง
    5. **Answer Generation** — Gemini สร้างคำตอบจาก context ที่ดีที่สุด
    """
    if not rag_service or not llm_service:
        raise HTTPException(status_code=500, detail="AI services not initialized. Verify GOOGLE_API_KEY is active.")
        
    try:
        question = request.question

        # Step 1: Query Rewriting — expand with tax keywords
        rewritten_query = llm_service.rewrite_query(question)

        # Step 2: HyDE — generate hypothetical answer for better vector matching
        hyde_doc = llm_service.generate_hyde_document(question)

        # Step 3: Hybrid Search — Vector (HyDE query) + BM25 (rewritten query) → RRF
        candidates = rag_service.hybrid_search(
            vector_query=hyde_doc,
            keyword_query=rewritten_query,
            k=20,
        )

        # Step 4: Re-ranking — cross-encoder scores (query, chunk) pairs
        if reranker:
            top_docs = reranker.rerank(question, candidates, top_k=5)
        else:
            top_docs = candidates[:5]

        logger.info(
            f"Pipeline: question='{question[:50]}...' | "
            f"candidates={len(candidates)} | reranked={len(top_docs)}"
        )

        # Step 5: Generate answer from reranked context
        answer = llm_service.generate_answer(top_docs, question)
        
        return AskResponse(
            answer=answer,
            context_sources=len(top_docs)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/chat/stream",
    summary="Streaming chat with tax advisor",
    response_description="SSE stream of answer tokens",
)
async def chat_stream(req: ChatRequest):
    """
    รับข้อความจากผู้ใช้พร้อมประวัติการสนทนา แล้วตอบแบบ streaming (Server-Sent Events)

    ใช้ pipeline เดียวกับ `/ask`:
    Query Rewriting → HyDE → Hybrid Search → Rerank → **Stream** answer tokens
    """
    if not rag_service or not llm_service:
        raise HTTPException(
            status_code=500,
            detail="AI services not initialized. Verify GOOGLE_API_KEY is active.",
        )

    try:
        question = req.message

        # Pre-retrieval: Query Rewriting + HyDE
        rewritten_query = llm_service.rewrite_query(question)
        hyde_doc = llm_service.generate_hyde_document(question)

        # Hybrid Search → RRF
        candidates = rag_service.hybrid_search(
            vector_query=hyde_doc,
            keyword_query=rewritten_query,
            k=20,
        )

        # Rerank
        if reranker:
            top_docs = reranker.rerank(question, candidates, top_k=5)
        else:
            top_docs = candidates[:5]

        logger.info(
            f"Stream pipeline: question='{question[:50]}...' | "
            f"candidates={len(candidates)} | reranked={len(top_docs)}"
        )

        # Stream answer via SSE
        async def event_generator():
            for chunk in llm_service.stream_answer(top_docs, question):
                yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))