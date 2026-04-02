"""
FastAPI Routes for standard RAG ingestion and querying endpoints.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
from app.services.rag_service import RAGService
from app.services.llm_service import LLMService
from langchain_ollama import OllamaEmbeddings

load_dotenv()


router = APIRouter(tags=["rag"])

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

try:
    embeddings = OllamaEmbeddings(
        model="all-minilm",
        base_url=OLLAMA_BASE_URL,
    )
    rag_service = RAGService(embeddings=embeddings)

    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is not set in the environment.")
    llm_service = LLMService(model_name="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)
except Exception as e:
    embeddings = None
    rag_service = None
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
    รับคำถามภาษาไทยเกี่ยวกับภาษีเงินได้บุคคลธรรมดา ค้นหา context จาก vector store แล้วสร้างคำตอบด้วย LLM

    **ขั้นตอน:**
    1. ดึง document chunks ที่เกี่ยวข้องจาก ChromaDB (top-k similarity)
    2. ส่ง context + คำถามให้ Gemini สร้างคำตอบ
    3. คืนคำตอบพร้อมจำนวน sources ที่ใช้
    """
    if not rag_service or not llm_service:
        raise HTTPException(status_code=500, detail="AI services not initialized. Verify GOOGLE_API_KEY is active.")
        
    try:
        # Retrieve context chunk documents
        docs = rag_service.similarity_search(request.question, k=4)

        print(f"Retrieved {len(docs)} context documents for question: {request.question}")
        print("Context documents:")
        for i, doc in enumerate(docs):
            print(f"Document {i+1}: {doc.page_content[:200]}...")  # Print first 200 chars of each doc
        
        # Generate the text completion answer
        answer = llm_service.generate_answer(docs, request.question)
        
        return AskResponse(
            answer=answer,
            context_sources=len(docs)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
