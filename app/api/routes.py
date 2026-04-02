"""
FastAPI Routes for standard RAG ingestion and querying endpoints.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os

from app.services.rag_service import RAGService
from app.services.llm_service import LLMService

from langchain_google_genai import GoogleGenerativeAIEmbeddings

router = APIRouter()

# Global instantiations mapping to our dependencies.
# Note: Google GenAI requires GOOGLE_API_KEY environment variable.
try:
    # Safe init block if keys aren't immediately present in environments
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    rag_service = RAGService(embeddings=embeddings)
    llm_service = LLMService(model_name="gemini-2.5-flash")
except Exception as e:
    embeddings = None
    rag_service = None
    llm_service = None
    print(f"Warning: Could not initialize AI services. Check environment for GOOGLE_API_KEY tokens: {e}")

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    context_sources: int

class IngestResponse(BaseModel):
    message: str
    chunks_ingested: int

@router.post("/ingest", response_model=IngestResponse)
async def ingest_documents():
    """
    Triggers the system to scan raw_documents directory and chunk them into ChromaDB.
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

@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Accepts a request payload query, retrieves similar context from Vector Store, and triggers LLM prediction.
    """
    if not rag_service or not llm_service:
        raise HTTPException(status_code=500, detail="AI services not initialized. Verify GOOGLE_API_KEY is active.")
        
    try:
        # Retrieve context chunk documents
        docs = rag_service.similarity_search(request.question, k=4)
        
        # Generate the text completion answer
        answer = llm_service.generate_answer(docs, request.question)
        
        return AskResponse(
            answer=answer,
            context_sources=len(docs)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
