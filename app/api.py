from __future__ import annotations
import json
import logging
import os
import sys
from typing import Any, Dict, List, Tuple

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from rag_chain import (
    MAX_QUESTION_LENGTH,
    answer_question,
    stream_answer,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Thai Tax Advisor Chatbot API",
    version="1.0.0",
    description="RAG-powered Thai tax Q&A API with hybrid search, Qwen 3.0, reranking, and streaming.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For presentation layer integration
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=MAX_QUESTION_LENGTH)     
    history: List[ChatMessage] = Field(default_factory=list)

class ChatResponse(BaseModel):
    answer: str
    citations: str
    domain: str
    risk: str

@app.get("/")
def read_root():
    return {"message": "Welcome to Thai Tax Advisor Chatbot API (Powered by Qwen 3.0)"}

@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    history_tuples = [(msg.role, msg.content) for msg in req.history]
    
    async def event_generator():
        for chunk in stream_answer(req.message, history_tuples):
            yield f"data: {json.dumps({'content': chunk})}\n\n"
        yield "data: [DONE]\n\n"
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    history_tuples = [(msg.role, msg.content) for msg in req.history]
    answer, citations = answer_question(req.message, history_tuples)
    return ChatResponse(
        answer=answer,
        citations=citations,
        domain="ภาษีเงินได้บุคคลธรรมดา",
        risk="ปานกลาง"
    )
