"""
Main application entry point for the FastAPI backend.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# from fastapi.openapi.utils import get_openapi
from app.api.routes import router

app = FastAPI(
    title="Thai Tax Advisor Chatbot API",
    description=(
        "RAG-powered Thai personal income tax Q&A API.\n\n"
        "**ขอบเขต:** ภาษีเงินได้บุคคลธรรมดา, ค่าลดหย่อน, แบบ ภ.ง.ด.90/91/94 (ปีภาษี 2567–2568)\n\n"
        "**ไม่ครอบคลุม:** ภาษีนิติบุคคล, VAT และคำปรึกษากฎหมายเชิงวิชาชีพ"
    ),
    version="1.0.0",
    contact={
        "name": "Thai Tax Advisor",
    },
    license_info={
        "name": "MIT",
    },
    openapi_tags=[
        {
            "name": "health",
            "description": "Health check endpoints.",
        },
        {
            "name": "rag",
            "description": "Retrieval-Augmented Generation — ingest documents and ask tax questions.",
        },
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

@app.get("/", tags=["health"], summary="Health check", response_description="Service status")
def read_root():
    """
    Returns the current health status of the API server.
    """
    return {"status": "healthy", "service": "Thai Tax Advisor Chatbot API"}
