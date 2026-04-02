"""
Main application entry point for the FastAPI backend.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router

app = FastAPI(
    title="Tax Law AI Assistant API",
    description="A robust backend server wrapping LangChain logic for vector ingestion and response generation.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

@app.get("/")
def read_root():
    """
    Health check mapping for basic up-time probing.
    """
    return {"status": "healthy", "service": "Tax Law AI Application Layer"}
