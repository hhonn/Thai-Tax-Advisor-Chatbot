from __future__ import annotations
import json
import logging
import os
import re
from typing import Generator, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Input constraints
MAX_QUESTION_LENGTH = 1000 
MAX_HISTORY_TURNS   = 6     
TOP_K_RERANK        = 8     
RETRIEVER_K         = 20    
MIN_RERANK_SCORE    = -3.0  

from dotenv import load_dotenv
from openai import OpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

load_dotenv()

# Qwen 3.0 Via OpenAI compatible API
OPENAI_API_KEY = os.environ.get("QWEN_API_KEY") or os.environ.get("OPENAI_API_KEY") 
OPENAI_BASE_URL = os.environ.get("QWEN_BASE_URL") or os.environ.get("OPENAI_BASE_URL") 
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen3-32b")

MODEL_ARGS = {
    "model": QWEN_MODEL,
    "temperature": 0.1,
    "max_tokens": 1500,
    "top_p": 0.1,
}

if OPENAI_API_KEY:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    except Exception as e:
        logger.error(f"Failed to load user's qwen model {e}")
        openai_client = None
else:
    logger.warning("No OPENAI_API_KEY provided.")
    openai_client = None

# Embeddings (BGE-M3 for semantic)
EMBEDDING_MODEL = "BAAI/bge-m3"
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Reranker (BGE-Reranker-V2-M3 for cross-encoder)
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
cross_encoder = CrossEncoder(RERANKER_MODEL)

# Setup Chroma
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_tax")
if os.path.exists(CHROMA_DIR):
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
else:
    logger.warning(f"Chroma DB not found at {CHROMA_DIR}. Ensure ingester has run.")
    vectorstore = None

# Setup BM25
BM25_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "bm25_index.pkl")
class MockBM25:
    def get_relevant_documents(self, query):
        return []

if os.path.exists(BM25_PATH):
    try:
        import pickle
        from pythainlp.tokenize import word_tokenize
        with open(BM25_PATH, "rb") as f:
            bm25_data = pickle.load(f)
            _bm25_instance = bm25_data["bm25"]
            _bm25_docs = bm25_data["docs"]
            
        class RealBM25:
            def get_relevant_documents(self, query, k=RETRIEVER_K):
                query_tokens = word_tokenize(query, engine="newmm")
                doc_scores = _bm25_instance.get_scores(query_tokens)
                
                # Get top K indices
                import numpy as np
                top_k = np.argsort(doc_scores)[::-1][:k]
                
                result = []
                for idx in top_k:
                    if doc_scores[idx] > 0:
                        row = _bm25_docs[idx]
                        result.append(Document(
                            page_content=row["text"],
                            metadata=row.get("metadata", {})
                        ))
                return result
                
        bm25_retriever = RealBM25()
        logger.info("Loaded real BM25 index.")
    except Exception as e:
        logger.error(f"Failed to load BM25, using mock. {e}")
        bm25_retriever = MockBM25()
else:
    logger.warning("BM25 index not found. Expected at " + BM25_PATH)
    bm25_retriever = MockBM25()

def _reciprocal_rank_fusion(dense_docs, sparse_docs, k=60):
    rrf_scores = {}
    
    for rank, doc in enumerate(dense_docs):
        text = doc.page_content
        rrf_scores[text] = rrf_scores.get(text, 0) + 1 / (rank + k)
        
    for rank, doc in enumerate(sparse_docs):
        text = doc.page_content
        rrf_scores[text] = rrf_scores.get(text, 0) + 1 / (rank + k)
    
    sorted_docs = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    return [Document(page_content=text) for text in sorted_docs]

def rr_search_with_rerank(query: str, k: int = 5) -> List[Document]:
    if vectorstore:
        dense_docs = vectorstore.similarity_search(query, k=RETRIEVER_K)
    else:
        dense_docs = []
        
    sparse_docs = bm25_retriever.get_relevant_documents(query)
    
    # RRF Fusion
    fused_docs = _reciprocal_rank_fusion(dense_docs, sparse_docs)
    fused_docs = fused_docs[:RETRIEVER_K]
    
    if not fused_docs:
        return []
    
    # Reranking
    pairs = [[query, doc.page_content] for doc in fused_docs]
    scores = cross_encoder.predict(pairs)
    
    scored_docs = list(zip(fused_docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # Filter by score
    top_docs = [doc for doc, score in scored_docs if score >= MIN_RERANK_SCORE][:k]
    return top_docs

def build_context(docs: List[Document]) -> str:
    parts = []
    for d in docs:
        content = d.page_content.strip().replace("\n", " ")
        # metadata extraction is simulated
        meta = d.metadata.get('tax_id', '') if d.metadata else ""
        parts.append(f"[อ้างอิง: {meta}] {content}")
    return "\n\n".join(parts)


from app.prompts import SYSTEM_PROMPT

def answer_question(question: str, user_history: List[Tuple[str, str]] = None) -> Tuple[str, str]:
    if not openai_client:
        return "Qwen 3.0 is not configured.", ""
        
    docs = rr_search_with_rerank(question, k=TOP_K_RERANK)
    context_str = build_context(docs)
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if user_history:
        for role, text in user_history[-MAX_HISTORY_TURNS:]:
            messages.append({"role": role, "content": text})
            
    prompt = f"Context:\n{context_str}\n\nQuestion: {question}"
    messages.append({"role": "user", "content": prompt})
    
    try:
        resp = openai_client.chat.completions.create(
            messages=messages,
            **MODEL_ARGS
        )
        return resp.choices[0].message.content, context_str
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return "เกิดข้อผิดพลาดในการประมวลผลคำตอบ", ""

def stream_answer(question: str, user_history: List[Tuple[str, str]] = None) -> Generator[str, None, None]:
    if not openai_client:
        yield "Qwen 3.0 is not configured."
        return
        
    docs = rr_search_with_rerank(question, k=TOP_K_RERANK)
    context_str = build_context(docs)
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if user_history:
        for role, text in user_history[-MAX_HISTORY_TURNS:]:
            messages.append({"role": role, "content": text})
            
    prompt = f"Context:\n{context_str}\n\nQuestion: {question}"
    messages.append({"role": "user", "content": prompt})
    
    try:
        resp = openai_client.chat.completions.create(
            messages=messages,
            stream=True,
            **MODEL_ARGS
        )
        
        # We need to yield citations first so client can parse it if needed
        # In a real implementation this might be SSE
        yield f"__CITATIONS::{context_str}::CITATIONS__\n"
        
        for chunk in resp:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except Exception as e:
        logger.error(f"Stream error: {e}")
        yield "เกิดข้อผิดพลาดในการประมวลผลคำตอบ"