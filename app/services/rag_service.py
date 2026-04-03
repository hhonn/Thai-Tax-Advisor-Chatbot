"""
RAG Service for managing document ingestion and retrieval.

Supports:
- Dense vector search (ChromaDB)
- Sparse keyword search (BM25 with PyThaiNLP tokenization)
- Hybrid search with Reciprocal Rank Fusion (RRF)
"""
import logging
import os
import pickle
from typing import List, Tuple

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

RETRIEVER_K = 20  # Default number of candidates from each retriever


class RAGService:
    def __init__(
        self,
        embeddings: Embeddings,
        raw_data_dir: str = "data/raw_documents",
        persist_dir: str = "data/vector_store",
        bm25_path: str = "data/processed/bm25_index.pkl",
    ):
        self.raw_data_dir = raw_data_dir
        self.persist_dir = persist_dir
        self.embeddings = embeddings
        self.vector_store = self._init_vector_store()
        self.bm25, self.bm25_docs = self._load_bm25(bm25_path)

    def _init_vector_store(self) -> Chroma:
        return Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
            collection_name="tax_law_docs",
        )

    @staticmethod
    def _load_bm25(path: str):
        if not os.path.exists(path):
            logger.warning(f"BM25 index not found at {path}")
            return None, []
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            logger.info(f"Loaded BM25 index from {path} ({len(data['docs'])} docs)")
            return data["bm25"], data["docs"]
        except Exception as e:
            logger.error(f"Failed to load BM25: {e}")
            return None, []

    # ── Dense (vector) search ────────────────────────────────────────────

    def vector_search(self, query: str, k: int = RETRIEVER_K) -> List[Document]:
        return self.vector_store.similarity_search(query, k=k)

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        return self.vector_store.similarity_search(query, k=k)

    def similarity_search_with_scores(
        self, query: str, k: int = 10
    ) -> List[Tuple[Document, float]]:
        return self.vector_store.similarity_search_with_relevance_scores(query, k=k)

    # ── Sparse (BM25 keyword) search ────────────────────────────────────

    def bm25_search(self, query: str, k: int = RETRIEVER_K) -> List[Document]:
        if not self.bm25:
            return []
        try:
            from pythainlp.tokenize import word_tokenize
            import numpy as np

            tokens = word_tokenize(query, engine="newmm")
            scores = self.bm25.get_scores(tokens)
            top_indices = np.argsort(scores)[::-1][:k]

            return [
                Document(
                    page_content=self.bm25_docs[idx]["text"],
                    metadata=self.bm25_docs[idx].get("metadata", {}),
                )
                for idx in top_indices
                if scores[idx] > 0
            ]
        except Exception as e:
            logger.error(f"BM25 search error: {e}")
            return []

    # ── Reciprocal Rank Fusion ──────────────────────────────────────────

    @staticmethod
    def reciprocal_rank_fusion(
        dense_docs: List[Document],
        sparse_docs: List[Document],
        rrf_k: int = 60,
    ) -> List[Document]:
        """
        Merge two ranked lists using RRF.
        score(d) = Σ 1 / (rank_i + k) for each list where d appears.
        """
        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for rank, doc in enumerate(dense_docs):
            text = doc.page_content
            rrf_scores[text] = rrf_scores.get(text, 0.0) + 1.0 / (rank + rrf_k)
            doc_map[text] = doc

        for rank, doc in enumerate(sparse_docs):
            text = doc.page_content
            rrf_scores[text] = rrf_scores.get(text, 0.0) + 1.0 / (rank + rrf_k)
            if text not in doc_map:
                doc_map[text] = doc

        sorted_texts = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
        return [doc_map[t] for t in sorted_texts]

    # ── Hybrid search (vector + BM25 → RRF) ────────────────────────────

    def hybrid_search(
        self,
        vector_query: str,
        keyword_query: str,
        k: int = RETRIEVER_K,
    ) -> List[Document]:
        """
        Run both dense and sparse retrieval, then fuse via RRF.

        Parameters:
        - vector_query: query for vector search (can be HyDE document)
        - keyword_query: query for BM25 (can be rewritten query)
        - k: max documents to return after fusion
        """
        dense_docs = self.vector_search(vector_query, k=k)
        sparse_docs = self.bm25_search(keyword_query, k=k)

        logger.info(
            f"Hybrid search: dense={len(dense_docs)}, sparse={len(sparse_docs)}"
        )

        fused = self.reciprocal_rank_fusion(dense_docs, sparse_docs)
        return fused[:k]

    # ── Ingest (backward compatibility) ─────────────────────────────────

    def load_and_chunk_documents(self) -> List[Document]:
        loader = PyPDFDirectoryLoader(self.raw_data_dir)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            add_start_index=True,
        )
        return splitter.split_documents(docs)

    def ingest_documents(self) -> int:
        chunks = self.load_and_chunk_documents()
        if not chunks:
            return 0
        self.vector_store.add_documents(chunks)
        return len(chunks)
