"""
Cross-Encoder Reranking Service using BGE-Reranker-V2-M3.

Re-scores (query, document) pairs with a cross-encoder that understands
deep semantic relationships better than bi-encoder similarity.
"""
import logging
from typing import List

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class RerankerService:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        min_score: float = -3.0,
    ):
        self.min_score = min_score
        try:
            self.model = CrossEncoder(model_name)
            logger.info(f"Loaded reranker model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            self.model = None

    def rerank(
        self,
        query: str,
        docs: List[Document],
        top_k: int = 5,
    ) -> List[Document]:
        """
        Re-rank documents by cross-encoder relevance score.

        Parameters:
        - query: the original user question
        - docs: candidate documents from hybrid search
        - top_k: max documents to return after reranking

        Returns:
        - List[Document]: top-k documents sorted by cross-encoder score,
          filtered by min_score threshold.
        """
        if not self.model or not docs:
            return docs[:top_k]

        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.model.predict(pairs)

        scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

        result = [doc for doc, score in scored if score >= self.min_score][:top_k]

        if result:
            logger.info(
                f"Reranked {len(docs)} → {len(result)} docs "
                f"(best={scored[0][1]:.3f}, worst_kept={scored[len(result)-1][1]:.3f})"
            )

        return result
