"""
LLM Service for prompt formatting and interfacing with the main AI model.

Includes:
- Answer generation from retrieved context
- Query rewriting (pre-retrieval optimization)
- HyDE: Hypothetical Document Embeddings (pre-retrieval optimization)
"""

import logging
from typing import Generator, List
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from app.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class LLMService:
    def __init__(
        self,
        model_name: str = "typhoon-v2.1-12b-instruct",
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 4096,
    ):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.1,
            max_tokens=max_tokens,
            top_p=0.1,
            api_key=api_key,
            base_url=base_url,
        )
        self.answer_prompt = PromptTemplate.from_template(
            SYSTEM_PROMPT +
            "บริบท:\n{context}\n\n"
            "คำถาม:\n{question}\n\n"
            "คำตอบ:"
        )
        self.rewrite_prompt = PromptTemplate.from_template(
            "คุณคือผู้ช่วยปรับปรุงคำค้นหาเกี่ยวกับภาษีไทย "
            "จงปรับปรุงคำถามต่อไปนี้ให้ชัดเจนขึ้น เพิ่มคำสำคัญที่เกี่ยวข้อง "
            "เช่น ชื่อมาตรา ชื่อกฎหมาย หรือศัพท์เทคนิคภาษี "
            "เพื่อให้ค้นหาข้อมูลภาษีได้ดีขึ้น\n\n"
            "คำถามเดิม: {question}\n\n"
            "คำถามที่ปรับปรุงแล้ว (ตอบเป็นคำถามเดียว ไม่ต้องอธิบาย):"
        )
        self.hyde_prompt = PromptTemplate.from_template(
            "คุณคือผู้เชี่ยวชาญด้านกฎหมายภาษีไทย "
            "จงเขียนเนื้อหาสั้นๆ (2-3 ประโยค) ที่น่าจะเป็นคำตอบ "
            "สำหรับคำถามต่อไปนี้ โดยใช้ภาษาเทคนิคกฎหมายภาษี "
            "อ้างอิงมาตราหรือประมวลรัษฎากรที่เกี่ยวข้อง "
            "แม้ไม่แน่ใจก็ให้ตอบอย่างมั่นใจ\n\n"
            "คำถาม: {question}\n\n"
            "เนื้อหา:"
        )

    def generate_answer(self, context_docs: List[Document], question: str) -> str:
        context_str = "\n\n---\n\n".join([doc.page_content for doc in context_docs])
        chain = self.answer_prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context_str, "question": question})

    def stream_answer(
        self, context_docs: List[Document], question: str
    ) -> Generator[str, None, None]:
        """
        Stream the LLM answer token-by-token.
        Yields string chunks as they are generated.
        """
        context_str = "\n\n---\n\n".join([doc.page_content for doc in context_docs])
        chain = self.answer_prompt | self.llm | StrOutputParser()
        yield from chain.stream({"context": context_str, "question": question})

    def rewrite_query(self, question: str) -> str:
        """
        Use LLM to rewrite/expand a user query with relevant tax keywords.
        Falls back to original question on failure.
        """
        try:
            chain = self.rewrite_prompt | self.llm | StrOutputParser()
            result = chain.invoke({"question": question}).strip()
            if result:
                logger.info(f"Query rewritten: '{question}' → '{result}'")
                return result
        except Exception as e:
            logger.warning(f"Query rewriting failed, using original: {e}")
        return question

    def generate_hyde_document(self, question: str) -> str:
        """
        Generate a Hypothetical Document (HyDE) that answers the question.
        The hypothetical answer is used as the search query for vector retrieval,
        often matching document language better than a direct question.
        Falls back to original question on failure.
        """
        try:
            chain = self.hyde_prompt | self.llm | StrOutputParser()
            result = chain.invoke({"question": question}).strip()
            if result:
                logger.info(f"HyDE generated ({len(result)} chars)")
                return result
        except Exception as e:
            logger.warning(f"HyDE generation failed, using original query: {e}")
        return question
