"""
LLM Service for prompt formatting and interfacing with the main AI model.
"""

from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

from app.prompts import SYSTEM_PROMPT


class LLMService:
    def __init__(
        self, model_name: str = "gemini-2.5-flash", google_api_key: str | None = None
    ):
        """
        Initializes the LLM Service integrating with Gemini.

        Parameters:
        - model_name (str): Specifies the base Gemini model id to utilize (defaults to gemini-2.5-flash).
        - google_api_key (str | None): Google API key; falls back to GOOGLE_API_KEY env var if not provided.
        """
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.1,
            google_api_key=google_api_key,
        )
        self.prompt = PromptTemplate.from_template(
            SYSTEM_PROMPT +
            "บริบท:\n{context}\n\n"
            "คำถาม:\n{question}\n\n"
            "คำตอบ:"
        )

    def generate_answer(self, context_docs: List[Document], question: str) -> str:
        """
        Leverages retrieved context and user queries to construct answers using LangChain pipelines.

        Parameters:
        - context_docs (List[Document]): Context document items dynamically retrieved from RAG.
        - question (str): Original user instruction or inquiry.

        Returns:
        - str: Clean string output directly from the language model prompt pipeline.
        """
        # Join the texts into a single string corpus
        context_str = "\n\n---\n\n".join([doc.page_content for doc in context_docs])

        # Link the chain
        chain = self.prompt | self.llm | StrOutputParser()

        # Execute chain synchronously
        return chain.invoke({"context": context_str, "question": question})
