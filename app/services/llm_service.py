"""
LLM Service for prompt formatting and interfacing with the main AI model.
"""
from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

class LLMService:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        """
        Initializes the LLM Service integrating with Gemini.
        
        Parameters:
        - model_name (str): Specifies the base Gemini model id to utilize (defaults to gemini-2.5-flash).
        """
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.1)
        self.prompt = PromptTemplate.from_template(
            "You are a helpful and knowledgeable Tax Law AI Assistant.\n"
            "Use the following context to answer the user's question accurately.\n"
            "If the answer is not contained in the context, say 'I cannot answer this based on the provided documents.'\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
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
