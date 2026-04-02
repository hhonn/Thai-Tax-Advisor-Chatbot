"""
RAG Service for managing document ingestion and retrieval.
"""
from typing import List
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

class RAGService:
    def __init__(self, embeddings: Embeddings, raw_data_dir: str = "data/raw_documents", persist_dir: str = "data/vector_store"):
        """
        Initializes the RAG Service.
        
        Parameters:
        - embeddings (Embeddings): An LangChain compatible embeddings instance for vectorization.
        - raw_data_dir (str): Path to the directory where raw PDF files are stored.
        - persist_dir (str): Path for ChromaDB to store its local vector representation.
        """
        self.raw_data_dir = raw_data_dir
        self.persist_dir = persist_dir
        self.embeddings = embeddings
        self.vector_store = self._init_vector_store()

    def _init_vector_store(self) -> Chroma:
        """
        Initializes and returns the Chroma vector store pointing to persist_directory.
        """
        return Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
            collection_name="tax_law_docs"
        )

    def load_and_chunk_documents(self) -> List[Document]:
        """
        Loads PDF documents directly from the specified directory and splits them into usable chunks.
        
        Returns:
        - List[Document]: A flat list of separated document chunks ready for ingestion.
        """
        loader = PyPDFDirectoryLoader(self.raw_data_dir)
        docs = loader.load()
        
        # We split recursively for efficiency
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        return splitter.split_documents(docs)

    def ingest_documents(self) -> int:
        """
        Reads raw document PDFs, chunks them, and embeds them into ChromaDB.
        
        Returns:
        - int: The number of new document chunks ingested. Returns 0 if none.
        """
        chunks = self.load_and_chunk_documents()
        if not chunks:
            return 0
        
        self.vector_store.add_documents(chunks)
        return len(chunks)

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Performs vector similarity search against the previously populated database.
        
        Parameters:
        - query (str): The search phrase to extract matching chunks for.
        - k (int): The number of top matching document chunks to return.
        
        Returns:
        - List[Document]: The list of related documents from the store.
        """
        return self.vector_store.similarity_search(query, k=k)
