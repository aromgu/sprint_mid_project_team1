from typing import List

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from configs.config import RAGConfig


def build_vector_store(chunks: List[Document], config: RAGConfig) -> Chroma:
    embedding_unit = OpenAIEmbeddings(model=config.embedding_model)
    chroma_client = chromadb.EphemeralClient()
    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding_unit,
        collection_name="ai11_policy_production",
        client=chroma_client,
    )
