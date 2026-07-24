import json
import re
from typing import List

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from configs.config import RAGConfig
from configs.prompt import RERANK_PROMPT

try:
    from langchain_classic.retrievers import EnsembleRetriever
except Exception:
    from langchain_community.retrievers import EnsembleRetriever


class ProductionRetriever:
    def __init__(self, chunks: List[Document], vector_store: Chroma, config: RAGConfig):
        self.config = config
        self.naive_vector_retriever = vector_store.as_retriever(search_kwargs={"k": config.naive_k})
        self.bm25_retriever = BM25Retriever.from_documents(chunks)
        self.bm25_retriever.k = config.naive_k
        self.hybrid_retriever = EnsembleRetriever(retrievers=[self.bm25_retriever, self.naive_vector_retriever], weights=[0.5, 0.5])
        self.bm25_wide = BM25Retriever.from_documents(chunks)
        self.bm25_wide.k = config.wide_k
        self.vector_wide = vector_store.as_retriever(search_kwargs={"k": config.wide_k})
        self.wide_hybrid = EnsembleRetriever(retrievers=[self.bm25_wide, self.vector_wide], weights=[0.5, 0.5])
        self.rerank_chain = RERANK_PROMPT | ChatOpenAI(model=config.llm_model, temperature=config.temperature) | StrOutputParser()

    def retrieve_naive(self, query: str) -> List[Document]:
        return self.naive_vector_retriever.invoke(query)

    def retrieve_hybrid(self, query: str) -> List[Document]:
        return self.hybrid_retriever.invoke(query)

    def retrieve_reranked(self, query: str) -> List[Document]:
        candidate_pool = self.wide_hybrid.invoke(query)[:12]
        if not candidate_pool:
            return []

        candidates_str = "\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(candidate_pool))
        output = self.rerank_chain.invoke({"question": query, "candidates": candidates_str, "top_n": self.config.rerank_top_n})

        try:
            matched_indices = json.loads(re.search(r"\[.*?\]", output, re.S).group())
        except Exception:
            matched_indices = list(range(min(self.config.rerank_top_n, len(candidate_pool))))

        picked_indices = []
        for idx in matched_indices:
            if isinstance(idx, int) and 0 <= idx < len(candidate_pool) and idx not in picked_indices:
                picked_indices.append(idx)

        picked_indices += [i for i in range(len(candidate_pool)) if i not in picked_indices]
        return [candidate_pool[i] for i in picked_indices[:self.config.rerank_top_n]]


def build_retrievers(chunks: List[Document], vector_store: Chroma, config: RAGConfig) -> ProductionRetriever:
    return ProductionRetriever(chunks, vector_store, config)
