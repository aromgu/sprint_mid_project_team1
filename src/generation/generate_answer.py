from typing import Any, Dict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from configs.config import RAGConfig
from configs.prompt import ANSWER_PROMPT, REWRITE_PROMPT
from src.retrieval.retriever import ProductionRetriever


class AdvancedRAGChain:
    def __init__(self, retrievers: ProductionRetriever, config: RAGConfig):
        self.retrievers = retrievers
        self.main_llm = ChatOpenAI(model=config.llm_model, temperature=config.temperature)
        self.rewriter_unit = REWRITE_PROMPT | self.main_llm | StrOutputParser()
        self.generator_unit = ANSWER_PROMPT | self.main_llm | StrOutputParser()
        self.execution_chain = (
            {
                "context": (lambda x: x["question"]) | RunnableLambda(self._advanced_retrieval_flow) | (lambda docs: "\n\n".join(d.page_content for d in docs)),
                "question": (lambda x: x["question"]),
            }
            | self.generator_unit
        )

    def _advanced_retrieval_flow(self, query: str):
        rewritten_query = self.rewriter_unit.invoke({"question": query})
        return self.retrievers.retrieve_reranked(rewritten_query)

    def invoke(self, inputs: Dict[str, Any]) -> str:
        return self.execution_chain.invoke(inputs)


def build_advanced_chain(retrievers: ProductionRetriever, config: RAGConfig) -> AdvancedRAGChain:
    return AdvancedRAGChain(retrievers, config)
