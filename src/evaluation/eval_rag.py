import os
import sys
import types
from typing import List

import pandas as pd

from configs.config import RAGConfig
from src.dataset import get_hitk_testset, get_ragas_testset
from src.generation.generate_answer import AdvancedRAGChain
from src.retrieval.retriever import ProductionRetriever

try:
    from langchain_google_vertexai import ChatVertexAI as _CVX
except Exception:
    class _CVX:
        def __init__(self, *a, **k):
            raise ImportError("stub")

if "langchain_community.chat_models.vertexai" not in sys.modules:
    _vx = types.ModuleType("langchain_community.chat_models.vertexai")
    _vx.ChatVertexAI = _CVX
    sys.modules["langchain_community.chat_models.vertexai"] = _vx

from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def _evaluate_hitk(retrievers: ProductionRetriever) -> pd.DataFrame:
    test_set = get_hitk_testset()
    stages = {"Naive(Vector)": retrievers.retrieve_naive, "①Hybrid": retrievers.retrieve_hybrid, "②+Rerank": retrievers.retrieve_reranked}
    rows = []
    for q, gold_id, typ in test_set:
        row = {"유형": typ, "질문": q[:26] + "…"}
        for name, fn in stages.items():
            row[name] = "✅" if any(d.metadata.get("id") == gold_id for d in fn(q)[:3]) else "❌"
        rows.append(row)
    return pd.DataFrame(rows)


def _run_ragas_metrics(advanced_chain: AdvancedRAGChain, config: RAGConfig) -> pd.DataFrame:
    dataset_raw = get_ragas_testset()
    records = []
    for q, gt in zip(dataset_raw["questions"], dataset_raw["ground_truths"]):
        contexts = [d.page_content for d in advanced_chain.retrievers.retrieve_reranked(q)]
        records.append({
            "user_input": q,
            "response": advanced_chain.invoke({"question": q}),
            "retrieved_contexts": contexts,
            "reference": gt,
        })

    result = evaluate(
        dataset=EvaluationDataset.from_list(records),
        metrics=[Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference(), LLMContextRecall()],
        llm=LangchainLLMWrapper(ChatOpenAI(model=config.llm_model, temperature=config.temperature)),
        embeddings=LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=config.embedding_model)),
    )
    return result.to_pandas()


def run_evaluation(advanced_chain: AdvancedRAGChain, retrievers: ProductionRetriever, config: RAGConfig):
    df_hit = _evaluate_hitk(retrievers)
    df_hit.to_csv(os.path.join(config.report_dir, "hit_scoreboard.csv"), index=False, encoding="utf-8-sig")
    df_ragas = _run_ragas_metrics(advanced_chain, config)
    df_ragas.to_csv(os.path.join(config.report_dir, "ragas_evaluation_result.csv"), index=False, encoding="utf-8-sig")
