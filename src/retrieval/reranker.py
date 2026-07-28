"""
# reranker.py
Qwen3-Reranker-0.6B 기반 리랭커 (클래스 기반, 싱글톤)
"""
import gc
from typing import Any, ClassVar, Optional

import torch
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"
RERANKER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 지금 RFP 데이터셋은 "실제 요구사항(기능/성능/데이터 요구사항 등)"과
# "서약서·서식·청렴서약문 같은 행정 보일러플레이트"가 같은 기관명/사업명을
# 반복해서 포함하고 있어서, 지시문 없이는 리랭커가 이 둘을 잘 구분하지 못하고
# 단순히 기관명·사업명이 많이 겹치는 보일러플레이트 문서에 더 높은 점수를 줄 수 있다.
# -> instruct로 "실제 요구사항 내용"을 우선하도록 명시적으로 알려준다.
RERANK_TASK_INSTRUCTION = (
    "Given a query about a government or public-sector RFP (Request for Proposal) document, "
    "retrieve the passage that describes actual project requirements such as functional, "
    "technical, performance, or data requirements. Do not prefer boilerplate administrative "
    "content such as pledges, non-disclosure oaths, compliance certificates, or form indexes, "
    "even if they repeat the same organization or project name as the query."
)


class Reranker:
    """Qwen3-Reranker(CrossEncoder)를 한 번만 로드해서 재사용하는 싱글톤 클래스.

    ※ 싱글톤이란: 이 클래스로 객체를 아무리 여러 번 만들어도(Reranker() 호출을
      몇 번을 하든) 실제로는 맨 처음 만든 객체 딱 하나만 계속 재사용된다는 뜻.
      __new__를 오버라이드해서 "이미 만든 인스턴스가 있으면 그걸 그대로 반환"하는
      방식으로 구현했다. 기존 @lru_cache(maxsize=1)와 동일한 효과를 클래스로 표현한 것.
    """

    _instance: ClassVar[Optional["Reranker"]] = None

    def __new__(cls, *args, **kwargs):
        # 클래스 변수 _instance가 비어있을 때(=최초 1회)만 실제로 새 객체를 만든다.
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False  # 아직 모델 로딩 전이라는 표시
        return cls._instance

    def __init__(
        self,
        model_name: str = RERANKER_MODEL,
        device: str = RERANKER_DEVICE,
        instruction: str = RERANK_TASK_INSTRUCTION,
    ):
        # __new__에서 이미 초기화된 인스턴스를 반환한 경우,
        # __init__은 파이썬이 자동으로 또 호출하지만 모델을 중복 로드하면 안 되므로 여기서 막는다.
        if self._initialized:
            return

        from sentence_transformers import CrossEncoder

        print(f"[Reranker] '{model_name}' 로딩 중... (device={device})")
        self.model = CrossEncoder(
            model_name,
            device=device,
            prompts={"rfp_requirement": instruction},
            default_prompt_name="rfp_requirement",
        )
        self.device = device
        print("[Reranker] 로딩 완료")
        self._initialized = True  # 다음부터는 __init__ 내용을 건너뜀

    def _predict(self, pairs: list[tuple[str, str]], batch_size: int = 32):
        """query-document 쌍들의 관련성 점수를 계산하는 내부 공용 함수.

        - torch.inference_mode(): 추론 전용 모드로, 그래디언트를 아예 추적하지 않아서
          학습 때 쓰는 auto-grad 관련 메모리를 절약함 (OOM 완화의 핵심 포인트).
        - torch.cuda.empty_cache(): 추론이 끝난 뒤 PyTorch가 내부적으로 쥐고 있던
          '예약(reserved)됐지만 안 쓰는' GPU 메모리를 시스템에 반환. 멀티턴처럼
          호출이 반복되는 상황에서 메모리가 조금씩 누적되는 걸 막아준다.
        """
        with torch.inference_mode():
            scores = self.model.predict(pairs, batch_size=batch_size)

        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        return scores

    def rerank_documents(
        self,
        query: str,
        documents: list[Document],
        top_n: int = 5,
    ) -> list[Document]:
        if not documents:
            return []

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self._predict(pairs)

        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda pair: pair[1], reverse=True)

        reranked: list[Document] = []
        for doc, score in scored_docs[:top_n]:
            new_doc = Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "rerank_score": float(score)},
            )
            reranked.append(new_doc)

        return reranked

    def rerank_search_results(
        self,
        query: str,
        results: list[dict[str, Any]],
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        if not results:
            return []

        pairs = [(query, r["text"]) for r in results]
        scores = self._predict(pairs, batch_size=32)

        scored = list(zip(results, scores))
        scored.sort(key=lambda pair: pair[1], reverse=True)

        reranked: list[dict[str, Any]] = []
        for r, score in scored[:top_n]:
            r = dict(r)
            r["score"] = float(score)
            reranked.append(r)

        return reranked