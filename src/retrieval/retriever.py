"""
# retriever.py
하이브리드 리트리버 (Advanced RAG)

구성:
- BM25 (키워드 검색)  : 미리 만들어둔 BM25 인덱스 pickle(bm25_index.pkl)을 그대로 로드해서 사용
                        (Kiwi 재토큰화도, BM25Okapi 재구축도 필요 없음 -> 매우 빠름)
                        질의(query)만 검색 시점에 korean_tokenize()로 실시간 토큰화
- 벡터 (의미 검색)    : 기존 Chroma 벡터스토어 (ai11_policy_advanced_v2)
- 결합               : EnsembleRetriever (RRF, weights로 비중 조절)
- 리랭킹             : reranker.py의 Qwen3-Reranker-0.6B (rerank_documents)
"""
import os
import pickle
import re
from typing import Any

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_classic.retrievers import EnsembleRetriever

from kiwipiepy import Kiwi
import time
from functools import lru_cache, wraps

from src.retrieval.reranker import Reranker

reranker = Reranker()

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")
# CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION_NAME", "ai11_policy_advanced_v2")
# CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "/home/data/chroma_advanced_v2")
# CHROMA_COLLECTION = "ai11_policy_advanced_v2"
# CHROMA_PERSIST_DIR = "/home/data/chroma_advanced_v2"
# BM25_INDEX_PATH = "/home/data/bm25_advanced_v2/bm25_index.pkl"
CHROMA_COLLECTION = "ai11_policy_advanced_v2_1024"
CHROMA_PERSIST_DIR = "/home/data/chroma_advanced_v2_1024"
BM25_INDEX_PATH = "/home/data/bm25_advanced_v2_1024/bm25_index.pkl"
DEFAULT_WEIGHTS: tuple[float, float] = (0.3, 0.7)

def measure_time(func):
    """함수 실행 시간을 재서 출력해주는 데코레이터.
    60초 이상이면 '분 초' 형태로, 미만이면 '초' 단위로 출력한다.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start

        print(f"[{func.__name__}] 실행 시간: {_format_elapsed(elapsed)}")

        return result
    return wrapper


def _format_elapsed(seconds: float) -> str:
    """초 단위 float를 '분 초' 또는 '초' 형식의 문자열로 변환한다.
    measure_time 데코레이터와, 함수 내부에서 구간별로 시간을 찍을 때 공통으로 사용한다.
    """
    if seconds >= 60:
        minutes = int(seconds // 60)   # 몫 -> 분
        secs = seconds % 60            # 나머지 -> 초
        return f"{minutes}분 {secs:.3f}초"
    return f"{seconds:.3f}초"

# ══════════════════════════════════════════════════════════════
# 1) 한국어 토큰화 (BM25의 핵심)
# ══════════════════════════════════════════════════════════════
_kiwi = Kiwi(num_workers=1)
# _kiwi.add_user_word("이러닝", "NNG", score=0.0)

# 팀 회의에서 합의된 BM25 토큰화 정책
# - 품사(POS)로는 전혀 걸러내지 않는다 -> 조사("은/는/이/가" 등), 어미("-다", "-고" 등)도 전부 토큰으로 남김
# - 복합명사 병합도 하지 않는다 -> Kiwi가 쪼갠 형태소 단위 그대로 사용
# - 대신 각 형태소의 표면형(form)에서 "문자/숫자/하이픈"이 아닌 특수문자만 제거
BM25_POS_POLICY_ID = "strip_special_characters_v1"
BM25_EXCLUDED_POS_PREFIXES: tuple[str, ...] = ()  # 품사 제외 없음
BM25_TOKEN_NORMALIZATION = "strip_special_casefold"
BM25_SPECIAL_CHARS = re.compile(r"[^\w-]", re.UNICODE)  # \w=문자/숫자/밑줄, -=하이픈 만 허용


def korean_tokenize(text: str) -> list[str]:
    """한국어 문장을 Kiwi로 형태소 분석한 뒤, 품사 필터링·복합명사 병합 없이
    특수문자만 제거해서 반환한다.
    처리 순서:
      1) Kiwi로 형태소 분석 (조사/어미 포함 전부)
      2) 표면형에서 문자·숫자·하이픈이 아닌 기호(특수문자, 괄호, 문장부호 등) 제거
      3) 앞뒤에 남은 하이픈/공백 제거 후 casefold(대소문자 통일)
      4) 특수문자만 있었던 토큰(제거 후 빈 문자열)은 버림
    """
    raw_tokens = _kiwi.tokenize(text)

    tokens: list[str] = []
    for t in raw_tokens:
        normalized = BM25_SPECIAL_CHARS.sub("", t.form).strip("-").strip().casefold()
        if normalized:  # 특수문자만 있던 토큰은 제거 후 빈 문자열이 되므로 여기서 걸러짐
            tokens.append(normalized)

    return tokens


# ══════════════════════════════════════════════════════════════
# 2) 청크 문서 로딩
# ══════════════════════════════════════════════════════════════
@measure_time
def load_chunk_documents(batch_size: int = 600, min_tokens: int = 95) -> list[Document]:
    """Chroma 컬렉션에 저장된 모든 청크를 LangChain Document 리스트로 로드한다.

    min_tokens: metadata["token_count"]가 이 값 이하인 청크는 로드에서 제외한다.
      너무 짧은 청크(제목 한 줄, 표 조각 등)는 정보량이 적어 BM25/벡터 검색
      양쪽 모두에서 노이즈로 작용하기 쉬워서 기본적으로 걸러낸다.
    """
    import chromadb

    # [1] Chroma DB에 직접 접속
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = client.get_collection(name=CHROMA_COLLECTION)

    # [2] 전체 청크 개수 확인 (몇 번 나눠서 가져와야 하는지 계산하기 위함)
    total_count = collection.count()
    print(f"[load_chunk_documents] 전체 청크 개수: {total_count}")

    docs: list[Document] = []
    skipped_short = 0  # token_count <= min_tokens 라서 제외한 청크 개수 (로그 확인용)

    # [3] offset을 batch_size만큼씩 옮겨가며 나눠서 조회
    for offset in range(0, total_count, batch_size):
        raw = collection.get(
            include=["documents", "metadatas"],
            limit=batch_size,
            offset=offset,
        )

        # [4] 이번 batch 결과를 LangChain Document로 변환
        for i in range(len(raw["ids"])):
            content = raw["documents"][i]        # 청크 본문
            metadata = raw["metadatas"][i] or {}  # 메타데이터 (None 방지용 or {})
            # metadata 안에는 chunk_id, source_filename, issuer, project_amount_won,
            # token_count 등이 들어있음

            # 본문이 비어있는 청크는 BM25에 노이즈만 되므로 건너뛰기
            if not content:
                continue

            # 토큰 수가 min_tokens 이하인 청크는 제외
            # (metadata에 token_count가 없는 예외 케이스는 0으로 간주해 함께 제외)
            token_count = metadata.get("token_count", 0)
            if token_count <= min_tokens:
                skipped_short += 1
                continue

            docs.append(
                Document(
                    page_content=content,
                    metadata=metadata,
                )
            )

    print(
        f"[load_chunk_documents] Chroma에서 {len(docs)}개 청크 로드 완료 "
        # f"(token_count<={min_tokens}인 청크 {skipped_short}개 제외)"
    )
    return docs


# ══════════════════════════════════════════════════════════════
# 2-1) 메타데이터 필터 (Chroma의 where 문법과 비슷하게 사용)
# ══════════════════════════════════════════════════════════════
def _matches_filter(metadata: dict, metadata_filter: dict) -> bool:
    """metadata가 metadata_filter 조건을 만족하는지 확인한다.

    지원하는 형태:
      {"source_filename": "abc.hwp"}                 -> 정확히 일치하는 것만
      {"source_filename": {"$in": ["a.hwp", "b.hwp"]}} -> 여러 값 중 하나라도 일치하면 통과
    여러 key를 같이 넣으면 전부 만족(AND)해야 통과한다.
    """
    for key, condition in metadata_filter.items():
        value = metadata.get(key)
        if isinstance(condition, dict) and "$in" in condition:
            if value not in condition["$in"]:
                return False
        else:
            if value != condition:
                return False
    return True


# ══════════════════════════════════════════════════════════════
# 3) BM25 리트리버 (키워드 검색) — 미리 만들어둔 pickle 인덱스 사용
# ══════════════════════════════════════════════════════════════
@lru_cache(maxsize=1)
def _load_bm25_artifact() -> dict:
    """팀원이 미리 만들어둔 BM25 인덱스 pickle을 한 번만 로드해서 캐싱한다.

    파일 안에는 다음 3개가 들어있다 (전부 같은 순서로 1:1 매칭됨):
      - chunk_ids       : 각 위치의 chunk_id 리스트
      - tokenized_corpus: 이미 토큰화된 코퍼스 (list[list[str]])
      - index           : 완성된 rank_bm25.BM25Okapi 객체 (재구축 불필요)
    """
    t0 = time.perf_counter()
    with open(BM25_INDEX_PATH, "rb") as f:
        artifact = pickle.load(f)
    print(
        f"[_load_bm25_artifact] BM25 인덱스 로드 완료: {len(artifact['chunk_ids'])}개 청크 "
        f"({_format_elapsed(time.perf_counter() - t0)})"
    )
    return artifact


def build_bm25_retriever(
    documents: list[Document],
    k: int = 5,
    metadata_filter: dict | None = None,
) -> BM25Retriever:
    """BM25 리트리버를 만든다."""
    t0 = time.perf_counter()

    artifact = _load_bm25_artifact()
    chunk_ids: list[str] = artifact["chunk_ids"]
    tokenized_corpus: list[list[str]] = artifact["tokenized_corpus"]

    # chunk_id로 실제 Document(원문+metadata)를 바로 찾을 수 있게 딕셔너리로 준비
    doc_by_chunk_id = {d.metadata.get("chunk_id"): d for d in documents}

    t1 = time.perf_counter()
    print(f"[build_bm25_retriever] (1) chunk_id 매핑 준비: {_format_elapsed(t1 - t0)}")

    if metadata_filter:
        # pickle 순서(index) 중, 조건에 맞는 문서의 위치만 골라낸다
        selected_positions = [
            i
            for i, cid in enumerate(chunk_ids)
            if (doc := doc_by_chunk_id.get(cid)) is not None
            and _matches_filter(doc.metadata, metadata_filter)
        ]
        target_chunk_ids = [chunk_ids[i] for i in selected_positions]
        target_tokens = [tokenized_corpus[i] for i in selected_positions]

        from rank_bm25 import BM25Okapi

        vectorizer = BM25Okapi(target_tokens)  # 이미 토큰화되어 있으므로 Kiwi 없이 즉시 생성됨
        print(
            f"[build_bm25_retriever] 필터 적용: {len(chunk_ids)}개 중 {len(target_chunk_ids)}개 대상 "
            f"-> 새 BM25Okapi 생성"
        )
    else:
        # 필터가 없으면 pickle에 있는 완성된 인덱스를 그대로 재사용 (재구축 없음)
        target_chunk_ids = chunk_ids
        vectorizer = artifact["index"]

    t2 = time.perf_counter()
    print(f"[build_bm25_retriever] (2) BM25Okapi 준비: {_format_elapsed(t2 - t1)}")

    # BM25 인덱스 내부 순서와 정확히 같은 순서로 Document 리스트를 맞춘다
    aligned_documents: list[Document] = []
    missing = 0
    for cid in target_chunk_ids:
        doc = doc_by_chunk_id.get(cid)
        if doc is None:
            # Chroma에 없는 chunk_id (버전 불일치 등) -> 순서 유지를 위해 빈 문서로 채움
            doc = Document(page_content="", metadata={"chunk_id": cid})
            missing += 1
        aligned_documents.append(doc)

    # if missing:
    #     print(f"[build_bm25_retriever] 경고: chunk_id {missing}개를 Chroma에서 못 찾음 (버전 불일치 가능성)")

    t3 = time.perf_counter()
    print(f"[build_bm25_retriever] (3) 문서 정렬: {_format_elapsed(t3 - t2)}")

    retriever = BM25Retriever(
        vectorizer=vectorizer,
        docs=aligned_documents,
        k=k,
        preprocess_func=korean_tokenize,  # 검색 질의(query)에만 적용됨
    )

    t4 = time.perf_counter()
    print(f"[build_bm25_retriever] 총 소요 시간: {_format_elapsed(t4 - t0)}")

    return retriever


# ══════════════════════════════════════════════════════════════
# 4) 벡터 리트리버 (의미 검색) — 기존 Chroma 재사용
# ══════════════════════════════════════════════════════════════
def _to_chroma_where(metadata_filter: dict) -> dict:
    """metadata_filter(dict)를 Chroma where 절 문법으로 변환한다.
    Chroma는 최상위 key가 1개여야 해서, 조건이 2개 이상이면 $and로 묶는다.
    """
    conditions = [{key: condition} for key, condition in metadata_filter.items()]
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}

def build_vector_retriever(k: int = 5, metadata_filter: dict | None = None):
    """Naive RAG에서 만든 Chroma 벡터스토어를 읽기 전용으로 로드해 리트리버로 반환.

    metadata_filter: Chroma가 원래 지원하는 필터라 그대로 search_kwargs의 'filter'로 넘기면 됨.
      예: {"source_filename": "abc.hwp"} 또는 {"source_filename": {"$in": ["a.hwp", "b.hwp"]}}
    """
    embeddings = OpenAIEmbeddings(model=EMBEDDINGS_MODEL, api_key=OPENAI_API_KEY)
    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    # search_kwargs = {"k": k}
    search_kwargs = {"k": k, "fetch_k": k * 4, "lambda_mult": 0.5}
    if metadata_filter:
        # search_kwargs["filter"] = metadata_filter  # langchain_chroma는 'filter' 키워드로 받음
        search_kwargs["filter"] = _to_chroma_where(metadata_filter) 
    return vectorstore.as_retriever(
        search_type="mmr",  # "관련성"과 "다양성"을 동시에 고려하는 검색 방식
        search_kwargs=search_kwargs,
    )
    # return vectorstore.as_retriever(
    #     search_type="similarity",  # 쿼리와 가장 가까운 문서 k개를 그냥 순서대로 뽑음
    #     search_kwargs=search_kwargs,
    # )


# ══════════════════════════════════════════════════════════════
# 5) 하이브리드 리트리버 (BM25 + 벡터 결합)
#    EnsembleRetriever가 내부적으로 RRF(순위 결합)로 두 결과를 섞는다.
# ══════════════════════════════════════════════════════════════
@lru_cache(maxsize=1)
def _get_cached_documents() -> list[Document]:
    """청크 전체 로딩(Chroma 조회, 제일 오래 걸리는 부분)만 따로 캐싱한다.

    metadata_filter는 매번 달라질 수 있어서(dict는 lru_cache에 못 넣음) build_hybrid_retriever
    자체는 더 이상 캐싱하지 않는다. 대신 제일 무거운 이 부분만 캐싱해서, 필터가 바뀌어도
    Chroma 재조회 없이 메모리에 있는 문서 목록만 다시 필터링하면 되게 만든다.
    """
    return load_chunk_documents()


def build_hybrid_retriever(
    k: int = 5,
    weights: tuple[float, float] = DEFAULT_WEIGHTS,
    metadata_filter: dict | None = None,
) -> EnsembleRetriever:
    """BM25 + 벡터를 결합한 하이브리드 리트리버를 만든다.
    """
    documents = _get_cached_documents()

    bm25_retriever = build_bm25_retriever(documents, k=k, metadata_filter=metadata_filter)
    vector_retriever = build_vector_retriever(k=k, metadata_filter=metadata_filter)

    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=list(weights),
    )

## 5-1) 하이브리드 리트리버 캐시 (metadata_filter 없는 경우만)
@lru_cache(maxsize=4)
def _build_hybrid_retriever_cached(k: int, weights: tuple[float, float]) -> EnsembleRetriever:
    """metadata_filter가 없는(가장 흔한) 경우를 위한 캐시.
    같은 (k, weights)로 다시 호출되면 BM25Okapi 재구축 없이 캐시된 리트리버를 즉시 반환한다.
    """
    return build_hybrid_retriever(k=k, weights=weights, metadata_filter=None)


def get_hybrid_retriever(
    k: int = 30,
    weights: tuple[float, float] = DEFAULT_WEIGHTS,
    metadata_filter: dict | None = None,
) -> EnsembleRetriever:

    if metadata_filter:
        return build_hybrid_retriever(k=k, weights=weights, metadata_filter=metadata_filter)
    return _build_hybrid_retriever_cached(k, weights)


# ══════════════════════════════════════════════════════════════
# 6) 검색 함수 (기존 search_documents와 같은 반환 형태로 맞춤)
# ══════════════════════════════════════════════════════════════
@measure_time
def search_documents(
    query: str,
    k: int = 5,
    candidate_k: int = 10,
    metadata_filter: dict | None = None,
    # use_score_fusion: bool = True,
) -> list[dict[str, Any]]:
    """하이브리드 검색 + 리랭킹을 한 번에 수행한다."""
    t0 = time.perf_counter()
    retriever = get_hybrid_retriever(k=candidate_k, metadata_filter=metadata_filter)
    candidate_docs = retriever.invoke(query)
    t1 = time.perf_counter()
    print(f"[검색 단계] {t1 - t0:.2f}초, 후보 {len(candidate_docs)}개")
    
    # 리랭커(Qwen3-Reranker)로 "후보 전체"를 채점한다.
    #     주의: top_n을 k가 아니라 후보 개수 전체로 줘야 함
    all_reranked = reranker.rerank_documents(query, candidate_docs, top_n=len(candidate_docs))
    
    t2 = time.perf_counter()
    print(f"[리랭킹 단계] {t2 - t1:.2f}초")

    # use_score_fusion에 따라 최종 k개를 결정하는 방식이 갈림
    # if use_score_fusion:
        # docs = apply_score_fusion(query, all_reranked, top_n=k)
        # t3 = time.perf_counter()
        # print(f"[점수 융합 단계] {t3 - t2:.2f}초")
    # else:
        # docs = all_reranked[:k]   # fusion 안 쓰면 리랭커 순서 그대로 top-k
    docs = all_reranked[:k]
    reranked_chunk_ids = [doc.metadata.get("chunk_id") for doc in docs]
    print(f"[리랭킹 결과 청크 ID] {reranked_chunk_ids}")

    results: list[dict[str, Any]] = []
    for doc in docs:
        table_html = doc.metadata.get("table_html")
        text_value = table_html if table_html else doc.page_content
        results.append(
            {
                "id": doc.metadata.get("chunk_id"), # 청크 고유 ID
                "text": text_value,                             # 실제 검색 결과로 보여줄 텍스트 (표가 있으면 표 HTML, 없으면 원문)
                "file_nm": doc.metadata.get("source_filename"), # 파일명
                "file_type": doc.metadata.get("file_type"), # 파일 확장자
                "score": doc.metadata.get("rerank_score"), # 리랭커 점수 (rerank_documents가 채워줌)
                "page_start": doc.metadata.get("page_start"),  # 페이지 시작번호
                "project_name": doc.metadata.get("project_name"),  # 사업명
                "project_amount_won": doc.metadata.get("project_amount_won"),  # 사업 금액 (원)
                "notice_number": doc.metadata.get("notice_number"),  # 공고번호
                "notice_round": doc.metadata.get("notice_round"), # 공고차수
                "issuer": doc.metadata.get("issuer"),      # 발주기관
                "published_at": doc.metadata.get("published_at"), # 공고 게시일시
                "bid_start_at": doc.metadata.get("bid_start_at"), # 입찰 시작일시
                "bid_end_at": doc.metadata.get("bid_end_at"),     # 입찰 마감일시
            }
        )
    return results