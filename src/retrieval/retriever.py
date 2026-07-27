"""
# retriever.py
하이브리드 리트리버 (Advanced RAG)

구성:
- BM25 (키워드 검색)  : LangChain BM25Retriever + kiwipiepy 한국어 형태소 토큰화
- 벡터 (의미 검색)    : 기존 Chroma 벡터스토어 
- 결합               : EnsembleRetriever (RRF, weights로 비중 조절)
"""
import os
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

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION_NAME", "ai11_policy_advanced_parent_child_v3")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "/home/data/chroma_advanced_parent_child_v3")

def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start

        if elapsed >= 60:
            minutes = int(elapsed // 60)   # 몫 -> 분
            seconds = elapsed % 60         # 나머지 -> 초
            print(f"[{func.__name__}] 실행 시간: {minutes}분 {seconds:.3f}초")
        else:
            print(f"[{func.__name__}] 실행 시간: {elapsed:.3f}초")

        return result
    return wrapper

# ══════════════════════════════════════════════════════════════
# 1) 한국어 토큰화 (BM25의 핵심)
# ══════════════════════════════════════════════════════════════
_kiwi = Kiwi(num_workers=2)

_CONTENT_TAG_PREFIXES = ("N", "V", "M", "X")
_CONTENT_S_TAGS = {"SL", "SH", "SN"}            # 외국어, 한자, 숫자만
_EXCLUDE_TAGS = {"VX", "XSV"}                    # 보조용언(있다/없다 등), 동사파생접미사(하다/되다 등) 제외
_NOUN_TAGS_FOR_MERGE = {"NNG", "NNP", "NNB"}     # 복합명사로 합칠 대상(일반/고유/의존명사)


def korean_tokenize(text: str) -> list[str]:
    """한국어 문장을 형태소 단위로 쪼개고,
    1) 조사/어미/문장부호/보조용언 등 문법 요소는 제외하고
    2) 공백 없이 붙어있는 명사끼리는 복합명사로 합쳐서 반환한다.
    """
    raw_tokens = _kiwi.tokenize(text)

    # [1단계] 필요 없는 품사 걸러내기
    filtered = []
    for t in raw_tokens:
        tag = t.tag
        if not tag or tag in _EXCLUDE_TAGS:
            continue
        if tag[0] in _CONTENT_TAG_PREFIXES or tag in _CONTENT_S_TAGS:
            filtered.append(t)

    # [2단계] 붙어있는 명사끼리 하나로 합치기 (예: '예산'+'액' -> '예산액')
    merged = []
    for t in filtered:
        prev = merged[-1] if merged else None
        if (
            prev is not None
            and prev["tag"] in _NOUN_TAGS_FOR_MERGE
            and t.tag in _NOUN_TAGS_FOR_MERGE
            and prev["end"] == t.start   # 원문에서 공백 없이 바로 이어질 때만 합침
        ):
            prev["form"] += t.form
            prev["end"] = t.start + t.len
        else:
            merged.append({"form": t.form, "tag": t.tag, "end": t.start + t.len})

    return [m["form"] for m in merged]

# ══════════════════════════════════════════════════════════════
# 2) 청크 문서 로딩
# ══════════════════════════════════════════════════════════════
@measure_time
def load_chunk_documents(batch_size: int = 500) -> list[Document]:
    import chromadb

    # [1] Chroma DB에 직접 접속
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = client.get_collection(name=CHROMA_COLLECTION)

    # [2] 전체 청크 개수 확인 (몇 번 나눠서 가져와야 하는지 계산하기 위함)
    total_count = collection.count()
    print(f"[load_chunk_documents] 전체 청크 개수: {total_count}")

    docs: list[Document] = []

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

            # 본문이 비어있는 청크는 BM25에 노이즈만 되므로 건너뛰기
            if not content:
                continue

            docs.append(
                Document(
                    page_content=content,
                    metadata=metadata,
                )
            )

    print(f"[load_chunk_documents] Chroma에서 {len(docs)}개 청크 로드 완료")
    return docs


# ══════════════════════════════════════════════════════════════
# 3) BM25 리트리버 (키워드 검색) — 한국어 토큰화 적용
# ══════════════════════════════════════════════════════════════
def build_bm25_retriever(documents: list[Document], k: int = 5) -> BM25Retriever:
    """kiwipiepy 토큰화를 적용한 BM25 리트리버를 만든다."""
    bm25 = BM25Retriever.from_documents(
        documents,
        preprocess_func=korean_tokenize,  # ← 한국어 형태소 토큰화 연결
    )
    bm25.k = k  # 상위 k개 반환
    return bm25


# ══════════════════════════════════════════════════════════════
# 4) 벡터 리트리버 (의미 검색) — 기존 Chroma 재사용
# ══════════════════════════════════════════════════════════════
def build_vector_retriever(k: int = 5):
    """Naive RAG에서 만든 Chroma 벡터스토어를 읽기 전용으로 로드해 리트리버로 반환."""
    embeddings = OpenAIEmbeddings(model=EMBEDDINGS_MODEL, api_key=OPENAI_API_KEY)
    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


# ══════════════════════════════════════════════════════════════
# 5) 하이브리드 리트리버 (BM25 + 벡터 결합)
#    EnsembleRetriever가 내부적으로 RRF(순위 결합)로 두 결과를 섞는다.
#    weights: [BM25 비중, 벡터 비중] — 합이 1이 되게. 처음엔 반반으로 시작.
# ══════════════════════════════════════════════════════════════
@lru_cache(maxsize=8)
def build_hybrid_retriever(
    k: int = 5,
    weights: tuple[float, float] = (0.3, 0.7),
) -> EnsembleRetriever:
    documents = load_chunk_documents()

    bm25_retriever = build_bm25_retriever(documents, k=k)
    vector_retriever = build_vector_retriever(k=k)

    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=list(weights),
    )

# ══════════════════════════════════════════════════════════════
# 6) 검색 함수 (기존 search_documents와 같은 반환 형태로 맞춤)
#    → generation 단계(BidMateRAGSession.ask)가 그대로 받을 수 있게 인터페이스 통일
# ══════════════════════════════════════════════════════════════
@measure_time
def hybrid_search(query: str, k: int = 5) -> list[dict[str, Any]]:

    retriever = build_hybrid_retriever(k=k)
    docs = retriever.invoke(query)[:k]  # [Document, ...]

    results: list[dict[str, Any]] = []
    for doc in docs:
        results.append(
            {
                "id": doc.metadata.get("chunk_id"),
                "text": doc.page_content,
                "file_nm": doc.metadata.get("file_nm"),
                "score": None, 
                # "metadata": doc.metadata,
            }
        )
    return results