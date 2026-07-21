import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
embeddings_model = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")


# ─────────────────────────────────────────────
# 1) 임베딩 모델 준비
# ─────────────────────────────────────────────
embeddings = OpenAIEmbeddings(model=embeddings_model, api_key=openai_api_key)

# ─────────────────────────────────────────────
# 2) 기존에 저장된 Chroma DB 불러오기 (읽기 전용으로 사용)
# ─────────────────────────────────────────────
vectorstore = Chroma(
    collection_name="ai11_policy",  # 저장할 때 썼던 컬렉션 이름과 동일해야 함
    embedding_function=embeddings,  # 쿼리를 벡터로 바꿔줄 임베딩 모델
    persist_directory="/home/data/chroma",  # 기존 DB가 저장된 폴더 경로
)

# ─────────────────────────────────────────────
# 3) 체인(chain)에 연결할 때 쓰는 표준 리트리버
# ─────────────────────────────────────────────
retriever = vectorstore.as_retriever(
    search_type="similarity",  # 단순 유사도 기반 검색
    search_kwargs={"k": 5},  # 상위 5개 문서 반환
)


# ─────────────────────────────────────────────
# 4) 실제 컬렉션 스키마에 맞춘 검색 함수
# ─────────────────────────────────────────────
def search_documents(query: str, k: int = 5) -> list[dict]:
    """
    Args:
        query: 검색할 질문 문자열
        k: 가져올 문서 개수 (기본값 5)

    Returns:
        [
            {
                "id": "20250012_01",              # 공고번호_공고차수
                "text": "...",                     # raw_text (청크 본문)
                "file_nm": "rfp_001.pdf",           # 파일명
                "score": 0.93,                      # 유사도 점수 (클수록 유사)
                "metadata": {...},                  # 원본 메타데이터 전체 (혹시 몰라 함께 보관)
            },
            ...
        ]
    """
    # 정규화된 유사도 점수(0~1, 클수록 유사)를 함께 반환하는 메서드 사용
    results = vectorstore.similarity_search_with_relevance_scores(query, k=k)

    retrieved_docs = []

    # for doc, score in results:
    #     print(doc.metadata.keys())  # 실제로 어떤 키 이름들이 있는지 확인
    #     break  # 하나만 보고 멈추기

    for doc, score in results:
        retrieved_docs.append(
            {
                "id": doc.metadata.get("chunk_id"),
                "text": doc.page_content,
                "file_nm": doc.metadata.get("file_nm"),
                "score": float(score),
                "metadata": doc.metadata,
            }
        )

    return retrieved_docs

