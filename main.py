import logging
import time
from datetime import datetime

from logging_config import setup_logging
from src.generation.generate_answer import BidMateRAGSession

from src.retrieval.retriever import search_documents

logger = logging.getLogger(__name__)


def main():
    """
    프로그램 실행 진입점

    여기에는 실제 실행 코드만 넣는다.
    즉,
    - 세션 객체 생성
    - retriever 결과 준비
    - 질문 실행
    - 결과 출력

    같은 코드가 들어간다.
    """

    logger.info("프로그램 시작")

    # 세션 객체 생성
    # 실제 환경에서는 "YOUR_API_KEY" 대신 환경변수 사용 권장
    session = BidMateRAGSession(api_key="YOUR_API_KEY")

    # retriever가 top-k로 뽑아줬다고 가정한 문서 조각 예시
    # 실제 프로젝트에서는 이 부분이 FAISS/Chroma/BM25 등의 검색 결과로 대체됨

    # 사용자 질문 정의
    query = "이 사업의 예산, 수행기간, 제출기한을 알려줘"
    logger.info("사용자 질문: %s", query)

    # retriever를 통해 top-k 문서 검색
    retrieval_start_dt = datetime.now()
    retrieval_start_perf = time.perf_counter()
    logger.info(
        "문서 검색 시작 | 시작시각=%s",
        retrieval_start_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
    )

    retrieved_docs = search_documents(query, k=5)

    retrieval_end_dt = datetime.now()
    retrieval_elapsed = time.perf_counter() - retrieval_start_perf
    logger.info("검색된 문서 수: %d", len(retrieved_docs))
    logger.info(
        "문서 검색 종료 | 종료시각=%s | 소요시간=%.3f초",
        retrieval_end_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        retrieval_elapsed,
    )

    # 세션 객체를 사용해 질문 수행
    ask_start_dt = datetime.now()
    ask_start_perf = time.perf_counter()
    logger.info(
        "session.ask 시작 | 시작시각=%s",
        ask_start_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
    )

    result = session.ask(query, retrieved_docs)

    ask_end_dt = datetime.now()
    ask_elapsed = time.perf_counter() - ask_start_perf
    logger.info("session.ask 완료")
    logger.info(
        "session.ask 종료 | 종료시각=%s | 소요시간=%.3f초",
        ask_end_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        ask_elapsed,
    )
    logger.info("신뢰도: %s", result.get("confidence"))

    # 결과 출력
    print("===== 직접 답변 =====")
    print(result["answer"])
    print()

    print("===== 요약 =====")
    print(result["summary"])
    print()

    print("===== 추출 필드 =====")
    print(result["fields"])
    print()

    print("===== 근거 문서 =====")
    print(result["citations"])
    print()

    print("===== 근거 인용 =====")
    print(result["evidence_quotes"])
    print()

    print("===== 신뢰도 =====")
    print(result["confidence"])
    print()

    print("===== 추가 확인 필요 여부 =====")
    print(result["needs_clarification"])
    print()

    print("===== 확인 질문 =====")
    print(result["clarification_question"])
    print()

    print("===== 충돌 정보 =====")
    print(result["conflicts"])
    print()


if __name__ == "__main__":
    setup_logging()
    main()
