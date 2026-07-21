# main.py

import logging
import os
import time
from datetime import datetime

from dotenv import load_dotenv

from logging_config import setup_logging
from src.generation.generate_answer import BidMateRAGSession
from src.retrieval.retriever import search_documents

load_dotenv()
logger = logging.getLogger(__name__)


def main():
    """
    멀티턴 RAG 실행 함수

    동작 방식:
    1. 프로그램 시작 시 세션 객체를 한 번만 생성한다.
    2. 사용자가 질문을 입력할 때마다 retriever로 관련 문서를 다시 검색한다.
    3. 같은 session 객체로 session.ask()를 반복 호출한다.
    4. session 내부의 previous_response_id가 유지되므로 멀티턴이 된다.
    """

    logger.info("프로그램 시작")

    # 환경변수에서 OpenAI API 키 읽기
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # API 키가 없으면 예외 처리
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY가 설정되어 있지 않습니다.")

    # 세션 객체는 프로그램 시작 시 한 번만 생성
    # 이 객체가 살아있는 동안 previous_response_id가 유지됨
    session = BidMateRAGSession(api_key=openai_api_key)

    print("입찰메이트 RAG 멀티턴 Q&A를 시작합니다.")
    print("종료하려면 'exit' 또는 'quit'를 입력하세요.")
    print("새 대화를 시작하려면 'reset'을 입력하세요.")
    print()

    # 멀티턴 대화 루프 시작
    while True:
        # 사용자 질문 입력
        query = input("질문 > ").strip()

        # 종료 명령 처리
        if query.lower() in ["exit", "quit"]:
            print("프로그램을 종료합니다.")
            break

        # 세션 초기화 명령 처리
        if query.lower() == "reset":
            session.reset()
            print("대화 세션이 초기화되었습니다.")
            print()
            continue

        # 빈 입력 방지
        if not query:
            print("질문을 입력해주세요.")
            print()
            continue

        logger.info("사용자 질문: %s", query)

        # ----------------------------
        # 1. Retrieval 단계
        # ----------------------------
        retrieval_start_dt = datetime.now()
        retrieval_start_perf = time.perf_counter()

        logger.info(
            "문서 검색 시작 | 시작시각=%s",
            retrieval_start_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        )

        # 현재 질문 기준으로 관련 문서 검색
        retrieved_docs = search_documents(query, k=5)

        retrieval_end_dt = datetime.now()
        retrieval_elapsed = time.perf_counter() - retrieval_start_perf

        logger.info("검색된 문서 수: %d", len(retrieved_docs))
        logger.info(
            "문서 검색 종료 | 종료시각=%s | 소요시간=%.3f초",
            retrieval_end_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            retrieval_elapsed,
        )

        # 검색 결과가 없으면 바로 다음 턴으로 넘어감
        if not retrieved_docs:
            print("검색된 문서가 없습니다.")
            print()
            continue

        # ----------------------------
        # 2. Generation 단계
        # ----------------------------
        ask_start_dt = datetime.now()
        ask_start_perf = time.perf_counter()

        logger.info(
            "session.ask 시작 | 시작시각=%s",
            ask_start_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        )

        # 같은 session 객체를 계속 사용하므로 멀티턴 유지
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
        logger.info("현재 previous_response_id: %s", session.previous_response_id)

        # ----------------------------
        # 3. 결과 출력
        # ----------------------------
        print()
        print("===== 직접 답변 =====")
        print(result["answer"])
        print()

        print("===== 요약 =====")
        print(result["summary"])
        print()

        # print("===== 추출 필드 =====")
        # print(result["fields"])
        # print()

        print("===== 근거 문서 =====")
        for citation in result["citations"]:
            print(
                f"- source: {citation['source']} | "
                f"chunk_id: {citation['chunk_id']} | "
                f"score: {citation['score']}"
            )
        print()

        print("===== 근거 인용 =====")
        for item in result["evidence_quotes"]:
            print(
                f"- source: {item['source']} | "
                f"chunk_id: {item['chunk_id']} | "
                f"quote: {item['quote']}"
            )
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
