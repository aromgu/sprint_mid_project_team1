import asyncio
import logging
import os
import time
from datetime import datetime

from dotenv import load_dotenv

from logging_config import setup_logging
from src.generation.generate_answer import BidMateRAGSession
from src.retrieval.retriever import search_documents

load_dotenv()

# -------------------------------------------------------------
# 로거 설정
# -------------------------------------------------------------
setup_logging()
logger = logging.getLogger(__name__)

# httpx 및 openai 라이브러리의 HTTP 요청 로그(HTTP Request: POST ...) 차단
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


async def main():
    """
    멀티턴 RAG 실행 함수

    동작 방식:
    1. 프로그램 시작 시 세션 객체를 한 번만 생성한다.
    2. 사용자가 질문을 입력할 때마다 retriever로 관련 문서를 다시 검색한다.
    3. 같은 session 객체로 session.ask()를 반복 호출한다.
    4. session 내부의 previous_response_id가 유지되므로 멀티턴이 된다.
    """

    logger.info("프로그램 시작")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY가 설정되어 있지 않습니다.")

    session = BidMateRAGSession(api_key=openai_api_key)

    print("입찰메이트 RAG 멀티턴 Q&A를 시작합니다.")
    print("종료하려면 'exit' 또는 'quit'를 입력하세요.")
    print("새 대화를 시작하려면 'reset'을 입력하세요.")
    print()

    while True:
        query = input("질문 > ").strip()

        if query.lower() in ["exit", "quit"]:
            print("프로그램을 종료합니다.")
            break

        if query.lower() == "reset":
            session.reset()
            print("대화 세션이 초기화되었습니다.")
            print()
            continue

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

        rewritten_query = await session.rewrite_query(query)
        logger.info("재작성 질문: %s", rewritten_query)

        retrieved_docs = search_documents(rewritten_query, k=5)

        retrieval_end_dt = datetime.now()
        retrieval_elapsed = time.perf_counter() - retrieval_start_perf

        logger.info("검색된 문서 수: %d", len(retrieved_docs))
        logger.info(
            "문서 검색 종료 | 종료시각=%s | 소요시간=%.3f초",
            retrieval_end_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            retrieval_elapsed,
        )

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

        result = await session.ask(
            query=query,
            retrieved_docs=retrieved_docs,
            rewritten_query=rewritten_query,
        )

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
        # 3. 근거 문서 포맷팅 및 출력
        # ----------------------------
        evidence_list = result.get("evidence", [])
        if evidence_list:
            evidence_lines = []
            for item in evidence_list:
                source = item.get("source", "N/A")
                page = f"p.{item['page']}" if item.get("page") is not None else ""
                chunk_id = (
                    f"chunk:{item['chunk_id']}"
                    if item.get("chunk_id") is not None
                    else ""
                )
                score = (
                    f"score:{item['score']:.4f}"
                    if item.get("score") is not None
                    else ""
                )
                quote = item.get("quote", "")

                meta_info = " | ".join(filter(None, [source, page, chunk_id, score]))
                evidence_lines.append(f'- [{meta_info}]\n  인용: "{quote}"')
            evidence_text = "\n".join(evidence_lines)
        else:
            evidence_text = "(근거 인용 없음)"

        answer_block = "\n\n".join(
            [
                "===== 직접 답변 =====",
                str(result.get("answer", "")),
                "===== 요약 =====",
                str(result.get("summary", "")),
                "===== 근거 문서 및 인용 =====",
                evidence_text,
                "===== 신뢰도 =====",
                str(result.get("confidence", "")),
                "===== 추가 확인 필요 여부 =====",
                str(result.get("needs_clarification", "")),
                "===== 확인 질문 =====",
                str(result.get("clarification_question", "")),
                "===== 충돌 정보 =====",
                str(result.get("conflicts", "")),
            ]
        )

        # 터미널 화면 출력
        # print(answer_block)
        # print()

        # 로그 파일 및 로거 기록
        logger.info(
            "--- [답변 생성 결과 시작] ---\n%s\n--- [답변 생성 결과 끝] ---",
            answer_block,
        )


if __name__ == "__main__":
    asyncio.run(main())
