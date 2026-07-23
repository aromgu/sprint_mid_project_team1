# ══════════════════════════════════════════════════════════════
# app.py  ―  입찰메이트 RAG 멀티턴 Q&A (Streamlit 버전)
# 기존 main.py 의 while 루프 / input / print 를
# Streamlit 화면으로 바꾼 것입니다.
# RAG 로직(search_documents, BidMateRAGSession)은 그대로 재사용합니다.
# 실행:
#   uv add streamlit
#   uv run streamlit run app.py
# ══════════════════════════════════════════════════════════════

import logging
import os

import streamlit as st
from dotenv import load_dotenv

# ─ 기존 main.py 와 똑같이 님의 모듈들을 그대로 import ─
from logging_config import setup_logging
from src.generation.generate_answer import BidMateRAGSession
from src.retrieval.retriever import search_documents


# ─ 초기 설정 (프로그램에서 딱 한 번만 실행되게 처리) ─
# @st.cache_resource : Streamlit이 재실행돼도 이 함수 결과를 재사용(캐시)함.
#                      로깅 설정이 매번 중복 실행되는 걸 막아줌.
@st.cache_resource
def init():
    load_dotenv()  # .env 파일에서 환경변수 로드
    setup_logging()  # 로깅 설정
    return logging.getLogger(__name__)


logger = init()


# ──────────────────────────────────────────────────────────────
# 세션(session) 객체 준비  ★이번 수정의 핵심★
#
# 기존 main.py: session = BidMateRAGSession(...) 를 한 번 만들고 재사용
# Streamlit  : 입력마다 코드가 처음부터 다시 실행되므로,
#              그냥 만들면 매 질문마다 session이 새로 생겨 멀티턴이 깨짐.
#   → st.session_state 에 저장해두면 재실행돼도 유지됨.
# ──────────────────────────────────────────────────────────────
def get_session():
    # session_state 안에 아직 session이 없을 때만 새로 생성
    if "rag_session" not in st.session_state:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("OPENAI_API_KEY가 설정되어 있지 않습니다. .env 파일을 확인하세요.")
            st.stop()  # 여기서 실행 중단
        # 딱 한 번만 생성되어 session_state에 보관됨
        st.session_state.rag_session = BidMateRAGSession(api_key=api_key)
    return st.session_state.rag_session


# ──────────────────────────────────────────────────────────────
# 화면 상단
# ──────────────────────────────────────────────────────────────
st.title("입찰메이트 RAG Q&A")
st.caption("문서를 검색해 답변하는 멀티턴 챗봇입니다.")

# ─ 사이드바: 새 대화 시작(reset) 버튼 ─
#   기존 main.py 의 'reset' 명령을 버튼으로 옮긴 것
with st.sidebar:
    st.header("설정")
    if st.button(" 새 대화 시작 (reset)"):
        if "rag_session" in st.session_state:
            st.session_state.rag_session.reset()  # 세션 내부 초기화
        st.session_state.messages = []  # 화면 대화기록도 비움
        st.success("대화 세션이 초기화되었습니다.")

# ─ 화면에 그릴 대화기록 저장소 준비 ─
if "messages" not in st.session_state:
    st.session_state.messages = []


# ──────────────────────────────────────────────────────────────
# 화면에 그려주는 함수
#   보여주는 것: ① LLM 답변  ② 검색된 청크 ID 전체  ③ 인용된 청크(ID + 내용)
#
#   record = {"result": ..., "retrieved_ids": [...], "retrieved_docs": [...]}
# ──────────────────────────────────────────────────────────────
def render_result(record):
    result = record["result"]
    retrieved_ids = record["retrieved_ids"]
    retrieved_docs = record["retrieved_docs"]  # 본문을 찾기 위해 원본 문서도 보관

    # ① LLM 답변
    st.markdown("### 답변")
    st.write(result["answer"])

    # ② 리트리버가 검색한 청크 ID 전체
    st.markdown("** 검색된 청크 ID**")
    st.write(", ".join(str(cid) for cid in retrieved_ids))

    # ③ LLM이 인용한 청크: ID + 본문 내용
    cited_ids = [c["chunk_id"] for c in result.get("citations", [])]
    st.markdown("**인용된 청크 (내용 포함)**")

    if not cited_ids:
        st.caption("인용된 청크가 없습니다.")
    else:
        # chunk_id → 원본 문서를 빠르게 찾기 위한 딕셔너리 만들기
        #   {청크ID: doc} 형태. 검색된 문서들로 미리 인덱스를 만들어 둠
        docs_by_id = {doc["metadata"]["chunk_id"]: doc for doc in retrieved_docs}

        # 인용된 청크마다 ID와 본문을 함께 표시
        for cid in cited_ids:
            doc = docs_by_id.get(cid)  # 그 ID의 원본 문서 찾기
            if doc is None:
                # 인용된 청크가 검색결과에 없을 때 (드묾)
                st.markdown(f"- **chunk_id: {cid}** — (검색결과에서 본문을 찾지 못함)")
                continue

            # 리트리버가 주는 본문은 text 필드에 있음
            text = doc["text"]
            # 접이식으로 정리 → ID는 제목, 펼치면 본문
            with st.expander(f"chunk_id: {cid}"):
                st.write(text)


# ──────────────────────────────────────────────────────────────
# 지금까지의 대화를 화면에 다시 그리기
#   (Streamlit은 재실행되므로 매번 전체 대화를 다시 그려야 함)
# ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.write(msg["content"])  # 사용자 질문은 텍스트 그대로
        else:
            render_result(msg["content"])  # 답변은 딕셔너리 → 함수로 그림


# ──────────────────────────────────────────────────────────────
# 입력창 + 실제 RAG 실행 (기존 while 루프 안쪽 로직)
# ──────────────────────────────────────────────────────────────
query = st.chat_input("질문을 입력하세요")

if query:
    session = get_session()  # 유지되고 있는 session 객체 가져오기

    # ─ 사용자 질문 표시 & 기록 ─
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    logger.info("사용자 질문: %s", query)

    # ─ 답변 생성 (검색 → session.ask) ─
    with st.chat_message("assistant"):
        # ① Retrieval: 관련 문서 검색
        with st.spinner("문서 검색 중..."):
            retrieved_docs = search_documents(query, k=5)

        # 검색 결과가 없으면 안내하고 종료
        if not retrieved_docs:
            st.warning("검색된 문서가 없습니다.")
        else:
            # 검색된 청크 ID 전체 추출 (main.py 의 로그와 동일한 값)
            retrieved_ids = [doc["metadata"]["chunk_id"] for doc in retrieved_docs]
            logger.info("검색된 문서: %s", retrieved_ids)
            logger.info("검색된 문서 수: %d", len(retrieved_docs))

            # ② Generation: 같은 session으로 ask → 멀티턴 유지
            with st.spinner("답변 생성 중..."):
                result = session.ask(query, retrieved_docs)

            logger.info("신뢰도: %s", result.get("confidence"))
            logger.info("현재 previous_response_id: %s", session.previous_response_id)

            # ③ 답변 + 검색된 청크 ID + 원본 문서(본문 표시용) 를 하나로 묶음
            record = {
                "result": result,
                "retrieved_ids": retrieved_ids,
                "retrieved_docs": retrieved_docs,  # 인용 청크 본문을 찾기 위해 보관
            }

            # ④ 화면에 그림
            render_result(record)

            # ⑤ 기록에 저장 → 다음 턴에서 위쪽에 다시 그려짐
            st.session_state.messages.append({"role": "assistant", "content": record})
