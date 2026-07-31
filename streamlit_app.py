"""
입찰메이트(BidMate) RAG Q&A — Streamlit 웹 화면

main.py는 터미널(콘솔)에서 input()으로 질문을 받는 방식이었는데,
이 파일은 같은 로직(질의 재작성 → 하이브리드 검색 → 답변 생성)을
브라우저에서 채팅창처럼 쓸 수 있게 그대로 옮긴 버전입니다.

바뀐 것: "화면(UI)"만 터미널 → 웹으로 바뀌었고,
RAG 파이프라인 자체(rewrite_query → search_documents → session.ask)는 main.py와 동일합니다.

실행 방법:
    uv run streamlit run streamlit_app.py
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv

from logging_config import setup_logging
from src.generation.generate_answer import BidMateRAGSession
from src.retrieval.retriever import search_documents

# -----------------------------------------------------------------
# 0. 초기 설정 (main.py와 동일한 부분)
# -----------------------------------------------------------------
load_dotenv()
setup_logging()

st.set_page_config(
    page_title="입찰메이트 · RFP RAG Q&A",
    page_icon="📄",
    layout="wide",
)


# -----------------------------------------------------------------
# 1. 비동기(async) 함수를 Streamlit에서 실행하기 위한 도우미
# -----------------------------------------------------------------
# session.rewrite_query()와 session.ask()는 main.py에서 'await'로 호출하던
# 비동기 함수다. 하지만 Streamlit은 기본적으로 동기(sync) 코드로 스크립트를
# 위에서 아래로 실행한다. 그래서 asyncio 이벤트 루프를 직접 만들어서 실행해줘야 한다.
#
# 주의: 질문마다 asyncio.run()을 새로 부르면, 매번 "새 이벤트 루프"가 만들어졌다가
# 닫히는데, OpenAI 비동기 클라이언트(AsyncOpenAI)는 내부적으로 처음 사용한 이벤트
# 루프에 연결을 물고 있어서 두 번째 질문부터 "Event loop is closed" 같은 에러가
# 날 수 있다. 그래서 세션(브라우저 탭)당 이벤트 루프를 하나만 만들어 재사용한다.
def get_event_loop() -> asyncio.AbstractEventLoop:
    if "event_loop" not in st.session_state:
        st.session_state.event_loop = asyncio.new_event_loop()
    return st.session_state.event_loop


def run_async(coro):
    """비동기 함수(coroutine)를 세션 전용 이벤트 루프에서 실행하고 결과를 돌려준다."""
    loop = get_event_loop()
    return loop.run_until_complete(coro)


# Streamlit 버전에 따라 fragment API 이름이 다르다 (1.37+ : st.fragment,
# 1.33~1.36 : st.experimental_fragment). 아주 오래된 버전이라 둘 다 없으면
# 그냥 원본 함수를 그대로 쓰도록(=아무 효과 없음) 안전하게 처리한다.
_fragment = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None)
if _fragment is None:
    def _fragment(func):
        return func


# -----------------------------------------------------------------
# 2. 세션 상태(state) 초기화
# -----------------------------------------------------------------
# Streamlit은 사용자가 버튼을 누르거나 채팅을 입력할 때마다 이 파일 전체를
# 처음부터 다시 실행한다(rerun). 그래서 "대화 세션 객체"와 "대화 기록"을
# st.session_state에 저장해둬야 재실행돼도 이전 대화가 사라지지 않는다.
if "session" not in st.session_state:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error(
            "`OPENAI_API_KEY`가 설정되어 있지 않습니다. "
            "프로젝트 루트의 `.env` 파일을 확인해주세요."
        )
        st.stop()  # API 키가 없으면 여기서 화면 렌더링을 멈춘다.

    model_name = os.getenv("OPENAI_MODEL", "gpt-5-mini")
    st.session_state.session = BidMateRAGSession(api_key=api_key, model=model_name)
    st.session_state.model_name = model_name

if "messages" not in st.session_state:
    # 화면에 그릴 대화 기록.
    # user 메시지  : {"role": "user", "content": "질문 텍스트"}
    # assistant 메시지: {"role": "assistant", "result": {...} 또는 None, "retrieved_docs": [...]}
    st.session_state.messages = []


# -----------------------------------------------------------------
# 3. 결과를 예쁘게 그려주는 함수들
# -----------------------------------------------------------------
def render_result(result: Dict[str, Any], key_prefix: str = "") -> None:
    """session.ask()가 돌려준 결과 dict를 화면에 표시한다.

    key_prefix: 메시지마다 고유한 문자열(예: 'msg_3')을 넘겨받아서,
    아래 expander의 key로 사용한다. 이게 없으면 근거 인용 개수가 같은
    두 메시지의 expander가 이름이 겹쳐서 오류가 날 수 있다.
    """
    # main.py에서는 이 값들을 전부 텍스트로 이어붙여서 로그로만 찍었는데,
    # 여기서는 각각을 구분해서 보기 좋게 나눠서 보여준다.
    confidence = result.get("confidence", "")
    confidence_badge = {
        "high": "🟢 높음",
        "medium": "🟡 보통",
        "low": "🔴 낮음",
    }.get(confidence, confidence or "-")

    # 1) 직접 답변
    st.markdown(result.get("answer", "") or "_(답변 없음)_")

    # 2) 요약 + 신뢰도
    col1, col2 = st.columns([4, 1])
    with col1:
        summary = result.get("summary", "")
        if summary:
            st.caption(f"요약: {summary}")
    with col2:
        st.caption(f"신뢰도: {confidence_badge}")

    # 3) 추가 확인이 필요한 경우 (needs_clarification)
    if result.get("needs_clarification"):
        question = result.get("clarification_question", "")
        st.info(f"❓ 추가 확인이 필요합니다: {question}")

    # 4) 문서 간 충돌 정보가 있는 경우
    conflicts = result.get("conflicts")
    if conflicts:
        st.warning(f"⚠️ 문서 간 충돌 정보가 발견됐습니다: {conflicts}")

    # 5) 근거 인용 (evidence) — 접었다 펼 수 있게 expander로
    evidence_list: List[Dict[str, Any]] = result.get("evidence", []) or []
    if evidence_list:
        with st.expander(
            f"🔎 근거 인용 {len(evidence_list)}건 보기",
            key=f"{key_prefix}_evidence" if key_prefix else None,
        ):
            for i, item in enumerate(evidence_list, start=1):
                source = item.get("source", "N/A")
                page = f"p.{item['page']}" if item.get("page") is not None else "-"
                chunk_id = item.get("chunk_id", "-")
                score = item.get("score")
                score_text = (
                    f"{score:.4f}" if isinstance(score, (int, float)) else "-"
                )
                quote = item.get("quote", "")

                st.markdown(
                    f"**[{i}]** `{source}` · {page} · chunk:`{chunk_id}` · score:`{score_text}`"
                )
                st.markdown(f"> {quote}")
                if i < len(evidence_list):
                    st.divider()


def render_retrieved_docs(retrieved_docs: List[Dict[str, Any]], key_prefix: str = "") -> None:
    """search_documents()가 반환한 검색 결과(리랭킹 후 top-k)를 표시한다.

    RFP 문서 메타데이터(사업명·발주기관·마감일 등)까지 함께 보여줘서,
    '컨설턴트가 실제로 궁금해할 정보'를 답변과 별도로 확인할 수 있게 한다.
    key_prefix는 render_result와 마찬가지로 expander의 key를 고유하게 만들기 위함이다.
    """
    if not retrieved_docs:
        return

    with st.expander(
        f"📂 검색된 문서 {len(retrieved_docs)}건 (리랭킹 후 상위 k개)",
        key=f"{key_prefix}_docs" if key_prefix else None,
    ):
        for doc in retrieved_docs:
            title = doc.get("project_name") or doc.get("file_nm") or "제목 없음"
            score = doc.get("score")
            score_text = f"{score:.3f}" if isinstance(score, (int, float)) else "-"

            st.markdown(f"**{title}**  ·  score: `{score_text}`")

            meta_line = " · ".join(
                filter(
                    None,
                    [
                        f"발주기관: {doc.get('issuer')}" if doc.get("issuer") else None,
                        f"금액: {doc.get('project_amount_won')}"
                        if doc.get("project_amount_won")
                        else None,
                        f"마감: {doc.get('bid_end_at')}" if doc.get("bid_end_at") else None,
                    ],
                )
            )
            if meta_line:
                st.caption(meta_line)

            st.caption(
                f"chunk_id: {doc.get('id')} · 파일: {doc.get('file_nm')} "
                f"(p.{doc.get('page_start')})"
            )

            text = doc.get("text") or ""
            preview = text[:300] + ("…" if len(text) > 300 else "")
            st.text(preview)
            st.divider()


def render_assistant_turn(msg: Dict[str, Any], key_prefix: str = "") -> None:
    """messages 리스트에 저장된 assistant 턴 하나를 화면에 그린다."""
    if msg.get("result") is None:
        # main.py의 "검색된 문서가 없습니다." 분기와 동일한 상황
        st.warning("검색된 문서가 없습니다. 다른 질문으로 다시 시도해보세요.")
        return

    render_result(msg["result"], key_prefix=key_prefix)
    render_retrieved_docs(msg.get("retrieved_docs", []), key_prefix=key_prefix)


@_fragment
def render_message(idx: int, msg: Dict[str, Any]) -> None:
    """메시지 1개(user 또는 assistant)를 화면에 그린다.

    @_fragment로 감싸는 이유:
    expander(검색된 문서 / 근거 인용)를 열고 닫을 때마다 앱 전체가 다시
    실행되면(rerun), 화면이 통째로 다시 그려지면서 스크롤이 아래로
    내려갔다가 다시 위로 튀는 현상이 생긴다. fragment 안에서는 '이
    메시지 하나'만 다시 그려지기 때문에 다른 메시지와 스크롤 위치가
    그대로 유지된다.
    """
    key_prefix = f"msg_{idx}"
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_assistant_turn(msg, key_prefix=key_prefix)


# -----------------------------------------------------------------
# 4. 사이드바 — 설정 & 초기화(reset) 버튼
# -----------------------------------------------------------------
with st.sidebar:
    st.title("📄 입찰메이트 (BidMate)")
    st.caption("RFP 문서 기반 RAG Q&A")

    st.markdown("---")
    st.markdown(f"**사용 모델**: `{st.session_state.model_name}`")

    top_k = st.slider(
        "검색할 문서 개수 (k)",
        min_value=1,
        max_value=10,
        value=5,
        help="main.py의 search_documents(rewritten_query, k=5)와 같은 값입니다.",
    )

    st.markdown("---")
    if st.button("🔄 대화 초기화 (reset)", use_container_width=True):
        # main.py에서 'reset'을 입력했을 때와 같은 동작:
        # session 내부 상태(previous_response_id 등)를 지우고,
        # 화면에 보이는 대화 기록도 함께 지운다.
        st.session_state.session.reset()
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("멀티턴 대화이므로, 이전 질문의 맥락을 이어서 물어볼 수 있습니다.")


# -----------------------------------------------------------------
# 5. 메인 화면 — 지금까지의 대화 기록 표시
# -----------------------------------------------------------------
st.header("입찰메이트 RAG 멀티턴 Q&A")
st.caption("RFP(제안요청서) 내용에 대해 자유롭게 질문해보세요.")

for idx, msg in enumerate(st.session_state.messages):
    render_message(idx, msg)


# -----------------------------------------------------------------
# 6. 사용자 입력 처리 (main.py의 while 루프에 해당하는 부분)
# -----------------------------------------------------------------
query: Optional[str] = st.chat_input("RFP에 대해 궁금한 점을 질문해보세요")

if query:
    session: BidMateRAGSession = st.session_state.session

    # 6-1) 사용자 메시지 저장 + 화면 표시 (history 루프와 동일한 render_message 재사용)
    st.session_state.messages.append({"role": "user", "content": query})
    render_message(len(st.session_state.messages) - 1, st.session_state.messages[-1])

    # 6-2) assistant 턴 처리 (main.py와 동일한 순서)
    with st.chat_message("assistant"):
        with st.spinner("질문을 분석하고 문서를 검색하는 중..."):
            # ① 후속 질문을 검색하기 좋은 standalone 질문으로 재작성
            rewritten_query = run_async(session.rewrite_query(query))
            # ② 하이브리드 검색(BM25 + 벡터 + 리랭킹)
            retrieved_docs = search_documents(rewritten_query, k=top_k)

        if not retrieved_docs:
            st.warning("검색된 문서가 없습니다. 다른 질문으로 다시 시도해보세요.")
            st.session_state.messages.append(
                {"role": "assistant", "result": None, "retrieved_docs": []}
            )
        else:
            with st.spinner("답변을 생성하는 중..."):
                # ③ 검색 결과를 근거로 답변 생성 (멀티턴 문맥 유지)
                result = run_async(
                    session.ask(
                        query=query,
                        retrieved_docs=retrieved_docs,
                        rewritten_query=rewritten_query,
                    )
                )

            # 다음 리런부터는 render_message(new_idx, ...)로 그려질 메시지이므로,
            # key_prefix를 미리 같은 규칙("msg_{idx}")으로 맞춰준다.
            new_idx = len(st.session_state.messages)
            render_result(result, key_prefix=f"msg_{new_idx}")
            render_retrieved_docs(retrieved_docs, key_prefix=f"msg_{new_idx}")

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "result": result,
                    "retrieved_docs": retrieved_docs,
                }
            )