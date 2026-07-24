from langchain_core.prompts import ChatPromptTemplate
REWRITE_PROMPT = ChatPromptTemplate.from_template(
    "다음 사용자 질문을 사내 규정 문서 검색에 적합하게 재작성하세요.\n\n"
    "규칙:\n- SW-210 같은 모델명·코드는 반드시 그대로 유지하세요.\n"
    "- \"아프다\", \"몸이 안 좋다\", \"사흘/나흘\"은 병가, 3일/4일 등 규정 표현으로 바꾸세요.\n"
    "- 현행 규정을 묻는 것이므로 \"현행\"을 포함하세요.\n"
    "원본 질문: {question}\n재작성:"
)
RERANK_PROMPT = ChatPromptTemplate.from_template(
    "질문과 후보 문단들이 주어집니다. 질문에 **실제로 답할 수 있는 구체적 정보**를 담은 문단을 고르세요.\n"
    "주의: \"어디로 문의하라\" 등 안내문이나 [구버전 — 폐지] 문단은 관련성이 낮습니다.\n\n"
    "질문: {question}\n\n후보:\n{candidates}\n\n"
    "가장 관련 높은 문단 번호 {top_n}개를 관련도 순으로 JSON 배열로만 출력하세요. 예: [3, 0, 7]"
)
ANSWER_PROMPT = ChatPromptTemplate.from_template(
    "당신은 사내 규정 안내 도우미입니다. 아래 [문서]에 근거해서만 답변하세요.\n\n"
    "규칙:\n- \"[구버전 — 폐지]\" 표시가 있는 문서는 근거로 사용하지 마세요.\n"
    "- 문서에 전혀 근거가 없으면 \"규정에서 확인되지 않습니다\"라고 답하세요.\n\n"
    "[문서]\n{context}\n\n질문: {question}\n답변:"
)
