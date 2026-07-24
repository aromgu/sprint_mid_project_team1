from typing import List

from langchain_core.documents import Document

from src.preprocessing.clean_text import clean_text


def load_documents() -> List[Document]:
    warranty = [
        ("보증-03", "SW-200 스마트워치의 보증 기간은 구매일로부터 24개월이며, 정품 등록 시 6개월이 추가 연장됩니다. 보증 기간 내 정상 사용 중 발생한 고장은 무상 수리됩니다."),
        ("보증-04", "SW-210 스마트워치의 보증 기간은 구매일로부터 15개월이며, 리퍼비시 제품으로 보증 연장은 불가능합니다. 보증 기간 내 정상 사용 중 발생한 고장은 무상 수리됩니다."),
        ("보증-08", "SW-410 스마트워치의 보증 기간은 구매일로부터 24개월입니다. 단, 해외 구매분은 국내 무상 보증 대상에서 제외되며 유상 수리만 가능합니다."),
    ]
    refund = [("환불-02", "상품에 하자가 있는 경우 수령 후 30일 이내에 무상 교환 또는 환불이 가능하며, 이때 배송비는 회사가 부담합니다.")]
    hr = [
        ("인사-연차", "정규직 직원의 연차 휴가는 입사 첫 해에 총 15일이 부여됩니다. 3년 이상 근속 시 2년마다 1일씩 가산되며, 최대 25일까지 늘어납니다."),
        ("인사-병가", "병가는 연간 최대 10일까지 유급으로 사용할 수 있으며, 3일 이상 연속 사용 시 의사 진단서를 제출해야 합니다."),
        ("인사-육아", "출산 전후 휴가는 90일(다태아 120일)이 부여되며, 육아휴직은 만 8세 이하 자녀 1명당 최대 1년까지 사용할 수 있습니다."),
        ("인사-출장국내", "국내 출장 시 일비는 1일 3만원이며, 숙박비는 실비로 정산됩니다. 법인카드 사용을 원칙으로 합니다."),
    ]
    faq = [("FAQ-환불", "환불 신청, 교환 신청, 환불 배송비, 교환 배송비 관련 문의가 많습니다. 자세한 환불 기준은 환불 규정 문서를 확인하세요.")]
    old = [("구버전-연차", "※ [구버전 — 2023년 12월 폐지, 효력 없음] 정규직 직원의 연차 휴가는 입사 첫 해에 총 10일이 부여된다.")]

    raw_data = [("보증규정", warranty), ("환불규정", refund), ("인사규정", hr), ("고객센터FAQ", faq), ("구버전(폐지)", old)]

    documents = []
    for source, items in raw_data:
        for cid, text in items:
            cleaned_text = clean_text(text)
            documents.append(Document(page_content=cleaned_text, metadata={"id": cid, "source": source}))

    return documents
