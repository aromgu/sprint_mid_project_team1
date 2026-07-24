# RAG 시스템 구현 로드맵
## 9개 RFP 난이도별 단계적 알고리즘 적용 전략

**작성일**: 2026-07-20  
**목표**: 난이도 하 → 중 → 상으로 점진적 난이도를 높이며 알고리즘 고도화

---

## 📊 전체 구조

```
테스트 단계: 난이도 하 (3개) → 난이도 중 (3개) → 난이도 상 (3개)
알고리즘:   기본 검색 → 의미 검색 → 하이브리드 통합 → 고도화
```

---

## PHASE 1: 난이도 하 (기본 검색 단계)
### 대상 RFP (3개)

1. **한국연구재단** - 산학협력 실태조사 UICC (1.3억원)
   - 구조: 표준화된 양식, 단순 필드
   - 문제: 메타데이터 검색만으로 충분

2. **한국생산기술연구원** - 고압가스 안전 관리 (4,000만원)
   - 구조: 명확한 카테고리, 규정 기반
   - 문제: 정확한 매칭 필요

3. **한국전기안전공사** - 전기안전 관제 보안 모듈 (2.2억원)
   - 구조: 기술 스펙 중심, 표준 용어
   - 문제: 키워드 기반 검색으로 해결

### 추천 알고리즘

#### Step 1: 기본 키워드 검색 (BM25)
```python
from rank_bm25 import BM25Okapi
import PyPDF2

# 1. PDF 텍스트 추출
def extract_text_from_pdf(pdf_path):
    texts = []
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            texts.append(page.extract_text())
    return ' '.join(texts)

# 2. 문서 청크 분할
def chunk_text(text, chunk_size=300, overlap=50):
    chunks = []
    words = text.split()
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

# 3. BM25 인덱싱
def create_bm25_index(chunks):
    tokenized = [chunk.split() for chunk in chunks]
    return BM25Okapi(tokenized), chunks

# 4. 검색
def search_bm25(query, bm25, chunks, top_k=3):
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [chunks[i] for i in top_indices]

# 사용 예
pdf_path = "/mnt/project/한국연구재단_2024년_대학산학협력활동_실태조사_시스템UICC_기능개선.pdf"
text = extract_text_from_pdf(pdf_path)
chunks = chunk_text(text)
bm25, chunk_list = create_bm25_index(chunks)

# 쿼리
query = "시스템 기능 요구사항"
results = search_bm25(query, bm25, chunk_list)
```

#### Step 2: 메타데이터 기반 필터링
```python
# 문서 메타데이터 추출
def extract_metadata(pdf_path):
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        metadata = reader.metadata
        return {
            'title': metadata.title if metadata else '',
            'pages': len(reader.pages),
            'subject': metadata.subject if metadata else ''
        }

# 필터링
def filter_by_metadata(query, metadata_list, filters):
    filtered = []
    for i, meta in enumerate(metadata_list):
        if all(filter_func(meta) for filter_func in filters):
            filtered.append(i)
    return filtered
```

#### Step 3: 간단한 점수 계산
```python
def calculate_relevance_score(query, text, bm25_score):
    # BM25 스코어 (0-1 정규화)
    normalized_bm25 = min(bm25_score / 100, 1.0)
    
    # 키워드 매칭 스코어
    query_words = set(query.lower().split())
    text_words = set(text.lower().split())
    overlap = len(query_words & text_words) / len(query_words) if query_words else 0
    
    # 통합 점수
    score = 0.7 * normalized_bm25 + 0.3 * overlap
    return score
```

### 난이도 하 테스트 계획

| 문서 | 쿼리 예시 | 기대 결과 | 성공 기준 |
|------|---------|---------|---------|
| UICC | "시스템 기능 요구사항" | 기능명세서 섹션 | Top-3 정확도 > 70% |
| 고압가스 | "안전 관리 기준" | 안전 규정 관련 항목 | Top-3 정확도 > 75% |
| 전기안전 | "보안 모듈 스펙" | 기술 사양서 | Top-3 정확도 > 70% |

### 난이도 하 구현 체크리스트

- [ ] PDF 텍스트 추출 성공
- [ ] BM25 인덱싱 구현
- [ ] 기본 검색 테스트 (쿼리 10개)
- [ ] 정확도 70% 이상 달성
- [ ] 응답 시간 < 1초

---

## PHASE 2: 난이도 중 (의미 검색 단계)
### 대상 RFP (3개)

1. **한영대학교** - 학사정보시스템 고도화 (1.3억원)
   - 구조: 교육 도메인 용어, 학사 규정
   - 문제: "수강신청"과 "등록"의 의미 구분

2. **인천광역시** - 도시계획위원회 통합관리시스템 (1.5억원)
   - 구조: 도시 규제, 법령 참고
   - 문제: 도시계획 용어의 동의어 처리

3. **스포츠윤리센터** - LMS 기능개선 (4,600만원)
   - 구조: 교육 플랫폼 용어
   - 문제: "학습" vs "교육" vs "훈련"의 차이

### 추천 알고리즘

#### Step 1: 한국어 특화 임베딩 모델
```python
from sentence_transformers import SentenceTransformer

# 한국어 전용 모델 (KoBERT 기반)
model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')

# 또는 더 가벼운 모델
# model = SentenceTransformer('jhgan/ko-sroberta-multitask')

def create_embeddings(chunks):
    """모든 청크를 임베딩"""
    embeddings = model.encode(chunks, convert_to_tensor=True)
    return embeddings

# 사용
chunks = chunk_text(text)
embeddings = create_embeddings(chunks)
```

#### Step 2: 벡터 검색 (Cosine Similarity)
```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def search_semantic(query, chunks, embeddings, top_k=3):
    """의미 기반 검색"""
    query_embedding = model.encode([query], convert_to_tensor=False)
    
    # 코사인 유사도 계산
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    # Top-K 선택
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    return [(chunks[i], similarities[i]) for i in top_indices]
```

#### Step 3: 의미론적 유사도 개선 (동의어 처리)
```python
# 한국어 동의어 맵
KOREAN_SYNONYMS = {
    '수강신청': ['신청', '등록', '수강'],
    '학점': ['점수', '성적', '학위'],
    '도시계획': ['도시', '계획', '규제'],
    '학습': ['교육', '훈련', '강의'],
}

def expand_query_with_synonyms(query):
    """쿼리 확장"""
    expanded = [query]
    for word in query.split():
        if word in KOREAN_SYNONYMS:
            expanded.extend(KOREAN_SYNONYMS[word])
    return expanded

# 사용
expanded_queries = expand_query_with_synonyms("수강신청 프로세스")
# 결과: ["수강신청 프로세스", "신청", "등록", "수강"]
```

#### Step 4: 하이브리드 검색 (BM25 + 벡터)
```python
def hybrid_search(query, bm25, chunks, embeddings, top_k=3, 
                  bm25_weight=0.3, semantic_weight=0.7):
    """하이브리드 검색: BM25 + 의미 검색"""
    
    # 1. BM25 점수
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_scores = bm25_scores / (max(bm25_scores) + 1e-10)  # 정규화
    
    # 2. 의미 검색 점수
    query_embedding = model.encode([query], convert_to_tensor=False)
    semantic_scores = cosine_similarity(query_embedding, embeddings)[0]
    
    # 3. 통합 점수
    combined_scores = (bm25_weight * bm25_scores + 
                      semantic_weight * semantic_scores)
    
    # 4. Top-K
    top_indices = np.argsort(combined_scores)[-top_k:][::-1]
    
    return [(chunks[i], combined_scores[i]) for i in top_indices]
```

### 난이도 중 테스트 계획

| 문서 | 쿼리 예시 | 기대 결과 | 성공 기준 |
|------|---------|---------|---------|
| 학사정보 | "학생 수강 등록 절차" | 수강신청 프로세스 | Top-3 정확도 > 80% |
| 도시계획 | "도시 규제 및 제한" | 도시계획 규정 | Top-3 정확도 > 80% |
| LMS | "학습 콘텐츠 관리" | 콘텐츠 관리 기능 | Top-3 정확도 > 75% |

### 난이도 중 구현 체크리스트

- [ ] 한국어 임베딩 모델 선택 및 설치
- [ ] 벡터 변환 파이프라인 구현
- [ ] 동의어 사전 구축 (50개 이상)
- [ ] 하이브리드 검색 구현
- [ ] 정확도 75% 이상 달성
- [ ] 응답 시간 < 2초

---

## PHASE 3: 난이도 상 (고도화 단계)
### 대상 RFP (3개)

1. **한국농수산** - 농산물가격안정기금 정부예산회계연계 (3.9억원)
   - 구조: 정부회계 규정 + 농산물기금 + 국가결산
   - 문제: 4개 규정 통합, 복잡한 의존성

2. **한국수자원공사** - 건설통합시스템(CMS) 고도화 (7.8억원)
   - 구조: 기술 요구사항 + 운영 프로세스 통합
   - 문제: 도메인 전문성 필요

3. **국가과학기술** - 통합정보시스템 고도화 (1.4억원)
   - 구조: 학제간 정보 통합, 다양한 용어
   - 문제: 여러 분야의 용어 통합

### 추천 알고리즘

#### Step 1: 멀티-레이어 벡터 DB (Pinecone)
```python
import pinecone

# Pinecone 초기화
pinecone.init(api_key="YOUR_API_KEY", environment="us-west4-gcp")

# 인덱스 생성
index_name = "rag-hnongsu"  # 한국농수산
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=384, metric="cosine")

index = pinecone.Index(index_name)

# 벡터 업로드
def upload_to_pinecone(chunks, embeddings, metadata_list, batch_size=100):
    """Pinecone에 벡터 업로드"""
    vectors_to_upload = []
    
    for i, (chunk, embedding, metadata) in enumerate(
        zip(chunks, embeddings, metadata_list)):
        vectors_to_upload.append({
            'id': f'{i}',
            'values': embedding.tolist(),
            'metadata': {
                'text': chunk,
                'doc_type': metadata.get('type'),
                'source': metadata.get('source'),
                'page': metadata.get('page')
            }
        })
        
        # 배치 업로드
        if len(vectors_to_upload) >= batch_size:
            index.upsert(vectors_to_upload)
            vectors_to_upload = []
    
    # 남은 데이터 업로드
    if vectors_to_upload:
        index.upsert(vectors_to_upload)
```

#### Step 2: 문맥 인식 청킹 (Context-Aware Chunking)
```python
def smart_chunk_text(text, sections_map=None, chunk_size=400, overlap=100):
    """문서 구조를 인식한 청킹"""
    chunks = []
    metadata = []
    
    # 섹션 기반 분할 (정부 문서는 보통 명확한 섹션이 있음)
    if sections_map:
        for section_title, section_text in sections_map.items():
            # 각 섹션을 독립적으로 청킹
            section_chunks = chunk_text(section_text, chunk_size, overlap)
            for chunk in section_chunks:
                chunks.append(chunk)
                metadata.append({
                    'section': section_title,
                    'type': 'regulation',  # 규정
                    'source': 'hnongsu'
                })
    
    return chunks, metadata

# 사용 예: 농산물가격안정기금 문서 구조
sections = {
    '1. 기금 조성 및 관리': '기금 조성 관련 내용...',
    '2. 정부예산 회계 연계': '회계 연계 관련 내용...',
    '3. 국가결산 체계': '결산 관련 내용...',
    '4. 다부처 지침': '지침 관련 내용...'
}

chunks, metadata = smart_chunk_text(text, sections)
```

#### Step 3: 고급 쿼리 변환 (Query Expansion & Rewriting)
```python
import re
from collections import defaultdict

def advanced_query_expansion(query, domain_dict):
    """도메인 특화 쿼리 확장"""
    expanded_queries = [query]
    
    # 1. 도메인 용어 확장
    for term, related_terms in domain_dict.items():
        if term.lower() in query.lower():
            expanded_queries.extend(related_terms)
    
    # 2. 문법적 변형
    # "결산" -> "결산하다", "결산된", "결산 항목"
    grammatical_variants = generate_grammatical_variants(query)
    expanded_queries.extend(grammatical_variants)
    
    # 3. 핵심 개념 추출 및 재조합
    entities = extract_domain_entities(query)
    for entity_combo in generate_entity_combinations(entities):
        expanded_queries.append(entity_combo)
    
    return list(set(expanded_queries))  # 중복 제거

# 한국농수산 도메인 사전
hnongsu_domain = {
    '결산': ['재정 결산', '정산', '회계 결산', '결산 항목'],
    '기금': ['펀드', '자금', '적립금', '기금 조성'],
    '정부예산': ['국가예산', '예산 연계', '예산 과목'],
    '농산물': ['농산물 가격', '농산물 수급', '농산물 유통'],
}
```

#### Step 4: Re-ranking (LLM 기반)
```python
from anthropic import Anthropic

def rerank_with_llm(query, search_results, top_k=3):
    """LLM을 사용한 재정렬"""
    client = Anthropic()
    
    # 검색 결과 정렬
    results_text = "\n\n".join([
        f"결과 {i+1}:\n{result[0][:300]}..."
        for i, result in enumerate(search_results)
    ])
    
    # LLM에 평가 요청
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": f"""
다음 검색 결과를 쿼리 "{query}"와의 관련성에 따라 평가해주세요.
상위 {top_k}개만 선택하고, JSON 형식으로 반환해주세요.
형식: {{"rank": 1, "score": 0.95, "reason": "...이유..."}}

{results_text}
"""
        }]
    )
    
    # 응답 파싱 및 재정렬
    return message.content[0].text
```

#### Step 5: 마ル티-모달 RAG (표, 이미지 처리)
```python
import pymupdf
import pytesseract
from PIL import Image

def extract_tables_from_pdf(pdf_path):
    """PDF에서 표 추출"""
    doc = pymupdf.open(pdf_path)
    tables = []
    
    for page_num, page in enumerate(doc):
        # 표 감지
        table_list = page.find_tables()
        for table in table_list:
            table_data = table.extract()
            tables.append({
                'page': page_num,
                'data': table_data,
                'text': ' '.join([' '.join(row) for row in table_data])
            })
    
    return tables

def extract_images_from_pdf(pdf_path):
    """PDF에서 이미지 추출 및 OCR"""
    doc = pymupdf.open(pdf_path)
    images = []
    
    for page_num, page in enumerate(doc):
        for img in page.get_images():
            xref = img[0]
            pix = pymupdf.Pixmap(doc, xref)
            
            # OCR
            if pix.n - pix.alpha < 4:  # 그레이스케일
                text = pytesseract.image_to_string(
                    Image.frombytes("L", (pix.width, pix.height), pix.samples),
                    lang='kor'
                )
            else:  # 컬러
                text = pytesseract.image_to_string(
                    Image.frombytes("RGB", (pix.width, pix.height), pix.samples),
                    lang='kor'
                )
            
            images.append({
                'page': page_num,
                'text': text,
                'size': f'{pix.width}x{pix.height}'
            })
    
    return images
```

#### Step 6: 엔드-투-엔드 파이프라인
```python
class AdvancedRAG:
    def __init__(self, pinecone_index, model, llm_client):
        self.index = pinecone_index
        self.model = model
        self.llm = llm_client
    
    def retrieve_and_generate(self, query, top_k=5, use_rerank=True):
        """검색 및 생성"""
        
        # 1단계: 쿼리 확장
        expanded_queries = advanced_query_expansion(
            query, 
            domain_dict=hnongsu_domain
        )
        
        # 2단계: 다중 검색
        all_results = []
        for exp_query in expanded_queries[:3]:  # 상위 3개만 사용
            # Pinecone 검색
            query_embedding = self.model.encode([exp_query])[0]
            results = self.index.query(
                vector=query_embedding.tolist(),
                top_k=top_k,
                include_metadata=True
            )
            all_results.extend(results['matches'])
        
        # 중복 제거 및 점수 집계
        unique_results = self._deduplicate_and_score(all_results)
        
        # 3단계: Re-ranking (선택)
        if use_rerank:
            ranked_results = rerank_with_llm(
                query,
                unique_results,
                top_k=3
            )
        else:
            ranked_results = unique_results[:3]
        
        # 4단계: 생성
        context = "\n\n".join([
            f"출처: {r['metadata']['source']}\n{r['metadata']['text']}"
            for r in ranked_results
        ])
        
        response = self.llm.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1500,
            messages=[{
                "role": "user",
                "content": f"""
다음 문서 내용을 바탕으로 질문에 답변해주세요.

문서 내용:
{context}

질문: {query}

답변:
"""
            }]
        )
        
        return {
            'answer': response.content[0].text,
            'sources': ranked_results,
            'confidence': sum([r.get('score', 0) for r in ranked_results]) / len(ranked_results)
        }
    
    def _deduplicate_and_score(self, results):
        """결과 중복 제거 및 점수 집계"""
        unique_dict = {}
        for result in results:
            doc_id = result['id']
            if doc_id not in unique_dict:
                unique_dict[doc_id] = result
            else:
                # 점수 평균 계산
                unique_dict[doc_id]['score'] = (
                    unique_dict[doc_id]['score'] + result['score']
                ) / 2
        
        return sorted(
            unique_dict.values(),
            key=lambda x: x['score'],
            reverse=True
        )
```

### 난이도 상 테스트 계획

| 문서 | 쿼리 예시 | 기대 결과 | 성공 기준 |
|------|---------|---------|---------|
| 농산물기금 | "농산물 가격 결산 처리 절차" | 4개 규정 통합 설명 | 정확도 > 85%, 신뢰도 > 0.8 |
| CMS | "건설 프로젝트 통합 관리 프로세스" | 기술+운영 통합 | 정확도 > 85%, 응답 < 3초 |
| 통합정보 | "학제간 정보 표준화 방안" | 다양한 분야 용어 통합 | 정확도 > 80%, 포괄성 > 0.85 |

### 난이도 상 구현 체크리스트

- [ ] Pinecone 인덱스 설정 및 업로드
- [ ] 문서 구조 기반 스마트 청킹 구현
- [ ] 도메인별 동의어 사전 구축 (각 문서 100+ 항목)
- [ ] 고급 쿼리 변환 파이프라인 구현
- [ ] LLM 기반 Re-ranking 구현
- [ ] 표/이미지 추출 및 처리 구현
- [ ] 엔드-투-엔드 파이프라인 통합
- [ ] 정확도 80% 이상, 신뢰도 0.8 이상 달성
- [ ] 응답 시간 < 3초 달성

---

## 📈 성능 메트릭

### 난이도별 목표

| 단계 | 모델 | 정확도 | 응답시간 | 신뢰도 |
|------|------|--------|---------|--------|
| **난이도 하** | BM25 | > 70% | < 1s | N/A |
| **난이도 중** | 하이브리드 | > 75% | < 2s | N/A |
| **난이도 상** | 고도화 | > 80% | < 3s | > 0.8 |

### 평가 방법

```python
def evaluate_rag_system(test_queries, ground_truth):
    """RAG 시스템 평가"""
    
    metrics = {
        'accuracy': 0,      # 정답률
        'precision': 0,     # 정밀도 (Top-3)
        'recall': 0,        # 재현율
        'f1_score': 0,      # F1 점수
        'mrr': 0,          # Mean Reciprocal Rank
        'ndcg': 0          # Normalized DCG
    }
    
    results = []
    for query, expected_answer in test_queries:
        result = rag_system.retrieve_and_generate(query)
        
        # 정확도 계산 (간단한 문자열 매칭)
        is_correct = expected_answer.lower() in result['answer'].lower()
        
        results.append({
            'query': query,
            'predicted': result['answer'],
            'correct': is_correct,
            'confidence': result['confidence']
        })
    
    # 집계
    metrics['accuracy'] = sum([r['correct'] for r in results]) / len(results)
    
    return metrics, results
```

---

## 🔄 구현 순서 (추천)

### Week 1: 난이도 하 기반 구축
```
Day 1-2: PDF 처리 및 청킹
Day 3-4: BM25 인덱싱 및 검색
Day 5: 테스트 및 평가
```

### Week 2: 난이도 중 고도화
```
Day 1-2: 한국어 임베딩 모델 선택 및 벡터화
Day 3-4: 하이브리드 검색 구현
Day 5: 동의어 사전 구축 및 테스트
```

### Week 3-4: 난이도 상 완성
```
Day 1-2: Pinecone 설정 및 업로드
Day 3-4: 고급 쿼리 변환 및 Re-ranking
Day 5-6: 표/이미지 처리 (선택)
Day 7-8: 통합 테스트 및 최적화
```

---

## 💾 권장 기술 스택

### 필수
- **Python 3.9+**
- **PyPDF2 또는 pymupdf**: PDF 처리
- **rank-bm25**: BM25 검색
- **sentence-transformers**: 임베딩

### 선택 (난이도별)
- **난이도 중**: pinecone-client, chromadb
- **난이도 상**: claude API (Anthropic), OpenAI API

### 개발 환경
```bash
pip install PyPDF2 rank-bm25 sentence-transformers
pip install pinecone-client python-dotenv
pip install anthropic
pip install pandas numpy scikit-learn
```

---

## 📝 체크리스트

### 전체 구현 체크리스트
- [ ] PDF 파일 9개 준비 완료
- [ ] 각 난이도별 테스트 쿼리 세트 준비 (최소 10개)
- [ ] 정답 데이터셋 구축
- [ ] 난이도 하 구현 완료 (정확도 > 70%)
- [ ] 난이도 중 구현 완료 (정확도 > 75%)
- [ ] 난이도 상 구현 완료 (정확도 > 80%)
- [ ] 통합 테스트 완료
- [ ] 성능 최적화 완료
- [ ] 문서화 완료

---

**다음 단계**: 각 단계별 구현 코드 작성 및 테스트
