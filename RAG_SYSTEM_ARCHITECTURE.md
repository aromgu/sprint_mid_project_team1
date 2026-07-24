# RAG 시스템 구축 전략
## 버전 분리 vs 통합 시스템 (추천: 통합)

**결론**: 3개 버전을 따로 만들지 말고, **하나의 통합 RAG 시스템**으로 만들되 **설정 기반으로 단계별 기능을 활성화**하는 방식을 추천합니다.

---

## ❌ 비추천: 3개 버전 분리

```
version_1_bm25/
├── main.py (BM25만)
├── requirements.txt
└── config.yaml

version_2_hybrid/
├── main.py (BM25 + 벡터)
├── requirements.txt
└── config.yaml

version_3_advanced/
├── main.py (Pinecone + Re-ranking)
├── requirements.txt
└── config.yaml
```

### 문제점

- ❌ **코드 중복**: PDF 처리, 청킹, 인덱싱 로직이 3번 반복
- ❌ **유지보수 어려움**: 버그 수정 시 3개 파일을 모두 수정해야 함
- ❌ **메모리 낭비**: 같은 라이브러리를 여러 번 로드
- ❌ **테스트 코드 3배**: 각 버전마다 테스트 코드 작성 필요
- ❌ **성능 비교 어려움**: 코드 베이스가 다름

---

## ✅ 추천: 하나의 통합 시스템

```
rag_system/
├── main.py                    # 진입점
├── config.yaml                # 설정 (버전 선택)
├── config_phase1.yaml         # PHASE 1 설정
├── config_phase2.yaml         # PHASE 2 설정
├── config_phase3.yaml         # PHASE 3 설정
├── requirements.txt           # 모든 의존성
│
├── core/                       # 공통 모듈 (모든 버전)
│   ├── __init__.py
│   ├── pdf_processor.py       # PDF 처리
│   ├── chunker.py             # 텍스트 청킹
│   └── base_retriever.py      # 기본 인터페이스
│
├── retrievers/                # 버전별 리트리버
│   ├── __init__.py
│   ├── bm25_retriever.py      # PHASE 1
│   ├── hybrid_retriever.py    # PHASE 2 (BM25Retriever 상속)
│   └── advanced_retriever.py  # PHASE 3 (HybridRetriever 상속)
│
├── utils/                      # 선택적 유틸리티
│   ├── __init__.py
│   ├── embeddings.py          # 임베딩 (PHASE 2+)
│   ├── synonyms.py            # 동의어 (PHASE 2+)
│   ├── reranker.py            # Re-ranking (PHASE 3)
│   └── metrics.py             # 평가 메트릭
│
└── tests/                      # 테스트
    ├── test_bm25.py           # PHASE 1
    ├── test_hybrid.py         # PHASE 2
    └── test_advanced.py       # PHASE 3
```

### 장점

- ✅ **코드 중복 최소화**: 공통 로직은 한 곳에만 작성
- ✅ **쉬운 유지보수**: 버그 수정 한 번만
- ✅ **메모리 효율적**: 필요한 모듈만 로드
- ✅ **테스트 코드 관리 용이**: 통일된 테스트 전략
- ✅ **성능 비교 쉬움**: 같은 코드베이스에서 비교
- ✅ **점진적 개발**: PHASE 1 → 2 → 3으로 자연스럽게 확장

---

## 📋 설정 기반 단계별 활성화

### config.yaml (PHASE 1)
```yaml
# 리트리버 선택
retriever_type: "bm25"

# PHASE 1: BM25 설정
bm25:
  enabled: true
  chunk_size: 300
  overlap: 50
  top_k: 3

# PHASE 2: 임베딩 설정 (비활성화)
embedding:
  enabled: false
  model: "sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens"
  device: "cuda"

# PHASE 3: 고급 설정 (비활성화)
advanced:
  enabled: false
  pinecone_api_key: "${PINECONE_API_KEY}"
  pinecone_index: "rag-hnongsu"
  use_reranking: false
```

### config_phase2.yaml (PHASE 2)
```yaml
retriever_type: "hybrid"

bm25:
  enabled: true
  chunk_size: 300
  overlap: 50
  top_k: 3

embedding:
  enabled: true  # ✅ 활성화
  model: "sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens"
  device: "cuda"
  
  # 동의어 설정
  synonyms:
    enabled: true
    dictionary_path: "data/synonyms.json"

advanced:
  enabled: false
```

### config_phase3.yaml (PHASE 3)
```yaml
retriever_type: "advanced"

bm25:
  enabled: true
  chunk_size: 300
  overlap: 50

embedding:
  enabled: true
  model: "sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens"
  device: "cuda"
  synonyms:
    enabled: true

advanced:
  enabled: true  # ✅ 활성화
  pinecone_api_key: "${PINECONE_API_KEY}"
  pinecone_index: "rag-hnongsu"
  use_reranking: true  # LLM Re-ranking
  use_image_ocr: true  # 이미지 처리
  llm_model: "claude-3-5-sonnet-20241022"
```

### 실행

```bash
# PHASE 1: BM25
python main.py --config config_phase1.yaml

# PHASE 2: 하이브리드
python main.py --config config_phase2.yaml

# PHASE 3: 고도화
python main.py --config config_phase3.yaml
```

---

## 💻 코드 구조 예시

### main.py (진입점)
```python
import yaml
import sys
from core.pdf_processor import PDFProcessor
from retrievers import get_retriever

def load_config(config_file):
    """YAML 설정 파일 로드"""
    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main(config_file):
    config = load_config(config_file)
    
    # 1. 공통 처리 (모든 버전 동일)
    processor = PDFProcessor()
    text = processor.extract_text("한국농수산_농산물가격안정기금.pdf")
    chunks = processor.chunk_text(
        text,
        chunk_size=config['bm25']['chunk_size'],
        overlap=config['bm25']['overlap']
    )
    
    # 2. 리트리버 선택 (설정에 따라 동적으로)
    retriever_type = config['retriever_type']
    retriever = get_retriever(retriever_type, config)
    
    # 3. 인덱싱
    retriever.index(chunks)
    
    # 4. 검색
    query = "농산물 가격 결산 처리"
    results = retriever.retrieve(query)
    
    # 5. 결과 출력
    print(f"[{retriever_type.upper()}] Query: {query}")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result[:100]}...")

if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    main(config_file)
```

### core/pdf_processor.py (공통)
```python
import PyPDF2

class PDFProcessor:
    """PDF 처리 (모든 버전에서 동일)"""
    
    def extract_text(self, pdf_path):
        """PDF에서 텍스트 추출"""
        texts = []
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                texts.append(page.extract_text())
        return '\n'.join(texts)
    
    def chunk_text(self, text, chunk_size=300, overlap=50):
        """텍스트를 청크로 분할"""
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
```

### core/base_retriever.py (기본 인터페이스)
```python
from abc import ABC, abstractmethod

class BaseRetriever(ABC):
    """모든 리트리버의 기본 인터페이스"""
    
    @abstractmethod
    def index(self, chunks):
        """청크를 인덱싱"""
        pass
    
    @abstractmethod
    def retrieve(self, query, top_k=3):
        """쿼리 기반 검색"""
        pass
```

### retrievers/bm25_retriever.py (PHASE 1)
```python
from core.base_retriever import BaseRetriever
from rank_bm25 import BM25Okapi

class BM25Retriever(BaseRetriever):
    """BM25 기반 키워드 검색 (PHASE 1)"""
    
    def __init__(self, config):
        self.config = config
        self.bm25 = None
        self.chunks = None
    
    def index(self, chunks):
        """BM25 인덱싱"""
        tokenized = [chunk.split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized)
        self.chunks = chunks
    
    def retrieve(self, query, top_k=3):
        """BM25 검색"""
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]
        
        return [self.chunks[i] for i in top_indices]
```

### retrievers/hybrid_retriever.py (PHASE 2)
```python
from retrievers.bm25_retriever import BM25Retriever
from utils.embeddings import get_embeddings
from utils.synonyms import expand_query
import numpy as np

class HybridRetriever(BM25Retriever):
    """하이브리드 검색: BM25 + 벡터 (PHASE 2)"""
    
    def __init__(self, config):
        super().__init__(config)
        
        # 임베딩 모델 로드
        self.model = get_embeddings(
            config['embedding']['model']
        )
        self.embeddings = None
    
    def index(self, chunks):
        """BM25 + 벡터 인덱싱"""
        # 부모 클래스의 BM25 인덱싱
        super().index(chunks)
        
        # 벡터 변환
        self.embeddings = self.model.encode(chunks)
    
    def retrieve(self, query, top_k=3):
        """하이브리드 검색"""
        
        # 1. 쿼리 확장 (동의어)
        expanded_queries = expand_query(query)
        
        # 2. BM25 검색
        bm25_scores = self._bm25_search(expanded_queries)
        
        # 3. 의미 검색
        semantic_scores = self._semantic_search(query)
        
        # 4. 점수 통합 (BM25 30% + 벡터 70%)
        combined = 0.3 * bm25_scores + 0.7 * semantic_scores
        
        top_indices = np.argsort(combined)[-top_k:][::-1]
        return [self.chunks[i] for i in top_indices]
    
    def _bm25_search(self, queries):
        """BM25 점수 계산"""
        max_score = 0
        for q in queries:
            tokenized = q.split()
            scores = self.bm25.get_scores(tokenized)
            max_score = max(max_score, max(scores))
        return np.array([max_score] * len(self.chunks))  # 단순화
    
    def _semantic_search(self, query):
        """의미 검색 점수"""
        from sklearn.metrics.pairwise import cosine_similarity
        query_embedding = self.model.encode([query])
        return cosine_similarity(query_embedding, self.embeddings)[0]
```

### retrievers/advanced_retriever.py (PHASE 3)
```python
from retrievers.hybrid_retriever import HybridRetriever
from utils.reranker import LLMReranker
import pinecone

class AdvancedRetriever(HybridRetriever):
    """고도화 RAG: Pinecone + Re-ranking (PHASE 3)"""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Pinecone 초기화
        pinecone.init(
            api_key=config['advanced']['pinecone_api_key']
        )
        self.index = pinecone.Index(
            config['advanced']['pinecone_index']
        )
        
        # Re-ranker 초기화
        self.reranker = LLMReranker(
            config['advanced']['llm_model']
        )
    
    def index(self, chunks, metadata=None):
        """하이브리드 인덱싱 + Pinecone 업로드"""
        
        # 부모 클래스 인덱싱
        super().index(chunks)
        
        # Pinecone에도 업로드
        vectors = self.embeddings.tolist()
        for i, (vec, chunk) in enumerate(zip(vectors, chunks)):
            self.index.upsert([
                (str(i), vec, {'text': chunk})
            ])
    
    def retrieve(self, query, top_k=3):
        """고도화 검색: 하이브리드 + Re-ranking"""
        
        # 1. 하이브리드 검색으로 후보 수집 (Top-5)
        candidates = super().retrieve(query, top_k=5)
        
        # 2. LLM Re-ranking
        ranked = self.reranker.rerank(
            query,
            candidates,
            top_k=top_k
        )
        
        # 3. 결과 포맷팅 (신뢰도 포함)
        results = []
        for rank, (text, score) in enumerate(ranked, 1):
            results.append({
                'rank': rank,
                'text': text,
                'confidence': score
            })
        
        return results
```

### retrievers/__init__.py (팩토리)
```python
def get_retriever(retriever_type, config):
    """설정에 따라 적절한 리트리버 반환 (팩토리 패턴)"""
    
    if retriever_type == "bm25":
        from retrievers.bm25_retriever import BM25Retriever
        return BM25Retriever(config)
    
    elif retriever_type == "hybrid":
        from retrievers.hybrid_retriever import HybridRetriever
        return HybridRetriever(config)
    
    elif retriever_type == "advanced":
        from retrievers.advanced_retriever import AdvancedRetriever
        return AdvancedRetriever(config)
    
    else:
        raise ValueError(
            f"Unknown retriever type: {retriever_type}. "
            f"Choose from: bm25, hybrid, advanced"
        )
```

---

## 🧪 테스트 전략

### tests/test_bm25.py
```python
import pytest
from core.pdf_processor import PDFProcessor
from retrievers.bm25_retriever import BM25Retriever

def test_bm25_basic():
    config = {'bm25': {'chunk_size': 300, 'overlap': 50}}
    retriever = BM25Retriever(config)
    
    chunks = [
        "한국농수산식품유통공사는 농산물 가격을 관리합니다",
        "정부예산회계는 투명성을 보장합니다",
        "농산물 결산 절차는 매년 진행됩니다"
    ]
    
    retriever.index(chunks)
    results = retriever.retrieve("농산물 가격")
    
    assert len(results) == 3
    assert "농산물" in results[0]

def test_bm25_performance():
    """PHASE 1 성능 목표: 정확도 > 70%"""
    # 테스트 데이터셋
    test_cases = [
        ("시스템 기능", "기능", True),
        ("요구사항", "요구", True),
        ("보안", "보안", True),
    ]
    
    retriever = BM25Retriever(config)
    accuracy = evaluate_retriever(retriever, test_cases)
    
    assert accuracy > 0.70, f"정확도 {accuracy} < 70%"
```

### tests/test_hybrid.py
```python
import pytest
from retrievers.hybrid_retriever import HybridRetriever

def test_hybrid_synonyms():
    config = {'embedding': {'enabled': True}}
    retriever = HybridRetriever(config)
    
    # "수강신청" 및 동의어들이 올바르게 처리되는지 확인
    results = retriever.retrieve("학생 수강 등록")
    assert any("신청" in r or "등록" in r for r in results)

def test_hybrid_performance():
    """PHASE 2 성능 목표: 정확도 > 75%"""
    test_cases = [
        ("학사정보 고도화", 0.8),
        ("도시계획 규정", 0.85),
        ("LMS 기능", 0.75),
    ]
    
    retriever = HybridRetriever(config)
    accuracy = evaluate_retriever(retriever, test_cases)
    
    assert accuracy > 0.75, f"정확도 {accuracy} < 75%"
```

### tests/test_advanced.py
```python
import pytest
from retrievers.advanced_retriever import AdvancedRetriever

def test_advanced_reranking():
    """Re-ranking 검증"""
    config = {'advanced': {'use_reranking': True}}
    retriever = AdvancedRetriever(config)
    
    results = retriever.retrieve("농산물 가격 결산")
    
    # 신뢰도 점수 확인
    for result in results:
        assert 'confidence' in result
        assert 0 <= result['confidence'] <= 1

def test_advanced_performance():
    """PHASE 3 성능 목표: 정확도 > 80%, 신뢰도 > 0.8"""
    test_cases = [
        ("농산물 기금 규정", 0.85, 0.85),
        ("건설 시스템 통합", 0.85, 0.82),
        ("정보 표준화", 0.80, 0.78),
    ]
    
    retriever = AdvancedRetriever(config)
    accuracy, confidence = evaluate_advanced(retriever, test_cases)
    
    assert accuracy > 0.80
    assert confidence > 0.80
```

### 테스트 실행
```bash
# 모든 테스트 실행
pytest tests/

# PHASE별 테스트
pytest tests/test_bm25.py
pytest tests/test_hybrid.py
pytest tests/test_advanced.py

# 상세 리포트
pytest tests/ -v --tb=short
```

---

## 📊 장점 비교 표

| 기준 | 3개 버전 분리 | 통합 시스템 (추천) |
|------|-------------|------------------|
| **코드 중복** | 높음 ❌ | 낮음 ✅ |
| **유지보수** | 어려움 ❌ | 쉬움 ✅ |
| **메모리 사용** | 많음 ❌ | 적음 ✅ |
| **테스트 코드** | 3배 필요 ❌ | 1배만 ✅ |
| **버그 수정** | 3곳 수정 ❌ | 1곳만 ✅ |
| **성능 비교** | 어려움 ❌ | 쉬움 ✅ |
| **확장성** | 낮음 ❌ | 높음 ✅ |
| **점진적 개발** | 비효율 ❌ | 효율적 ✅ |

---

## 🚀 구현 순서

### Step 1: 기본 구조 설정 (Day 1)
```bash
mkdir -p rag_system/{core,retrievers,utils,tests}
touch rag_system/{main.py,config.yaml,requirements.txt}
touch rag_system/core/__init__.py
touch rag_system/retrievers/__init__.py
touch rag_system/utils/__init__.py
```

### Step 2: 공통 모듈 구현 (Day 2)
- `core/pdf_processor.py` ✓
- `core/base_retriever.py` ✓
- `config.yaml` ✓

### Step 3: BM25Retriever 구현 (Day 3)
- `retrievers/bm25_retriever.py` ✓
- `tests/test_bm25.py` ✓
- `config_phase1.yaml` ✓

### Step 4: HybridRetriever 구현 (Day 4)
- `retrievers/hybrid_retriever.py` ✓
- `utils/embeddings.py` ✓
- `utils/synonyms.py` ✓
- `tests/test_hybrid.py` ✓

### Step 5: AdvancedRetriever 구현 (Day 5-6)
- `retrievers/advanced_retriever.py` ✓
- `utils/reranker.py` ✓
- `tests/test_advanced.py` ✓

### Step 6: 통합 테스트 (Day 7-8)
- 3가지 리트리버 모두 테스트 ✓
- 성능 비교 ✓
- 최적화 ✓

---

## ✅ 체크리스트

- [ ] 프로젝트 구조 설정
- [ ] `core/` 모듈 완성
- [ ] BM25Retriever 완성 + 테스트
- [ ] HybridRetriever 완성 + 테스트
- [ ] AdvancedRetriever 완성 + 테스트
- [ ] 설정 파일 3개 준비 (phase1, 2, 3)
- [ ] 통합 테스트 완료
- [ ] 성능 목표 달성
  - [ ] PHASE 1: > 70%
  - [ ] PHASE 2: > 75%
  - [ ] PHASE 3: > 80%
- [ ] 문서화 완료

---

## 🎯 결론

**하나의 통합 RAG 시스템**으로 만들고, **설정 파일**로 버전을 관리하는 것이 훨씬 효율적입니다!

```bash
# 간단한 실행
python main.py --config config_phase1.yaml  # PHASE 1
python main.py --config config_phase2.yaml  # PHASE 2
python main.py --config config_phase3.yaml  # PHASE 3
```
