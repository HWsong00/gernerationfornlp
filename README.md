# 수능형 문제 풀이 Agent with Adaptive RAG

> 문제 유형을 분석하여 외부 지식이 필요한 경우에만 RAG를 사용하는 시스템

## 목차

- [특징](#특징)
- [시스템 다이어그램](#시스템-다이어그램)
- [실행 방법](#실행-방법)
- [결과 분석](#결과-분석)


## 특징


#### 1. **Adaptive RAG**
기존의 모든 문제에 RAG를 적용하는 방식과 달리, **문제 분석 단계**를 통해 외부 지식 필요 여부를 먼저 판단합니다.

```
문제 입력 → [분석 노드] → 외부 지식 필요?
                           ├─ Yes → RAG 검색 → 풀이
                           └─ No → 직접 풀이
```

#### 2. **신뢰도를 통한 추론 의사결정 효율화**
각 선택지별 신뢰도를 산출하게끔 지시하여, 모호한 선지들에 대해 추론을 반복하는 현상을 완화하였습니다.

```json
{
  "1번": 0.85,
  "2번": 0.15,
  "3번": 0.60,
  "4번": 0.30
}
→ 프롬프트의 Decision Logic에 따라 최종 답안 선택

효과
:train set의 한국사 문제 73개에 대하여 추론 속도 약 10분 단축
```

#### 3. **지문 및 선택지 독립 검색 전략**
- **지문 + 각 선택지** 별로 독립적으로 검색
- **Cross-Encoder Reranking**으로 정확도 향상
- 중복 제거 및 출처 명시




## 시스템 다이어그램

### 전체 워크플로우

<div align=center>
Example History Router Flowchart
</div>
<div align=center>
<img width="1290" height="781" alt="image" src="https://github.com/user-attachments/assets/ed38276a-672e-4f51-9f1e-91add239cd7a" />
</div>
<img width="8192" height="2798" alt="Image" src="https://github.com/user-attachments/assets/5107692e-5032-4d0a-9bab-eeb4d4e9f5be"/>


### 모듈 구조

```
project_root/
├── config.yaml                 # 설정 파일
├── prompts.py                  # 프롬프트 관리
├── main.py                     # 진입점
├── models/
│   ├── __init__.py
│   ├── schemas.py             # Pydantic 스키마
│   └── llm_manager.py         # LLM 관리
├── rag/
│   ├── __init__.py
│   └── retriever.py           # RAG 시스템
├── workflow/
│   ├── __init__.py
│   ├── state.py               # State 정의
│   ├── nodes.py               # 노드 함수
│   ├── routing.py             # 라우팅
│   └── graph.py               # 그래프 구성
└── utils/
    ├── __init__.py
    ├── data_parser.py         # 데이터 파싱
    └── batch_processor.py     # 배치 처리
```


## 실행 방법

#### 의존성 설치

```bash
pip install -r requirements.txt
```

#### 설정 파일 구성

```yaml
# LLM 모델 설정
llm:
  repo_id: "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF"
  filename: "Qwen3-30B-A3B-Instruct-2507-UD-Q5_K_XL.gguf"
  n_ctx: 32769
  n_gpu_layers: -1  # GPU 레이어 수 (-1: 모두 사용)

# RAG API 키 및 경로 설정
rag:
  upstage_api_key: "YOUR_UPSTAGE_API_KEY"
  vectorstore:
    persist_directory: "/path/to/chromadb"
    collection_name: "CHRONICLE_solar"

# 데이터 경로 설정
data:
  input_csv: "/path/to/test.csv"
  output_csv: "results.csv"
```
#### 실행

```bash
python main.py
```


## 결과 분석

### 성능 비교

<img width="590" height="390" alt="Image" src="https://github.com/user-attachments/assets/3a83ff01-c44c-436f-ba39-85cbab87426d" />

> 최종 private score에 대해 3.43%p 향상

<img width="3869" height="1769" alt="Image" src="https://github.com/user-attachments/assets/47fcd66a-8473-4565-8287-41a2f8f3eb1f" />

> validation 셋 추론 결과 1.52%p 향상


#### 주요 기술 

- **LangGraph**: 워크플로우 오케스트레이션
- **llama-cpp-python**: LLM 추론
- **Upstage Embeddings**: 임베딩 생성
- **ChromaDB**: 벡터 데이터베이스
- **Cross-Encoder Reranker**: 검색 결과 재정렬




