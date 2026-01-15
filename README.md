# Qwen3 8B 기반 한국어 추론 능력 향상 프로젝트 (Gemini 3 Distillation)

![Project Status](https://img.shields.io/badge/Status-Experimental-orange)
![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 프로젝트 소개 (Project Overview)

본 프로젝트는 **Gemini 3**의 우수한 추론 능력을 **Qwen3 8B** 모델에 증류(Distillation)하여, 한국어 수능(CSAT), KMMLU(한국사), MMMLU, KLUE MRC 등 고난도 과제에서의 성능을 극대화하는 것을 목표로 합니다.

제한된 컴퓨팅 자원 환경에서 합성 데이터(Synthetic Data)를 적극적으로 활용하였으며, 단순한 Fine-tuning을 넘어 **지식 주입(Knowledge Injection)**, **CoT(Chain of Thought) 증류**, **추상화(Abstraction) 기반 학습** 등 다양한 실험적 방법론을 적용하고 그 한계와 가능성을 분석했습니다.

## 🎯 주요 목표 (Target Tasks)

*   **한국어 대학수학능력시험 (Korean CSAT)**
*   **KMMLU (Korean History 등)**
*   **MMMLU**
*   **KLUE MRC** (Work in Progress)

## 🏗️ 폴더 구조 (Project Structure)

```bash
.
├── configs/            # 학습 및 모델 설정 파일 (YAML)
├── src/                # 소스 코드
│   ├── data.py         # 데이터 전처리 및 로딩
│   ├── model.py        # 모델 로드 및 설정
│   └── trainer.py      # 학습 루프 및 트레이너
├── main.py             # 실행 진입점 (Entry point)
├── inference.py        # 추론 및 테스트 스크립트
├── requirements.txt    # 의존성 패키지 목록
└── README.md           # 프로젝트 문서
```

## 🚀 설치 및 실행 (Installation & Usage)

### 1. 환경 설정

```bash
# Repository Clone
git clone https://github.com/boostcampaitech8/pro-nlp-generationfornlp-nlp-05.git
cd pro-nlp-generationfornlp-nlp-05

# Checkout Branch
git checkout feature/finetune-qwen3

# Install Dependencies
pip install -r requirements.txt
```

### 2. 학습 실행 (Training)

설정 파일(`configs/config.yaml`)을 수정하여 하이퍼파라미터를 조정한 후 아래 명령어로 학습을 시작합니다.

```bash
python main.py --config configs/config.yaml
```

*   `--skip_data`: 데이터 전처리 과정을 생략하고 기존 캐시된 데이터를 사용할 경우 추가합니다.

### 3. 추론 실행 (Inference)

```bash
python inference.py
```

## 🧪 실험 내용 및 결과 (Experiments & Analysis)

본 프로젝트에서는 성능 향상을 위해 다음과 같은 4단계의 주요 실험을 진행했습니다.

### 1. 실험 개요 (Overview)
Gemini 3의 추론 과정을 Qwen 모델에 학습시키고자 했으나, 학습 지표(Training Metric)의 상승이 테스트 셋(Test Set)의 성능 향상으로 직결되지 않는 **일반화(Generalization) 문제**가 지속적으로 관찰되었습니다.

### 2. 지식 주입 및 스타일 정렬 (Knowledge Injection & Style Alignment)
*   **시도:** 합성 데이터를 통해 지식을 직접 주입하거나 오답을 반박(Refutation)하는 방식 시도.
*   **결과:** 모델이 사실 관계(Fact)를 학습하기보다 데이터의 표면적 스타일만 모방하는 **모델 붕괴(Model Collapse)** 현상 발생.
*   **대응:** 원본 데이터 분포를 따르도록 전략을 수정하여 훈련 셋 성능은 방어했으나, 테스트 셋에서의 성능 하락은 해결되지 않음 (Embedding Space 불일치 문제).

### 3. 추론 증류 및 증강 고도화 (Reasoning Distillation)
*   **시도:** 'Gold Reasoning (CoT)' 데이터를 집중 학습하고, 도메인별 커리큘럼 및 내적 독백(Internal Monologue) 도입.
*   **결과:** 모델이 논리적 사고 과정을 내재화하기보다, 교사 모델(Teacher Model)의 **추론 스타일을 단순 암기**하는 과적합(Overfitting) 경향을 보임.

### 4. 추상화 기반 학습 (Abstraction Based Learning)
*   **시도:** 인지 부하를 줄이기 위해 **추상화(Abstraction) 도출**과 **해답 생성** 과정을 분리하는 Warmstart SFT 시도 (Reinforcement Learning through Abstraction Discovery).
*   **방법:** 별도의 LoRA 어댑터를 사용하여 각 과정을 독립적으로 학습.
*   **결과:** 합성 데이터의 품질 관리 난이도로 인해 유의미한 성능 개선 달성 실패.

---
*Created by [Team NLP-05] for Boostcamp AI Tech 8.*