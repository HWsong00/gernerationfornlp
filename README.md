# Project Hub (main)

이 레포의 `main` 브랜치는 **단일 구현을 합치는 목적이 아니라**,  
각자 다른 접근/실험/구현을 **브랜치 단위로 분리 관리**하기 위한 **허브(Entry) 브랜치**입니다.

> ✅ 처음 오신 분은 이 README를 보고 목적에 맞는 브랜치를 체크아웃해서 진행하세요.  
> ❗ `main` 자체에는 실행 가능한 완성본이 없을 수 있습니다(가이드 역할).

---

## TL;DR — 어디로 가면 되나요?

- **ver-1**: wikipedia api 기반 RAG QA
- **ver-3**: ORPO를 통한 강화 학습
- **ver-4**: Gemini3의 답변결과를 기반으로 qwen3-8b로 지식증류


---

## How to Checkout

```bash
git fetch --all --prune

# baseline
git checkout feature/ver-1

# approach A
git checkout ver-3

# approach B
git checkout ver-4
