<div align=center>
Example History Router Flowchart
</div>
<div align=center>
<img width="1290" height="781" alt="image" src="https://github.com/user-attachments/assets/ed38276a-672e-4f51-9f1e-91add239cd7a" />
</div>


# Project Hub (main)

이 레포의 `main` 브랜치는 **단일 구현을 합치는 목적이 아니라**,  
각자 다른 접근/실험/구현을 **브랜치 단위로 분리 관리**하기 위한 **허브(Entry) 브랜치**입니다.

> ✅ 처음 오신 분은 이 README를 보고 목적에 맞는 브랜치를 체크아웃해서 진행하세요.  
> ❗ `main` 자체에는 실행 가능한 완성본이 없을 수 있습니다(가이드 역할).

---

## TL;DR — 어디로 가면 되나요?

- **ver-1**: (예: baseline / 가장 단순한 구현 / 빠르게 흐름 파악)
- **ver-3**: (예: A 방식 / 성능 개선 / 구조 리팩토링)
- **ver-4**: (예: B 방식 / 실험적 접근 / 최신 시도)

아래 표에서 목적에 맞게 고르세요.

---

## Branch Directory

| 브랜치 | 누가/목적 | 핵심 특징 | 실행 방법 | 추천 대상 |
|---|---|---|---|---|
| `feature/ver-1` | (작성자/목적) | (키워드 2~3개) | 브랜치 README 참고 | 처음 보는 사람 / baseline 필요 |
| `ver-3` | (작성자/목적) | (키워드 2~3개) | 브랜치 README 참고 | (예: 구조/성능 관심) |
| `ver-4` | (작성자/목적) | (키워드 2~3개) | 브랜치 README 참고 | (예: 최신 시도 확인) |



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
