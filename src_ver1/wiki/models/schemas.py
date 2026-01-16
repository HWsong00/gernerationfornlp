from pydantic import BaseModel, Field


class ProblemAnalysis(BaseModel):
    """문제 분석 결과를 구조화된 형태로 반환"""
    problem_type: str = Field(
        description=(
            "문제 유형 분류"
        )
    )
    needs_external_knowledge: bool = Field(
        description=(
            "이 문제를 풀기 위해 외부 지식(한국사, 정치, 경제, 사회 등)이 필요한지 판단"
            "KLUE MRC처럼 지문 내에 모든 정보가 있는 경우 False"
            "한국사, 정치, 경제 등 특정 지식을 알아야 풀 수 있는 경우 True"
        )
    )
    reasoning: str = Field(
        description="왜 외부 지식이 필요한지/필요 없는지 설명"
    )
    confidence: float = Field(
        description="이 판단에 대한 신뢰도 (0.0 ~ 1.0)",
        ge=0.0,
        le=1.0
    )