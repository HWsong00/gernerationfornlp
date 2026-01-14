import re
from typing import List, Optional
from kiwipiepy import Kiwi

# 기본 태그 설정 (필요시 settings.py에서 관리해도 좋습니다)
DEFAULT_TAG_INCLUDE = [
    'NNG', 'NNP', 'NNB', 'NR', 'VV', 'VA', 'MM', 'XR', 
    'SW', 'SL', 'SH', 'SN', 'SB'
]

# 1. Kiwi 싱글톤 팩토리
_kiwi_instance = None

def get_kiwi():
    """Kiwi 인스턴스를 단 하나만 생성하여 공유합니다."""
    global _kiwi_instance
    if _kiwi_instance is None:
        print("🔍 [Tokenizer] Kiwi 형태소 분석기 로드 중...")
        _kiwi_instance = Kiwi()
    return _kiwi_instance

def _fallback_tokenize(text: str) -> List[str]:
    """분석 실패 시 안전장치"""
    return re.findall(r'\b\w+\b', text, re.UNICODE)

# 2. 핵심 토큰화 로직 (장(Jang)님의 로직 유지)
def tokenize_kiwi(
    text: str,
    text_type: str, # "corpus" 또는 "query"
    kiwi: Optional[Kiwi] = None,
    tag_include: Optional[List[str]] = None,
    top_n: int = 3,
    score_threshold: float = 1.2,
) -> List[str]:
    # 인자가 없으면 싱글톤 인스턴스와 기본 태그 사용
    kiwi = kiwi or get_kiwi()
    tag_include = tag_include or DEFAULT_TAG_INCLUDE
    
    try:
        if text_type == "corpus":
            # 색인 시: 본문 길이에 따라 유동적으로 후보군 확장
            analyzed = kiwi.analyze(text, top_n=top_n + len(text) // 200)
            if not analyzed: return _fallback_tokenize(text)
            
            num_candi = 1
            # 1위 대비 점수차가 크지 않은 후보들 포함 (재현율 확보)
            while (num_candi < len(analyzed) and 
                   analyzed[num_candi][1] > score_threshold * analyzed[0][1]):
                num_candi += 1
                
        elif text_type == "query":
            # 검색 시: 정밀한 상위 후보 사용
            analyzed = kiwi.analyze(text, top_n=top_n)
            if not analyzed: return _fallback_tokenize(text)
            num_candi = min(3, len(analyzed))

        # 형태소/태그 결합 추출
        all_tokenized = [
            f"{t.form}/{t.tag}"
            for nc in range(num_candi)
            for t in analyzed[nc][0]
            if t.tag in tag_include
        ]

        unique_tokens = list(set(all_tokenized))
        return unique_tokens if unique_tokens else _fallback_tokenize(text)
    
    except Exception as e:
        print(f"⚠️ [Tokenizer] 에러 발생: {e}")
        return _fallback_tokenize(text)