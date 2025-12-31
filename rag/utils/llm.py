import os
from huggingface_hub import hf_hub_download
from langchain_community.chat_models import ChatLlamaCpp

# 다른 파일에서 이 변수들을 참조할 수 있게 상수로 빼둡니다.
REPO_ID = "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF"
FILENAME = "Qwen3-30B-A3B-Instruct-2507-UD-Q6_K_XL.gguf"

# 실제 모델 객체를 담을 변수 (처음엔 비워둠)
_llm_instance = None

def get_llm():
    """모델이 필요할 때만 로드하고, 이미 로드되었다면 기존 객체를 반환하는 함수"""
    global _llm_instance
    
    if _llm_instance is None:        
        # 모델 다운로드/경로 확인
        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
        )

        # LlamaCpp 인스턴스 생성
        _llm_instance = ChatLlamaCpp(
            model_path=model_path,
            n_gpu_layers=-1,      # L4 GPU 전체 사용
            n_ctx=32768,          # 긴 문맥 지원
            max_tokens=2048,
            temperature=0.7,
            top_p=0.90,
            repeat_penalty=1.1,
            verbose=False,
        )
        
    return _llm_instance

# 만약 이 파일을 직접 실행(python llm.py)했을 때만 테스트로 동작하게 함
if __name__ == "__main__":
    test_llm = get_llm()
    print("테스트 호출 완료:", test_llm)