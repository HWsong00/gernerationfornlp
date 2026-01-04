import os
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

def get_langfuse_handler():
    """
    코랩 보안 비밀(Secrets) 또는 .env에서 환경 변수를 로드하여 
    Langfuse 핸들러를 반환합니다.
    """
    # 1. 키값 가져오기 (코랩 환경 우선, 실패 시 환경변수/os.getenv 시도)
    try:
        from google.colab import userdata
        public_key = userdata.get("LANGFUSE_PUBLIC_KEY")
        secret_key = userdata.get("LANGFUSE_SECRET_KEY")
        host = userdata.get("LANGFUSE_HOST")
    except (ImportError, Exception):
        # 코랩이 아닐 경우 os.environ에서 가져옴 (dotenv 로드 가정)
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        host = os.getenv("LANGFUSE_HOST")

    if not all([public_key, secret_key, host]):
        print("⚠️ [Langfuse] 키값이 설정되지 않았습니다. 모니터링 없이 진행합니다.")
        return None

    try:
        langfuse_client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
        
        if langfuse_client.auth_check():
            print("✅ [Langfuse] 연결 성공! 대시보드에서 로그를 확인하세요.")
            return CallbackHandler(public_key=public_key, secret_key=secret_key, host=host)
        else:
            print("❌ [Langfuse] 인증 실패!")
            return None
    except Exception as e:
        print(f"❌ [Langfuse] 에러: {e}")
        return None

# 전역 핸들러 인스턴스
langfuse_handler = get_langfuse_handler()