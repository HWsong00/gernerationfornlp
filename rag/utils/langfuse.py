import os
from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.callback import CallbackHandler

# 1. 환경 변수 로드
load_dotenv()

def get_langfuse_handler():
    """
    Langfuse 연결을 확인하고 LangChain용 CallbackHandler를 반환합니다.
    """
    # 환경 변수 확인
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST")

    if not all([public_key, secret_key, host]):
        print("⚠️ [Langfuse] 환경 변수가 설정되지 않았습니다. 로그 없이 진행합니다.")
        return None

    try:
        # 인증 체크용 클라이언트
        langfuse_client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host
        )

        if langfuse_client.auth_check():
            print("✅ [Langfuse] 연결 성공! 대시보드에서 로그를 확인하세요.")
            # LangChain용 핸들러 생성
            return CallbackHandler(
                public_key=public_key,
                secret_key=secret_key,
                host=host
            )
        else:
            print("❌ [Langfuse] 인증 실패: 키값을 확인해주세요.")
            return None

    except Exception as e:
        print(f"❌ [Langfuse] 연결 중 에러 발생: {e}")
        return None

# 전역 핸들러 인스턴스 (main.py에서 임포트하여 사용)
langfuse_handler = get_langfuse_handler()