import os
from langfuse.langchain import CallbackHandler

def get_langfuse_handler():
    try:
        # 1. 코랩 보안 비밀(Secrets) 우선 로드
        from google.colab import userdata
        pk = userdata.get("LANGFUSE_PUBLIC_KEY")
        sk = userdata.get("LANGFUSE_SECRET_KEY")
        host = userdata.get("LANGFUSE_HOST")
    except (ImportError, Exception):
        # 2. 로컬 환경(.env 등) 대응
        pk = os.getenv("LANGFUSE_PUBLIC_KEY")
        sk = os.getenv("LANGFUSE_SECRET_KEY")
        host = os.getenv("LANGFUSE_HOST")

    if not all([pk, sk, host]):
        print("⚠️ [Langfuse] 키값이 설정되지 않았습니다. 로그 없이 진행합니다.")
        return None

    # 환경 변수로 세팅하여 핸들러가 자동으로 인지하게 함
    os.environ["LANGFUSE_PUBLIC_KEY"] = pk
    os.environ["LANGFUSE_SECRET_KEY"] = sk
    os.environ["LANGFUSE_HOST"] = host

    try:
        # langfuse.langchain의 CallbackHandler는 환경 변수를 자동으로 참조합니다.
        handler = CallbackHandler() 
        print("✅ [Langfuse] 핸들러 생성 성공!")
        return handler
    except Exception as e:
        print(f"❌ [Langfuse] 생성 실패: {e}")
        return None

# 전역 핸들러 인스턴스
langfuse_handler = get_langfuse_handler()