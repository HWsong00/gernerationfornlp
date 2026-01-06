import os
from openai import OpenAI

# 1. 프로젝트 전역에서 사용할 모델 식별자 (llama-server는 어떤 문자열이든 수용합니다)
MODEL_NAME = "Qwen3-30B-A3B-Instruct-2507"

def get_llm_client():
    """
    llama-server(http://localhost:8080)와 통신하는 
    순수 OpenAI SDK 클라이언트를 생성하여 반환합니다.
    """
    # llama-server는 기본적으로 별도의 API Key가 필요 없으나, 
    # SDK 규격상 임의의 값을 넣어줍니다.
    client = OpenAI(
        base_url="http://localhost:8080/v1", 
        api_key="sk-no-key-required"
    )
    return client

# 2. (옵션) 서버가 정상인지 간단히 확인하는 유틸리티
def check_server_status():
    import requests
    try:
        response = requests.get("http://localhost:8080/health")
        return response.status_code == 200
    except:
        return False

if __name__ == "__main__":
    # 파일 직접 실행 시 테스트
    if check_server_status():
        print("✅ llama-server가 정상적으로 작동 중입니다.")
        client = get_llm_client()
        print(f"🚀 클라이언트 준비 완료 (Base URL: {client.base_url})")
    else:
        print("❌ 서버 연결 실패. llama-server가 8080 포트에서 실행 중인지 확인하세요.")