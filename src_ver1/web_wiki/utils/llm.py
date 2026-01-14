import os
import time
import subprocess
import requests
from openai import OpenAI

# 1. 프로젝트 전역에서 사용할 모델 식별자 (llama-server는 어떤 문자열이든 수용합니다)
MODEL_NAME = "Qwen3-30B-A3B-Instruct-2507"

from openai import OpenAI

# 서버 실행 시 설정한 포트와 동일해야 합니다!
PORT = 8081 

def get_llm_client():
    """llama-server와 통신하는 Native OpenAI 클라이언트"""
    client = OpenAI(
        base_url=f"http://localhost:{PORT}/v1", 
        api_key="sk-no-key-required"
    )
    return client

def check_server_status():
    import requests
    try:
        # 헬스 체크 엔드포인트도 같은 포트를 바라봐야 함
        response = requests.get(f"http://localhost:{PORT}/health")
        return response.status_code == 200
    except:
        return False

def start_llama_server():
    """서버가 없으면 직접 nohup으로 실행"""
    if check_server_status():
        print("✅ 서버가 이미 실행 중입니다.")
        return None

    print("🚀 서버가 꺼져 있습니다. 새로 가동합니다...")
    # 질문자님이 아까 쓰셨던 그 명령어입니다.
    cmd = f"""nohup /content/llama-server \
        --model "models/Qwen3-30B-A3B-Instruct-2507-UD-Q6_K_XL.gguf" \
        --n-gpu-layers -1 \
        --ctx-size 14400 \
        --parallel 2 \
        --cont-batching \
        --flash-attn on \
        --port {PORT} \
        --host 0.0.0.0 \
        > server.log 2>&1 &"""
    
    # 셸 명령어로 실행
    process = subprocess.Popen(cmd, shell=True)
    
    # 서버가 준비될 때까지 대기 
    for i in range(240):
        if check_server_status():
            print("✅ 서버 로드 완료!")
            return process
        if i % 5 == 0:
            print(f"⏳ 모델 로딩 중... ({i*5}초 경과)")
        time.sleep(5)
    
    raise TimeoutError("서버 가동에 실패했습니다. server.log를 확인하세요.")


if __name__ == "__main__":
    # 파일 직접 실행 시 테스트
    if check_server_status():
        print("✅ llama-server가 정상적으로 작동 중입니다.")
        client = get_llm_client()
        print(f"🚀 클라이언트 준비 완료 (Base URL: {client.base_url})")
    else:
        print("❌ 서버 연결 실패. llama-server가 8080 포트에서 실행 중인지 확인하세요.")