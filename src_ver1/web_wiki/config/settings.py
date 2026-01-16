import os
from pathlib import Path

# 1. 경로 설정
# 현재 settings.py 파일의 위치를 기준으로 프로젝트 루트 폴더를 잡습니다.
BASE_DIR = Path(__file__).resolve().parent.parent

# 데이터 경로: 코랩이나 서버 환경에 맞춰 환경 변수 또는 직접 경로를 수정합니다.
DATA_PATH = "/content/drive/MyDrive/수능 풀이/history_final_dataset/*.json"
