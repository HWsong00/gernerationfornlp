"""
수능형 문제 풀이 Agent with RAG On/Off Decision
메인 실행 파일
"""

import yaml
from utils.data_parser import parse_dataset
from rag.utils.processor import process
from retriever.retriever import initialize_rag


def load_config():
    """config.yaml 파일 로드"""
    with open('config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    """메인 실행 함수"""
    # 설정 로드
    config = load_config()
    data_config = config['data']
    
    # 데이터셋 파싱
    df = parse_dataset(data_config['input_csv'])
    
    # RAG 초기화
    initialize_rag()
    
    results = process(df, data_config['output_csv'])
    
    print(f"총 {len(results)}개 문제 처리 완료")
    
if __name__ == "__main__":
    main()