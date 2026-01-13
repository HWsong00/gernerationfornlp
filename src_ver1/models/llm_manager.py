"""
LLM 초기화 및 관리
"""

from llama_cpp import Llama
import yaml


_llm_instance = None


def load_config():
    """config.yaml 파일 로드"""
    with open('../config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_llm():
    """Llama.cpp 모델 인스턴스 반환 - 싱글톤"""
    global _llm_instance
    if _llm_instance is None:
        try:
            config = load_config()
            llm_config = config['llm']
            
            _llm_instance = Llama.from_pretrained(
                repo_id=llm_config['repo_id'],
                filename=llm_config['filename'],
                n_ctx=llm_config['n_ctx'],
                n_gpu_layers=llm_config['n_gpu_layers'],
                flash_attn=llm_config['flash_attn'],
                verbose=llm_config['verbose']
            )
        except Exception as e:
            print(f"LLM 초기화 오류: {e}")
            raise
    return _llm_instance


def initialize_llm():
    """LLM 초기화 (명시적 호출용)"""
    return get_llm()


def get_generation_params():
    """생성 파라미터 반환"""
    config = load_config()
    return config['llm']['generation']