"""
RAG 시스템 초기화 및 관리
"""

from langchain_upstage import UpstageEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import yaml


_compression_retriever = None


def load_config():
    """config.yaml 파일 로드"""
    with open('../config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def initialize_rag():
    """RAG 시스템 초기화"""
    global _compression_retriever

    if _compression_retriever is None:
        print("RAG 시스템 초기화 중 (최초 1회)...")
        
        config = load_config()
        rag_config = config['rag']

        query_embeddings = UpstageEmbeddings(
            api_key=rag_config['upstage_api_key'],
            model=rag_config['embedding_model']
        )

        vectorstore_query = Chroma(
            persist_directory=rag_config['vectorstore']['persist_directory'],
            embedding_function=query_embeddings,
            collection_name=rag_config['vectorstore']['collection_name']
        )

        model = HuggingFaceCrossEncoder(
            model_name=rag_config['retriever']['reranker_model']
        )
        compressor = CrossEncoderReranker(
            model=model, 
            top_n=rag_config['retriever']['reranker_top_n']
        )
        retriever = vectorstore_query.as_retriever(
            search_kwargs={"k": rag_config['retriever']['search_k']}
        )

        _compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=retriever
        )

    return _compression_retriever