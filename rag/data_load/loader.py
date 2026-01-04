import glob
import json
import os
import gc
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split_data(data_path: str, chunk_size=1000, chunk_overlap=200) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " "], 
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    
    final_splits = []
    file_list = glob.glob(data_path)
    
    print(f"📚 [Data] 데이터 로드 및 청킹 시작... (대상 파일: {len(file_list)}개)")

    for file_path in file_list:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 1. 파일 하나를 메모리에 올림
            data = json.load(f)
            
            for item in data:
                page_content = (
                    f"제목: {item.get('제목', '')}\n"
                    f"한자명: {item.get('한자명', '')}\n"
                    f"정의: {item.get('[정의]', '')}\n"
                    f"내용: {item.get('[내용]', '')}"
                )
                
                metadata = {
                    "title": item.get('제목', ''),
                    "source": os.path.basename(file_path)
                }
                
                # 2. 즉시 청킹하여 결과만 final_splits에 추가
                chunks = text_splitter.split_text(page_content)
                title_prefix = f"문서제목: {metadata['title']}\n"
                
                for i, chunk in enumerate(chunks):
                    content = title_prefix + chunk if i > 0 else chunk
                    final_splits.append(Document(page_content=content, metadata={**metadata, "chunk_id": i}))
            
    if 'data' in locals():
        del data
    
    gc.collect() 
    print(f"✅ [Data] 총 {len(final_splits)}개의 청크 생성 완료. 메모리 정리 후 리턴합니다.")
    
    return final_splits