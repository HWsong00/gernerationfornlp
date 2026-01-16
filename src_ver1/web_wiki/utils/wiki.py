import wikipedia
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict

class WikipediaAPI:
    def __init__(self, lang: str = "ko", max_pages_per_keyword: int = 3):
        """
        위키백과 검색 및 본문 추출 클래스
        """
        wikipedia.set_lang(lang) #
        self.max_pages_per_keyword = max_pages_per_keyword #

    def search_and_fetch(self, keywords: List[str]) -> List[dict]:
        """
        키워드 리스트를 받아 위키 문서를 가져옴
        """
        docs = []
        seen_titles = set()

        for kw in keywords:
            try:
                # 키워드당 최대 설정된 페이지 수만큼 검색 결과 가져오기
                search_results = wikipedia.search(kw)[: self.max_pages_per_keyword] #
            except Exception:
                continue

            for title in search_results:
                if title in seen_titles:
                    continue
                try:
                    page = wikipedia.page(title) #
                except Exception:
                    continue
                
                # 제목과 본문 저장
                docs.append({"title": page.title, "content": page.content}) #
                seen_titles.add(page.title) #
        return docs

class WikiChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        긴 위키 본문을 작은 단위(Chunk)로 나누는 클래스
        """
        self.splitter = RecursiveCharacterTextSplitter( #
            chunk_size=chunk_size, #
            chunk_overlap=chunk_overlap, #
            separators=["\n\n", "\n", ". ", " "], #
        )

    def chunk(self, wiki_docs: List[Dict]) -> List[Dict]:
        """
        본문을 청킹하여 ID와 함께 리스트로 반환
        """
        chunks = []
        idx = 0
        for doc in wiki_docs:
            text_chunks = self.splitter.split_text(doc["content"]) #
            for ch in text_chunks:
                chunks.append({
                    "id": idx,
                    "title": doc["title"],
                    "text": ch,
                }) #
                idx += 1
        return chunks