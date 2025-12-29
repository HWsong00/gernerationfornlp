"""
Retriever Node: 전략 분류, 쿼리 생성, Dual Search 수행
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .state import MCQState
from utils.llm import llm_with_params

def retrieve_node(state: MCQState):
    """
    한국사 문제에 대해 전략을 분류하고, 키워드를 생성하며, Dual Search를 수행하는 노드
    
    Args:
        state: MCQState
        
    Returns:
        dict: {
            "strategy": str,
            "summary": str,
            "optimized_query": str,
            "retrieved_context": str
        }
    """
    if not state.get('is_history'):
        return {"retrieved_context": "한국사 문제가 아니므로 검색을 생략합니다."}

    # Phase 1: INFERENCE vs GENERAL 분류
    router_prompt = ChatPromptTemplate.from_template(
        "<|im_start|>system\n문제를 분류하세요: \n"
        "- **INFERENCE**: (가), '이 왕', '이 단체' 등 주어가 생략되어 추론이 필요한 경우\n"
        "- **GENERAL**: 대상이 명확한 사실 확인 문제\n"
        "단어 하나만 출력하세요.<|im_end|>\n"
        "<|im_start|>user\n지문: {paragraph}\n질문: {question}\n분류:<|im_end|>\n<|im_start|>assistant\n"
    )
    
    # Note: llm_with_params는 외부에서 정의되어야 함
    strategy = (router_prompt | llm_with_params | StrOutputParser()).invoke(state).strip()

    # Phase 2: 요약 및 10대 키워드 생성 (병렬 처리 권장이나 여기선 순차 구현)
    gen_prompt = ChatPromptTemplate.from_template(
        "<|im_start|>system\n당신은 역사 전문가입니다. 다음 지침을 따르세요:\n"
        "1. 지문을 2문장 이내로 핵심 요약하세요.\n"
        "2. 검색을 위한 핵심 키워드를 '콤마'로 구분하여 10개 이내로 뽑으세요.\n"
        "형식: 요약: [내용] / 키워드: [키워드들]<|im_end|>\n"
        "<|im_start|>user\n지문: {paragraph}\n질문: {question}\n결과:<|im_end|>\n<|im_start|>assistant\n"
    )
    gen_res = (gen_prompt | llm_with_params | StrOutputParser()).invoke(state)

    summary = gen_res.split("요약:")[1].split("/ 키워드:")[0].strip()
    keywords = gen_res.split("키워드:")[1].strip()

    print(f"   🚦 [전략]: {strategy} | ✨ [키워드]: {keywords}")

    # Phase 3: Dual Search (키워드 쿼리 + 지문 요약)
    # Note: hybrid_retriever는 외부에서 정의되어야 함
    docs_query = hybrid_retriever.invoke(keywords)
    docs_summary = hybrid_retriever.invoke(summary)

    # 중복 제거 및 컨텍스트 조립
    combined = {d.page_content: d for d in (docs_query + docs_summary)}.values()
    para_context = "\n".join([f"- {d.page_content}" for d in list(combined)[:6]])

    return {
        "strategy": strategy,
        "summary": summary,
        "optimized_query": keywords,
        "retrieved_context": f"전략: {strategy}\n요약: {summary}\n참고자료:\n{para_context}"
    }