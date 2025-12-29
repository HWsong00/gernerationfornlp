"""
Classifier Node: 한국사 문제 여부 판별
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .state import MCQState
from utils.llm import llm_with_params

def classifier_node(state: MCQState):
    """
    한국사 문제 여부를 판별하는 노드
    
    Args:
        state: MCQState
        
    Returns:
        dict: {"is_history": bool}
    """
    print(f"🧐 [Classifier] 한국사 문제 여부 판별 중...")
    prompt = ChatPromptTemplate.from_template(
        "<|im_start|>system\n주어진 문제가 '한국사(Korean History)'와 관련되었는지 판단하여 YES 또는 NO로만 답하세요.<|im_end|>\n"
        "<|im_start|>user\n지문: {paragraph}\n질문: {question}\n판별:<|im_end|>\n<|im_start|>assistant\n"
    )
    
    # Note: llm_with_params는 외부에서 정의되어야 함
    # 예: from utils.llm import llm_with_params
    chain = prompt | llm_with_params | StrOutputParser()
    res = chain.invoke({"paragraph": state['paragraph'], "question": state['question']}).strip().upper()

    return {"is_history": "YES" in res}