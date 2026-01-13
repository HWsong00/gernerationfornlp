import json
import re
from workflow.state import ProblemState
from models.llm_manager import get_llm, get_generation_params
from rag.retriever import initialize_rag
from prompts import (
    PROBLEM_ANALYSIS_PROMPT,
    PROBLEM_SOLVING_SYSTEM_MESSAGE,
    SOLVE_WITHOUT_RAG_USER_MESSAGE,
    SOLVE_WITH_RAG_USER_MESSAGE
)


def analyze_problem(state: ProblemState) -> dict:
    """
    문제를 분석하여 외부 지식 필요 여부를 판단
    """
    llm = get_llm()
    gen_params = get_generation_params()

    context = state['context']
    question = state['question']

    prompt = PROBLEM_ANALYSIS_PROMPT.format(
        context=context,
        question=question
    )

    messages = [
        {"role": "user", "content": prompt}
    ]

    out = llm.create_chat_completion(
        messages=messages,
        max_tokens=gen_params['max_tokens'],
        temperature=gen_params['temperature'],
        top_p=gen_params['top_p'],
        top_k=gen_params['top_k'],
        min_p=gen_params['min_p'],
        repeat_penalty=gen_params['repeat_penalty']
    )

    response_text = out['choices'][0]['message']['content']

    def extract_json(text):
        """텍스트에서 JSON 객체를 추출 (중첩된 중괄호 처리)"""
        # ```json ... ``` 형식 제거
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)

        # 첫 번째 { 찾기
        start_idx = text.find('{')
        if start_idx == -1:
            return None

        # 중괄호 카운트로 JSON 끝 찾기
        brace_count = 0
        for i in range(start_idx, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    return text[start_idx:i+1]
        return None

    json_str = extract_json(response_text)

    if json_str:
        try:
            analysis = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"⚠️SON 파싱 오류: {e}")
            print(f"추출된 JSON 문자열: {json_str[:200]}...")
            analysis = {
                "problem_type": "unknown",
                "needs_external_knowledge": True,
                "reasoning": f"JSON 파싱 실패: {str(e)}",
                "confidence": 0.5
            }
    else:
        print(f"JSON 형식을 찾을 수 없음. 응답 텍스트: {response_text[:200]}...")
        analysis = {
            "problem_type": "unknown",
            "needs_external_knowledge": True,
            "reasoning": "JSON 형식이 아님. 기본값 사용.",
            "confidence": 0.5
        }

    return {
        "problem_type": analysis.get("problem_type", "unknown"),
        "needs_external_knowledge": analysis.get("needs_external_knowledge", True),
        "reasoning": analysis.get("reasoning", ""),
        "confidence": analysis.get("confidence", 0.5)
    }


def retrieve_rag_context(state: ProblemState) -> dict:
    """
    RAG를 사용하여 보강 자료 검색
    needs_external_knowledge가 True일 때만 호출됨
    """
    compression_retriever = initialize_rag()

    context = state['context']
    choices = state['choices']

    rag_docs = []
    queries = [context] + choices

    for q in queries:
        refined_docs = compression_retriever.invoke(q)
        rag_docs.extend(refined_docs)

    # 중복 제거
    seen = set()
    unique_docs = []
    for doc in rag_docs:
        source_id = doc.metadata.get('한글명칭') or doc.metadata.get('source', 'Unknown')
        doc_id = (source_id, doc.page_content[:100])
        if doc_id not in seen:
            seen.add(doc_id)
            unique_docs.append(doc)

    # 출처 표시
    rag_context_parts = []
    for d in unique_docs:
        한글명칭 = d.metadata.get('한글명칭')
        if 한글명칭:
            rag_context_parts.append(f"출처: {한글명칭}, 내용: {d.page_content}")
        else:
            rag_context_parts.append(f"내용: {d.page_content}")
    rag_context = "\n".join(rag_context_parts)

    return {"rag_context": rag_context}


def solve_without_rag(state: ProblemState) -> dict:
    """
    RAG 없이 지문만으로 문제 풀이
    KLUE MRC 같은 경우
    """
    llm = get_llm()
    gen_params = get_generation_params()

    context = state['context']
    question = state['question']

    system_msg = PROBLEM_SOLVING_SYSTEM_MESSAGE.strip()
    user_msg = SOLVE_WITHOUT_RAG_USER_MESSAGE.strip().format(
        context=context,
        question=question
    )

    messages = [
        {"role": "user", "content": system_msg},
        {"role": "system", "content": user_msg}
    ]

    out = llm.create_chat_completion(
        messages=messages,
        max_tokens=gen_params['max_tokens'],
        temperature=gen_params['temperature'],
        top_p=gen_params['top_p'],
        top_k=gen_params['top_k'],
        min_p=gen_params['min_p'],
        repeat_penalty=gen_params['repeat_penalty']
    )

    answer_text = out['choices'][0]['message']['content'].strip()

    return {
        "final_answer": answer_text,
        "rag_context": ""
    }


def solve_with_rag(state: ProblemState) -> dict:
    """
    RAG 보강 자료를 사용하여 문제 풀이
    외부 지식이 필요한 경우
    """
    llm = get_llm()
    gen_params = get_generation_params()

    context = state['context']
    question = state['question']
    rag_context = state.get('rag_context', '')

    system_msg = PROBLEM_SOLVING_SYSTEM_MESSAGE.strip()
    user_msg = SOLVE_WITH_RAG_USER_MESSAGE.strip().format(
        rag_context=rag_context,
        context=context,
        question=question
    )

    messages = [
        {"role": "user", "content": system_msg},
        {"role": "system", "content": user_msg}
    ]

    out = llm.create_chat_completion(
        messages=messages,
        max_tokens=gen_params['max_tokens'],
        temperature=gen_params['temperature'],
        top_p=gen_params['top_p'],
        top_k=gen_params['top_k'],
        min_p=gen_params['min_p'],
        repeat_penalty=gen_params['repeat_penalty']
    )

    answer_text = out['choices'][0]['message']['content'].strip()

    return {"final_answer": answer_text}