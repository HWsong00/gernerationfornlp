from langchain_community.llms import LlamaCpp
import os
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
import glob
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langfuse import Langfuse
import ast
import pandas as pd
from tqdm import tqdm

MODEL_PATH = hf_hub_download(
    repo_id="unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF",
    filename="Qwen3-30B-A3B-Instruct-2507-UD-Q6_K_XL.gguf",
)

# LlamaCpp로 직접 모델 로드
base_llm = LlamaCpp(
        model_path=MODEL_PATH,
        n_gpu_layers=-1,      # L4 GPU 사용
        n_ctx=32768,          # 긴 문맥(RAG) 지원
        max_tokens=2048,
        temperature=0.7,
        top_p=0.90,
        repeat_penalty=1.1,
        verbose=False,        # 상세 로그 출력 여부
)
    
print("=====모델 로드 완료=====")

print("=====데이터 로딩 시작=====")

DATA_PATH = "...한국사 RAG 자료 데이터 경로..."
DB_PATH = "...chromaDB 저장 경로..."
COLLECTION_NAME = "korean_history_2"

# 임베딩 모델 설정
embedding_model = HuggingFaceEmbeddings(
    model_name="dragonkue/BGE-m3-ko",
    model_kwargs={'device': 'cuda'}
)

if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
    print(f"기존 벡터 DB를 '{DB_PATH}'에서 불러옵니다...")
    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME
    )
    print("=====기존 DB 로드 완료=====")
    
else:
    print(f"기존 DB가 없습니다. '{DATA_PATH}'에서 새 DB 구축을 시작합니다.")
    
    files = glob.glob(DATA_PATH)
    documents = []

    def metadata_func(record: dict, metadata: dict) -> dict:
        metadata["volume"] = record.get("volume")
        metadata["title"] = record.get("title")
        return metadata

    for file in files:
        try:
            loader = JSONLoader(
                file_path=file,
                jq_schema='.[]',
                content_key='text',
                metadata_func=metadata_func
            )
            documents.extend(loader.load())
        except Exception as e:
            print(f"{file} 로드 실패: {e}")

    if documents:
        print(f"총 {len(documents)}개의 문서 로드 완료")
        
        # 청킹
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        print(f"{len(splits)}개의 청크로 분할됨")

        # 벡터 DB 생성 및 저장
        print(f"=====신규 벡터 DB를 '{DB_PATH}'에 생성 및 저장 중=====")
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embedding_model,
            collection_name=COLLECTION_NAME,
            persist_directory=DB_PATH
        )
        print("=====신규 DB 생성 및 저장 완료=====")
    else:
        print("로드된 문서가 없어 DB를 생성할 수 없습니다.")
        vectorstore = None

# 리트리버 준비
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
print("=====리트리버 준비 완료=====")

# .env 파일의 내용을 환경 변수로 로드합니다.
load_dotenv()

# 환경 변수에서 가져와 설정 (값이 없으면 None 반환)
os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY")
os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY")
os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_HOST")

try:
    langfuse = Langfuse()
    if langfuse.auth_check():
        print("=====랭퓨즈 연결 완료=====")
    else:
        print("=====랭퓨즈 연결 실패=====")
except Exception as e:
    print(f"에러: {e}")
    
    
langfuse_handler = None
try:
    from langfuse.langchain import CallbackHandler
    langfuse_handler = CallbackHandler()
    print("=====LangFuse 핸들러 연결 성공! (로그 적재 중)=====")

except ImportError:
    print("=====LangFuse 패키지 경로 에러: 로그 없이 진행=====")
except Exception as e:
    print(f"=====LangFuse 연결 실패 ({e}): 로그 없이 진행=====")
    

llm_with_params = base_llm
print("=====LLM 설정 완료=====")


import re
from typing import TypedDict, List
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

parser = StrOutputParser()

# State 정의 
class MCQState(TypedDict):
    id: str
    paragraph: str
    question: str
    choices: List[str]
    is_korean_history: bool # 한국사 문제인지
    retrieved_context: str  # 최종 수집된 증거 자료
    full_response: str      # LLM의 풀이 과정 (Raw)
    final_answer: str       # 추출된 정답 번호


ROUTER_PROMPT = PromptTemplate.from_template(
    """<|im_start|>system
당신은 과목 분류 전문가입니다. 다음 문제가 "한국사" 과목의 문제인지 판단하세요.

[분류 기준]
- 한국사 지식이 꼭 필요한 문제: 'KOREAN_HISTORY'
- 세계사, 일반 논리, 문학, 단순 상식 등: 'GENERAL'

결과는 반드시 'KOREAN_HISTORY' 또는 'GENERAL' 중 한 단어로만 답하세요.
<|im_end|>
<|im_start|>user
[지문]
{paragraph}

[질문]
{question}
<|im_end|>
<|im_start|>assistant
"""
)

def router_node(state: MCQState):
    """
    ==== 한국사 문제인지 / 아닌지 분기하는 노드 =====
    """
    chain = ROUTER_PROMPT | llm_with_params | parser
    result = chain.invoke({
        "paragraph": state["paragraph"], 
        "question": state["question"]
    }).strip().upper()
    
    is_history = "KOREAN_HISTORY" in result
    
    return {"is_korean_history": is_history, "retrieved_context": ""}

# 분기 결정을 위한 조건부 함수
def route_decision(state: MCQState):
    if state.get("is_korean_history", False):
        return "retrieve"
    return "general_solve"


def retrieve_node(state: MCQState):
    """
    ===== 문제 유형별 쿼리 생성 및 RAG 실행 노드 =====
    """

    # 문제 유형 분류
    router_prompt = PromptTemplate.from_template(
        """<|im_start|>system
당신은 '문제 해결 전략가'입니다. 주어진 문제를 보고 최적의 검색 전략을 하나 선택하십시오.

[분류 기준]
1. **INFERENCE**: (가), (나), "이 인물", "이 단체" 처럼 주어가 가려져 있어 문맥 묘사를 통해 대상을 찾아야 하는 경우.
2. **SEQUENCE**: "(가)와 (나) 사이", "순서대로 나열", "연표", "시기" 등 시간의 흐름이나 연도를 묻는 경우.
3. **GENERAL**: 위 두 경우에 해당하지 않는 일반적인 사실 확인 문제 (대상이 명확한 경우).

반드시 **INFERENCE**, **SEQUENCE**, **GENERAL** 중 단어 하나만 출력하십시오.
<|im_start|>user
[지문]
{paragraph}

[질문]
{question}

[선지]
{choices}

전략:<|im_end|>
<|im_start|>assistant
"""
    )

    router_chain = router_prompt | llm_with_params | parser
    
    # 전략
    strategy = router_chain.invoke({
        "paragraph": state['paragraph'],
        "question": state['question'],
        "choices": str(state['choices'])
    }).strip()

    # 전략에 따라 system message를 다르게
    if "INFERENCE" in strategy:
        # INFERENCE: 몽타주 검색 (이름 추측 금지)
        sys_msg = """당신은 '역사 탐정'이다. [지문]과 [선지]을 보고, 숨겨진 주어((가), (나) 등)를 찾기 위한 '몽타주 검색어'를 만들어라.
**규칙** 
- (가)가 누구인지 절대 추측하여 특정 이름(예: 김구, 신라)을 넣지마라.
- 오직 지문에 묘사된 '행동', '사건 내용', '단체명', '장소'를 나열하라.
- 검색어만을 대답하여라.
"""

    elif "SEQUENCE" in strategy:
        # SEQUENCE: 연도/타임라인 검색
        sys_msg = """당신은 '연표 분석가'이다. [지문]과 [선지]을 보고, 사건의 순서나 시기를 파악하기 위한 검색어를 만들어라.
**규칙**
- 지문에 나온 사건들의 '발생 연도', '시대적 배경', '왕의 재위 기간' 관련 키워드를 반드시 포함하라.
- 검색어만을 대답하여라.
"""
    else: 
        # GENERAL: 핵심 요약 검색
        sys_msg = """당신은 '검색 키워드 요약 전문가'이다. [지문]과 [선지]을 보고, 질문에 답하기 위한 핵심 키워드를 추출하라.
**규칙** 
- 불필요한 조사를 빼고, '인물', '사건', '핵심 용어' 위주로 간결하게 요약하라.
- 키워드만을 대답하여라."""

    query_gen_prompt = PromptTemplate.from_template(
        """<|im_start|>system
{sys_msg}
<|im_start|>user
[지문]
{paragraph}

[선지]
{choices}

최적 검색어:<|im_end|>
<|im_start|>assistant
"""
    )

    gen_chain = query_gen_prompt | llm_with_params | parser
    generated_query = gen_chain.invoke({
        "sys_msg": sys_msg,
        "paragraph": state['paragraph'],
        "choices": str(state['choices'])
    }).strip()

    print(f"[생성된 쿼리]: {generated_query}")

    # [Dual Search] - llm이 만든 쿼리로 증강한 결과 + 지문을 쿼리로 증강한 결과
    # llm이 만든 쿼리로 증강한 결과
    llm_query_result = retriever.invoke(generated_query)

    # 지문을 쿼리로 증강한 결과
    para_query_result = retriever.invoke(state['paragraph'] + str(state['choices']))

    # 두 검색 결과 합치기 (정밀 검색 우선)
    combined_docs = llm_query_result + para_query_result
    unique_docs = [] 
    seen = set()
    for d in combined_docs:  # 두 결과가 가져온 문서들 중 unique 문서들만 
        if d.page_content not in seen:
            unique_docs.append(d)
            seen.add(d.page_content)

    # ----리랭커 사용하여 보완할 지점-----
    final_para_docs = unique_docs[:5] # 상위 5개만 사용
    para_context = "\n".join([f"- {d.page_content}" for d in final_para_docs])

    # # -----------------------------------------------------
    # # Phase 4: [Choices Search] 선지별 교차 검증
    # # -----------------------------------------------------
    # choices_evidence = []
    # for idx, choice in enumerate(state['choices']):
    #     # 검색어 = 최적화된 쿼리 + 선지 내용
    #     combined_q = f"{generated_query} {choice}"
    #     choice_docs = retriever.invoke(combined_q)

    #     if choice_docs:
    #         evi = " / ".join([d.page_content for d in choice_docs[:2]]) # 선지당 2개만
    #     else:
    #         evi = "관련 정보 없음"
    #     choices_evidence.append(f"[선지 {idx+1}]: {evi}")

    # -----------------------------------------------------
    # Phase 5: 문맥 조립
    # -----------------------------------------------------
    full_context = f"""
=== [배경 지식 (전략: {strategy})] ===
{para_context}
"""
    return {"retrieved_context": full_context}


# =========================================================
# 3. Solver Node (문제 풀이)
# =========================================================
def ko_history_solver_node(state: MCQState):
    """
    한국사 문제 풀이 노드 
    """
    choices_str = "\n".join(
        [f"{i+1}. {c}" for i, c in enumerate(state['choices'])]
    )

    system_prompt = """당신은 논리적인 한국사 전문가입니다.
제공된 <개념 보충 자료>를 근거로 문제를 해결하십시오.
- 자료에 없는 내용은 추측하지 말고, 자료를 바탕으로 논리적으로 오답을 소거하십시오.
- 마지막 줄에는 반드시 {"정답": "번호"} 형식으로 답을 출력하십시오."""

    prompt = PromptTemplate.from_template(
        """<|im_start|>system
{system_msg}<|im_end|>
<|im_start|>user
<지문>
{paragraph}

<개념 보충 자료>
{retrieved_context}

<질문>
{question}

<선지>
{choices}
<|im_end|>
<|im_start|>assistant
"""
    )

    chain = prompt | llm_with_params | parser

    response = chain.invoke({
        "system_msg": system_prompt,
        "paragraph": state['paragraph'],
        "question": state['question'],
        "choices": choices_str,
        "retrieved_context": state['retrieved_context']
    })

    return {"full_response": response}


def general_solver_node(state: MCQState):
    """
    한국사를 제외한 과목 문제 풀이 노드
    """
    choices_str = "\n".join([f"{i+1}. {c}" for i, c in enumerate(state['choices'])])
    prompt = PromptTemplate.from_template(
      """<|im_start|>system
      당신은 논리적이고 꼼꼼한 학생입니다. 주어진 제시문과 질문을 분석하여 객관식 문제를 해결해야 합니다.

### 지시사항
1. **문제 분석**: 질문이 요구하는 핵심이 무엇인지 먼저 정의하십시오.
2. **사고 과정 (CoT)**:
   - 각 선택지가 정답이거나 오답인 이유를 제시문에서 근거를 찾아 명확히 설명하십시오.
   - 단순히 정답만 맞히지 말고, 왜 나머지 선택지는 정답이 될 수 없는지(오답 소거)를 논리적으로 서술하십시오.
3. **형식 준수**:
   - 풀이 과정은 줄글로 작성하되, 불필요한 반복을 피하십시오.
   - **가장 마지막 줄**에는 아래 JSON 형식으로 답안을 제출하십시오.
{{"정답": "번호"}}
<|im_end|>
<|im_start|>user
[지문]
{paragraph}

[질문]
{question}

[선지]
{choices}
<|im_end|>
<|im_start|>assistant
"""
    )
    response = (prompt | llm_with_params | parser).invoke({
        "paragraph": state['paragraph'],
        "question": state['question'],
        "choices": choices_str
    })

    return {'full_response': response}


def parser_node(state: MCQState):
    """
    정답 추출 노드
    """
    text = state['full_response']
    answer = None
    # JSON 형식 찾기
    match = re.search(r'{"정답":\s*"(\d+)"}', text)
    if match:
        answer = match.group(1)
    else:
        # 못 찾으면 텍스트 내 마지막 숫자 추출 (최후의 수단)
        nums = re.findall(r'\d+', text)
        if nums: answer = nums[-1]
        
    return {"final_answer": answer}


# 워크플로우 조립
workflow = StateGraph(MCQState)

workflow.add_node("router", router_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("korean_solve", ko_history_solver_node)
workflow.add_node("general_solve", general_solver_node)
workflow.add_node("parse", parser_node)

# 흐름 연결
workflow.set_entry_point("router")

workflow.add_conditional_edges(
    "router",
    route_decision,
    {
        "retrieve": "retrieve",
        "general_solve": "general_solve"
    }
)

# 한국사 경로: Retrieve -> Korean Solve -> Parse
workflow.add_edge("retrieve", "korean_solve")
workflow.add_edge("korean_solve", "parse")

# 일반 경로: General Solve -> Parse
workflow.add_edge("general_solve", "parse")

# 종료
workflow.add_edge("parse", END)

# 컴파일
app = workflow.compile()
print("=====워크플로우 조립 완료=====")

def main():
    # 데이터 로드 (73개 테스트)
    csv_path = "...csv 경로..."
    is_train = os.path.basename(csv_path).startswith("train")
    
    data = pd.read_csv(csv_path)

    results = []
    sub = []  # 최종 제출 형식 csv 용
    print("🔥 최종 RAG 모델 평가 시작...")

    for idx, row in tqdm(data.iterrows(), total=len(data)):
        try:
            # 데이터 전처리
            problem = ast.literal_eval(row['problems']) if isinstance(row['problems'], str) else row['problems']

            inputs = {
                "id": row['id'],
                "paragraph": row['paragraph'],
                "question": problem['question'],
                "choices": problem['choices']
            }

            # LangFuse 콜백 설정
            config = {"callbacks": [langfuse_handler]} if 'langfuse_handler' in globals() else {}

            # 그래프 실행
            output = app.invoke(inputs, config=config)

            # 결과 저장
            result = {
                "id": inputs['id'],
                "question": inputs['question'],
                "predicted_answer": output['final_answer'],
                "retrieved_context": output['retrieved_context'],
                "full_response": output['full_response'],
            }

            # train.csv일 때만 실제 answer 관련 필드 추가
            if is_train:
                ground_truth = problem['answer']
                result.update({
                    "real_answer": str(ground_truth),
                    "is_correct": str(output['final_answer']) == str(ground_truth)
                })

            results.append(result)

            sub.append({
                "id": inputs['id'],
                "answer": output['final_answer']
            })

        except Exception as e:
            print(f"❌ Error at {idx}: {e}")

    # 결과 저장 및 출력
    final_df = pd.DataFrame(results)
    final_df.to_csv("/content/drive/MyDrive/csat/rag_results.csv", index=False, encoding='utf-8-sig')

    sub_df = pd.DataFrame(sub)
    sub_df.to_csv("/content/drive/MyDrive/csat/sub_lang.csv", index=False, encoding='utf-8-sig')

    acc = final_df['is_correct'].mean() * 100
    print(f"\n🏆 최종 정답률: {acc:.2f}%")
    
if __name__ == "__main__":
    main()



