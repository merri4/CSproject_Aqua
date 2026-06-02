import json
import chromadb
import ollama

DB_DIR = "./chroma_db"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.1:8b"

# 이미 빌드된 폴더에서 로드 (속도가 매우 빠름)
chroma_client = chromadb.PersistentClient(path=DB_DIR)
collection = chroma_client.get_collection(name="salmon_farm_manual")


def run_rag_control_loop(sensor_json_str):
    """Omniverse 센서 데이터를 받아 저장된 DB에서 지침을 찾아 제어 명령을 반환합니다."""
    sensor_data = json.loads(sensor_json_str)

    # 1. 센서 수치 기반 검색 쿼리 생성
    query_text = (
        f"현재 암모니아: {sensor_data.get('ammonia')} mg/L, "
        f"용존산소(DO): {sensor_data.get('DO')} mg/L, "
        f"질산염: {sensor_data.get('nitrate')} mg/L. 조치 사항 검색."
    )

    # 2. 질문 임베딩 및 기구축된 로컬 DB 검색
    query_embedding = ollama.embeddings(model=EMBED_MODEL, prompt=query_text)[
        "embedding"
    ]
    results = collection.query(query_embeddings=[query_embedding], n_results=1)

    if not results["documents"][0]:
        return {"error": "참고할 매뉴얼을 찾지 못했습니다. DB를 먼저 구축해주세요."}

    retrieved_doc = results["documents"][0][0]

    # 3. LLM에게 프롬프트 전달 및 JSON 응답 강제
    system_instruction = f"""
    당신은 실내 연어 양식장(RAS)의 자동 제어 에이전트입니다.
    오직 아래 제공된 [양식장 매뉴얼]만을 근거로 [현재 센서 데이터]를 분석하여 최적의 제어 가이드를 작성하세요.
    
    [양식장 매뉴얼]
    {retrieved_doc}
    
    출력은 반드시 아래의 JSON 포맷으로만 하세요. 그 외의 텍스트나 설명은 절대 하지 마세요.
    {{
        "control_action": "가동할 설비와 액션 명확히 기재 (ex: FEEDER_STOP, OXYGEN_PUMP_UP, WATER_EXCHANGE, NORMAL)",
        "reason": "매뉴얼의 수질 기준치를 근거로 한 판단 이유"
    }}
    """

    output = ollama.generate(
        model=LLM_MODEL,
        prompt=f"[현재 센서 데이터]\n{sensor_json_str}",
        system=system_instruction,
        stream=False,
    )

    return output["response"]

if __name__ == "__main__":
    
    # 1. 환수가 필요한 상황 가정 (질산염 수치 폭발)
    test_sensor_data = '{"ammonia": 0.02, "DO": 7.5, "nitrate": 55.0}'

    print("[Omniverse 로부터 센서 데이터 수신]")
    print(f"데이터: {test_sensor_data}")

    print("\n[로컬 Vector DB 기반 AI 의사결정 요청 중...]")
    ai_response = run_rag_control_loop(test_sensor_data)

    print("\n[AI 에이전트가 Omniverse로 반환할 최종 제어 명령 JSON]")
    print(ai_response)