import json
import os
import chromadb
import ollama
import pandas as pd

try:
    import pandasai as pai
    from pandasai_litellm.litellm import LiteLLM
except ImportError:
    pai = None
    LiteLLM = None

DB_DIR = "./chroma_db"
EMBED_MODEL = "nomic-embed-text"
PANDASAI_MODEL = os.getenv("PANDASAI_MODEL", "gemma4:latest")
QUERY_MODEL = os.getenv("QUERY_MODEL", "exaone3.5:7.8b")
ANALYSIS_MODE = os.getenv("ANALYSIS_MODE", "pandas").lower()
OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
FINAL_NUM_CTX = int(os.getenv("FINAL_NUM_CTX", "4096"))
FINAL_NUM_PREDICT = int(os.getenv("FINAL_NUM_PREDICT", "320"))

# 1. 기구축된 Chroma DB 로드
chroma_client = chromadb.PersistentClient(path=DB_DIR)
collection = chroma_client.get_collection(name="salmon_farm_manual")


def _load_sensor_dataframe(file_path):
    if not os.path.exists(file_path):
        return None

    sensor_df = pd.read_csv(file_path, parse_dates=["timestamp"])
    required_columns = {
        "timestamp",
        "temperature",
        "DO",
        "CO2",
        "pH",
        "nitrate",
        "ammonia",
    }
    missing_columns = required_columns - set(sensor_df.columns)
    if missing_columns:
        raise ValueError(f"CSV에 필요한 컬럼이 없습니다: {sorted(missing_columns)}")

    return sensor_df.sort_values("timestamp")


def _trend_label(delta, stable_threshold):
    if abs(delta) <= stable_threshold:
        return "stable"
    return "rising" if delta > 0 else "falling"


def analyze_sensor_csv_with_pandas(file_path):
    """pandas로 CSV를 읽고 계산 기반 수질 트렌드 요약을 생성합니다."""
    sensor_df = _load_sensor_dataframe(file_path)
    if sensor_df is None:
        return None

    if sensor_df.empty:
        raise ValueError("CSV에 분석할 센서 데이터가 없습니다.")

    first = sensor_df.iloc[0]
    latest = sensor_df.iloc[-1]
    elapsed_min = max(
        (latest["timestamp"] - first["timestamp"]).total_seconds() / 60,
        1,
    )

    specs = {
        "temperature": {"unit": "C", "stable": 0.05, "bad": "either"},
        "DO": {"unit": "mg/L", "stable": 0.05, "bad": "falling"},
        "CO2": {"unit": "mg/L", "stable": 0.1, "bad": "rising"},
        "pH": {"unit": "", "stable": 0.03, "bad": "falling"},
        "nitrate": {"unit": "mg/L", "stable": 0.5, "bad": "rising"},
        "ammonia": {"unit": "mg/L", "stable": 0.005, "bad": "rising"},
    }

    lines = [
        f"분석 구간: {first['timestamp']}부터 {latest['timestamp']}까지 약 {elapsed_min:.0f}분, 데이터 {len(sensor_df)}건.",
    ]
    worsening = []

    for column, spec in specs.items():
        start_value = float(first[column])
        latest_value = float(latest[column])
        delta = latest_value - start_value
        rate_per_hour = delta / elapsed_min * 60
        trend = _trend_label(delta, spec["stable"])
        unit = spec["unit"]

        lines.append(
            f"{column}: 최신 {latest_value:g}{unit}, 변화 {delta:+.3g}{unit}, "
            f"시간당 변화율 {rate_per_hour:+.3g}{unit}/h, 추세 {trend}."
        )

        bad_direction = spec["bad"]
        if (
            (bad_direction == "rising" and trend == "rising")
            or (bad_direction == "falling" and trend == "falling")
            or (bad_direction == "either" and trend != "stable")
        ):
            worsening.append((column, abs(rate_per_hour), trend))

    worsening.sort(key=lambda item: item[1], reverse=True)
    if worsening:
        lines.append(
            "악화 우선순위: "
            + ", ".join(f"{name}({trend})" for name, _, trend in worsening[:3])
            + "."
        )
    else:
        lines.append("악화 우선순위: 뚜렷한 악화 추세 없음.")

    return "\n".join(lines)


def analyze_sensor_csv_with_pandasai(file_path):
    """PandasAI로 CSV를 DataFrame 형태로 분석해 수질 트렌드 요약을 생성합니다."""
    if not os.path.exists(file_path):
        return None

    if pai is None or LiteLLM is None:
        raise ImportError(
            "PandasAI가 설치되어 있지 않습니다. Python 3.11 환경에서 "
            "`pip install pandasai pandasai-litellm`를 실행하세요."
        )

    sensor_df = _load_sensor_dataframe(file_path)

    llm = LiteLLM(
        model=f"ollama/{PANDASAI_MODEL}",
        api_base=OLLAMA_API_BASE,
    )
    pai.config.set({"llm": llm, "verbose": False, "max_retries": 2})

    smart_df = pai.DataFrame(sensor_df)
    return str(
        smart_df.chat(
            """
            You are analyzing recent RAS salmon-farm water-quality sensor data.
            Use the actual dataframe columns, not a prewritten text summary.
            Summarize the last 1 hour trend in Korean.

            Include:
            - latest value for temperature, DO, CO2, pH, nitrate, ammonia
            - whether each variable is rising, falling, or stable
            - the variables that are worsening fastest
            - a short risk summary for vector database guideline retrieval

            Return concise plain text only.
            """
        )
    )


def run_trend_analysis_loop(csv_file_path):
    """1시간 동안의 센서 변화량 CSV를 분석하여 가이드라인을 검색하고 제어 명령을 반환합니다."""

    # 1. CSV 데이터 로드 및 트렌드 분석
    try:
        if ANALYSIS_MODE == "pandasai":
            trend_analysis_text = analyze_sensor_csv_with_pandasai(csv_file_path)
        else:
            trend_analysis_text = analyze_sensor_csv_with_pandas(csv_file_path)
    except (ImportError, ValueError) as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)

    if not trend_analysis_text:
        return json.dumps(
            {"error": f"CSV 파일을 찾을 수 없습니다: {csv_file_path}"},
            ensure_ascii=False,
        )

    # 2. 변화 트렌드를 요약하여 DB 검색을 위한 쿼리 텍스트 생성
    query_text = (
        "최근 1시간 동안의 수질 변화 트렌드 분석 및 경고 상태 조회:\n"
        f"{trend_analysis_text}"
    )

    # 3. 로컬 Vector DB에서 가장 관련 깊은 글로벌/국내 매뉴얼 지침 검색
    query_embedding = ollama.embeddings(model=EMBED_MODEL, prompt=query_text)[
        "embedding"
    ]
    results = collection.query(query_embeddings=[query_embedding], n_results=1)

    retrieved_doc = (
        results["documents"][0][0]
        if results["documents"][0]
        else "기본 수질 지침을 따르십시오."
    )

    # 4. 프롬프트 엔지니어링: 단순 수치 비교가 아닌 '변화량(추세)'을 진단하도록 지시
    system_instruction = f"""
    당신은 실내 연어 양식장(RAS)의 스마트 트렌드 분석 에이전트입니다.
    제공된 [수질 트렌드 분석]을 검토하고, [양식장 매뉴얼]의 임계치와 비교하여 시급한 제어 명령을 내리세요.
    
    [양식장 매뉴얼]
    {retrieved_doc}
    
    [진단 가이드라인]
    - 단일 수치뿐만 아니라 수치가 '상승 중'인지 '하강 중'인지 추세를 파악하세요.
    - 예를 들어 암모니아가 임계치에 근접하며 급상승 중이거나, DO가 지속해서 급감 중이라면 선제적 조치 명령을 내려야 합니다.
    
    출력은 반드시 compact JSON 객체 하나로만 하세요. 마크다운 블록(```json)이나 다른 설명은 절대 하지 마세요.
    각 값은 짧게 작성하세요. control_action은 명령어만 쉼표로 구분하세요.
    {{
        "trend_diagnosis": "짧은 분석 요약",
        "control_action": "FEEDER_STOP, OXYGEN_PUMP_UP 등",
        "urgency": "LOW 또는 MEDIUM 또는 HIGH"
    }}
    """

    # 5. Ollama 호출 및 결과 반환
    output = ollama.generate(
        model=QUERY_MODEL,
        prompt=f"[수질 트렌드 분석]\n{trend_analysis_text}",
        system=system_instruction,
        stream=False,
        format="json",
        options={
            "temperature": 0,
            "num_ctx": FINAL_NUM_CTX,
            "num_predict": FINAL_NUM_PREDICT,
        },
    )

    return output["response"]


# --- 🧪 비동기 CSV 수신 및 분석 테스트 ---
if __name__ == "__main__":
    # 테스트용 가상 CSV 파일 경로
    CSV_PATH = "./sensor_history.csv"

    print(f"🔄 비동기 생성된 수질 히스토리 파일({CSV_PATH}) 로드 중...")

    print("\n[AI 에이전트 시계열 트렌드 진단 요청...]")
    ai_response = run_trend_analysis_loop(CSV_PATH)

    print("\n[AI 에이전트가 Omniverse로 반환할 최종 시나리오 진단 결과]")
    print(ai_response)