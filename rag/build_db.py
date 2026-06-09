import os
import glob
import chromadb
import ollama
from pypdf import PdfReader


# 1. 설정 변수
PDF_DIR = "./manuals"
DB_DIR = "./chroma_db"
EMBED_MODEL = "nomic-embed-text"

# 2. 로컬 디렉토리에 영구 저장되는 Chroma 클라이언트 설정
chroma_client = chromadb.PersistentClient(path=DB_DIR)
collection = chroma_client.get_or_create_collection(name="salmon_farm_manual")

def summarize_text_with_llm(raw_text):
    prompt = f"""
    다음은 실내 연어 양식장 관련 문서의 일부입니다. 
    이 내용 중에서 '암모니아, 용존산소(DO), 질산염, pH' 등의 수질 임계치와 
    그에 따른 '설비 제어 지침(액션)'이 포함된 핵심 문장들만 추출해서 짧게 요약해줘.
    
    [문서 내용]
    {raw_text[:4000]}  # 텍스트가 너무 길어 제한을 둡니다.
    """
    response = ollama.generate(
        model=EMBED_MODEL,
        prompt=prompt,
        system="당신은 실내 연어 양식장 매뉴얼을 요약하는 사람입니다.",
        stream=False,
    )
    return response["response"]


def build_vector_database():
    pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))

    if not pdf_files:
        print(
            f"{PDF_DIR} 폴더에 PDF 파일이 없습니다. 파일을 넣고 다시 실행해주세요."
        )
        return

    print(f"총 {len(pdf_files)}개의 PDF 문서를 발견했습니다. 분석을 시작합니다.")

    doc_id_counter = 0
    for file_path in pdf_files:
        print(f"파일 읽는 중: {os.path.basename(file_path)}")

        # PDF 텍스트 추출
        try:
            reader = PdfReader(file_path)
            full_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"

            if not full_text.strip():
                print(f"{file_path}에서 텍스트를 추출하지 못했습니다. 건너뜁니다.")
                continue

            # LLM을 활용한 데이터 전처리 (주요 지침 요약)
            print("LLM이 핵심 수질 지침을 요약하는 중...")
            summary = summarize_text_with_llm(full_text)
            print(f"요약 완료:\n{summary}\n" + "-" * 40)

            # Ollama 임베딩 생성 후 ChromaDB에 영구 저장
            response = ollama.embeddings(model=EMBED_MODEL, prompt=summary)
            embedding = response["embedding"]

            collection.add(
                ids=[f"pdf_doc_{doc_id_counter}"],
                embeddings=[embedding],
                documents=[summary],
                metadatas=[{"source": os.path.basename(file_path)}],
            )
            doc_id_counter += 1

        except Exception as e:
            print(f"{file_path} 처리 중 에러 발생: {str(e)}")

    print(f"모든 문서가 요약되어 '{DB_DIR}'에 영구 저장되었습니다.")

    documents = [
        "실내 순환여과식 양식(RAS)에서 암모니아(NH3) 수치가 0.05 mg/L 이상으로 상승하면 사료 투여량을 즉시 50% 감축하거나 중단해야 한다.",
        "용존산소(DO) 수치가 6.0 mg/L 이하로 떨어지면 연어의 호흡 부전이 발생하므로 비상 액체 산소 주입 밸브를 열고 순환 펌프 속도를 올려야 한다.",
        "질산염(NO3-) 수치가 50 mg/L을 초과하여 누적되면 대량 폐사 위험이 있으므로 즉시 10% 내외의 스마트 자동 환수(물 교환) 시스템을 가동한다.",
        "pH 수치가 7.8 이상으로 높아지면 암모니아의 독성이 급격히 강해지므로 pH 조절제를 투입하여 수질을 6.8 ~ 7.2 사이로 유지해야 한다.",
        "바이오필터(여과조) 효율 저하로 암모니아가 질산염으로 분해되지 않을 때는 환수율을 20% 상향하고 아질산염 수치를 함께 점검해야 한다.",
    ]

    for i, doc in enumerate(documents):
        # Ollama를 통해 텍스트를 벡터(숫자 배열)로 변환
        response = ollama.embeddings(model=EMBED_MODEL, prompt=doc)
        embedding = response["embedding"]

        # Chroma DB에 저장
        collection.add(
            ids=[f"doc_{i}"], embeddings=[embedding], documents=[doc]
        )
    print("DB 구축 완료!")

if __name__ == "__main__":
    if not os.path.exists(PDF_DIR):
        os.makedirs(PDF_DIR)
        print(f"{PDF_DIR} 폴더를 생성했습니다. 여기에 PDF를 넣어주세요.")
    else:
        build_vector_database()