import os
import re
import numpy as np
import faiss
import anthropic
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from loader import load_all_jobs, load_faqs, load_trend_summaries, load_intent_examples

# ============================
# 초기 설정
# ============================

# .env 파일에서 토큰 불러오기
load_dotenv()
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

# Claude API 클라이언트 초기화
client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

# 한국어 지원 임베딩 모델 로드 (벡터 변환용)
# 성능 가장 좋은 한국어 포함 다국어 지원 임베딩 모델 사용
print("임베딩 모델 로딩 중...")
embedding_model = SentenceTransformer("intfloat/multilingual-e5-large")
print("임베딩 모델 로딩 완료!")


# ============================
# 데이터 로드
# ============================

# Intent 예시 질문 데이터 로드
intent_examples = load_intent_examples()

# 공고 데이터 로드
print("공고 데이터 벡터 인덱스 구축 중...")
jobs = load_all_jobs()

# 트렌드 데이터 로드
print("트렌드 데이터 벡터 인덱스 구축 중...")
trends = load_trend_summaries()

# FAQ 데이터 로드
faqs = load_faqs()


# ============================
# 인덱스 구축 함수
# ============================

def build_intent_index(intent_examples: list):
    """Intent 예시 질문들을 벡터로 변환하고 FAISS 인덱스 구축"""
    all_examples = []   # 전체 예시 질문 텍스트
    example_labels = [] # 각 예시 질문의 카테고리 레이블

    for category in intent_examples:
        intent = category.get("intent", "")
        examples = category.get("examples", [])

        for example in examples:
            all_examples.append(example)
            example_labels.append(intent)  # 예시 질문과 카테고리 매핑

    # 예시 질문들을 벡터로 변환
    embeddings = embedding_model.encode(all_examples)
    embeddings = np.array(embeddings).astype("float32")

    # FAISS 인덱스 생성
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, example_labels  # 인덱스와 레이블 같이 반환


def build_faq_index(faqs: list):
    """FAQ 질문들을 벡터로 변환하고 FAISS 인덱스 구축"""
    faq_texts = [faq.get("question", "") for faq in faqs]

    # FAQ 질문들을 벡터로 변환
    embeddings = embedding_model.encode(faq_texts)
    embeddings = np.array(embeddings).astype("float32")

    # FAISS 인덱스 생성
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index


def build_index(jobs: list):
    """공고 데이터를 벡터로 변환하고 FAISS 인덱스 구축"""
    texts = []

    for job in jobs:
        # 각 공고의 주요 정보를 하나의 텍스트로 합치기
        title = job.get("position", {}).get("title", "")
        company = job.get("company", {}).get("name", "")
        skills = " ".join(job.get("skills", []))
        category = job.get("category", "")
        location = job.get("work_condition", {}).get("location", {}).get("sido", "")
        career_type = job.get("position", {}).get("career", {}).get("type", "")

        # 주요업무, 자격조건도 포함해서 검색 정확도 향상
        main_tasks = " ".join(job.get("detail", {}).get("main_tasks", []))
        requirements = " ".join(job.get("detail", {}).get("requirements", []))

        text = f"{title} {company} {category} {skills} {location} {career_type} {main_tasks} {requirements}"
        texts.append(text)

    # 텍스트를 벡터로 변환
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    # FAISS 인덱스 생성 및 벡터 추가
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 거리 기반 유사도 검색
    index.add(embeddings)

    return index


def build_trend_index(trends: list):
    """트렌드 데이터를 벡터로 변환하고 FAISS 인덱스 구축"""
    trend_texts = []

    for trend in trends:
        # 트렌드 정보를 하나의 텍스트로 합치기 (검색 정확도 향상)
        text = f"{trend.get('category', '')} 트렌드 " + \
               f"인기 프레임워크: {' '.join(trend.get('hot_frameworks', []))} " + \
               f"인기 언어: {' '.join(trend.get('hot_languages', []))} " + \
               f"인기 도구: {' '.join(trend.get('hot_tools', []))} " + \
               f"연봉: {trend.get('salary_trend', '')} " + \
               f"복지: {trend.get('welfare_trend', '')}"
        trend_texts.append(text)

    # 텍스트를 벡터로 변환
    embeddings = embedding_model.encode(trend_texts)
    embeddings = np.array(embeddings).astype("float32")

    # FAISS 인덱스 생성 및 벡터 추가
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index


# ============================
# 인덱스 구축 실행
# ============================

# Intent 인덱스 구축
intent_index, intent_labels = build_intent_index(intent_examples)
print(f"Intent 인덱스 구축 완료! 총 {len(intent_labels)}개 예시 질문 인덱싱됨")

# 공고 인덱스 구축
faiss_index = build_index(jobs)
print(f"공고 인덱스 구축 완료! 총 {len(jobs)}개 공고 인덱싱됨")

# 트렌드 인덱스 구축
trend_index = build_trend_index(trends)
print(f"트렌드 인덱스 구축 완료! 총 {len(trends)}개 카테고리 인덱싱됨")

# FAQ 인덱스 구축
faq_index = build_faq_index(faqs)
print(f"FAQ 인덱스 구축 완료! 총 {len(faqs)}개 FAQ 인덱싱됨")


# ============================
# Intent 분류
# ============================

def classify_intent_by_vector(query: str, threshold: float = 0.2):
    """1차: 벡터 유사도로 Intent 분류"""

    # 쿼리를 벡터로 변환
    query_vector = embedding_model.encode([query])
    query_vector = np.array(query_vector).astype("float32")

    # 가장 유사한 예시 질문 검색
    distances, indices = intent_index.search(query_vector, 1)

    distance = distances[0][0]
    idx = indices[0][0]

    # 유사도가 임계값 이하면 해당 카테고리 반환
    if distance < threshold:
        return intent_labels[idx], distance

    # 임계값 초과면 None 반환 (2차 분류 필요)
    return None, distance


def classify_intent_by_llm(query: str) -> str:
    """2차: Claude API로 Intent 분류 (1차에서 확신도 낮을 때)"""

    prompt = f"""당신은 취업 플랫폼 챗봇의 질문 분류기입니다.
아래 질문을 다음 카테고리 중 하나로 분류하세요.

카테고리:
- CUSTOM: 유저 개인 맞춤 공고 추천 요청 (나한테, 내 경력, 내 스킬 등)
- GENERAL: 일반적인 공고 검색 요청
- TREND: IT 취업 시장 트렌드 관련 질문
- COMPANY: 특정 기업 정보 질문
- CAREER_TIP: 취업 준비 관련 질문 (자소서, 면접, 포트폴리오 등)
- PUBLIC_DATA: 공공기관/공기업 채용 관련 질문
- INVALID: 취업/채용과 무관한 질문

반드시 카테고리 이름만 답하세요. 다른 텍스트는 절대 포함하지 마세요.

질문: {query}"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=20,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text.strip().upper()


def classify_intent(query: str) -> str:
    """하이브리드 Intent 분류: 1차(벡터) + 2차(LLM) 2중 검사"""

    # 1차: 벡터 유사도로 빠르게 분류
    intent_vector, distance = classify_intent_by_vector(query)
    print(f"1차 분류 결과: {intent_vector} (거리: {distance:.2f})")

    # 2차: LLM으로 항상 분류 (정확도 향상)
    intent_llm = classify_intent_by_llm(query)
    print(f"2차 분류 결과: {intent_llm}")

    # 1차랑 2차 결과 비교
    if intent_vector == intent_llm:
        # 둘 다 일치하면 확정
        print(f"1차 2차 일치 → {intent_llm} 확정")
    else:
        # 다르면 2차 LLM 우선
        print(f"1차 2차 불일치 → 2차 LLM 결과 {intent_llm} 우선 적용")

    return intent_llm


# ============================
# 검색 함수
# ============================

def search_jobs(query: str, user_info: dict = None, top_k: int = 3):
    """벡터 유사도 기반으로 관련 공고 검색"""

    # 유저 질문에서 원하는 공고 개수 추출 (예: "3개만", "1개", "5개 추천")
    number_match = re.search(r'(\d+)\s*개', query)
    if number_match:
        top_k = min(int(number_match.group(1)), 10)  # 최대 10개로 제한

    # 검색 쿼리 구성 - 질문 + 유저 정보 합치기
    search_text = query

    # 유저 정보가 있으면 검색 쿼리에 추가해서 맞춤 검색
    if user_info:
        search_text += " " + " ".join([
            user_info.get("job_type", ""),
            user_info.get("region", ""),
            user_info.get("career_type", ""),
            user_info.get("occupation", "")
        ])

    # 쿼리를 벡터로 변환
    query_vector = embedding_model.encode([search_text])
    query_vector = np.array(query_vector).astype("float32")

    # FAISS로 가장 유사한 공고 top_k개 검색
    distances, indices = faiss_index.search(query_vector, top_k)

    # 검색된 인덱스로 실제 공고 데이터 반환
    results = []
    for idx in indices[0]:
        if idx < len(jobs):
            results.append(jobs[idx])

    return results


def search_trends(query: str, top_k: int = 3):
    """벡터 유사도 기반으로 관련 트렌드 데이터 검색"""

    # 쿼리를 벡터로 변환
    query_vector = embedding_model.encode([query])
    query_vector = np.array(query_vector).astype("float32")

    # FAISS로 가장 유사한 트렌드 top_k개 검색
    distances, indices = trend_index.search(query_vector, top_k)

    # 검색된 인덱스로 실제 트렌드 데이터 반환
    results = []
    for idx in indices[0]:
        if idx < len(trends):
            results.append(trends[idx])

    return results


def find_similar_faq(query: str, threshold: float = 30.0):
    """유저 질문과 가장 유사한 FAQ 케이스 찾기"""

    # 쿼리를 벡터로 변환
    query_vector = embedding_model.encode([query])
    query_vector = np.array(query_vector).astype("float32")

    # 가장 유사한 FAQ 검색
    distances, indices = faq_index.search(query_vector, 1)

    distance = distances[0][0]
    idx = indices[0][0]

    # 유사도가 임계값 이하면 해당 FAQ 반환
    if distance < threshold:
        return faqs[idx], distance

    return None, distance


# ============================
# 답변 생성
# ============================

def generate_answer(query: str, related_jobs: list, user_info: dict = None, history: list = [], related_trends: list = []):
    """검색된 공고/트렌드와 유저 정보, 대화 히스토리를 바탕으로 LLM이 답변 생성"""

    # 검색된 공고 정보를 텍스트로 변환
    jobs_text = ""
    for i, job in enumerate(related_jobs, 1):
        title = job.get("position", {}).get("title", "")
        company = job.get("company", {}).get("name", "")
        location = job.get("work_condition", {}).get("location", {}).get("sido", "")
        category = job.get("category", "")
        skills = ", ".join(job.get("skills", []))
        salary = job.get("work_condition", {}).get("salary", {})
        salary_text = f"{salary.get('min', 0)//10000}만원 ~ {salary.get('max', 0)//10000}만원" if salary else "협의"
        career_type = job.get("position", {}).get("career", {}).get("type", "")
        benefits = ", ".join(job.get("detail", {}).get("benefits", []))
        work_type = job.get("position", {}).get("work_type", "")

        jobs_text += f"{i}. [{category}] {company} - {title}\n"
        jobs_text += f"   위치: {location} | 연봉: {salary_text} | 경력: {career_type}\n"
        jobs_text += f"   근무형태: {work_type}\n"
        jobs_text += f"   스킬: {skills}\n"
        jobs_text += f"   복리후생: {benefits}\n\n"

    # 트렌드 데이터를 텍스트로 변환
    trends_text = ""
    for trend in related_trends:
        trends_text += f"[{trend.get('category', '')}] 트렌드\n"
        trends_text += f"   인기 프레임워크: {', '.join(trend.get('hot_frameworks', []))}\n"
        trends_text += f"   인기 언어: {', '.join(trend.get('hot_languages', []))}\n"
        trends_text += f"   인기 도구: {', '.join(trend.get('hot_tools', []))}\n"
        trends_text += f"   연봉 트렌드: {trend.get('salary_trend', '')}\n"
        trends_text += f"   복지 트렌드: {trend.get('welfare_trend', '')}\n\n"

    # 유저 맞춤 정보가 있으면 프롬프트에 반영
    user_context = ""
    if user_info:
        user_context = f"""
사용자 정보:
- 희망 직무: {user_info.get('job_type', '미설정')}
- 희망 지역: {user_info.get('region', '미설정')}
- 직종: {user_info.get('occupation', '미설정')}
- 경력: {user_info.get('career_type', '미설정')}
- 학력: {user_info.get('education', '미설정')}
"""

    # 시스템 프롬프트
    system_prompt = """당신은 일로온(일로ON) 취업 플랫폼의 AI 챗봇 어시스턴트입니다.
반드시 한국어로만 답변하세요. 다른 언어는 절대 사용하지 마세요.
사용자의 취업 관련 질문에 친절하고 전문적으로 답변해주세요.
이전 대화 내용을 참고하여 문맥에 맞는 답변을 해주세요.
제공된 공고 목록을 바탕으로 답변하고, 목록에 없는 외부 사이트(Indeed, LinkedIn 등)는 언급하지 마세요.
공고의 스킬 목록에 있는 기술명은 한국어와 영어가 같은 의미입니다. 사용자가 한국어로 기술명을 말해도 영어로 된 스킬 목록에서 찾아서 답변해주세요.
공고 추천 시에는 핵심 정보를 간결하게 전달하고, 사용자가 추가로 궁금한 점을 물어볼 수 있도록 유도하세요.
사용자가 물어본 것만 답변하고 묻지 않은 내용은 절대 추가하지 마세요.
트렌드 데이터가 제공된 경우 해당 데이터만 바탕으로 답변하고 데이터에 없는 내용은 절대 추가하지 마세요."""

    # 메시지 구성 - 시스템 프롬프트 + 이전 대화 히스토리 + 현재 질문
    # Claude API는 system 파라미터 별도로 분리
    user_messages = []

    # 이전 대화 히스토리 추가 (최근 5개만 유지해서 토큰 절약)
    if history:
        user_messages.extend(history[-5:])

    # 현재 유저 메시지 구성
    current_message = f"""{user_context}
사용자 질문: {query}

아래는 일로온 플랫폼의 관련 공고 목록입니다:
{jobs_text if jobs_text else "관련 공고를 찾지 못했습니다."}

{f"아래는 관련 트렌드 정보입니다:{chr(10)}{trends_text}" if trends_text else ""}
위 공고 목록과 트렌드 정보를 바탕으로 사용자에게 도움이 되는 답변을 한국어로 해주세요."""

    user_messages.append({"role": "user", "content": current_message})

    # Claude API 호출해서 답변 생성
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=system_prompt,
        messages=user_messages
    )

    return response.content[0].text