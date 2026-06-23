import os
import re
import json
import numpy as np
import faiss
import anthropic
import requests
from dotenv import load_dotenv
import voyageai
from loader import (
    load_all_jobs,
    load_quick_replies,
    load_trend_summaries,
    load_intent_examples,
    get_category_stats,
    load_busan_youth_policies,
    load_busan_job_services
)

# ============================
# 초기 설정
# ============================

# .env 파일에서 토큰 불러오기
load_dotenv()
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
YOUTHCENTER_API_KEY = os.getenv("YOUTHCENTER_API_KEY")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")

# Claude API 클라이언트 초기화
client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

# Voyage AI 임베딩 클라이언트 초기화 (API 기반, 서버 RAM 점유 없음)
print("Voyage AI 클라이언트 초기화 중...")
vo = voyageai.Client(api_key=VOYAGE_API_KEY)
print("Voyage AI 클라이언트 초기화 완료!")


# ============================
# 임베딩 헬퍼 함수
# ============================

def embed_documents(texts: list) -> np.ndarray:
    """문서 텍스트를 벡터로 변환 (인덱스 구축용)"""
    result = vo.embed(texts, model="voyage-4", input_type="document")
    return np.array(result.embeddings).astype("float32")


def embed_query(query: str) -> np.ndarray:
    """검색 쿼리를 벡터로 변환"""
    result = vo.embed([query], model="voyage-4", input_type="query")
    return np.array(result.embeddings).astype("float32")


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

# 퀵리플라이 데이터 로드
quick_replies = load_quick_replies()

# 카테고리별 통계 데이터 로드
category_stats = get_category_stats()

# 부산 공공데이터
busan_youth_policies = load_busan_youth_policies()
busan_job_services = load_busan_job_services()

print(f"부산 청년정책 데이터: {len(busan_youth_policies)}건")
print(f"부산 일자리지원서비스 데이터: {len(busan_job_services)}건")

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
    embeddings = embed_documents(all_examples)

    # FAISS 인덱스 생성
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, example_labels  # 인덱스와 레이블 같이 반환


def build_quick_replies_index(quick_replies: list):
    """퀵리플라이 질문들을 벡터로 변환하고 FAISS 인덱스 구축"""
    quick_reply_texts = [quick_reply.get("question", "") for quick_reply in quick_replies]

    # 퀵리플라이 질문들을 벡터로 변환
    embeddings = embed_documents(quick_reply_texts)

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
    embeddings = embed_documents(texts)

    # FAISS 인덱스 생성 및 벡터 추가
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 거리 기반 유사도 검색
    index.add(embeddings)

    return index


def build_trend_index(trends: list):
    """트렌드 데이터를 벡터로 변환하고 FAISS 인덱스 구축"""
    trend_texts = []

    for trend in trends:
        text = f"{trend.get('category', '')} 트렌드 " + \
               f"인기 프레임워크: {' '.join(trend.get('hot_frameworks', []))} " + \
               f"인기 언어: {' '.join(trend.get('hot_languages', []))} " + \
               f"인기 도구: {' '.join(trend.get('hot_tools', []))} " + \
               f"연봉: {trend.get('salary_trend', '')} " + \
               f"복지: {trend.get('welfare_trend', '')}"
        trend_texts.append(text)

    # 트렌드 데이터가 없으면 빈 인덱스 생성
    if not trend_texts:
        dimension = 1024
        index = faiss.IndexFlatL2(dimension)
        return index

    embeddings = embed_documents(trend_texts)

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

# 퀵리플라이 인덱스 구축
quick_replies_index = build_quick_replies_index(quick_replies)
print(f"퀵리플라이 인덱스 구축 완료! 총 {len(quick_replies)}개 퀵리플라이 인덱싱됨")


# ============================
# Intent 분류
# ============================

def classify_intent_by_vector(query: str, threshold: float = 0.2):
    """1차: 벡터 유사도로 Intent 분류"""

    # 쿼리를 벡터로 변환
    query_vector = embed_query(query)

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
- PUBLIC_DATA: 정부 지원 제도, 정책, 지원금, 청년내일채움공제, 국민취업지원제도 등 정부/공공기관 지원 관련 질문. "정책", "제도", "지원금", "지원사업" 키워드가 있으면 이걸로 분류
- CAREER_TIP: 이력서, 자소서, 면접, 포트폴리오 등 취업 준비 방법 관련 질문
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
# 개체명 추출 (NLU 3단계)
# ============================

def normalize_job_type(job_type: str) -> str:
    """user_info.job_type(자유형식)을 표준 카테고리로 정규화
    예: "백엔드 개발" → "백엔드/서버", "데이터 엔지니어" → "데이터"
    토큰 분리 시 "개발", "엔지니어" 같은 범용 단어가 너무 넓게 매칭되는 문제 방지
    """
    if not job_type:
        return None

    # 표준 카테고리 매핑 테이블 (키워드가 job_type에 포함되면 해당 카테고리로 정규화)
    매핑 = {
        "백엔드": "백엔드/서버",
        "서버": "백엔드/서버",
        "프론트엔드": "프론트엔드",
        "프론트": "프론트엔드",
        "AI": "AI/ML",
        "ML": "AI/ML",
        "머신러닝": "AI/ML",
        "딥러닝": "AI/ML",
        "데이터": "데이터 엔지니어링",
        "DevOps": "DevOps/인프라",
        "인프라": "DevOps/인프라",
        "클라우드": "DevOps/인프라",
        "iOS": "모바일",
        "Android": "모바일",
        "모바일": "모바일",
        "Flutter": "모바일",
        "Swift": "모바일",
        "Kotlin": "모바일",
        "게임": "게임",
        "Unity": "게임",
        "Unreal": "게임",
        "보안": "보안",
        "QA": "QA/테스트",
        "테스트": "QA/테스트",
        "PM": "PM/기획",
        "기획": "PM/기획",
        "프로덕트": "PM/기획",
    }

    for keyword, category in 매핑.items():
        if keyword in job_type:
            return category

    return job_type  # 매핑 없으면 원본 반환


def extract_entities(query: str, intent: str) -> dict:
    """인텐트에 맞게 유저 질문에서 개체명 추출"""

    prompt = f"""아래 질문에서 필요한 정보를 추출하세요.
질문 유형: {intent}

추출 규칙:
CUSTOM/GENERAL/QUICK_REPLIES: 회사명, 공고명, 직무, 지역, 경력, 기술스택, 고용형태(재택/정규직/계약직 등), 연봉조건(높은순/낮은순), 마감조건(임박) 추출
- TREND: 직무분야, 기술스택 추출
- COMPANY: 회사명 추출
- CAREER_TIP: 준비유형(자소서/면접/포트폴리오 등) 추출
- PUBLIC_DATA: 정책유형(취업/창업/null), 세부키워드(구체적인 지원 종류, 예: "면접비", "교육비", "인턴", "창업자금". 포괄적인 단어는 null), 지역(질문에서 언급된 지역명. 없으면 null)
없으면 null로 표시. 반드시 JSON 형식으로만 답하세요.

직무 추출 시 반드시 아래 표준 카테고리 중 하나로만 답하세요:
백엔드/서버, 프론트엔드, AI/ML, 데이터 엔지니어링, DevOps/인프라, 모바일 개발, 보안, QA/테스트, PM/기획
(예: "iOS 개발" → "모바일", "머신러닝" → "AI/ML", "백엔드 개발" → "백엔드/서버")

질문: {query}

예시 출력:
{{"회사명": null, "공고명": null, "직무": "백엔드/서버", "지역": null, "경력": "신입", "기술스택": "Python", "고용형태": "재택", "연봉조건": null, "마감조건": null}}

중요:
- 반드시 JSON만 출력하세요.
- JSON 앞뒤에 설명을 쓰지 마세요.
- 문자열 값은 한 줄로만 작성하세요.
- 줄바꿈(\\n)을 포함하지 마세요.
- 회사명과 공고명은 짧게 추출하세요.
- JSON 형식이 아니면 안 됩니다.
"""
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}]
    )
    print("=== 개체명 원본 응답 ===")
    print(response.content[0].text) 
    
    try:
        text = response.content[0].text.strip()

        # 코드블록 제거
        text = text.replace("```json", "").replace("```", "").strip()

        # 혹시 앞뒤에 설명이 섞여도 JSON 객체만 추출
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            text = json_match.group(0)

        result = json.loads(text)
        return result

    except Exception as e:
        print(f"개체명 추출 파싱 실패: {response.content[0].text.strip()}")
        print(f"파싱 에러: {e}")
        return {}
# ============================
# History 기반 질문 재작성
# ============================

def rewrite_query_with_history(query: str, history: list) -> str:
    """이전 대화 맥락을 반영해 현재 질문을 검색 가능한 독립 질문으로 재작성"""

    if not history:
        return query

    recent_history = history[-6:]

    history_text = ""
    for msg in recent_history:
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")
        else:
            role = getattr(msg, "role", "")
            content = getattr(msg, "content", "")

        history_text += f"{role}: {content}\n"

    prompt = f"""당신은 취업 플랫폼 챗봇의 검색 질의 재작성기입니다.

사용자의 현재 질문이 이전 대화의 특정 공고, 회사, 순번, 정책, 조건을 가리키는 경우,
검색에 사용할 수 있도록 독립적인 질문으로 바꾸세요.

규칙:
- 현재 질문만으로 의미가 충분하면 그대로 반환하세요.
- "첫 번째/두 번째/세 번째/1번/2번/3번/그 회사/거기/해당 공고/지원서류/연봉/채용절차/마감일" 같은 표현은 이전 대화를 참고해 구체적인 회사명, 공고명, 직무명으로 바꾸세요.
- 존재하지 않는 정보는 새로 만들지 마세요.
- 반드시 재작성된 질문 한 문장만 출력하세요.
- 설명, 따옴표, JSON, 마크다운은 출력하지 마세요.
- 회사명이 포함되어 있어도 "공고", "채용", "연봉", "지원서류", "채용절차", "마감일", "직무", "상세"를 묻는 질문이면 기업 정보 질문이 아니라 채용공고 검색 질문으로 재작성하세요.

이전 대화:
{history_text}

현재 질문:
{query}

재작성된 질문:"""

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}]
        )

        rewritten = response.content[0].text.strip()

        if rewritten:
            print(f"[HISTORY REWRITE] {query} -> {rewritten}")
            return rewritten

        return query

    except Exception as e:
        print(f"[HISTORY REWRITE 실패] {e}")
        return query

# ============================
# 검색 함수
# ============================

def search_jobs(query: str, user_info: dict = None, top_k: int = 3, entities: dict = {}):
    """벡터 유사도 + 필터 기반으로 관련 공고 검색"""

    # 유저 질문에서 원하는 공고 개수 추출
    number_match = re.search(r'(\d+)\s*개', query)
    if number_match:
        top_k = min(int(number_match.group(1)), 10)

    # ============================
    # 1단계: 필터링 (조건 기반)
    # ============================
    filtered_jobs = jobs  # 전체 공고에서 시작

    if entities:
                # 회사명 / 공고명 필터
        if entities.get("회사명") or entities.get("공고명"):
            company_keyword = entities.get("회사명") or ""
            title_keyword = entities.get("공고명") or ""

            matched_jobs = [
                j for j in filtered_jobs
                if (
                    company_keyword
                    and company_keyword in j.get("company", {}).get("name", "")
                ) or (
                    title_keyword
                    and title_keyword in j.get("position", {}).get("title", "")
                )
            ]

            if matched_jobs:
                print(f"회사명/공고명 필터 결과: {len(matched_jobs)}개")
                return matched_jobs[:top_k]

            print("회사명/공고명 필터 결과 없음 → 기존 필터 유지")


        # 직무 필터 (토큰 기반 부분 매칭으로 일반화)
        # LLM이 표준 카테고리로 뽑아주지만, user_info.job_type 같은 자유형식도 처리
        # 예: "백엔드 개발" → ["백엔드", "개발"] → "백엔드" in "백엔드/서버" → True
        if entities.get("직무"):
            키워드 = entities["직무"]
            키워드_토큰 = 키워드.replace("/", " ").split()
            filtered_jobs = [j for j in filtered_jobs if
                            키워드 in j.get("category", "") or
                            j.get("category", "") in 키워드 or
                            any(t in j.get("category", "") for t in 키워드_토큰) or
                            any(t in j.get("position", {}).get("title", "") for t in 키워드_토큰) or
                            j.get("position", {}).get("job_category", {}).get("mid", "") in 키워드]

        # 지역 필터
        if entities.get("지역"):
            filtered_jobs = [j for j in filtered_jobs if
                            entities["지역"] in j.get("work_condition", {}).get("location", {}).get("sido", "") or
                            entities["지역"] in j.get("work_condition", {}).get("location", {}).get("sigungu", "") or
                            j.get("work_condition", {}).get("location", {}).get("sido", "") in entities["지역"] or
                            j.get("work_condition", {}).get("location", {}).get("sigungu", "") in entities["지역"]]

        # 경력 필터
        if entities.get("경력"):
            filtered_jobs = [j for j in filtered_jobs if entities["경력"] in j.get("position", {}).get("career", {}).get("type", "")]

        # 고용형태 필터 (재택근무 등)
        if entities.get("고용형태"):
            filtered_jobs = [j for j in filtered_jobs if entities["고용형태"] in j.get("position", {}).get("work_type", "")]

        # 연봉 정렬
        if entities.get("연봉조건") == "높은순":
            filtered_jobs = sorted(filtered_jobs, key=lambda x: x.get("work_condition", {}).get("salary", {}).get("max", 0), reverse=True)

        # 마감 임박 정렬
        if entities.get("마감조건") == "임박":
            filtered_jobs = sorted(
                filtered_jobs,
                key=lambda x: x.get("dates", {}).get("deadline") or "9999-12-31"
            )

    # 필터 결과가 있으면 top_k개 반환
    if filtered_jobs and entities and any([
        entities.get("직무"), entities.get("지역"), entities.get("경력"),
        entities.get("고용형태"), entities.get("연봉조건"), entities.get("마감조건")
    ]):
        print(f"필터 검색 결과: 총 {len(filtered_jobs)}개 → top {top_k}개 반환")
        for job in filtered_jobs[:top_k]:
            print(f"  - {job.get('company', {}).get('name', '')} | 연봉: {job.get('work_condition', {}).get('salary', {}).get('max', 0)//10000}만원")
        return filtered_jobs[:top_k]

    # ============================
    # 2단계: 벡터 검색 (필터 조건 없을 때)
    # ============================
    search_text = query

    if entities:
        search_text += " " + " ".join([
            entities.get("직무") or "",
            entities.get("기술스택") or "",
            entities.get("지역") or "",
            entities.get("경력") or "",
            entities.get("고용형태") or "",
        ])

    if user_info:
        search_text += " " + " ".join([
            user_info.get("job_type") or "",
            user_info.get("region") or "",
            user_info.get("career_type") or "",
            user_info.get("occupation") or "",
            user_info.get("major") or "",
            user_info.get("career_years") or "",
        ])

    query_vector = embed_query(search_text)

    distances, indices = faiss_index.search(query_vector, top_k)

    results = []
    for idx in indices[0]:
        if idx < len(jobs):
            results.append(jobs[idx])

    # 벡터 검색 결과를 인기도 기반으로 정렬 (조회수 + 지원수 + 스크랩수)
    results.sort(key=lambda x: (
        x.get("stats", {}).get("view_count", 0) +
        x.get("stats", {}).get("apply_count", 0) +
        x.get("stats", {}).get("bookmark_count", 0)
    ), reverse=True)        

    return results

def search_trends(query: str, top_k: int = 3, entities: dict = {}):
    """벡터 유사도 기반으로 관련 트렌드 데이터 검색"""

    # 검색 쿼리 구성 - 질문 + 개체명
    search_text = query

    # 개체명 있으면 검색 쿼리에 추가
    if entities:
        search_text += " " + " ".join([
            entities.get("직무분야") or "",
            entities.get("기술스택") or "",
        ])

    # 쿼리를 벡터로 변환
    query_vector = embed_query(search_text)

    # FAISS로 가장 유사한 트렌드 top_k개 검색
    distances, indices = trend_index.search(query_vector, top_k)

    # 검색된 인덱스로 실제 트렌드 데이터 반환
    results = []
    for idx in indices[0]:
        if idx < len(trends):
            results.append(trends[idx])

    return results


def find_similar_quick_reply(query: str, threshold: float = 30.0):
    """유저 질문과 가장 유사한 퀵리플라이 케이스 찾기"""

    # 쿼리를 벡터로 변환
    query_vector = embed_query(query)

    # 가장 유사한 퀵리플라이 검색
    distances, indices = quick_replies_index.search(query_vector, 1)

    distance = distances[0][0]
    idx = indices[0][0]

    # 유사도가 임계값 이하면 해당 퀵리플라이 반환
    if distance < threshold:
        return quick_replies[idx], distance

    return None, distance


# ============================
# 공공데이터 검색
# ============================

def contains_any(text: str, keywords: list) -> bool:
    """텍스트에 키워드 중 하나라도 포함되는지 확인"""
    if not text:
        return False
    return any(keyword for keyword in keywords if keyword and keyword in text)


def search_busan_policies(query: str, entities: dict = {}, user_info: dict = None, top_k: int = 3):
    """부산광역시 청년지원정책 CSV에서 관련 정책 검색"""
    results = []

    policy_type = entities.get("정책유형") or ""
    detail_keyword = entities.get("세부키워드") or ""

    keywords = []
    if detail_keyword:
        keywords.extend([k.strip() for k in detail_keyword.split(",") if k.strip()])

    # 질문에서 자주 쓰는 키워드도 보조 검색어로 사용
    common_keywords = ["취업", "창업", "일자리", "면접", "교육", "상담", "청년", "지원", "인턴", "자격증"]
    keywords.extend([kw for kw in common_keywords if kw in query])

    if policy_type:
        keywords.append(policy_type)

    for item in busan_youth_policies:
        score = 0
        reasons = []

        name = item.get("세부사업명", "")
        content = item.get("주요내용", "")
        field = item.get("분야", "")
        student = item.get("학생 여부", "")
        income = item.get("소득 기준", "")
        vulnerable = item.get("취약계층사업 여부", "")
        vulnerable_type = item.get("취약계층 유형", "")
        employment_status = item.get("취창업 여부", "")
        age_target = item.get("연령 구분", "")
        support_type = item.get("지원형태", "")
        law = item.get("관련 법령", "")
        budget = item.get("예산", "")

        search_text = f"{name} {content} {field} {student} {income} {vulnerable} {vulnerable_type} {employment_status} {age_target} {support_type}"

        # 부산 공공데이터 기본 가점
        score += 2
        reasons.append("부산광역시 청년지원정책 데이터 기반")

        # 질문 키워드 매칭
        for kw in keywords:
            if kw and kw in search_text:
                score += 3
                reasons.append(f"'{kw}' 키워드와 관련")
                break

        # 정책유형 매칭
        if policy_type and policy_type in employment_status:
            score += 3
            reasons.append(f"{policy_type} 관련 정책")

        # 일자리/취업 질문이면 일자리 분야 우대
        if any(word in query for word in ["취업", "일자리", "구직", "면접", "인턴"]):
            if "일자리" in field or "취업" in search_text:
                score += 3
                reasons.append("일자리·취업 분야 정책")

        # 창업 질문이면 창업 관련 우대
        if "창업" in query:
            if "창업" in search_text:
                score += 3
                reasons.append("창업 관련 정책")

        # 사용자 정보가 있으면 보조 반영
        if user_info:
            education = user_info.get("education", "")
            career_type = user_info.get("career_type", "")

            if education and "학생" in student and ("대학" in education or "학생" in education):
                score += 2
                reasons.append("사용자 학력/학생 여부와 관련")

            if career_type and career_type in employment_status:
                score += 2
                reasons.append("사용자 취업 상태와 관련")

        if score > 2:
            results.append({
                "name": name,
                "description": content,
                "support": support_type,
                "keyword": field,
                "period": "",
                "url": "",
                "score": score,
                "reasons": reasons,
                "source": "부산광역시 청년지원정책 현황",
                "extra": {
                    "분야": field,
                    "학생 여부": student,
                    "소득 기준": income,
                    "취창업 여부": employment_status,
                    "연령 구분": age_target,
                    "관련 법령": law,
                    "예산": budget
                }
            })

    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    return results[:top_k]


def search_busan_services(query: str, entities: dict = {}, top_k: int = 3):
    """부산광역시 청년일자리지원서비스 CSV에서 관련 지원기관/서비스 검색"""
    results = []

    service_intent_keywords = ["상담", "센터", "기관", "어디", "위치", "문의", "방문", "지원서비스", "일자리센터", "취업지원"]
    detail_keyword = entities.get("세부키워드") or ""

    keywords = []
    if detail_keyword:
        keywords.extend([k.strip() for k in detail_keyword.split(",") if k.strip()])

    keywords.extend([kw for kw in service_intent_keywords if kw in query])
    keywords.extend([kw for kw in ["취업", "일자리", "청년", "면접", "교육", "상담"] if kw in query])

    for item in busan_job_services:
        score = 0
        reasons = []

        service_name = item.get("서비스명", "")
        organization = item.get("운영기관", "")
        address = item.get("상세주소", "")
        phone = item.get("문의처", "")
        homepage = item.get("홈페이지", "")
        x = item.get("X좌표", "")
        y = item.get("Y좌표", "")

        search_text = f"{service_name} {organization} {address} {phone} {homepage}"

        # 기관/상담 관련 질문이면 기본 가점
        if contains_any(query, service_intent_keywords):
            score += 4
            reasons.append("부산 청년일자리 지원기관 안내가 필요한 질문")

        # 키워드 매칭
        for kw in keywords:
            if kw and kw in search_text:
                score += 3
                reasons.append(f"'{kw}' 키워드와 관련")
                break

        # 위치/연락처가 있으면 서비스 활용성 가점
        if address:
            score += 1
        if phone:
            score += 1
        if homepage:
            score += 1
        if x and y:
            score += 1

        if score >= 3:
            results.append({
                "name": service_name,
                "description": f"{organization}에서 운영하는 부산 청년일자리 지원서비스입니다.",
                "support": f"주소: {address} / 문의처: {phone}",
                "keyword": "부산 청년일자리지원서비스",
                "period": "",
                "url": homepage,
                "score": score,
                "reasons": reasons,
                "source": "부산광역시 청년일자리지원서비스 현황",
                "extra": {
                    "운영기관": organization,
                    "상세주소": address,
                    "문의처": phone,
                    "홈페이지": homepage,
                    "X좌표": x,
                    "Y좌표": y
                }
            })

    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    return results[:top_k]

def search_youth_policies(query: str, entities: dict = {}, user_info: dict = None):
    """온통청년 청년정책API에서 취업/창업 관련 정책 검색
    지역/키워드 기준 간단 랭킹 → 상위 5개 반환
    """

        # 부산 공공데이터 CSV 검색
    busan_policy_results = search_busan_policies(
        query=query,
        entities=entities,
        user_info=user_info,
        top_k=3
    )

    busan_service_results = search_busan_services(
        query=query,
        entities=entities,
        top_k=2
    )

    print(f"[PUBLIC_DATA] 부산 정책 검색 결과: {len(busan_policy_results)}건")
    print(f"[PUBLIC_DATA] 부산 지원서비스 검색 결과: {len(busan_service_results)}건")
    
    url = "https://www.youthcenter.go.kr/go/ythip/getPlcy"

    # 엔터티에서 추출
    policy_type = entities.get("정책유형") or ""
    세부키워드 = entities.get("세부키워드") or ""
    지역 = entities.get("지역") or entities.get("거주지역") or ""
    keywords = [k.strip() for k in 세부키워드.split(",") if k.strip()] if 세부키워드 else []

    print(f"[PUBLIC_DATA] 정책유형: {policy_type}, 지역: {지역}, 키워드: {keywords}")

    # 1단계: 전체 개수 파악
    params_count = {
        "apiKeyNm": YOUTHCENTER_API_KEY,
        "lclsfNm": "일자리",
        "pageSize": 1,
        "rtnType": "json"
    }
    if policy_type in ["취업", "창업"]:
        params_count["mclsfNm"] = policy_type

    try:
        count_response = requests.get(url, params=params_count)
        total_count = count_response.json().get("result", {}).get("pagging", {}).get("totCount", 100)
        print(f"API 전체 정책 수: {total_count}개")
    except Exception as e:
        print(f"정책 개수 조회 실패: {e}")
        total_count = 100

    # 2단계: 전체 가져오기
    params = {
        "apiKeyNm": YOUTHCENTER_API_KEY,
        "lclsfNm": "일자리",
        "pageSize": total_count,
        "rtnType": "json"
    }
    if policy_type in ["취업", "창업"]:
        params["mclsfNm"] = policy_type

    try:
        response = requests.get(url, params=params)
        data = response.json()
        raw_items = data.get("result", {}).get("youthPolicyList", [])

        # 마감(0057003) 제외
        items = [i for i in raw_items if i.get("aplyPrdSeCd") != "0057003"]
        print(f"전체 {len(raw_items)}개 중 마감 제외 후: {len(items)}개")

        if not items:
            print("온통청년 API 결과 없음")
            return []
                
        # 특정 정책명 질문 매칭
       # 특정 정책명 질문 매칭
        def find_policy_by_name(items, query):
            for item in items:
                policy_name = item.get("plcyNm", "")

                # 사용자가 정책명을 그대로 질문에 포함한 경우만 상세 검색으로 처리
                if policy_name and policy_name in query:
                    return item

            return None


        matched_policy = find_policy_by_name(items, query)

        if matched_policy:
            matched_policy["_score"] = 999
            matched_policy["_reasons"] = ["사용자가 특정 정책명을 질문한 것으로 판단됨"]

            matched_result = [{
                "name": matched_policy.get("plcyNm", ""),
                "description": matched_policy.get("plcyExplnCn", ""),
                "support": matched_policy.get("plcySprtCn", ""),
                "keyword": matched_policy.get("plcyKywdNm", ""),
                "period": matched_policy.get("aplyYmd", ""),
                "url": matched_policy.get("aplyUrlAddr", ""),
                "score": matched_policy.get("_score", 999),
                "reasons": matched_policy.get("_reasons", []),
                "source": "온통청년 정책 API"
            }]

            return matched_result + busan_service_results
        
        # 3단계: 간단 점수 계산
        def calc_score(item):
            score = 0
            reasons = []

            plcy_nm = item.get("plcyNm", "")
            plcy_expln = item.get("plcyExplnCn", "")
            plcy_sprt = item.get("plcySprtCn", "")
            plcy_kwyd = item.get("plcyKywdNm", "")
            pvsn = item.get("pvsnInstGroupCd", "")
            aply_prd = item.get("aplyPrdSeCd", "")

            if 지역 and (지역 in plcy_nm or 지역 in plcy_expln or 지역 in plcy_sprt or 지역 in plcy_kwyd):
                score += 5
                reasons.append(f"{지역} 지역과 관련된 정책")

            if pvsn == "0054001":
                score += 3
                reasons.append("전국 단위 신청 가능")

            if aply_prd == "0057002":
                score += 5
                reasons.append("상시 신청 가능")

            for kw in keywords:
                if kw in plcy_nm or kw in plcy_expln or kw in plcy_sprt or kw in plcy_kwyd:
                    score += 3
                    reasons.append(f"'{kw}' 키워드와 관련")
                    break

            if not reasons:
                reasons.append("일자리 분야 정책 중 관련도가 높은 정책")

            return score, reasons
        
        # 4단계: 점수 정렬 → 상위 5개
        scored = []

        for item in items:
            score, reasons = calc_score(item)
            item["_score"] = score
            item["_reasons"] = reasons
            scored.append(item)

        scored.sort(key=lambda x: x.get("_score", 0), reverse=True)
        top5 = scored[:5]

        print(f"상위 5개: {[(i.get('plcyNm',''), i.get('_score', 0), i.get('_reasons', [])) for i in top5]}")
        
        policies = []
        for item in top5:
            policies.append({
                "name": item.get("plcyNm", ""),
                "description": item.get("plcyExplnCn", ""),
                "support": item.get("plcySprtCn", ""),
               "keyword": item.get("plcyKywdNm", ""),
                "period": item.get("aplyYmd", ""),
                "url": item.get("aplyUrlAddr", ""),
                "score": item.get("_score", 0),
                "reasons": item.get("_reasons", []),
                "source": "온통청년 정책 API"
            })

        # 부산 공공데이터 + 온통청년 API 결과 통합
        combined = busan_policy_results + policies + busan_service_results

        # 점수 기준 정렬 후 최대 7개 반환
        combined.sort(key=lambda x: x.get("score", 0), reverse=True)

        print(f"[PUBLIC_DATA] 통합 반환 결과: {len(combined[:7])}건")

        return combined[:7]

    except Exception as e:
        print(f"온통청년 API 호출 실패: {e}")
        return busan_policy_results + busan_service_results


# ============================
# 답변 생성
# ============================

def generate_answer(query: str, related_jobs: list, user_info: dict = None, history: list = [], related_trends: list = [], related_policies: list = [], intent: str = "", category_stats: dict = {}):
    """검색된 공고/트렌드/정책과 유저 정보, 대화 히스토리를 바탕으로 LLM이 답변 생성"""

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
        deadline = job.get("dates", {}).get("deadline", "")
        documents = ", ".join(job.get("apply", {}).get("document", []))
        preferred = ", ".join(job.get("detail", {}).get("preferred", []))
        talent = ", ".join(job.get("detail", {}).get("talent", []))

        steps = job.get("recruitment_process", {}).get("steps", [])
        process = " → ".join([step.get("name", "") for step in steps if step.get("name")])

        jobs_text += f"{i}. [{category}] {company} - {title}\n"
        jobs_text += f"   위치: {location} | 연봉: {salary_text} | 경력: {career_type}\n"
        jobs_text += f"   근무형태: {work_type} | 마감일: {deadline}\n"
        jobs_text += f"   스킬: {skills}\n"
        jobs_text += f"   지원서류: {documents}\n"
        jobs_text += f"   채용절차: {process}\n"
        jobs_text += f"   우대사항: {preferred}\n"
        jobs_text += f"   인재상: {talent}\n"
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

    # 정책 데이터를 텍스트로 변환
    policies_text = ""
    for i, policy in enumerate(related_policies, 1):
        reasons = policy.get("reasons", [])
        reason_text = ", ".join(reasons) if reasons else "관련 정책으로 판단됨"

        policies_text += f"{i}. {policy.get('name', '')}\n"
        policies_text += f"   추천근거: {reason_text}\n"
        policies_text += f"   설명: {policy.get('description', '')}\n"
        policies_text += f"   지원내용: {policy.get('support', '')}\n"
        policies_text += f"   신청기간: {policy.get('period', '')}\n"
        policies_text += f"   신청URL: {policy.get('url', '')}\n\n"

    # 카테고리별 통계 텍스트 변환 (TREND일 때만)
    stats_text = ""
    if intent == "TREND" and category_stats:
        sorted_stats = sorted(category_stats.items(), key=lambda x: x[1]["apply_count"], reverse=True)
        stats_text += "현재 플랫폼 채용 통계:\n"
        for category, stat in sorted_stats:
            stats_text += f"- {category}: 공고 {stat['count']}개, 지원수 {stat['apply_count']}명, 조회수 {stat['view_count']}회\n"

    # 유저 맞춤 정보가 있으면 프롬프트에 반영
    user_context = ""

    # 유저 맞춤 정보가 있으면 프롬프트에 반영
    user_context = ""
    if user_info:
        user_context = f"""
사용자 정보:
- 희망 직무: {user_info.get('job_type', '미설정')}
- 희망 지역: {user_info.get('region', '미설정')}
- 직종: {user_info.get('occupation', '미설정')}
- 경력 유형: {user_info.get('career_type', '미설정')}
- 학력: {user_info.get('education', '미설정')}
- 대학교: {user_info.get('university', '미설정')}
- 전공: {user_info.get('major', '미설정')}
- 경력 연수: {user_info.get('career_years', '미설정')}
- 재직 중인 회사: {user_info.get('company_name', '미설정')}
"""

    # 시스템 프롬프트
    system_prompt = f"""당신은 일로온(일로ON) 취업 플랫폼의 AI 챗봇 어시스턴트입니다.
반드시 한국어로만 답변하세요. 다른 언어는 절대 사용하지 마세요.
사용자의 취업 관련 질문에 친절하고 전문적으로 답변해주세요.
이전 대화 내용을 참고하여 문맥에 맞는 답변을 해주세요.
제공된 공고 목록을 바탕으로 답변하고, 목록에 없는 외부 사이트(Indeed, LinkedIn 등)는 언급하지 마세요.
공고의 스킬 목록에 있는 기술명은 한국어와 영어가 같은 의미입니다. 사용자가 한국어로 기술명을 말해도 영어로 된 스킬 목록에서 찾아서 답변해주세요.
공고 추천 시에는 핵심 정보를 간결하게 전달하고, 사용자가 추가로 궁금한 점을 물어볼 수 있도록 유도하세요.
사용자가 물어본 것만 답변하고 묻지 않은 내용은 절대 추가하지 마세요.
트렌드 데이터가 제공된 경우 해당 데이터만 바탕으로 답변하고 데이터에 없는 내용은 절대 추가하지 마세요.
현재 질문 유형은 {intent}입니다. 반드시 이 유형에 해당하는 데이터만 활용해서 답변하고, 관련 없는 데이터는 절대 언급하지 마세요.
정책별 핵심 내용을 간략하게 요약하고 신청 URL이 있으면 안내해주세요.
PUBLIC_DATA 답변에서는 반드시 제공된 추천근거를 바탕으로 왜 추천했는지 함께 설명하세요."""

    # 메시지 구성 - 시스템 프롬프트 + 이전 대화 히스토리 + 현재 질문
    # Claude API는 system 파라미터 별도로 분리
    user_messages = []

    # 이전 대화 히스토리 추가 (최근 5개만 유지해서 토큰 절약)
    if history:
        user_messages.extend(history[-5:])

    # 현재 유저 메시지 구성
    current_message = f"""{user_context}
사용자 질문: {query}

{f"아래는 일로온 플랫폼의 관련 공고 목록입니다:{chr(10)}{jobs_text}" if jobs_text else ""}
{f"아래는 관련 트렌드 정보입니다:{chr(10)}{trends_text}" if trends_text else ""}
{f"아래는 관련 청년 정책 정보입니다:{chr(10)}{policies_text}" if policies_text else ""}
{f"아래는 플랫폼 채용 통계입니다:{chr(10)}{stats_text}" if stats_text else ""}

위 제공된 정보를 바탕으로 사용자에게 도움이 되는 답변을 한국어로 해주세요.
{"\n\n더 알고 싶은 정책이 있으시면 말씀해주세요! 😊" if intent == "PUBLIC_DATA" else ""}"""

    user_messages.append({"role": "user", "content": current_message})

    # Claude API 호출해서 답변 생성
    response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=1024,
    system=system_prompt,
    messages=user_messages
    )

    answer = response.content[0].text

    if intent == "PUBLIC_DATA":
        answer += "\n\n📌 보다 많은 정보와 자세한 내용은 온통청년 홈페이지를 참고해주세요."

    return answer