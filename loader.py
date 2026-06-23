import json
import os

# ============================
# 경로 설정
# ============================

# 데이터 폴더 경로 설정 (현재 파일 기준으로 data 폴더를 찾음)
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# 각 데이터 파일 경로
SURVEY_FILE = os.path.join(DATA_DIR, "dummy_surveys.json")
QUICK_REPLIES_FILE = os.path.join(DATA_DIR, "quick_replies.json")
INTENT_EXAMPLES_FILE = os.path.join(DATA_DIR, "intent_examples.json")
BUSAN_YOUTH_POLICY_FILE = os.path.join(DATA_DIR, "부산광역시_청년지원정책 현황.csv")
BUSAN_JOB_SERVICE_FILE = os.path.join(DATA_DIR, "청년일자리지원 서비스.csv")

# 공고 데이터 로드 시 제외할 파일 목록
# 새로운 비공고 데이터 파일 추가 시 여기에 파일명 추가하면 됨
EXCLUDED_FILES = ["dummy_surveys.json", "quick_replies.json", "intent_examples.json"]


# ============================
# Intent 예시 질문 데이터
# ============================

def load_intent_examples():
    """Intent Classification 예시 질문 데이터 로드"""
    with open(INTENT_EXAMPLES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("categories", [])  # 카테고리 목록 반환


# ============================
# 공고 데이터
# ============================

def load_all_jobs():
    """data 폴더의 모든 JSON 파일에서 공고 데이터 로드"""
    all_jobs = []

    for filename in os.listdir(DATA_DIR):

        # JSON 파일이 아니면 건너뜀
        if not filename.endswith(".json"):
            continue

        # 제외 목록에 있는 파일이면 건너뜀
        if filename in EXCLUDED_FILES:
            continue

        # 파일 전체 경로 만들기
        filepath = os.path.join(DATA_DIR, filename)

        # JSON 파일 열어서 데이터 읽기
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 파일에서 카테고리 이름 가져오기 (예: AI/ML, 백엔드 등)
        category = data.get("category", "")

        for job in data.get("jobs", []):
            job_category_mid = job.get("position", {}).get("job_category", {}).get("mid", "")
            job["category"] = category or job_category_mid
            all_jobs.append(job)

    return all_jobs  # 모든 공고 데이터 반환


# ============================
# 트렌드 데이터
# ============================

def load_trend_summaries():
    """카테고리별 트렌드 데이터 로드"""
    trends = []

    for filename in os.listdir(DATA_DIR):

        # JSON 파일이 아니면 건너뜀
        if not filename.endswith(".json"):
            continue

        # 제외 목록에 있는 파일이면 건너뜀
        if filename in EXCLUDED_FILES:
            continue

        # 파일 전체 경로 만들기
        filepath = os.path.join(DATA_DIR, filename)

        # JSON 파일 열어서 데이터 읽기
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 트렌드 데이터와 카테고리 가져오기
        trend = data.get("trend_summary", {})
        category = data.get("category", "")

        # 트렌드 데이터가 있는 파일만 추가
        if trend:
            trends.append({
                "category": category,                                 # 직무 카테고리 (예: AI/ML, 백엔드)
                "hot_frameworks": trend.get("hot_frameworks", []),    # 인기 프레임워크
                "hot_languages": trend.get("hot_languages", []),      # 인기 언어
                "hot_tools": trend.get("hot_tools", []),              # 인기 도구
                "talent_keywords": trend.get("talent_keywords", []),  # 인재상 키워드
                "salary_trend": trend.get("salary_trend", ""),        # 연봉 트렌드
                "welfare_trend": trend.get("welfare_trend", "")       # 복지 트렌드
            })

    return trends  # 모든 카테고리의 트렌드 데이터 반환


# ============================
# 유저 설문 데이터
# ============================

def load_user_surveys():
    """유저 설문 데이터 로드"""
    with open(SURVEY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def get_user_info(user_id: str):
    """user_id로 특정 유저의 설문 데이터 가져오기"""
    surveys = load_user_surveys()

    # user_id 일치하는 유저 찾기
    for survey in surveys:
        if survey.get("user_id") == user_id:
            return survey

    # 해당 유저 없으면 None 반환
    return None


# ============================
# 퀵리플라이 데이터
# ============================

def load_quick_replies():
    """퀵리플라이 질문 데이터 로드"""
    with open(QUICK_REPLIES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("quick_replies", [])

# ============================
# 카테고리별 통계 집계
# ============================

def get_category_stats():
    """카테고리별 공고 수, 조회수, 지원수, 스크랩수 집계"""
    all_jobs = load_all_jobs()
    stats = {}

    for job in all_jobs:
        category = job.get("category", "")
        if category not in stats:
            stats[category] = {
                "count": 0,          # 공고 수
                "view_count": 0,     # 총 조회수
                "apply_count": 0,    # 총 지원수
                "bookmark_count": 0  # 총 스크랩수
            }
        stats[category]["count"] += 1
        stats[category]["view_count"] += job.get("stats", {}).get("view_count", 0)
        stats[category]["apply_count"] += job.get("stats", {}).get("apply_count", 0)
        stats[category]["bookmark_count"] += job.get("stats", {}).get("bookmark_count", 0)

    return stats

# ============================
# 부산 공공데이터 CSV
# ============================

def load_csv_records(file_path: str):
    """CSV 파일을 읽어서 dict 리스트로 반환"""
    import csv

    if not os.path.exists(file_path):
        print(f"[WARN] CSV 파일을 찾을 수 없습니다: {file_path}")
        return []

    encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]

    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding, newline="") as f:
                reader = csv.DictReader(f)
                records = []

                for row in reader:
                    cleaned_row = {}
                    for key, value in row.items():
                        if key is None:
                            continue

                        clean_key = key.strip()
                        clean_value = value.strip() if isinstance(value, str) else value
                        cleaned_row[clean_key] = clean_value or ""

                    records.append(cleaned_row)

                print(f"[LOAD] CSV 로드 완료: {os.path.basename(file_path)} ({len(records)}건)")
                return records

        except UnicodeDecodeError:
            continue

    print(f"[ERROR] CSV 인코딩을 확인해주세요: {file_path}")
    return []


def load_busan_youth_policies():
    """부산광역시 청년지원정책 현황 CSV 로드"""
    return load_csv_records(BUSAN_YOUTH_POLICY_FILE)


def load_busan_job_services():
    """부산광역시 청년일자리지원 서비스 CSV 로드"""
    return load_csv_records(BUSAN_JOB_SERVICE_FILE)