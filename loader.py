import json
import os

# 데이터 폴더 경로 설정 (현재 파일 기준으로 data 폴더를 찾음)
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def load_all_jobs():
    """data 폴더의 모든 JSON 파일에서 공고 데이터 로드"""
    all_jobs = []
    
    # 공고 데이터가 아닌 파일들은 제외 (유저 설문, FAQ 등)
    # 새로운 데이터 파일 추가 시 여기에 파일명 추가하면 됨
    excluded_files = ["dummy_surveys.json", "faq.json"]
    
    for filename in os.listdir(DATA_DIR):
        
        # JSON 파일이 아니면 건너뜀
        if not filename.endswith(".json"):
            continue
        
        # 제외 목록에 있는 파일이면 건너뜀
        if filename in excluded_files:
            continue
        
        # 파일 전체 경로 만들기 (예: D:\illoon_chatbot\data\dummy_jobs_ai_ml.json)
        filepath = os.path.join(DATA_DIR, filename)
        
        # JSON 파일 열어서 데이터 읽기
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 파일에서 카테고리 이름 가져오기 (예: AI/ML, 백엔드 등)
        category = data.get("category", "")
        
        # 해당 파일의 공고들을 하나씩 꺼내서 전체 리스트에 추가
        for job in data.get("jobs", []):
            job["category"] = category  # 각 공고에 카테고리 정보 추가
            all_jobs.append(job)
    
    return all_jobs  # 모든 공고 데이터 반환

# 유저 설문 데이터 파일 경로
SURVEY_FILE = os.path.join(DATA_DIR, "dummy_surveys.json")

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

# FAQ 파일 경로
FAQ_FILE = os.path.join(DATA_DIR, "faq.json")

def load_faqs():
    """FAQ 질문 데이터 로드"""
    with open(FAQ_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("faqs", [])


def load_trend_summaries():
    """카테고리별 트렌드 데이터 로드"""
    trends = []
    
    # 공고 데이터가 아닌 파일들은 제외 (유저 설문, FAQ 등)
    excluded_files = ["dummy_surveys.json", "faq.json"]
    
    for filename in os.listdir(DATA_DIR):
        
        # JSON 파일이 아니면 건너뜀
        if not filename.endswith(".json"):
            continue
        
        # 제외 목록에 있는 파일이면 건너뜀
        if filename in excluded_files:
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
                "category": category,                                    # 직무 카테고리 (예: AI/ML, 백엔드)
                "hot_frameworks": trend.get("hot_frameworks", []),       # 인기 프레임워크
                "hot_languages": trend.get("hot_languages", []),         # 인기 언어
                "hot_tools": trend.get("hot_tools", []),                 # 인기 도구
                "talent_keywords": trend.get("talent_keywords", []),     # 인재상 키워드
                "salary_trend": trend.get("salary_trend", ""),           # 연봉 트렌드
                "welfare_trend": trend.get("welfare_trend", "")          # 복지 트렌드
            })
    
    return trends  # 모든 카테고리의 트렌드 데이터 반환