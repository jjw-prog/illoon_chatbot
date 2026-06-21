from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from rag import search_jobs, search_trends, search_youth_policies, generate_answer, classify_intent, extract_entities, category_stats, normalize_job_type
from loader import get_user_info, load_quick_replies

# ============================
# FastAPI 앱 설정
# ============================

# FastAPI 앱 생성
app = FastAPI()

# CORS 설정 - Spring 백엔드 서버에서 이 서버로 요청 보낼 수 있게 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================
# 요청 데이터 형식 정의
# ============================

# 대화 히스토리 한 턴의 형식 정의 (role: user/assistant, content: 메시지 내용)
class ChatMessage(BaseModel):
    role: str
    content: str

# 유저 요청 데이터 형식 정의
class ChatRequest(BaseModel):
    message: str                                  # 유저가 보낸 메시지
    user_id: Optional[str] = None                 # 유저 ID
    user_info: Optional[dict] = None              # 직접 유저 정보 넘길 때 (선택사항)
    is_quick_replies: Optional[bool] = False      # 퀵리플라이 버튼 클릭 여부
    quick_replies_category: Optional[str] = None  # 퀵리플라이 카테고리
    history: Optional[List[ChatMessage]] = []     # 이전 대화 히스토리


# ============================
# 유틸 함수
# ============================

def extract_sources(jobs: list) -> list:
    """검색된 공고 리스트에서 job_id만 추출해서 반환"""
    return [job.get("job_id", "") for job in jobs if job.get("job_id")]


# ============================
# 엔드포인트
# ============================

# 서버 상태 확인 엔드포인트
@app.get("/health")
def health_check():
    return {"status": "ok"}


# 퀵리플라이 버튼 목록 반환 엔드포인트 - 프론트에서 버튼 목록 가져올 때 사용
@app.get("/chat/quick-replies")
def get_quick_replies():
    quick_replies = load_quick_replies()

    # 카테고리별로 묶기 (keywords 제외)
    result = {}
    for quick_reply in quick_replies:
        category = quick_reply.get("category", "")
        if category not in result:
            result[category] = []
        result[category].append({
            "id": quick_reply.get("id"),
            "question": quick_reply.get("question")
        })

    return result


# 챗봇 메인 엔드포인트
@app.post("/chat")
def chat(request: ChatRequest):
    query = request.message

    # 입력 필터링 - 너무 짧거나 의미없는 질문 차단
    if len(query.strip()) < 2:
        return {"answer": "질문을 좀 더 구체적으로 입력해주세요 😊"}

    # 히스토리를 딕셔너리 형태로 변환 (LLM에 넘기기 위해)
    history = [{"role": msg.role, "content": msg.content} for msg in request.history]

    # user_id로 유저 설문 데이터 가져오기
    user_info = request.user_info
    if request.user_id and not user_info:
        user_info = get_user_info(request.user_id)

    # ============================
    # 퀵리플라이 버튼 클릭 처리 (Intent 분류 안 거침)
    # ============================
    if request.is_quick_replies:
        print(f"퀵리플라이 처리: {query} (카테고리: {request.quick_replies_category})")

        # 퀵리플라이 개체명 추출
        entities = extract_entities(query, "QUICK_REPLIES")
        print(f"추출된 개체명: {entities}")

        # 나의 검색 → 유저 정보 반영
        if request.quick_replies_category == "나의 검색":
            related_jobs = search_jobs(query, user_info=user_info, entities=entities)
            quick_replies_user_info = user_info
        # 공고/채용정보 → 유저 정보 없이
        else:
            related_jobs = search_jobs(query, user_info=None, entities=entities)
            quick_replies_user_info = None

        related_trends = search_trends(query, entities=entities)
        answer = generate_answer(
            query=query,
            related_jobs=related_jobs,
            user_info=quick_replies_user_info,
            history=history,
            related_trends=related_trends,
            related_policies=[],
            intent="QUICK_REPLIES"
        )
        return {
            "answer": answer,
            "sources": extract_sources(related_jobs),
            "intent": "QUICK_REPLIES",
            "quick_replies_category": request.quick_replies_category,
            "user_message": {"role": "user", "content": query},
            "assistant_message": {"role": "assistant", "content": answer}
        }

    # ============================
    # Intent 분류 후 처리
    # ============================
    intent = classify_intent(query)
    print(f"최종 Intent: {intent}")

    # 개체명 추출 (인텐트에 맞게)
    entities = extract_entities(query, intent)
    print(f"추출된 개체명: {entities}")

    # INVALID → 취업 무관 질문 차단
    if intent == "INVALID":
        return {
            "answer": "저는 취업 관련 질문만 답변할 수 있어요! 공고 추천, 취업 정보 등을 물어봐주세요 😊",
            "sources": [],
            "intent": intent
        }

    # COMPANY → 기업 정보 안내
    if intent == "COMPANY":
        return {
            "answer": "정확한 기업 정보는 공고 상세페이지를 확인해주세요 😊",
            "sources": [],
            "intent": intent
        }

    # PUBLIC_DATA → 온통청년 청년정책 데이터로 답변
    if intent == "PUBLIC_DATA":
        policies = search_youth_policies(query, entities=entities, user_info=user_info)
        answer = generate_answer(
            query=query,
            related_jobs=[],
            user_info=None,
            history=history,
            related_trends=[],
            related_policies=policies,
            intent=intent
        )
        return {
            "answer": answer,
            "sources": [],
            "intent": intent,
            "user_message": {"role": "user", "content": query},
            "assistant_message": {"role": "assistant", "content": answer}
        }

    # CAREER_TIP → Claude 자체 지식으로 답변 (공고/트렌드 데이터 없이)
    if intent == "CAREER_TIP":
        answer = generate_answer(
            query=query,
            related_jobs=[],
            user_info=None,
            history=history,
            related_trends=[],
            related_policies=[],
            intent=intent
        )
        return {
            "answer": answer,
            "sources": [],
            "intent": intent,
            "user_message": {"role": "user", "content": query},
            "assistant_message": {"role": "assistant", "content": answer}
        }

    # TREND → 트렌드 데이터로 답변
    if intent == "TREND":
        related_trends = search_trends(query, entities=entities)
        answer = generate_answer(
            query=query,
            related_jobs=[],
            user_info=None,
            history=history,
            related_trends=related_trends,
            related_policies=[],
            intent=intent,
            category_stats=category_stats
        )
        return {
            "answer": answer,
            "sources": [],
            "intent": intent,
            "user_message": {"role": "user", "content": query},
            "assistant_message": {"role": "assistant", "content": answer}
        }

    # CUSTOM → 설문 데이터 + 공고 데이터로 맞춤 답변
    if intent == "CUSTOM":
         # user_info 정보를 entities에 병합 (entities에 없는 값만)
        if user_info:
            entities["직무"] = entities.get("직무") or user_info.get("job_type")
            entities["지역"] = entities.get("지역") or user_info.get("region")
            entities["경력"] = entities.get("경력") or user_info.get("career_type")
       
        print(f"병합 후 entities: {entities}")  # 추가
        related_jobs = search_jobs(query, user_info=user_info, entities=entities)
        answer = generate_answer(
            query=query,
            related_jobs=related_jobs,
            user_info=user_info,
            history=history,
            related_trends=[],
            related_policies=[],
            intent=intent
        )
        return {
            "answer": answer,
            "sources": extract_sources(related_jobs),
            "intent": intent,
            "user_message": {"role": "user", "content": query},
            "assistant_message": {"role": "assistant", "content": answer}
        }

    # GENERAL → 공고 데이터로 일반 답변 (설문 데이터 X)
    related_jobs = search_jobs(query, user_info=None, entities=entities)
    answer = generate_answer(
        query=query,
        related_jobs=related_jobs,
        user_info=None,
        history=history,
        related_trends=[],
        related_policies=[],
        intent=intent
    )
    return {
        "answer": answer,
        "sources": extract_sources(related_jobs),
        "intent": intent,
        "user_message": {"role": "user", "content": query},
        "assistant_message": {"role": "assistant", "content": answer}
    }