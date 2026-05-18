from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from rag import search_jobs, search_trends, generate_answer, find_similar_faq, classify_intent
from loader import get_user_info, load_faqs

# ============================
# FastAPI 앱 설정
# ============================

# FastAPI 앱 생성
app = FastAPI()

# CORS 설정 - 프론트엔드에서 이 서버로 요청 보낼 수 있게 허용
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
    message: str                                # 유저가 보낸 메시지
    user_id: Optional[str] = None               # 유저 ID
    user_info: Optional[dict] = None            # 직접 유저 정보 넘길 때 (선택사항)
    is_faq: Optional[bool] = False              # FAQ 버튼 클릭 여부
    history: Optional[List[ChatMessage]] = []   # 이전 대화 히스토리


# ============================
# 엔드포인트
# ============================

# 서버 상태 확인 엔드포인트
@app.get("/health")
def health_check():
    return {"status": "ok"}


# FAQ 목록 반환 엔드포인트 - 프론트에서 버튼 목록 가져올 때 사용
@app.get("/faq")
def get_faqs():
    faqs = load_faqs()
    return {"faqs": faqs}


# 챗봇 메인 엔드포인트
@app.post("/chat")
def chat(request: ChatRequest):
    query = request.message

    # 입력 필터링 - 너무 짧거나 의미없는 질문 차단
    if len(query.strip()) < 2:
        return {"reply": "질문을 좀 더 구체적으로 입력해주세요 😊"}

    # 히스토리를 딕셔너리 형태로 변환 (LLM에 넘기기 위해)
    history = [{"role": msg.role, "content": msg.content} for msg in request.history]

    # user_id로 유저 설문 데이터 가져오기
    user_info = request.user_info
    if request.user_id and not user_info:
        user_info = get_user_info(request.user_id)

    # ============================
    # FAQ 버튼 클릭 처리 (Intent 분류 안 거침)
    # ============================
    if request.is_faq:
        related_jobs = search_jobs(query, user_info=user_info)
        related_trends = search_trends(query)
        reply = generate_answer(
            query=query,
            related_jobs=related_jobs,
            user_info=user_info,
            history=history,
            related_trends=related_trends
        )
        return {
            "reply": reply,
            "user_message": {"role": "user", "content": query},
            "assistant_message": {"role": "assistant", "content": reply}
        }

    # ============================
    # Intent 분류 후 처리
    # ============================
    intent = classify_intent(query)
    print(f"최종 Intent: {intent}")

    # INVALID → 취업 무관 질문 차단
    if intent == "INVALID":
        return {"reply": "저는 취업 관련 질문만 답변할 수 있어요! 공고 추천, 취업 정보 등을 물어봐주세요 😊"}

    # COMPANY → 기업 정보 안내
    if intent == "COMPANY":
        return {"reply": "정확한 기업 정보는 공고 상세페이지를 확인해주세요 😊"}

    # PUBLIC_DATA → 공공데이터 준비 중 안내
    if intent == "PUBLIC_DATA":
        return {"reply": "공공기관 채용 정보는 현재 준비 중입니다 😊"}

    # CAREER_TIP → Claude 자체 지식으로 답변 (공고/트렌드 데이터 없이)
    if intent == "CAREER_TIP":
        reply = generate_answer(
            query=query,
            related_jobs=[],
            user_info=None,
            history=history,
            related_trends=[]
        )
        return {
            "reply": reply,
            "user_message": {"role": "user", "content": query},
            "assistant_message": {"role": "assistant", "content": reply}
        }

    # TREND → 트렌드 데이터로 답변
    if intent == "TREND":
        related_trends = search_trends(query)
        reply = generate_answer(
            query=query,
            related_jobs=[],
            user_info=None,
            history=history,
            related_trends=related_trends
        )
        return {
            "reply": reply,
            "user_message": {"role": "user", "content": query},
            "assistant_message": {"role": "assistant", "content": reply}
        }

    # CUSTOM → 설문 데이터 + 공고 데이터로 맞춤 답변
    if intent == "CUSTOM":
        related_jobs = search_jobs(query, user_info=user_info)
        reply = generate_answer(
            query=query,
            related_jobs=related_jobs,
            user_info=user_info,
            history=history,
            related_trends=[]
        )
        return {
            "reply": reply,
            "user_message": {"role": "user", "content": query},
            "assistant_message": {"role": "assistant", "content": reply}
        }

    # GENERAL → 공고 데이터로 일반 답변 (설문 데이터 X)
    related_jobs = search_jobs(query, user_info=None)
    reply = generate_answer(
        query=query,
        related_jobs=related_jobs,
        user_info=None,
        history=history,
        related_trends=[]
    )
    return {
        "reply": reply,
        "user_message": {"role": "user", "content": query},
        "assistant_message": {"role": "assistant", "content": reply}
    }