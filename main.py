from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from rag import search_jobs, search_trends, generate_answer, is_valid_query, find_similar_faq
from loader import get_user_info, load_faqs

# FastAPI 앱 생성
app = FastAPI()

# CORS 설정 - 프론트엔드에서 이 서버로 요청 보낼 수 있게 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 대화 히스토리 한 턴의 형식 정의 (role: user/assistant, content: 메시지 내용)
class ChatMessage(BaseModel):
    role: str
    content: str

# 유저 요청 데이터 형식 정의
class ChatRequest(BaseModel):
    message: str                           # 유저가 보낸 메시지
    user_id: Optional[str] = None          # 유저 ID (비로그인 시 None)
    user_info: Optional[dict] = None       # 직접 유저 정보 넘길 때 (선택사항)
    history: Optional[List[ChatMessage]] = []  # 이전 대화 히스토리

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
    
    if not is_valid_query(query):
        return {"reply": "저는 취업 관련 질문만 답변할 수 있어요! 공고 추천, 취업 정보 등을 물어봐주세요 😊"}

    # FAQ 케이스 분류 - 유저 질문과 유사한 FAQ 찾기
    matched_faq, distance = find_similar_faq(query)
    if matched_faq:
        print(f"FAQ 매칭됨: {matched_faq.get('question')} (거리: {distance:.2f})")
    
    # user_id가 있으면 DB(현재는 JSON)에서 유저 설문 데이터 자동으로 가져오기
    user_info = request.user_info
    if request.user_id and not user_info:
        user_info = get_user_info(request.user_id)

    # 히스토리를 딕셔너리 형태로 변환 (LLM에 넘기기 위해)
    history = [{"role": msg.role, "content": msg.content} for msg in request.history]

    # RAG 검색 - 질문과 관련된 공고 찾기
    related_jobs = search_jobs(query, user_info=user_info)
    
    # 트렌드 검색 - 질문과 관련된 트렌드 데이터 찾기
    related_trends = search_trends(query)

    # LLM으로 자연스러운 답변 생성 (히스토리 포함)
    reply = generate_answer(
        query=query,
        related_jobs=related_jobs,
        user_info=user_info,
        history=history,
        related_trends=related_trends  # 트렌드 데이터 추가
    )

    return {
        "reply": reply,
        # 프론트에서 다음 요청 시 히스토리에 추가할 수 있게 현재 대화 반환
        "user_message": {"role": "user", "content": query},
        "assistant_message": {"role": "assistant", "content": reply}
    }