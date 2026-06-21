# 일로온 AI 챗봇 API 명세서

## 1. AI 챗봇

### 주요 기능

* 사용자 질문 기반 채용 공고 검색
* 사용자 설문 정보 기반 맞춤 공고 추천
* IT 직무 트렌드 안내
* 기업 정보 조회
* 청년 정책 추천
* 취업 준비 관련 가이드 제공
* 퀵리플라이 버튼 지원

---

## 챗봇 연동 구조

```text
프론트
  ↓
백엔드 서버
  ↓
일로온 AI 서버
  ↓
백엔드 서버
  ↓
프론트
```
---

# POST /chat

AI 챗봇 메인 응답 API

## Request Body

```json
{
  "message": "백엔드 개발자 공고 추천해줘",
  "user_id": "27379f57-57c5-45f3-8d62-f431d2f7ccb0",
  "user_info": null,
  "is_quick_replies": false,
  "quick_replies_category": null,
  "history": []
}
```

| 필드                     | 타입      | 필수 | 설명          |
| ---------------------- | ------- | -- | ----------- |
| message                | string  | Y  | 사용자 질문      |
| user_id                | string  | N  | 사용자 식별자     |
| user_info              | object  | N  | 사용자 설문 정보   |
| is_quick_replies       | boolean | N  | 퀵리플라이 클릭 여부 |
| quick_replies_category | string  | N  | 퀵리플라이 카테고리  |
| history                | array   | N  | 이전 대화 기록    |

---

## history 형식

```json
[
  {
    "role": "user",
    "content": "이전 질문"
  },
  {
    "role": "assistant",
    "content": "이전 답변"
  }
]
```

---

## Response

```json
{
  "answer": "AI 응답 내용",
  "sources": [],
  "intent": "GENERAL",
  "user_message": {
    "role": "user",
    "content": "사용자 질문"
  },
  "assistant_message": {
    "role": "assistant",
    "content": "AI 응답 내용"
  }
}
```

| 필드                | 타입     | 설명        |
| ----------------- | ------ | --------- |
| answer            | string | AI 응답     |
| sources           | array  | 참조 데이터    |
| intent            | string | 질문 분류 결과  |
| user_message      | object | 사용자 메시지   |
| assistant_message | object | AI 응답 메시지 |

---

# GET /chat/quick-replies

퀵리플라이 버튼 목록 조회 API

## Response

```json
{
  "나의 검색": [
    {
      "id": 1,
      "question": "나에게 맞는 공고 추천해줘"
    }
  ],
  "공고/채용정보": [
    {
      "id": 2,
      "question": "연봉이 높은 공고 알려줘"
    }
  ]
}
```

---

# GET /health

서버 상태 확인 API

## Response

```json
{
  "status": "ok"
}
```

---

# Intent 분류

| Intent      | 설명            |
| ----------- | ------------- |
| CUSTOM      | 사용자 맞춤 공고 추천  |
| GENERAL     | 일반 공고 검색      |
| TREND       | IT 취업 트렌드     |
| COMPANY     | 기업 정보 조회      |
| PUBLIC_DATA | 정부 정책 / 청년 정책 |
| CAREER_TIP  | 취업 준비 가이드     |
| INVALID     | 취업과 무관한 질문    |

---

# 연동 시 주의사항

## 1. user_info 필드 규격

일로온 AI 서버는 아래 필드명을 기준으로 사용자 설문 데이터를 처리한다.

```json
{
  "job_type": "",
  "region": "",
  "occupation": "",
  "career_type": "",
  "education": "",
  "university": "",
  "major": "",
  "career_years": "",
  "company_name": ""
}
```

### 중요

필드 추가는 가능하다.

예)

```json
{
  "job_type": "백엔드 개발",
  "region": "부산",
  "mbti": "INTJ",
  "certificate": "SQLD"
}
```

추가 필드는 AI 서버에서 무시되므로 연동에 문제가 없다.

하지만 기존 필드명을 변경할 경우 AI 서버 수정이 필요하다.

예)

```json
{
  "jobType": "백엔드 개발"
}
```

위와 같이 변경되면 AI 서버는 값을 읽지 못한다.

---

## 2. user_info 전달 방식

백엔드는 사용자 설문 데이터를 조회한 뒤 FastAPI 호출 시 user_info(유저데이터 전체) 객체로 함께 전달해야 한다.

권장 방식

```json
{
  "message": "나에게 맞는 공고 추천해줘",
  "user_id": "uuid",
  "user_info": {
    ...
  }
}
```

---

## 3. history 형식

history는 반드시 아래 구조를 따라야 한다.

```json
[
  {
    "role": "user",
    "content": "이전 질문"
  },
  {
    "role": "assistant",
    "content": "이전 답변"
  }
]
```

role 값은 다음 중 하나여야 한다.

```text
user
assistant
```

---

## 4. 현재 AI 서버에서 실제 활용하는 주요 설문 필드

추천 정확도에 직접 영향을 주는 필드

```text
job_type
region
career_type
```

보조 정보로 활용되는 필드

```text
occupation
education
university
major
career_years
company_name
```

---

## 5. 향후 DB 변경 시 협의 필요 항목

아래 항목 변경 시 AI 서버 수정이 필요할 수 있다.

* user_info 필드명 변경
* history 구조 변경
* job_type 값 체계 변경
* region 저장 형식 변경
* career_type 값 체계 변경

필드 추가만으로는 AI 서버 수정이 필요하지 않다.

---

# 백엔드 / DB 협의사항

현재 AI 서버는 아래 설문 정보를 활용합니다.

* job_type
* region
* occupation
* career_type
* education
* university
* major
* career_years
* company_name

### 우려 사항

* user_info 필드명 변경 시 AI 서버 수정이 필요할 수 있습니다.
* history 데이터 구조 변경 시 AI 서버 수정이 필요할 수 있습니다.
* job_type, region, career_type 값 체계 변경 시 사전 공유가 필요합니다.
* user_id 형식(UUID, 숫자형 등)이 변경되는 경우 공유가 필요합니다.
* 백엔드에서 사용자 설문 데이터를 조회한 후 user_info 객체로 함께 전달하는 방식을 권장합니다.

### 사용자 식별자(user_id) 관련

현재 개발 단계에서 사용한 더미 설문 데이터는 UUID 형식의 user_id를 사용하였습니다.