# LLM 기반 상호작용형 동화 서비스
**(LLM-Based Interactive Fairy Tale Service)**

> 동화를 읽으며 실시간 질문, 독후 활동, 학습 피드백이 가능한 AI 기반 아동 인터랙션 학습 시스템  

---

## 프로젝트 개요

동화를 읽으며 아동이 질문을 하고, LLM이 동화 문맥을 기반으로 답변을 제공하는 **상호작용형 동화 학습 시스템**

질문 임베딩과 유사도 기반 검색을 활용하여 반복 질문에는 빠르게 응답하고, 새로운 질문에는 관련 동화 문단을 함께 전달하여 LLM이 문맥 기반 답변을 생성하도록 설계

또한 독후 활동과 학습 피드백을 자동 생성하여 아동의 이해도, 표현력, 창의성을 평가하고 독서 학습 경험을 확장할 수 있도록 구현

---

## 시스템 구성 및 기술 요소

### 시스템 구조

본 시스템은 **사용자 인터페이스(UI), 백엔드 서버, 데이터베이스(DB), LLM 모듈**로 구성

- **UI** : 동화 읽기, 질문 입력, 추천 질문 확인, 독후 활동 수행  
- **Backend** : 질문 임베딩 생성, 유사도 계산, 동화 문단 검색, LLM 호출  
- **Database** : 동화 문단, 질문·답변 데이터, 임베딩 벡터, 독후 활동 결과 저장  
- **LLM Module** : 문맥 기반 답변 생성, 독후 활동 문제 생성, 학습 피드백 생성  

이 구조를 통해 **동화 읽기 → 질문 → 답변 → 독후 활동 → 피드백**의 학습 흐름을 하나의 상호작용 시스템으로 통합

---

### 프로젝트 구조

```
project
│
├ main.py
├ app_emb_gpt.py
├ insert_story_chunks.py
├ embed_story_chunks.py
├ QA.sql
│
├ templates
│   └ ButterflyDream.html
│
├ static
│   ├ audio
│   ├ images
│   └ css
```

---

### 사용 기술

| Category | Technology |
|---|---|
| Backend | Flask |
| Database | MySQL |
| Embedding Model | SentenceTransformers(all-MiniLM-L6-v2) |
| LLM | Gemma(gemma-3n-e4b) |
| Vector Search | Cosine Similarity |
| Frontend | HTML / JavaScript |
| Speech Recognition | Web Speech API |
| TTS | gTTS |

---

## 주요 기능

### 1. 실시간 질문 응답 시스템
- 사용자가 질문을 입력하면 질문을 임베딩 벡터로 변환하고 기존 질문들과의 코사인 유사도 계산 
- 유사도가 임계값 이상일 경우 기존 답변을 재사용하고, 새로운 질문인 경우 LLM을 호출하여 답변 생성

### 2. 동화 문맥 기반 답변 생성
- 새로운 질문이 들어오면 질문 임베딩과 동화 문단 임베딩 간 유사도를 계산하여 가장 관련성이 높은 문단 추출
- 해당 문단과 질문을 함께 LLM에 전달하여 동화 문맥을 반영한 답변 생성

### 3. 추천 질문 기능
- 페이지별 질문 데이터를 임베딩 기반으로 클러스터링하여 의미적으로 유사한 질문을 그룹화하고, 각 그룹의 대표 질문을 선정하여 페이지당 최대 3개의 추천 질문 제공

### 4. 독후 활동 기능
- 동화 읽기가 끝난 후 아동은 독후 활동 페이지에서 문제를 풀 수 있음
- 문항은 이해도, 표현력, 창의성 평가를 중심으로 구성되며 일부 문항은 LLM을 통해 동화 내용과 질문 데이터를 기반으로 생성됨

### 5. 학습 피드백 생성
- 아동이 독후 활동 답변을 제출하면 LLM이 이해도, 표현력, 창의성, 종합 평가를 포함한 학습 피드백 생성
- 모든 결과는 데이터베이스에 저장되어 부모는 아동의 독서 활동 기록을 관리할 수 있음

---

## 실행 방법

> python main.py
`main.py` 실행 시 다음 과정이 자동으로 수행된다.

1. 동화 HTML 파일에서 문단을 추출하여 DB에 저장 (`insert_story_chunks.py`)
2. 문단 임베딩 생성 (`embed_story_chunks.py`)
3. Flask 서버 실행
4. 브라우저에서 동화 페이지 자동 실행

---

## 참고 문헌

[1] 나미영. (2010).  
상호작용적 동화 읽어주기 활동이 유아의 이야기 이해력과 기억력에 미치는 영향.  
계명대학교 교육대학원 석사학위논문.

[2] Morrow, L. M. (1985).  
Retelling stories: A strategy for improving young children's comprehension, concept of story structure, and oral language complexity.  
*The Elementary School Journal*, 85(5), 647–661.

[3] 박나연, 최정민. (2023).  
질문 유도를 통해 유아의 창의적 사고를 증진하는 인터랙티브 동화 콘텐츠 컨셉 제안.  
*Journal of Integrated Design Research*, 22(4), 117–134.

[4] 안현주, 배지호, 이수안. (2024).  
생성형 AI 기반 사용자 맞춤형 동화책 생성 및 구연 서비스.  
*한국정보과학회 학술발표논문집*, 2, 135–137.

[5] 정용태, 김선혁, 김세훈, 정기현. (2024).  
생성형 AI를 활용한 인터랙티브 동화책 생성 서비스.  
*Proceedings of KIT Conference*, 934–938.

[6] Google DeepMind. (2025).  
Introducing Gemma-3n: The Developer Guide.  
https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/

[7] Hugging Face. (2025).  
Gemma-3n: Lightweight Multimodal Models.  
https://huggingface.co/blog/gemma3n

[8] 권경문, 최숙기. (2025).  
AI 기반 한국어 글쓰기 자동 피드백의 질 평가 연구.  
*청람어문교육*, 104, 205–244.

[9] 최숙기. (2025).  
생성형 AI 기반 한국어 글쓰기 피드백의 질 평가 도구 개발 및 타당화 연구.  
*청람어문교육*, 103, 227–263.

[10] Liu, Z., Ping, W., Roy, R., Xu, P., Lee, C., Shoeybi, M., & Catanzaro, B. (2024).  
ChatQA: Surpassing GPT-4 on conversational QA and RAG.  
*Proceedings of the 38th International Conference on Neural Information Processing Systems (NeurIPS 2024).*
