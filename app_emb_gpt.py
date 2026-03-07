import os
import json
import pymysql
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from gtts import gTTS
from flask import send_file
import io
from io import BytesIO

from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from sklearn.cluster import DBSCAN
import random

app = Flask(__name__, template_folder="templates")
CORS(app)

# DB 연결
connection = pymysql.connect(
    host='127.0.0.1',
    port=3306,
    user='root',
    password='1234',
    db='fairy_db',
    charset='utf8mb4',
    cursorclass=pymysql.cursors.DictCursor
)
cursor = connection.cursor()

# 임베딩 모델 로딩
model = SentenceTransformer("all-MiniLM-L6-v2")

# LM Studio 클라이언트 설정
client = OpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio"  # 아무 문자열
)

# 코사인 유사도
def cosine_similarity(vec1, vec2):
    v1, v2 = np.array(vec1), np.array(vec2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

# created_at을 'YYYY-MM-DD HH:MM' 단위로 묶기 위한 키 생성
def minute_key(dt_str_or_dt):
    """
    created_at을 'YYYY-MM-DD HH:MM' 단위로 묶기 위한 키 생성
    (서로 다른 테이블의 비슷한 시간 기록을 "같은 세션"으로 묶기 위함)
    """
    if isinstance(dt_str_or_dt, str):
        try:
            dt = datetime.strptime(dt_str_or_dt, "%Y-%m-%d %H:%M:%S")
        except:
            try:
                dt = datetime.strptime(dt_str_or_dt, "%Y-%m-%d %H:%M")
            except:
                return dt_str_or_dt[:16]
    else:
        dt = dt_str_or_dt
    return dt.strftime("%Y-%m-%d %H:%M")

# 질문 저장
def save_question(text, slide_index, story_title):
    try:
        page_num = slide_index + 1  # HTML상 페이지 번호와 맞추기 위한 보정
        cursor.execute("INSERT INTO questions (story_title, questions, page_num) VALUES (%s, %s, %s)", (story_title, text, page_num))
        connection.commit()
        return cursor.lastrowid
    except Exception as e:
        print("❌ 질문 저장 오류:", e)
        return None

# 질문 임베딩 저장
def save_question_embedding(qid, embedding):
    try:
        cursor.execute("INSERT INTO question_embeddings (question_id, embedding) VALUES (%s, %s)", (qid, json.dumps(embedding)))
        connection.commit()
    except Exception as e:
        print("❌ 질문 임베딩 저장 오류:", e)

# 답변 저장
def save_answer(qid, answer):
    try:
        cursor.execute("INSERT INTO answers (question_id, answers) VALUES (%s, %s)", (qid, answer))
        connection.commit()
    except Exception as e:
        print("❌ 답변 저장 오류:", e)

# 유사 질문 찾기
def find_similar_question(embedding):
    cursor.execute("SELECT question_id, embedding FROM question_embeddings")
    best_score, best_id = 0, None
    for row in cursor.fetchall():
        e = json.loads(row['embedding'])
        score = cosine_similarity(embedding, e)
        if score > best_score:
            best_score, best_id = score, row['question_id']
    return best_id if best_score > 0.9 else None

# 해당 질문의 답변 가져오기
def get_answer_by_question_id(qid):
    cursor.execute("SELECT answers FROM answers WHERE question_id = %s", (qid,))
    row = cursor.fetchone()
    return row["answers"] if row else None

# story_chunks에서 유사 문단 찾기
def find_relevant_story_chunks(embedding, story_title, top_n=3):
    cursor.execute("""
        SELECT id, chunk_text, embedding 
        FROM story_chunks 
        WHERE story_title = %s AND embedding IS NOT NULL
    """, (story_title,))
    chunks = cursor.fetchall()
    scored = []
    for row in chunks:
        chunk_emb = json.loads(row["embedding"])
        score = cosine_similarity(embedding, chunk_emb)
        scored.append((score, row["chunk_text"]))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [chunk for _, chunk in scored[:top_n]]

# GPT에게 답변 요청
def get_gpt_answer_with_context(question, context_chunks):
    story_context = "\n".join(context_chunks)
    messages = [
        {"role": "system", "content": f"이건 어린이 동화에 대한 질문이야. 아래 내용을 참고해서 아이가 이해하기 쉽고 따뜻하게 한국어로 대답해줘:\n\n{story_context}"},
        {"role": "user", "content": question}
    ]
    try:
        response = client.chat.completions.create(
            model="gemma-3n-e4b",          # LM Studio 실행 중인 모델 이름
            messages=messages,
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("❌ GPT 응답 오류:", e)
        return "GPT 답변을 가져오는 중 오류가 발생했습니다."

# 페이지별 질문 Top 3 클러스터 대표 질문 추출
def get_top3_questions_by_page(page_num):
    # 1. 해당 페이지의 질문과 임베딩 조회
    cursor.execute("""
        SELECT q.id, q.questions, qe.embedding
        FROM questions q
        JOIN question_embeddings qe ON q.id = qe.question_id
        WHERE q.page_num = %s
    """, (page_num,))
    rows = cursor.fetchall()

    if not rows:
        return []
    
    # 질문이 1개뿐이면 그냥 반환
    if len(rows) == 1:
        return [{"question": rows[0]['questions'], "count": 1}]

    embeddings = [json.loads(row['embedding']) for row in rows]
    questions = [row['questions'] for row in rows]

    # 2D 배열로 변환 (질문이 1개여도 안전하게 처리)
    embeddings = np.array(embeddings).reshape(-1, len(embeddings[0]))

    # 2. 클러스터링 (1개만 있어도 클러스터 인정)
    clustering = DBSCAN(eps=0.4, min_samples=1, metric='cosine').fit(embeddings)
    labels = clustering.labels_

    # 3. 클러스터별 대표 질문 추출
    cluster_map = defaultdict(list)
    for label, q in zip(labels, questions):
        if label != -1:
            cluster_map[label].append(q)

    cluster_counts = sorted(cluster_map.items(), key=lambda x: len(x[1]), reverse=True)
    top_questions = []

    # 대표 질문 추출
    for _, qs in cluster_counts[:3]:
        representative = Counter(qs).most_common(1)[0][0]
        top_questions.append({"question": representative, "count": len(qs)})

    # 4. 클러스터가 3개 미만이면, 나머지는 임의 질문으로 채우기
    if len(top_questions) < 3:
        remaining = 3 - len(top_questions)
        used_questions = set(q['question'] for q in top_questions)
        remaining_pool = [q for q in questions if q not in used_questions]
        random.shuffle(remaining_pool)
        for q in remaining_pool[:remaining]:
            top_questions.append({"question": q, "count": 1})

    return top_questions


#----------------------------------------------------------------------
# 기본 라우트
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/story", methods=["GET"])
def story_index():
    return render_template("ButterflyDream.html")

@app.route("/ButterflyDream")
def butterfly_dream():
    return render_template("ButterflyDream.html")

@app.route("/LittleMatch")
def little_match():
    return render_template("LittleMatch.html")

# 동화 중 Q&A(임베딩 기반)
@app.route("/ask", methods=["POST"])
def ask():
    print("📩 /ask 요청 들어옴!", flush=True)
    import time 
    try:
        data = request.get_json()
        story_title = data.get("story_title")
        question = data.get("question")
        slide_index = data.get("slide_index")
        if not question:
            return jsonify({"error": "질문이 없습니다."}), 400

        # [1] 임베딩 단계
        start = time.time()
        q_emb = model.encode(question).tolist()
        print(f"🔹 [1] 임베딩 완료 ({time.time() - start:.2f}s)")

        # [2] 기존 유사 질문 확인
        start = time.time()
        similar_id = find_similar_question(q_emb)
        print(f"🔹 [2] 유사 질문 검색 완료 ({time.time() - start:.2f}s)")

        q_emb = model.encode(question).tolist()
        similar_id = find_similar_question(q_emb)
        if similar_id:
            answer = get_answer_by_question_id(similar_id)
            qid = save_question(question, slide_index, story_title)
            if qid:
                save_question_embedding(qid, q_emb)
                save_answer(qid, answer)
            return jsonify({"question": question, "answer": answer})
        
        # [3] 문맥 검색 + GPT 호출 단계
        start = time.time()
        context_chunks = find_relevant_story_chunks(q_emb, story_title)
        print(f"🔹 [3] 문맥 검색 완료 ({time.time() - start:.2f}s)")

        start = time.time()
        answer = get_gpt_answer_with_context(question, context_chunks)
        print(f"🔹 [4] GPT 응답 완료 ({time.time() - start:.2f}s)")

        qid = save_question(question, slide_index, story_title)
        if qid:
            save_question_embedding(qid, q_emb)
            save_answer(qid, answer)

        return jsonify({"question": question, "answer": answer})

    except Exception as e:
        print("❌ /ask 처리 오류:", e)
        return jsonify({"error": "서버 오류 발생"}), 500

# top3_questions
@app.route("/top3_questions", methods=["GET"])
def top3_questions():
    page_num = request.args.get("page", type=int)
    if page_num is None:
        return jsonify({"error": "page 번호가 필요합니다."}), 400

    top_questions = get_top3_questions_by_page(page_num)
    return jsonify(top_questions)

# TTS
@app.route("/tts", methods=["POST"])
def tts():
    try:
        data = request.get_json()
        text = data.get("text")
        if not text:
            return jsonify({"error": "텍스트가 없습니다."}), 400
        tts_obj = gTTS(text, lang="ko")
        mp3_fp = BytesIO()
        tts_obj.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        return send_file(mp3_fp, mimetype="audio/mpeg")
    except Exception as e:
        print("❌ /tts 처리 오류:", e)
        return jsonify({"error": "TTS 오류 발생"}), 500

# 퀴즈 : 페이지 렌더
@app.route("/Quiz")
def quiz_main():
    return render_template("Quiz.html")

@app.route("/Quiz_activity")
def quiz_question_page():
    return render_template("Quiz_activity.html")

@app.route("/Quiz_home")
def quiz_home_page():
    cursor.execute("""
        SELECT DISTINCT story_title
        FROM quiz_activity
        ORDER BY story_title
    """)
    stories = [row["story_title"] for row in cursor.fetchall()]
    return render_template("Quiz_home.html", stories=stories)

@app.route("/Quiz_record")
def quiz_record_page():
    story_title = request.args.get("story")
    if not story_title:
        return render_template("Quiz_record.html", story=None, groups=[])

    cursor.execute("""
        SELECT 
            a.id, a.story_title, a.child_name, a.child_age, 
            a.question_texts, a.answers, a.created_at,
            f.comprehension_score, f.comprehension_comment,
            f.creativity_score, f.creativity_comment,
            f.expression_score, f.expression_comment,
            f.overall_comment
        FROM quiz_activity a
        LEFT JOIN quiz_feedback f ON a.id = f.activity_id
        WHERE a.story_title=%s
        ORDER BY a.created_at DESC
    """, (story_title,))
    activities = cursor.fetchall()

    groups = []
    for a in activities:
        qlist = json.loads(a["question_texts"]) if a["question_texts"] else []
        alist = json.loads(a["answers"]) if a["answers"] else []
        groups.append({
            "created_at": a["created_at"],
            "child_name": a["child_name"],
            "child_age": a["child_age"],
            "questions": qlist,
            "answers": alist,
            "feedback": {
                "comprehension_score": a.get("comprehension_score"),
                "comprehension_comment": a.get("comprehension_comment"),
                "creativity_score": a.get("creativity_score"),
                "creativity_comment": a.get("creativity_comment"),
                "expression_score": a.get("expression_score"),
                "expression_comment": a.get("expression_comment"),
                "overall_comment": a.get("overall_comment")
            }
        })

    return render_template("Quiz_record.html", story=story_title, groups=groups)

# === [자동 생성 질문 API] ===
@app.route("/quiz_auto_question_dynamic", methods=["GET"])
def quiz_auto_question_dynamic():
    story_title = request.args.get("story", "").strip()
    child_age = request.args.get("age", "").strip()

    if not story_title:
        return jsonify({"question": "이 동화를 읽고 느낀 점은 무엇인가요?"})

    try:
        # 1) 최근 질문 1개 가져오기
        cursor.execute("""
            SELECT id, questions
            FROM questions
            WHERE story_title=%s
            ORDER BY RAND() LIMIT 1
        """, (story_title,))
        row = cursor.fetchone()

        if not row:
            context = "(이 동화에 대한 과거 질문 기록이 없음)"
            selected_question = None
            selected_q_id = None
        else:
            selected_question = row["questions"]
            selected_q_id = row["id"]
            context = selected_question

        # 2) 질문 임베딩 가져오기
        if selected_q_id:
            cursor.execute("""
                SELECT embedding
                FROM question_embeddings
                WHERE question_id=%s
            """, (selected_q_id,))
            emb_row = cursor.fetchone()
        else:
            emb_row = None

        # 3) 관련 문단 top3 찾기
        relevant_chunks = []
        if emb_row and emb_row["embedding"]:
            q_emb = np.array(json.loads(emb_row["embedding"]))
            relevant_chunks = find_relevant_story_chunks(q_emb, story_title, top_n=3)

        # 문단 합치기
        chunk_context = "\n---\n".join(relevant_chunks) if relevant_chunks else "(관련 문단 없음)"

        # 4) 프롬프트
        prompt = f"""
너는 {child_age}세 아이에게 동화를 읽은 후 이해도를 확인하기 위한 질문을 만들어주는 친절한 선생님이야.

동화 제목: {story_title}
아래는 아이가 이 동화에서 실제로 물어봤던 질문이야:
{context}

또한 아래는 위 질문과 가장 관련 있는 동화 문단 3개야:
{chunk_context}

이 기록을 참고해서,
아이의 '이해력'을 평가할 수 있는 한 문장짜리 질문을 새로 만들어줘.
질문은 한국어로 짧고 명확하며, 아이의 눈높이에 맞춰야 해.
반드시 한 문장만 출력하세요.
앞에 어떤 설명도 쓰지 마세요.
문장 끝에 물음표로 끝나는 질문 한 개만 출력하세요.
질문 외의 설명, 괄호, 안내문, 대화말투는 절대 포함하지 마세요.
"""
        # 5) GPT 호출
        res = client.chat.completions.create(
            model="gemma-3n-e4b",
            messages=[
                {"role": "system", "content": "You are a friendly elementary teacher who creates comprehension questions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=60
        )

        q = res.choices[0].message.content.strip()

        # 6) fallback
        if len(q) < 3:
            q = f"{story_title}을(를) 읽고 느낀 점이 무엇인가요?"

        return jsonify({"question": q})

    except Exception as e:
        print("❌ /quiz_auto_question_dynamic 오류:", e)
        return jsonify({"question": f"{story_title}을(를) 읽고 느낀 점이 무엇인가요?"})

# === [활동 저장 및 피드백 생성] ===
@app.route("/quiz_activity/save", methods=["POST"])
def save_quiz_activity():
    """
    내장 질문3개 + 답변 저장
    """
    # 요청 감지
    print("📩 [save_quiz_activity] 요청 도착")

    try:
        data = request.get_json()
        print(f"📦 전달된 데이터: {data}")

        story_title = data.get("story_title")
        child_name = data.get("child_name")
        child_age = data.get("child_age")
        q1_text = data.get("q1_text")         # 질문 기록 기반(이해력)
        q2_text = data.get("q2_text")         # 인물에게 하고싶은말(표현력)
        q3_text = data.get("q3_text")         # 동화에 들어간다면(창의성)
        a1 = data.get("a1_answer")
        a2 = data.get("a2_answer")
        a3 = data.get("a3_answer")

        print(f"🔹 story_title={story_title}, child_name={child_name}, child_age={child_age}")  # 값 확인

        if not story_title:
            print("❌ story_title이 없음")    # 필수값 누락
            return jsonify({"error": "story_title이 없습니다."}), 400
        if child_age is None:
            print("❌ child_age가 없음")      # 필수값 누락
            return jsonify({"error": "child_age가 없습니다."}), 400

        questions = [
            {"text": q1_text or f"{story_title}을(를) 읽고 느낀 점이 무엇인가요?", "is_parent": False},
            {"text": q2_text or "동화 속 인물에게 해주고 싶은 말을 적어보세요.", "is_parent": False},
            {"text": q3_text or f"내가 만약 '{story_title}' 속에 들어가게 된다면, 이야기가 그 후에 어떻게 진행될지 상상해보세요.", "is_parent": False}
        ]
        answers = [a1 or "", a2 or "", a3 or ""]

        # DB insert 전 단계
        print("📝 quiz_activity INSERT 실행 전")

        cursor.execute("""
            INSERT INTO quiz_activity (story_title, child_name, child_age, question_texts, answers)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            story_title, child_name, int(child_age),
            json.dumps(questions, ensure_ascii=False),
            json.dumps(answers, ensure_ascii=False)
        ))
        connection.commit()
        activity_id = cursor.lastrowid
        print(f"✅ quiz_activity 저장 완료 (activity_id={activity_id})")    # INSERT 성공

        # 자동으로 GPT 피드백 생성
        print("⚙️ generate_feedback_for_activity 호출 시작")                # 함수 진입 확인
        generate_feedback_for_activity(activity_id, story_title, child_name, child_age)
        print("✅ generate_feedback_for_activity 완료")                    # 함수 완료 확인

        return jsonify({"id": activity_id, "message": "퀴즈 및 피드백이 저장되었습니다."})

    except Exception as e:
        print("❌ /quiz_activity/save 처리 중 오류 발생:", e)
        import traceback
        traceback.print_exc()      # 오류 상세 추적
        return jsonify({"error": "서버 오류 발생"}), 500

def generate_feedback_for_activity(activity_id, story_title, child_name, child_age):
    """
    GPT가 이해력/창의성/표현력/전체평가를 분석해 항목별 점수 및 코멘트를 DB에 저장
    """
    cursor.execute("SELECT * FROM quiz_activity WHERE id=%s", (activity_id,))
    act = cursor.fetchone()
    if not act:
        return

    qlist = json.loads(act["question_texts"]) if act["question_texts"] else []
    alist = json.loads(act["answers"]) if act["answers"] else []

    q1, q2, q3 = (qlist[i]["text"] if i < len(qlist) else "" for i in range(3))
    a1, a2, a3 = (alist[i] if i < len(alist) else "" for i in range(3))

    # GPT 피드백 프롬프트
    prompt = f"""

아래는 {child_age}세 아이가 '{story_title}'을 읽고 작성한 내용이야.
아이의 이해력, 표현력, 창의성을 각각 1~5점으로 평가하고, 각 항목에 대해 한 줄 코멘트를 써줘.
마지막에 아이의 수준에 맞춘 전체적인 칭찬 한 줄과 조언 한 줄을 작성해줘.

질문과 답변:
1. {q1} → {a1}
2. {q2} → {a2}
3. {q3} → {a3}

출력은 아래 형식만 유지하고(띄어쓰기나 줄바꿈 꼭 그대로 유지하고 다른 특수기호 쓰지마. 전체평가 다음에도 줄바꿈 해야돼.) 내용은 새롭게 작성해:
이해력: ⭐(점수)/코멘트
창의성: ⭐(점수)/코멘트
표현력: ⭐(점수)/코멘트
전체평가:
👍 칭찬
🌱 조언
"""
    try:
        res = client.chat.completions.create(
            model="gemma-3n-e4b",  
            messages=[
                {"role": "system", "content": "You are a kind teacher who evaluates children's story comprehension."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        fb_text = res.choices[0].message.content.strip()
    except Exception as e:
        print("❌ GPT 피드백 오류:", e)
        fb_text = "GPT 피드백 생성 중 오류가 발생했습니다."

    # GPT 응답 파싱
    import re

    def parse_star(line):
        """'⭐4' 형태에서 숫자만 추출"""
        import re
        m = re.search(r"⭐?(\d)", line)
        return int(m.group(1)) if m else None

    comp_score = creat_score = expr_score = None
    comp_comment = creat_comment = expr_comment = ""
    overall_comment = ""

    lines = [line.strip() for line in fb_text.splitlines() if line.strip()]
    for line in [l.strip() for l in fb_text.splitlines() if l.strip()]:
        if line.startswith("이해력:"):
            comp_score = parse_star(line)
            comp_comment = line.split("/", 1)[1].strip() if "/" in line else line
        elif line.startswith("표현력:"):
            expr_score = parse_star(line)
            expr_comment = line.split("/", 1)[1].strip() if "/" in line else line
        elif line.startswith("창의성:"):
            creat_score = parse_star(line)
            creat_comment = line.split("/", 1)[1].strip() if "/" in line else line
        # --- 전체평가가 한 줄 형태로 들어오는 경우 ---
        elif line.startswith("전체평가:"):
            body = line.replace("전체평가:", "").strip()
            # 👍 칭찬 추출
            if "👍" in body:
                praise = body.split("🌱")[0].strip()
                overall_comment += praise + "\n"
            # 🌱 조언 추출
            if "🌱" in body:
                advice = "🌱 " + body.split("🌱")[1].strip()
                overall_comment += advice

        # --- 혹시 GPT가 줄바꿈을 해서 보내는 경우 대비 ---
        elif line.startswith("👍"):
            overall_comment += line + "\n"
        elif line.startswith("🌱"):
            overall_comment += line

    # DB 저장
    cursor.execute("""
        INSERT INTO quiz_feedback (
            activity_id, story_title, child_name, child_age,
            comprehension_score, comprehension_comment,
            expression_score, expression_comment,
            creativity_score, creativity_comment,
            overall_comment
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        activity_id, story_title, child_name, child_age,
        comp_score, comp_comment,
        expr_score, expr_comment,
        creat_score, creat_comment,
        overall_comment.strip()
    ))
    connection.commit()

@app.route("/quiz_feedback/analyze", methods=["POST"])
def analyze_feedback():
    """
    특정 활동(activity_id)의 GPT 피드백을 JSON으로 반환 (분석용)
    """
    data = request.get_json()
    activity_id = data.get("activity_id")

    if not activity_id:
        return jsonify({"error": "activity_id가 없습니다."}), 400

    cursor.execute("""
        SELECT 
            story_title, child_name, child_age,
            comprehension_score, comprehension_comment,
            creativity_score, creativity_comment,
            expression_score, expression_comment,
            overall_comment, 
            DATE_FORMAT(created_at, '%%Y-%%m-%%d %%H:%%i') AS created_at
        FROM quiz_feedback
        WHERE activity_id = %s
    """, (activity_id,))
    fb = cursor.fetchone()

    if not fb:
        return jsonify({"error": "해당 활동의 피드백이 없습니다."}), 404

    # JSON 응답
    return jsonify({
        "story_title": fb["story_title"],
        "child_name": fb["child_name"],
        "child_age": fb["child_age"],
        "scores": {
            "comprehension": fb["comprehension_score"],
            "creativity": fb["creativity_score"],
            "expression": fb["expression_score"]
        },
        "comments": {
            "comprehension": fb["comprehension_comment"],
            "creativity": fb["creativity_comment"],
            "expression": fb["expression_comment"],
            "overall": fb["overall_comment"]
        },
        "created_at": fb["created_at"]
    })

@app.route("/quiz_feedback/show/<int:activity_id>")
def show_feedback(activity_id):
    """
    특정 활동의 GPT 피드백 결과 페이지 렌더링
    """
    cursor.execute("""
        SELECT 
            f.story_title, f.child_name, f.child_age,
            f.comprehension_score, f.comprehension_comment,
            f.creativity_score, f.creativity_comment,
            f.expression_score, f.expression_comment,
            f.overall_comment, f.created_at
        FROM quiz_feedback f
        WHERE f.activity_id = %s
    """, (activity_id,))
    fb = cursor.fetchone()

    if not fb:
        return render_template("Quiz_feedback.html", feedback=None)

    return render_template("Quiz_feedback.html", feedback=fb)

@app.route("/quiz_feedback/story", methods=["GET"])
def get_feedback_by_story():
    """
    특정 동화(story_title)에 대한 모든 GPT 피드백 JSON 조회
    """
    story_title = request.args.get("story")
    if not story_title:
        return jsonify({"error": "story_title이 필요합니다."}), 400

    cursor.execute("""
        SELECT 
            id, activity_id, story_title, child_name, child_age,
            comprehension_score, creativity_score, expression_score,
            overall_comment,
            DATE_FORMAT(created_at, '%%Y-%%m-%%d %%H:%%i') AS created_at
        FROM quiz_feedback
        WHERE story_title = %s
        ORDER BY created_at DESC
    """, (story_title,))

    return jsonify(cursor.fetchall())


# 서버 실행
if __name__ == "__main__":
    app.run(debug=True)

#----------------------------------------------------------------------