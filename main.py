from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import requests
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from openai import OpenAI
import os
import json
import numpy as np
from models import RecommendationLog, WatchedMovie
from schemas import RecommendationLogSchema, WatchedMovieCreate, WatchedMovieSchema,ReviewResponse,ReviewRequest,MovieDetailResponse
from database import SessionLocal, engine, Base
from sqlalchemy.orm import Session
import time


# ───────────────────────────────
# 환경설정
load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# DB 테이블 생성 (앱 시작 시 1회)
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
# ───────────────────────────────
# 모델 및 DB 초기화
embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-sts")
vector_db = Chroma(
    persist_directory="./movie_vectorDB",
    embedding_function=embedding_model
)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ───────────────────────────────
# 요청/응답 모델
class RecommendRequest(BaseModel):
    message: str

class RecommendResponse(BaseModel):
    reply: str

# ───────────────────────────────
# GPT 응답 파서
def parse_gpt_json_response(raw_text: str):
    raw_text = raw_text.strip()
    if raw_text.startswith("```json"):
        raw_text = raw_text[7:]
    elif raw_text.startswith("```"):
        raw_text = raw_text[3:]
    if raw_text.endswith("```"):
        raw_text = raw_text[:-3]
    return json.loads(raw_text.strip())


# ───────────────────────────────
# API 엔드포인트
@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest,db:Session=Depends(get_db)):
    start=time.time()
    # 1. GPT로 분위기 태그 추출
    tag_prompt = f"""
    다음 문장에서 감정, 분위기, 장르 관련 태그를 2~4개만 추출해줘.
    "{req.message}"
    형식: ["힐링", "감동", "우울한"]
    """
    tag_response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "감정/분위기/장르 태그를 JSON 배열로 추출해줘. 코드블럭 없이."},
            {"role": "user", "content": tag_prompt}
        ]
    )
    user_tags = parse_gpt_json_response(tag_response.choices[0].message.content)
    
    print("🕒 GPT 태그 추출 소요:", time.time() - start); start = time.time()

    # 2. 벡터DB에서 mood_labels 기반 필터링
    all_docs = vector_db.get(include=["documents", "metadatas"])
    filtered = []
    for doc_text, meta in zip(all_docs["documents"], all_docs["metadatas"]):
        mood_tags = meta.get("mood_labels", "").split(", ")
        if any(tag in mood_tags for tag in user_tags):
            filtered.append((doc_text, meta))

    if not filtered:
        return {"reply": f"{user_tags} 분위기에 맞는 영화를 찾을 수 없었습니다."}

    # 3. 필터링된 문서 → 벡터화하여 유사도 정렬
    user_vector = embedding_model.embed_query(req.message)
    movie_vectors = embedding_model.embed_documents([doc[0] for doc in filtered])
    similarities = [np.dot(user_vector, mv) for mv in movie_vectors]

    sorted_docs = sorted(zip(filtered, similarities), key=lambda x: x[1], reverse=True)
    
    
    print("🕒 유사도 비교 소요:", time.time() - start); start = time.time()

    # 4. 중복 제거하여 상위 5개만 추출
    seen_titles = set()
    top_docs = []
    for ((doc_text, meta), _) in sorted_docs:
        title = meta.get("title")
        if title not in seen_titles:
            seen_titles.add(title)
            top_docs.append((doc_text, meta))
        if len(top_docs) == 5:
            break

    # 5. GPT에게 추천 설명 요청
    titles = [f"{meta['title']} ({meta.get('year', '연도 미정')})" for _, meta in top_docs]
    recommend_prompt = f"""
    사용자의 요청: "{req.message}"
    추출된 태그: {user_tags}

    새로운 영화를 소개할 때 마다 영화의 제목 앞에 무조건 🎬를 붙여줘. 
    아래 영화들은 추천 후보입니다. 각 영화에 대해 1~2문장 소개와 추천 이유를 작성해줘:

    {chr(10).join(titles)}
    """
    gpt_response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "당신은 영화 추천 전문가입니다."},
            {"role": "user", "content": recommend_prompt}
        ]
    )
    
    print("🕒 GPT 설명 소요:", time.time() - start)
    # 7. DB에 기록 저장
    log = RecommendationLog(
        query=req.message,
        tags=", ".join(user_tags),
        recommended_titles=", ".join([meta["title"] for _, meta in top_docs])
    )
    db.add(log)
    db.commit()

    return {"reply": gpt_response.choices[0].message.content,"log_id": log.id}

# ───────────────────────────────
# 추천 로그 목록 조회
@app.get("/logs", response_model=list[RecommendationLogSchema])
def get_logs(db: Session = Depends(get_db)):
    return db.query(RecommendationLog).order_by(RecommendationLog.created_at.desc()).limit(30).all()

# 시청 완료 등록 API
@app.post("/watched", response_model=WatchedMovieSchema)
def mark_as_watched(data: WatchedMovieCreate, db: Session = Depends(get_db)):
    # 중복 방지
    existing = db.query(WatchedMovie).filter(WatchedMovie.title == data.title).first()
    if existing:
        raise HTTPException(status_code=400, detail="이미 시청 완료한 영화입니다.")

    watched = WatchedMovie(title=data.title, from_log_id=data.from_log_id)
    db.add(watched)
    db.commit()
    db.refresh(watched)
    return watched

# 시청 완료 목록 조회
@app.get("/watched", response_model=list[WatchedMovieSchema])
def list_watched(db: Session = Depends(get_db)):
    return db.query(WatchedMovie).order_by(WatchedMovie.watched_at.desc()).all()

    
@app.post("/watched/review")
def register_review(request: ReviewRequest, db: Session = Depends(get_db)):
    movie = db.query(WatchedMovie).filter(WatchedMovie.id == request.movie_id).first()
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found")

    movie.review = request.review
    db.commit()
    return {"message": "Review registered successfully"}

@app.get("/watched/{movie_id}/review", response_model=ReviewResponse)
def get_review(movie_id: int, db: Session = Depends(get_db)):
    movie = db.query(WatchedMovie).filter(WatchedMovie.id == movie_id).first()
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found")

    return ReviewResponse(
        exists=bool(movie.review),
        content=movie.review
    )
    
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

TMDB_BASE_URL = "https://api.themoviedb.org/3"
API_KEY = os.getenv("TMDB_API_KEY")

@app.get("/movie/search")
def search_movie(title: str):
    search_url = f"{TMDB_BASE_URL}/search/movie"
    params = {"api_key": API_KEY, "query": title, "language": "ko-KR"}
    res = requests.get(search_url, params=params)
    data = res.json()
    
    if not data["results"]:
        raise HTTPException(status_code=404, detail="영화 없음")

    movie = data["results"][0]  # 가장 첫 번째 검색 결과

    tag_prompt = f"""
    다음 문장에서 감정, 분위기, 장르 관련 태그를 2~4개만 추출해줘.
    "{movie.get("overview", "")}"
    형식: ["힐링", "감동", "우울한"]
    """
    tag_response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "감정/분위기/장르 태그를 JSON 배열로 추출해줘. 코드블럭 없이."},
            {"role": "user", "content": tag_prompt}
        ]
    )
    tags = parse_gpt_json_response(tag_response.choices[0].message.content)
    
    return {
        "id": movie["id"],
        "title": movie["title"],
        "overview": movie.get("overview", ""),
        "releaseYear": movie.get("release_date", "")[:4],
        "posterPath": movie.get("poster_path", ""),
        "moodLabels": tags
    }