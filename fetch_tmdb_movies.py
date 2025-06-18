import os
import requests
import json
import time
from dotenv import load_dotenv
from openai import OpenAI
import sys

sys.stdout.reconfigure(encoding='utf-8')

# ───────────────────────────────
# 환경 변수 로딩
load_dotenv()
API_KEY = os.getenv("TMDB_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_KEY)

BASE_URL = "https://api.themoviedb.org/3"
OUTPUT_FILE = "movies.json"

# ───────────────────────────────
# GPT 응답 파서
def parse_gpt_json_response(raw_text: str):
    """
    GPT 응답에서 ```json 블록 제거 후 JSON 파싱
    """
    raw_text = raw_text.strip()
    if raw_text.startswith("```json"):
        raw_text = raw_text[7:]
    elif raw_text.startswith("```"):
        raw_text = raw_text[3:]
    if raw_text.endswith("```"):
        raw_text = raw_text[:-3]
    return json.loads(raw_text.strip())

# ───────────────────────────────
# GPT로 분위기 추출
def get_mood_labels(overview: str):
    
    prompt = f"""
    줄거리: "{overview}"
    
    이 영화의 분위기를 가장 잘 나타내는 키워드를 2~3개 뽑아주세요.
    예: 감동, 무서운, 유쾌한, 따뜻한, 잔잔한, 우울한, 자극적인 등
    이 영화의 특징을 가장 잘 나타내는 키워드를 2~3개 뽑아주세요.
    형식: ["키워드1", "키워드2"] 형태의 JSON 배열로만 출력하세요.
    코드블럭(예: ```json)은 포함하지 마세요.
    """

    try:
        res = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "당신은 영화 분위기를 분석하는 태깅 어시스턴트입니다. 출력은 JSON 배열 형식만, 코드블럭 없이."},
                {"role": "user", "content": prompt}
            ]
        )
        gpt_response_text = res.choices[0].message.content.strip()
        print("GPT raw response:", gpt_response_text)
        
        try:
            return parse_gpt_json_response(gpt_response_text)
        except json.JSONDecodeError as e:
            print(" GPT 태깅 파싱 오류:", e)
            return []

    except Exception as e:
        print(" GPT 태깅 호출 오류:", e)
        return []

# ───────────────────────────────
# TMDB에서 인기 영화 가져오기
def fetch_popular_movies(pages: int = 3):
    all_movies = []

    for page in range(1, pages + 1):
        url = f"{BASE_URL}/movie/popular"
        params = {
            "api_key": API_KEY,
            "language": "ko-KR",
            "page": page
        }

        response = requests.get(url, params=params)
        if response.status_code == 200:
            results = response.json().get("results", [])
            for movie in results:
                title = movie.get("title")
                overview = movie.get("overview")
                release_date = movie.get("release_date", "0000")[:4]

                if overview:
                    print(f" {title} ... GPT 태깅 중")
                    mood_labels = get_mood_labels(overview)
                    time.sleep(1.2)  # rate limit 조절
                    all_movies.append({
                        "title": title,
                        "overview": overview,
                        "release_year": release_date,
                        "mood_labels": mood_labels
                    })
        else:
            print(f" API 호출 실패 (page {page}):", response.status_code)

    return all_movies

def fetch_top_rated_movies(pages: int = 3):
    all_movies = []

    for page in range(1, pages + 1):
        url = f"{BASE_URL}/movie/top_rated"
        params = {
            "api_key": API_KEY,
            "language": "ko-KR",
            "page": page
        }

        response = requests.get(url, params=params)
        if response.status_code == 200:
            results = response.json().get("results", [])
            for movie in results:
                title = movie.get("title")
                overview = movie.get("overview")
                release_date = movie.get("release_date", "0000")[:4]

                if overview:
                    print(f" {title} ... GPT 태깅 중")
                    mood_labels = get_mood_labels(overview)
                    time.sleep(1.2)
                    all_movies.append({
                        "title": title,
                        "overview": overview,
                        "release_year": release_date,
                        "mood_labels": mood_labels
                    })
        else:
            print(f" API 호출 실패 (Top Rated page {page}):", response.status_code)

    return all_movies

def fetch_movies_by_genre(genre_id: int, pages: int = 2):
    all_movies = []

    for page in range(1, pages + 1):
        url = f"{BASE_URL}/discover/movie"
        params = {
            "api_key": API_KEY,
            "language": "ko-KR",
            "with_genres": genre_id,
            "sort_by": "popularity.desc",
            "page": page
        }

        response = requests.get(url, params=params)
        if response.status_code == 200:
            results = response.json().get("results", [])
            for movie in results:
                title = movie.get("title")
                overview = movie.get("overview")
                release_date = movie.get("release_date", "0000")[:4]

                if overview:
                    print(f"🎬 {title} ... GPT 태깅 중")
                    mood_labels = get_mood_labels(overview)
                    time.sleep(1.2)
                    all_movies.append({
                        "title": title,
                        "overview": overview,
                        "release_year": release_date,
                        "genre_id": genre_id,
                        "mood_labels": mood_labels
                    })
        else:
            print(f"❌ 장르 ID {genre_id} - page {page} 실패:", response.status_code)

    return all_movies

# ───────────────────────────────
# 저장 함수
def save_movies_to_json(movies, filepath=OUTPUT_FILE):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(movies, f, ensure_ascii=False, indent=2)
    print(f"\n 총 {len(movies)}개 영화 저장 완료 → {filepath}")

# ───────────────────────────────
# 실행
if __name__ == "__main__":
    movies=[]
    movies += fetch_popular_movies(pages=10)
    movies+=fetch_top_rated_movies(pages=10)
    
    GENRE_MAP = {
        28: "액션", 12: "모험", 16: "애니메이션", 35: "코미디", 80: "범죄",
        99: "다큐멘터리", 18: "드라마", 10751: "가족", 14: "판타지", 36: "역사",
        27: "공포", 10402: "음악", 9648: "미스터리", 10749: "로맨스", 878: "SF",
        10770: "TV 영화", 53: "스릴러", 10752: "전쟁", 37: "서부"
    }
    for gid in GENRE_MAP:
        print(f"\n📚 장르 ID {gid} 수집 시작...")
        movies += fetch_movies_by_genre(gid, pages=3)
    
    save_movies_to_json(movies)
