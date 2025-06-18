import requests
import sys
import io
import os
from dotenv import load_dotenv

# 출력 인코딩 설정 (Windows 콘솔에서 한글 깨짐 방지)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

API_KEY = os.getenv("TMDB_API_KEY") # TMDB API 키
BASE_URL = 'https://api.themoviedb.org/3'

def get_popular_movies():
    url = f'{BASE_URL}/movie/popular'
    params = {
        'api_key': API_KEY,
        'language': 'ko-KR',  # 한국어 결과
        'page': 1
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        movies = response.json()['results']
        for idx, movie in enumerate(movies[:5], 1):  # 상위 5개만 출력
            print(f"{idx}. {movie['title']} ({movie['release_date']}) - 평점: {movie['vote_average']}")
    else:
        print("API 호출 실패:", response.status_code, response.text)

get_popular_movies()
