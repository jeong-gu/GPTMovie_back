import os
import json
import sys
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import shutil

sys.stdout.reconfigure(encoding='utf-8')

# ───────────────────────────────
# JSON 로드 → LangChain 문서 변환
def load_movie_json(json_path: str) -> list[Document]:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"{json_path} 파일이 존재하지 않습니다.")

    with open(json_path, "r", encoding="utf-8") as f:
        movies = json.load(f)

    docs = []
    seen_titles = set()
    for movie in movies:
        title = movie.get("title", "제목 없음")
        overview = movie.get("overview", "")
        year = movie.get("release_year", "연도 없음")
        moods = movie.get("mood_labels", [])
        unique_key = f"{title} ({year})"
    
        if unique_key in seen_titles:
            continue  # 중복은 건너뜀
        seen_titles.add(unique_key)
        
        # ✅ 분위기 태그를 page_content에도 포함
        text = (
            f"[제목] {title} ({year})\n"
            f"[줄거리] {overview}\n"
            f"[분위기 태그] {', '.join(moods)}"
        )

        # ✅ 반드시 metadata에 mood_labels 포함
        docs.append(Document(
            page_content=text,
            metadata={
                "title": title,
                "year": year,
                "mood_labels": ", ".join(moods)
            }
        ))

    print(f"✅ 총 {len(docs)}개 영화 문서 로드 완료")
    return docs

# ───────────────────────────────
# 벡터DB 생성 및 테스트
def prepare_chroma_movie_db(json_path: str, persist_dir: str):
    if os.path.exists(persist_dir):
        print(f"🧹 기존 벡터 DB 제거: {persist_dir}")
        shutil.rmtree(persist_dir)  # 💥 DB 초기화
    print("📦 고성능 임베딩 모델 로딩 중... (KoSBERT)")
    embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-sts")

    print("📄 영화 문서 불러오는 중...")
    movie_docs = load_movie_json(json_path)

    print("💾 벡터 DB에 저장 중...")
    vector_db = Chroma.from_documents(
        documents=movie_docs,
        embedding=embedding_model,
        persist_directory=persist_dir
    )
    print(f"🎉 벡터 DB 저장 완료: {persist_dir} (총 {len(movie_docs)}개 문서)")

    # 테스트 쿼리
    print("\n🔍 [테스트 검색] '잔잔하고 인생을 되돌아보게 하는 영화'")
    vector_db = Chroma(
        embedding_function=embedding_model,
        persist_directory=persist_dir
    )
    query = "잔잔하고 인생을 되돌아보게 하는 영화"
    results = vector_db.similarity_search(query, k=5)

    print("\n📌 [테스트 검색 결과]")
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.page_content.splitlines()[0]}")  # 제목만 출력

def inspect_vector_db(vector_db):
    """
    저장된 Chroma 벡터 DB 내부 문서 및 메타데이터를 확인하고
    중복 제목 여부도 출력합니다.
    """
    print("📦 벡터 DB 문서 검사 중...\n")

    all_docs = vector_db.get(include=["documents", "metadatas"])
    docs = all_docs["documents"]
    metas = all_docs["metadatas"]

    title_counts = {}
    for doc, meta in zip(docs, metas):
        title = meta.get("title", "제목 없음")
        year = meta.get("year", "연도 없음")
        key = f"{title} ({year})"
        title_counts[key] = title_counts.get(key, 0) + 1

    print(f"✅ 총 문서 수: {len(docs)}")
    print(f"✅ 고유 제목 수: {len(title_counts)}\n")

    # 중복 출력
    duplicates = {k: v for k, v in title_counts.items() if v > 1}
    if duplicates:
        print("⚠️ 중복 제목들:")
        for k, v in duplicates.items():
            print(f"  - {k}: {v}개")
    else:
        print("✅ 중복 없음")

    # 예시로 처음 5개 출력
    print("\n📑 샘플 문서 5개:")
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        if i >= 5: break
        print(f"{i+1}. {meta.get('title')} ({meta.get('year')}) - {meta.get('mood_labels')}")


# ───────────────────────────────
if __name__ == "__main__":
    json_path = "./movies.json"
    persist_dir = "./movie_vectorDB"
    prepare_chroma_movie_db(json_path, persist_dir)

    print("\n🧪 벡터 DB 내용 검사:")
    vector_db = Chroma(
        persist_directory=persist_dir,
        embedding_function=HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-sts")
    )
    inspect_vector_db(vector_db)