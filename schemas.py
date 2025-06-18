from pydantic import BaseModel
from datetime import datetime
from typing import List

class RecommendRequest(BaseModel):
    message: str

class RecommendResponse(BaseModel):
    reply: str

class RecommendationLogSchema(BaseModel):
    id: int
    query: str
    tags: str
    recommended_titles: str
    created_at: datetime

    class Config:
        orm_mode = True

class WatchedMovieCreate(BaseModel):
    title: str
    from_log_id: int | None = None  # 로그 기반 확정

class WatchedMovieSchema(BaseModel):
    id: int
    title: str
    watched_at: datetime
    from_log_id: int | None

    class Config:
        orm_mode = True
        
class ReviewResponse(BaseModel):
    exists: bool
    content: str | None = None
    
class ReviewRequest(BaseModel):
    movie_id: int
    review: str
    
class MovieDetailResponse(BaseModel):
    id: int
    title: str
    overview: str
    release_year: str
    mood_labels: List[str]
    poster_path: str