from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.sql import func
from database import Base

class RecommendationLog(Base):
    __tablename__ = "recommendation_logs"

    id = Column(Integer, primary_key=True, index=True)
    query = Column(String, nullable=False)
    tags = Column(String, nullable=True)  # 쉼표 구분 문자열
    recommended_titles = Column(String, nullable=True)  # 쉼표 구분 문자열
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class WatchedMovie(Base):
    __tablename__ = "watched_movies"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    watched_at = Column(DateTime(timezone=True), server_default=func.now())
    from_log_id = Column(Integer, ForeignKey("recommendation_logs.id"), nullable=True)
    review = Column(String, nullable=True)
