from datetime import datetime, timezone
from sqlalchemy import Boolean, Column, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for SQLAlchemy declarative models."""

    pass


class QueryLog(Base):
    """SQLAlchemy model for query execution logging."""

    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Identity (hashed — never store raw API keys)
    api_key_hash = Column(String(64), nullable=False, index=True)

    # Query details
    query_text = Column(Text, nullable=False)
    ticker = Column(String(10), nullable=True, index=True)
    filing_type = Column(String(10), nullable=True)
    fiscal_period = Column(String(20), nullable=True)

    # Result metadata
    confidence_score = Column(Float, nullable=True)
    response_time_ms = Column(Integer, nullable=True)
    was_declined = Column(Boolean, default=False)
    cache_hit = Column(Boolean, default=False)
    chunk_count = Column(Integer, nullable=True)

    # Timestamp
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )

    # Composite index for analytics queries
    __table_args__ = (Index("ix_query_logs_ticker_created", "ticker", "created_at"),)


class Session(Base):
    """SQLAlchemy model for session analytics and usage tracking."""

    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    api_key_hash = Column(String(64), nullable=False, unique=True, index=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    last_seen_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    total_queries = Column(Integer, default=0)
