from datetime import datetime, timezone, timedelta
from sqlalchemy import Integer, desc, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession
from finrag.database.models import QueryLog, Session


class QueryLogRepository:
    """Repository for managing QueryLog entities."""

    async def create_log(
        self,
        session: AsyncSession,
        api_key_hash: str,
        query_text: str,
        ticker: str | None,
        filing_type: str | None,
        fiscal_period: str | None,
        confidence_score: float | None,
        response_time_ms: int,
        was_declined: bool,
        cache_hit: bool,
        chunk_count: int | None,
    ) -> QueryLog:
        """Insert a new query log record."""
        log = QueryLog(
            api_key_hash=api_key_hash,
            query_text=query_text,
            ticker=ticker,
            filing_type=filing_type,
            fiscal_period=fiscal_period,
            confidence_score=confidence_score,
            response_time_ms=response_time_ms,
            was_declined=was_declined,
            cache_hit=cache_hit,
            chunk_count=chunk_count,
        )
        session.add(log)
        await session.flush()
        return log

    async def get_analytics(
        self,
        session: AsyncSession,
        ticker: str | None = None,
        days: int = 7,
    ) -> dict:
        """Returns aggregated query metrics and analytics.

        Returns:
            Aggregated dictionary matching the expected structure.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        # Aggregate query
        stmt = select(
            func.count(QueryLog.id).label("total"),
            func.sum(func.cast(QueryLog.cache_hit, Integer)).label("cache_hits"),
            func.sum(func.cast(QueryLog.was_declined, Integer)).label("declines"),
            func.avg(QueryLog.response_time_ms).label("avg_latency"),
        ).where(QueryLog.created_at >= cutoff)

        if ticker:
            stmt = stmt.where(QueryLog.ticker == ticker)

        agg_res = await session.execute(stmt)
        total, cache_hits, declines, avg_latency = agg_res.fetchone()

        total = total or 0
        cache_hits = cache_hits or 0
        declines = declines or 0
        avg_latency = float(avg_latency) if avg_latency is not None else 0.0

        # Top tickers query
        ticker_stmt = (
            select(QueryLog.ticker, func.count(QueryLog.id).label("count"))
            .where(QueryLog.created_at >= cutoff, QueryLog.ticker.is_not(None))
            .group_by(QueryLog.ticker)
            .order_by(desc("count"))
            .limit(5)
        )

        ticker_res = await session.execute(ticker_stmt)
        top_tickers = [{"ticker": t, "count": c} for t, c in ticker_res.all()]

        return {
            "period_days": days,
            "total_queries": total,
            "cache_hit_rate": round(cache_hits / total, 2) if total > 0 else 0.0,
            "avg_response_time_ms": round(avg_latency, 2),
            "decline_rate": round(declines / total, 2) if total > 0 else 0.0,
            "top_tickers": top_tickers,
        }


class SessionRepository:
    """Repository for managing Session entities."""

    async def upsert_session(self, session: AsyncSession, api_key_hash: str) -> None:
        """Upserts a session tracking record.

        Creates session if not exists. Updates last_seen_at and increments
        total_queries if exists using native Postgres ON CONFLICT DO UPDATE.
        """
        now = datetime.now(timezone.utc)
        stmt = insert(Session).values(
            api_key_hash=api_key_hash,
            total_queries=1,
            created_at=now,
            last_seen_at=now,
        )

        stmt = stmt.on_conflict_do_update(
            index_elements=["api_key_hash"],
            set_={
                "total_queries": Session.total_queries + 1,
                "last_seen_at": now,
            },
        )

        await session.execute(stmt)
