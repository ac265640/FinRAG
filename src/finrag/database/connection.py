import os
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

DATABASE_URL = os.getenv("DATABASE_URL", "")

def get_engine_args(url: str):
    import urllib.parse
    connect_args = {}
    if not url:
        return url, connect_args
    
    if "sslmode=" in url:
        parsed = urllib.parse.urlparse(url)
        query_params = urllib.parse.parse_qs(parsed.query)
        query_params.pop("sslmode", None)
        new_query = urllib.parse.urlencode(query_params, doseq=True)
        parsed = parsed._replace(query=new_query)
        url = urllib.parse.urlunparse(parsed)
        connect_args["ssl"] = True
    elif "neon.tech" in url:
        connect_args["ssl"] = True
        
    return url, connect_args

cleaned_url, connect_args = get_engine_args(DATABASE_URL)

engine = create_async_engine(
    cleaned_url,
    connect_args=connect_args,
    pool_size=5,  # max 5 connections (free tier friendly)
    max_overflow=10,
    pool_pre_ping=True,  # test connections before using them
    echo=False,  # set True only for debugging
)

AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_db():
    """FastAPI dependency for database sessions."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
