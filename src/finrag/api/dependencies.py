from functools import lru_cache
from fastapi import Request
from finrag.config import get_settings
from finrag.core.cache import RedisCache


@lru_cache()
def get_redis_cache_instance() -> RedisCache:
    """Creates a single cached instance of RedisCache using Settings."""
    settings = get_settings()
    return RedisCache(
        redis_url=settings.redis_url,
        ttl=settings.cache_ttl_seconds,
    )


async def get_redis_cache(request: Request = None) -> RedisCache:
    """FastAPI dependency to retrieve the shared RedisCache instance.

    Reuses request.app.state.redis_cache if available (lifespan managed),
    otherwise falls back to the module-level lru_cached instance.
    """
    if request and hasattr(request.app.state, "redis_cache") and request.app.state.redis_cache is not None:
        return request.app.state.redis_cache
    return get_redis_cache_instance()
