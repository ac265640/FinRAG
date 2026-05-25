import hashlib
import json
import redis.asyncio as redis
import structlog

logger = structlog.get_logger(__name__)


class RedisCache:
    """Redis cache layer for caching RAG query responses."""

    def __init__(self, redis_url: str, ttl: int = 3600) -> None:
        """Initialize Redis connection settings.

        Args:
            redis_url: Connection string.
            ttl: Key expiration time in seconds.
        """
        self.redis_url = redis_url
        self.ttl = ttl
        self._client = None

    @property
    def client(self):
        """Lazy initialization of the Redis client."""
        if self._client is None:
            self._client = redis.from_url(self.redis_url, decode_responses=True)
        return self._client

    def _make_key(self, query: str, filters: dict | None) -> str:
        """Creates a deterministic cache key.

        Sorts filter keys before hashing to handle different ordering
        of filter dictionary keys as identical. Use SHA256 of the
        serialized string, prefixed with a namespace.

        Args:
            query: The user query string.
            filters: Query metadata filters.

        Returns:
            Namespace-prefixed cache key.
        """
        filters_str = ""
        if filters:
            sorted_filters = {k: filters[k] for k in sorted(filters.keys())}
            filters_str = json.dumps(sorted_filters, sort_keys=True)

        combined = f"query:{query.strip()}|filters:{filters_str}"
        hasher = hashlib.sha256(combined.encode("utf-8"))
        hashed = hasher.hexdigest()
        return f"finrag:query:{hashed}"

    async def get(self, query: str, filters: dict | None) -> dict | None:
        """Returns cached response dict if exists, None if miss.

        Never raises — on Redis failure, return None and let the
        pipeline run normally.

        Args:
            query: The user query string.
            filters: Query metadata filters.

        Returns:
            The cached response dict or None.
        """
        try:
            key = self._make_key(query, filters)
            cached_val = await self.client.get(key)
            if cached_val:
                logger.info("cache_hit", key=key)
                return json.loads(cached_val)
            logger.info("cache_miss", key=key)
            return None
        except Exception as e:
            logger.error("cache_get_error", error=str(e))
            return None

    async def set(self, query: str, filters: dict | None, response: dict) -> None:
        """Stores query response.

        Never raises — on Redis failure, logs error and proceeds.

        Args:
            query: The user query string.
            filters: Query metadata filters.
            response: The response dictionary to store.
        """
        try:
            key = self._make_key(query, filters)
            serialized = json.dumps(response)
            await self.client.set(key, serialized, ex=self.ttl)
            logger.info("cache_set_success", key=key, ttl=self.ttl)
        except Exception as e:
            logger.error("cache_set_error", error=str(e))

    async def invalidate_ticker(self, ticker: str) -> int:
        """Deletes all cached keys containing a specific ticker.

        Returns count of deleted keys. Uses SCAN + DEL pattern
        to avoid blocking Redis.

        Args:
            ticker: The ticker symbol to invalidate.

        Returns:
            Count of deleted keys.
        """
        try:
            deleted_count = 0
            cursor = 0
            match_pattern = "finrag:query:*"
            ticker_upper = ticker.upper()

            while True:
                cursor, keys = await self.client.scan(cursor=cursor, match=match_pattern, count=100)
                if not keys:
                    if cursor == 0:
                        break
                    continue

                keys_to_delete = []
                for key in keys:
                    val = await self.client.get(key)
                    if val and ticker_upper in val.upper():
                        keys_to_delete.append(key)

                if keys_to_delete:
                    await self.client.delete(*keys_to_delete)
                    deleted_count += len(keys_to_delete)

                if cursor == 0:
                    break

            logger.info("cache_invalidate_ticker", ticker=ticker, deleted_count=deleted_count)
            return deleted_count
        except Exception as e:
            logger.error("cache_invalidate_ticker_error", ticker=ticker, error=str(e))
            return 0

    async def health_check(self) -> bool:
        """Pings Redis. Used in the /healthz endpoint.

        Returns:
            True if Redis is online and reachable, False otherwise.
        """
        try:
            return await self.client.ping()
        except Exception as e:
            logger.error("cache_health_check_error", error=str(e))
            return False
