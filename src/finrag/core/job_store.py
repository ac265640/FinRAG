import uuid
import json
from datetime import datetime, timezone
from finrag.api.dependencies import get_redis_cache_instance

class JobStore:
    """Store async ingestion job states in Redis, reusing the existing connection settings."""
    KEY_PREFIX = "finrag:job:"
    JOB_TTL = 86400  # 24 hours

    def __init__(self) -> None:
        self.redis_cache = get_redis_cache_instance()

    @property
    def client(self):
        """Underlying redis-py async client."""
        return self.redis_cache.client

    async def create_job(self, ticker: str, filing_type: str) -> str:
        """Create a new job state in Redis with initial status 'pending'."""
        job_id = str(uuid.uuid4())
        key = f"{self.KEY_PREFIX}{job_id}"
        
        job_state = {
            "job_id": job_id,
            "ticker": ticker.upper(),
            "filing_type": filing_type.upper(),
            "status": "pending",
            "progress": 0,
            "message": "Ingestion job queued",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "error": None
        }
        
        await self.client.set(key, json.dumps(job_state), ex=self.JOB_TTL)
        return job_id

    async def update_job(
        self,
        job_id: str,
        status: str,
        progress: int = 0,
        message: str = "",
        error: str | None = None
    ) -> None:
        """Update job fields and set completed_at timestamp if final status."""
        key = f"{self.KEY_PREFIX}{job_id}"
        job_data = await self.client.get(key)
        if not job_data:
            return
            
        job_state = json.loads(job_data)
        job_state["status"] = status
        job_state["progress"] = progress
        job_state["message"] = message
        job_state["error"] = error
        
        if status in ("completed", "failed"):
            job_state["completed_at"] = datetime.now(timezone.utc).isoformat()
            
        await self.client.set(key, json.dumps(job_state), ex=self.JOB_TTL)

    async def get_job(self, job_id: str) -> dict | None:
        """Retrieve the job status dictionary from Redis."""
        key = f"{self.KEY_PREFIX}{job_id}"
        job_data = await self.client.get(key)
        if not job_data:
            return None
        return json.loads(job_data)
