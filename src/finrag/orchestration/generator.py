"""LLM-powered answer generation with citation enforcement.

Replaces the Day 7 generate stub with actual LLM calls.
Uses Google Gemini via langchain-google-genai for generation,
with Pydantic structured output to guarantee the response
format matches the CitedAnswer schema.

Pipeline position:
    retrieve → rerank → [pre-decline check] → generate → enforce → validate

Design decisions:
- Google Gemini (gemini-2.0-flash) for fast, cost-effective generation.
  Flash model is sufficient for grounded extraction — creativity doesn't
  help when the answer must come from the provided context.
- System prompt is explicit about citation rules: every claim must
  cite a chunk_id from the provided context. No external knowledge.
- Context formatting: each chunk is wrapped with its chunk_id and
  metadata, so the LLM can reference them by ID.
- Retry once with stricter prompt if citation enforcement fails.
  Second attempt uses a "correction" prompt that includes the
  specific enforcement errors from the first attempt.
- Pre-generation decline: if context quality is too poor (below
  relevance floor), we skip the LLM call entirely to save cost.

Debt: DAY-8-002 — Using gemini-2.0-flash for all queries. Harder
      queries (multi-hop, contradiction) may need gemini-pro.
      Evaluate on Day 13 with golden dataset.
"""

import os

import structlog
from langchain_google_genai import ChatGoogleGenerativeAI

from finrag.orchestration.citation import CitationEnforcer
from finrag.orchestration.schemas import CitedAnswer, build_filing_reference

logger = structlog.get_logger(__name__)


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# gemini-2.5-flash: highly capable and fast, has much higher free tier limits (1,500 requests/day).
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_TEMPERATURE = 0.1  # Low temp for factual extraction
DEFAULT_MAX_RETRIES = 1  # One retry with stricter prompt

# Fallback model order on RESOURCE_EXHAUSTED
MODEL_FALLBACKS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
]


# System prompt for citation-grounded generation
SYSTEM_PROMPT = """You are a financial research assistant that answers questions
using ONLY the provided context from SEC filings and earnings call transcripts.

CRITICAL RULES:
1. ONLY use information from the provided context chunks. Never use external knowledge.
2. Every factual claim MUST cite the specific chunk_id it comes from.
3. If the context does not contain enough information to answer, say so explicitly.
4. Never provide investment advice, stock recommendations, or market predictions.
5. Be precise with numbers — quote them exactly as they appear in the source.

Your response must be a JSON object with these fields:
- answer_text: Your answer in clear, professional language.
- citations: List of objects, each with:
  - chunk_id: The exact chunk_id from the context that supports this claim.
  - filing_reference: Human-readable source (e.g., "AAPL 10-K FY2024, Item 7").
  - section: The section name from the filing.
  - text_excerpt: A short quote (max 200 chars) from the source.
  - relevance_score: How relevant this source is (0.0 to 1.0).
- confidence: Your confidence that the answer is correct and fully supported (0.0 to 1.0).
- reasoning: Brief explanation of how you derived the answer from the sources."""

# Summary-optimized prompt: structures output as an executive summary
SUMMARY_SYSTEM_PROMPT = """You are a financial research analyst that creates structured
executive summaries of SEC filings using ONLY the provided context chunks.

CRITICAL RULES:
1. ONLY use information from the provided context chunks. Never use external knowledge.
2. Every factual claim MUST cite the specific chunk_id it comes from using [chunk_id].
3. Be precise with numbers — quote exact figures from the source.
4. Never provide investment advice or stock recommendations.
5. Structure your summary clearly with the following sections (include only if context covers it):
   • Business Overview — what the company does, key products/segments
   • Financial Highlights — revenue, net income, EPS, key metrics
   • Operational Performance — segment results, growth areas
   • Key Risks — major risk factors disclosed
   • Management Outlook — guidance, forward-looking statements

Your response must be a JSON object with these fields:
- answer_text: A structured executive summary using the sections above. Use markdown headers (##) for each section. Include cited numbers with [chunk_id] references.
- citations: List of ALL sources used, each with:
  - chunk_id: The exact chunk_id from the context.
  - filing_reference: Human-readable source.
  - section: The section name.
  - text_excerpt: A short quote (max 200 chars).
  - relevance_score: Relevance score (0.0 to 1.0).
- confidence: Overall confidence in completeness of summary (0.0 to 1.0).
- reasoning: Note which sections you could/could not cover based on the context."""

# Stricter prompt for retry after citation failure
RETRY_PROMPT_SUFFIX = """

IMPORTANT CORRECTION: Your previous answer had citation errors:
{errors}

Please fix these issues:
- Only cite chunk_ids that exist in the provided context.
- Ensure every factual claim has a supporting citation.
- If you cannot find supporting evidence, lower your confidence score.
- Do NOT fabricate or hallucinate chunk_ids."""


# --------------------------------------------------------------------------- #
# Context Formatter
# --------------------------------------------------------------------------- #


def format_context_for_llm(chunks: list[dict]) -> str:
    """Format reranked chunks as numbered context for the LLM.

    Each chunk is wrapped with its chunk_id and metadata so the
    LLM can reference it by ID in its citations.

    Args:
        chunks: Reranked chunks with chunk_id, text, and metadata.

    Returns:
        Formatted context string ready for the LLM prompt.
    """
    parts: list[str] = []

    for i, chunk in enumerate(chunks, start=1):
        chunk_id = chunk.get("chunk_id", f"chunk_{i}")
        text = chunk.get("text", "")
        metadata = chunk.get("metadata", {})
        score = chunk.get("reranker_score", 0.0)
        filing_ref = build_filing_reference(metadata)

        header = f"[CHUNK {i}] chunk_id: {chunk_id}"
        if filing_ref:
            header += f" | source: {filing_ref}"
        header += f" | relevance: {score:.2f}"

        parts.append(f"{header}\n{text}")

    return "\n\n---\n\n".join(parts)


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #


class RAGGenerator:
    """LLM-powered answer generator with citation enforcement.

    Generates answers using Google Gemini, validates citations
    against the retrieved context, and retries once with a
    stricter prompt if enforcement fails.

    Args:
        model_name: Gemini model name (default gemini-2.0-flash).
        temperature: Generation temperature (default 0.1).
        max_retries: Max retry attempts on enforcement failure.
        citation_enforcer: Optional custom CitationEnforcer instance.
        api_key: Optional Google API key (falls back to env var).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_retries: int = DEFAULT_MAX_RETRIES,
        citation_enforcer: CitationEnforcer | None = None,
        api_key: str | None = None,
    ) -> None:
        """Initialize the RAG generator.

        Args:
            model_name: Gemini model identifier.
            temperature: Sampling temperature (0-1).
            max_retries: Retry count on citation failure.
            citation_enforcer: Custom enforcer (default creates one).
            api_key: Google API key (default: GOOGLE_API_KEY env var).
        """
        self._model_name = model_name
        self._temperature = temperature
        self._max_retries = max_retries
        self._enforcer = citation_enforcer or CitationEnforcer()

        # Resolve API key
        resolved_key = api_key or os.environ.get("GOOGLE_API_KEY", "")

        # Resolve all available keys for fallback rotation
        self._keys = {
            "gemini_primary": resolved_key,
            "gemini_secondary": os.environ.get("GOOGLE_API_KEY_SECONDARY", "").strip(),
            "cohere": os.environ.get("COHERE_API_KEY", "").strip(),
            "openai": os.environ.get("OPENAI_API_KEY", "").strip(),
        }

        # Lazy-init the LLM (defer to first call to allow testing without key)
        self._api_key = resolved_key
        self._llm: ChatGoogleGenerativeAI | None = None

        logger.info(
            "generator_initialized",
            model=model_name,
            temperature=temperature,
            max_retries=max_retries,
            has_gemini_primary=bool(self._keys["gemini_primary"]),
            has_gemini_secondary=bool(self._keys["gemini_secondary"]),
            has_cohere=bool(self._keys["cohere"]),
            has_openai=bool(self._keys["openai"]),
        )


    def _get_llm(self) -> ChatGoogleGenerativeAI:
        """Lazy-load the LLM client.

        Returns:
            Initialized ChatGoogleGenerativeAI instance.

        Raises:
            ValueError: If no API key is configured.
        """
        if self._llm is None:
            if not self._api_key:
                msg = (
                    "No Google API key configured. Set GOOGLE_API_KEY "
                    "environment variable or pass api_key to RAGGenerator."
                )
                raise ValueError(msg)

            self._llm = ChatGoogleGenerativeAI(
                model=self._model_name,
                temperature=self._temperature,
                google_api_key=self._api_key,
            )
            logger.info("llm_client_initialized", model=self._model_name)
        return self._llm

    def generate(
        self,
        query: str,
        context_chunks: list[dict],
        query_intent: str = "factual_extraction",
    ) -> tuple[CitedAnswer, bool, list[str]]:
        """Generate a cited answer from context chunks.

        Workflow:
        1. Pre-decline check: skip LLM if context quality too poor
        2. Format context and call LLM with structured output
        3. Parse response into CitedAnswer
        4. Run citation enforcement
        5. If enforcement fails, retry once with stricter prompt
        6. Return final answer + enforcement status

        Args:
            query: User's question.
            context_chunks: Reranked chunks with chunk_id, text, metadata.
            query_intent: Intent tag from router (e.g. "summarize").

        Returns:
            Tuple of:
            - CitedAnswer: The structured answer
            - bool: Whether citation enforcement passed
            - list[str]: Any enforcement errors
        """
        # Pre-decline check
        should_decline, decline_reason = self._enforcer.should_decline(context_chunks)
        if should_decline:
            logger.info("pre_generation_decline", reason=decline_reason)
            return (
                CitedAnswer(
                    answer_text=decline_reason,
                    citations=[],
                    confidence=0.0,
                    reasoning="Declined due to insufficient context quality.",
                ),
                False,
                [decline_reason],
            )

        # Format context
        context_str = format_context_for_llm(context_chunks)

        # First attempt
        answer = self._call_llm(query, context_str, query_intent=query_intent)
        enforcement = self._enforcer.enforce(answer, context_chunks)

        if enforcement.is_valid:
            logger.info(
                "generation_accepted_first_attempt",
                confidence=answer.confidence,
                citations=len(answer.citations),
            )
            return answer, True, []

        # Retry with stricter prompt
        if self._max_retries > 0:
            logger.warning(
                "generation_retry",
                errors=enforcement.errors,
                attempt=2,
            )
            error_str = "\n".join(f"- {e}" for e in enforcement.errors)
            answer = self._call_llm(query, context_str, retry_errors=error_str, query_intent=query_intent)
            enforcement = self._enforcer.enforce(answer, context_chunks)

            if enforcement.is_valid:
                logger.info(
                    "generation_accepted_retry",
                    confidence=answer.confidence,
                    citations=len(answer.citations),
                )
                return answer, True, []

        # Both attempts failed — but if we have a real answer, still return it.
        # The node will mark enforcement_passed=False so the frontend can show
        # a lower confidence, but we don't silently blank the answer.
        logger.warning(
            "generation_enforcement_failed",
            errors=enforcement.errors,
            hallucinated_ids=enforcement.hallucinated_ids,
            has_answer=bool(answer.answer_text.strip()),
        )

        # If the answer is non-empty and not an error message, keep it.
        # Only replace with a decline if the answer itself is blank/error.
        error_indicators = (
            "generation failed",
            "rate-limited",
            "api error",
            "missing api key",
        )
        answer_lower = answer.answer_text.lower()
        is_error_answer = any(ind in answer_lower for ind in error_indicators)

        if answer.answer_text.strip() and not is_error_answer:
            # Real answer exists — return it with enforcement flag
            logger.info(
                "generation_returning_despite_enforcement_failure",
                answer_len=len(answer.answer_text),
                citations=len(answer.citations),
            )
            return answer, False, enforcement.errors

        return answer, False, enforcement.errors

    def _call_llm(
        self,
        query: str,
        context: str,
        retry_errors: str | None = None,
        query_intent: str = "factual_extraction",
    ) -> CitedAnswer:
        """Call the LLM with automatic multi-provider fallback.

        Tries each available provider in priority order:
          1. Gemini primary key
          2. Gemini secondary key
          3. Cohere (command-r-plus)
          4. OpenAI (gpt-4o-mini)

        Uses direct curl calls — no extra SDK dependencies required.

        Args:
            query: User's question.
            context: Formatted context string.
            retry_errors: If set, appends retry correction prompt.
            query_intent: Router intent tag. "summarize" uses a different system prompt.

        Returns:
            Parsed CitedAnswer from the first successful provider.
        """
        import json
        import subprocess

        # Priority-ordered list of (name, api_key, model_id)
        providers = [
            ("gemini_primary", self._keys["gemini_primary"], self._model_name),
            ("gemini_secondary", self._keys["gemini_secondary"], self._model_name),
            ("cohere", self._keys["cohere"], "command-a-03-2025"),
            ("openai", self._keys["openai"], "gpt-4o-mini"),
        ]

        # Build prompt
        system = SUMMARY_SYSTEM_PROMPT if query_intent == "summarize" else SYSTEM_PROMPT
        if retry_errors:
            system += RETRY_PROMPT_SUFFIX.format(errors=retry_errors)

        user_message = (
            f"QUESTION: {query}\n\nCONTEXT:\n{context}\n\n"
            "Generate a cited answer using ONLY the context above. "
            "RETURN ONLY VALID JSON MATCHING THE SCHEMA."
        )

        last_error = "No providers configured."

        for name, key, model in providers:
            if not key:
                logger.debug("provider_skipped_no_key", provider=name)
                continue

            logger.info("llm_provider_attempt", provider=name, model=model)
            try:
                json_text = self._invoke_provider(
                    name=name,
                    key=key,
                    model=model,
                    system=system,
                    user_message=user_message,
                    subprocess_module=subprocess,
                    json_module=json,
                )
                json_text = json_text.strip().replace("```json", "").replace("```", "").strip()
                answer = CitedAnswer.model_validate(json.loads(json_text, strict=False))
                logger.info("llm_provider_success", provider=name, confidence=answer.confidence)
                return answer

            except Exception as e:
                last_error = str(e)
                is_quota = "quota" in last_error.lower() or "resource_exhausted" in last_error.lower() or "429" in last_error
                logger.warning("llm_provider_failed", provider=name, error=last_error, is_quota=is_quota)
                # Always fall through to the next provider on any failure

        # All providers exhausted
        logger.error("all_providers_exhausted", last_error=last_error)
        return CitedAnswer(
            answer_text=(
                "The AI model is temporarily unavailable across all configured providers. "
                "Please wait a few minutes and try again."
            ),
            citations=[],
            confidence=0.0,
            reasoning=f"All providers failed. Last error: {last_error}",
        )

    def _invoke_provider(
        self,
        name: str,
        key: str,
        model: str,
        system: str,
        user_message: str,
        subprocess_module,
        json_module,
    ) -> str:
        """Invoke a single LLM provider and return the raw JSON text response.

        Args:
            name: Provider identifier ('gemini_primary', 'gemini_secondary', 'cohere', 'openai').
            key: API key for this provider.
            model: Model identifier string.
            system: System prompt text.
            user_message: User message text.
            subprocess_module: Injected subprocess module.
            json_module: Injected json module.

        Returns:
            Raw JSON string from the model's response.

        Raises:
            ValueError: If the provider returns an error or unexpected structure.
            RuntimeError: If the curl subprocess fails.
        """
        if name.startswith("gemini"):
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model}:generateContent?key={key}"
            )
            payload = {
                "system_instruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user_message}]}],
                "generationConfig": {
                    "temperature": self._temperature,
                    "response_mime_type": "application/json",
                },
            }
            result = subprocess_module.run(
                ["curl", "-s", "-X", "POST", url,
                 "-H", "Content-Type: application/json",
                 "-d", json_module.dumps(payload)],
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode != 0:
                raise RuntimeError(f"curl failed: {result.stderr}")
            data = json_module.loads(result.stdout)
            if "error" in data:
                err = data["error"]
                raise ValueError(f"{err.get('status', 'API_ERROR')}: {err.get('message', 'Unknown error')}")
            if not data.get("candidates"):
                raise ValueError(f"No candidates in response: {result.stdout[:200]}")
            return data["candidates"][0]["content"]["parts"][0]["text"]

        elif name == "cohere":
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_message},
                ],
                "response_format": {"type": "json_object"},
                "temperature": self._temperature,
            }
            result = subprocess_module.run(
                ["curl", "-s", "-X", "POST", "https://api.cohere.com/v2/chat",
                 "-H", f"Authorization: Bearer {key}",
                 "-H", "Content-Type: application/json",
                 "-d", json_module.dumps(payload)],
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode != 0:
                raise RuntimeError(f"curl failed: {result.stderr}")
            data = json_module.loads(result.stdout)
            if "message" not in data:
                raise ValueError(f"Cohere unexpected response: {result.stdout[:200]}")
            # data["message"] is a string when the API returns an error
            if isinstance(data["message"], str):
                raise ValueError(f"Cohere API error: {data['message']}")
            return data["message"]["content"][0]["text"]

        elif name == "openai":
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_message},
                ],
                "response_format": {"type": "json_object"},
                "temperature": self._temperature,
            }
            result = subprocess_module.run(
                ["curl", "-s", "-X", "POST", "https://api.openai.com/v1/chat/completions",
                 "-H", f"Authorization: Bearer {key}",
                 "-H", "Content-Type: application/json",
                 "-d", json_module.dumps(payload)],
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode != 0:
                raise RuntimeError(f"curl failed: {result.stderr}")
            data = json_module.loads(result.stdout)
            if "error" in data:
                raise ValueError(f"OpenAI error: {data['error'].get('message', 'Unknown')}")
            if not data.get("choices"):
                raise ValueError(f"OpenAI no choices: {result.stdout[:200]}")
            return data["choices"][0]["message"]["content"]

        raise ValueError(f"Unknown provider: {name}")
