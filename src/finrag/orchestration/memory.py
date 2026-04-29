"""Conversation memory for multi-turn financial research sessions.

Tracks discussed entities, filings, prior answers, and context
across turns. Enables follow-up questions like "how does that
compare to last quarter?" by maintaining a structured session state.

Why session memory matters for financial RAG:
- Follow-up questions are the norm in financial research. Analysts
  don't ask isolated questions — they drill down: "what was revenue?"
  → "how does that compare to last year?" → "what drove the change?"
- Without memory, each query is independent. The LLM can't resolve
  "it", "the company", "last quarter" without prior context.
- Tracking discussed filings prevents re-retrieving the same chunks
  and enables cross-filing comparisons.

Design decisions:
- In-memory session store: no external database required. Sessions
  live as long as the server process (or API session).
- Entity extraction is keyword-based (ticker symbols, filing types).
  An LLM-based entity extractor would be more accurate but adds
  200ms+ per turn. Good enough for now.
- Session has a max_turns limit (default 20) to prevent unbounded
  memory growth. Old turns are summarized, not deleted.
- Memory is injected into the system prompt via the conversation
  context template from the prompt config.

Debt: DAY-10-001 — Entity extraction is keyword-based. LLM-based
      extraction would catch implicit entity references ("the
      Cupertino company" → AAPL). Add on Day 14.
"""

import re
import time
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger(__name__)


# --------------------------------------------------------------------------- #
# Known Financial Entities
# --------------------------------------------------------------------------- #

# Common ticker patterns (all-caps, 1-5 chars)
TICKER_PATTERN = re.compile(r"\b([A-Z]{1,5})\b")

# Filing type patterns
FILING_TYPES = {"10-K", "10-Q", "8-K", "DEF 14A", "S-1"}

# Time period patterns
PERIOD_PATTERN = re.compile(
    r"\b(FY\d{4}|Q[1-4]\s*\d{4}|fiscal\s+year\s+\d{4}|"
    r"quarter\s+\d|first\s+quarter|second\s+quarter|"
    r"third\s+quarter|fourth\s+quarter|annual|quarterly)\b",
    re.IGNORECASE,
)

# Common non-ticker words that match the ticker pattern
TICKER_STOPWORDS = {
    "A", "I", "AM", "AN", "AS", "AT", "BE", "BY", "DO", "GO",
    "IF", "IN", "IS", "IT", "MY", "NO", "OF", "ON", "OR", "SO",
    "TO", "UP", "US", "WE", "THE", "AND", "FOR", "NOT", "BUT",
    "YOU", "ALL", "CAN", "HAD", "HER", "WAS", "ONE", "OUR",
    "OUT", "HAS", "HIS", "HOW", "ITS", "MAY", "NEW", "NOW",
    "OLD", "SEE", "WAY", "WHO", "DID", "GET", "HIM", "LET",
    "SAY", "SHE", "TOO", "USE", "SEC", "CEO", "CFO", "COO",
    "CTO", "VP", "SVP", "EVP", "MD", "YOY", "QOQ", "EPS",
    "PE", "PB", "ROE", "ROA", "EBITDA", "IPO", "GDP", "CPI",
    "WHAT", "WHEN", "HOW", "WHY", "MUCH", "MANY", "DOES",
    "WERE", "BEEN", "HAVE", "FROM", "THIS", "THAT", "THAN",
    "WITH", "WILL", "OVER", "ALSO", "MORE", "SOME", "VERY",
    "JUST", "LIKE", "MAKE", "LAST", "YEAR", "EACH", "SAME",
    "BOTH", "MOST", "ONLY", "SUCH", "THEM", "THEN", "WHAT",
    "RISK", "ITEM", "TOTAL", "GROSS", "WHICH", "ABOUT",
    "COULD", "THEIR", "WOULD", "THERE", "OTHER", "AFTER",
    "FIRST", "THOSE", "THESE", "BEING", "WHILE", "WHERE",
    "BASED", "SINCE",
}


# --------------------------------------------------------------------------- #
# Turn Record
# --------------------------------------------------------------------------- #


@dataclass
class TurnRecord:
    """A single conversation turn (question + answer).

    Attributes:
        query: The user's question.
        answer: The generated answer.
        entities: Entities mentioned in this turn (tickers, etc).
        filings: Filing types referenced.
        periods: Time periods mentioned.
        citations: Citation chunk_ids used.
        timestamp: When this turn occurred (Unix timestamp).
        metadata_filter: Any metadata filter applied.
    """

    query: str
    answer: str = ""
    entities: list[str] = field(default_factory=list)
    filings: list[str] = field(default_factory=list)
    periods: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    metadata_filter: dict | None = None


# --------------------------------------------------------------------------- #
# Session Memory
# --------------------------------------------------------------------------- #


@dataclass
class SessionMemory:
    """In-memory conversation session for multi-turn research.

    Tracks all entities, filings, and answers discussed across turns.
    Provides context for follow-up question resolution and prompt
    injection into the conversation-aware system prompt.

    Attributes:
        session_id: Unique session identifier.
        turns: List of conversation turns.
        all_entities: Accumulated entity set across all turns.
        all_filings: Accumulated filing type set.
        all_periods: Accumulated time periods.
        all_cited_chunks: Set of all cited chunk_ids.
        max_turns: Maximum turns before summarization.
        created_at: Session creation timestamp.
    """

    session_id: str = ""
    turns: list[TurnRecord] = field(default_factory=list)
    all_entities: set[str] = field(default_factory=set)
    all_filings: set[str] = field(default_factory=set)
    all_periods: set[str] = field(default_factory=set)
    all_cited_chunks: set[str] = field(default_factory=set)
    max_turns: int = 20
    created_at: float = field(default_factory=time.time)

    @property
    def turn_count(self) -> int:
        """Number of turns in this session."""
        return len(self.turns)

    @property
    def is_first_turn(self) -> bool:
        """Whether this is the first turn."""
        return len(self.turns) == 0

    @property
    def last_turn(self) -> TurnRecord | None:
        """The most recent turn, if any."""
        return self.turns[-1] if self.turns else None

    @property
    def last_query(self) -> str:
        """The last query asked."""
        return self.turns[-1].query if self.turns else ""

    @property
    def last_answer(self) -> str:
        """The last answer given."""
        return self.turns[-1].answer if self.turns else ""

    def add_turn(
        self,
        query: str,
        answer: str = "",
        citations: list[dict] | None = None,
        metadata_filter: dict | None = None,
    ) -> TurnRecord:
        """Record a conversation turn.

        Extracts entities, filings, and periods from the query
        and answer, then adds them to the session-level accumulators.

        Args:
            query: The user's question.
            answer: The generated answer.
            citations: List of citation dicts with chunk_id.
            metadata_filter: Metadata filter applied for this turn.

        Returns:
            The created TurnRecord.
        """
        # Extract entities from query + answer
        combined_text = f"{query} {answer}"
        entities = extract_entities(combined_text)
        filings = extract_filings(combined_text)
        periods = extract_periods(combined_text)
        cited_ids = [c.get("chunk_id", "") for c in (citations or []) if c.get("chunk_id")]

        turn = TurnRecord(
            query=query,
            answer=answer,
            entities=entities,
            filings=filings,
            periods=periods,
            citations=cited_ids,
            metadata_filter=metadata_filter,
        )

        self.turns.append(turn)

        # Accumulate across turns
        self.all_entities.update(entities)
        self.all_filings.update(filings)
        self.all_periods.update(periods)
        self.all_cited_chunks.update(cited_ids)

        # Trim if over max_turns
        if len(self.turns) > self.max_turns:
            self._summarize_old_turns()

        logger.info(
            "turn_recorded",
            session_id=self.session_id,
            turn_number=self.turn_count,
            entities=entities,
            filings=filings,
            periods=periods,
            cited_chunks=len(cited_ids),
        )

        return turn

    def get_context_for_prompt(self) -> dict:
        """Build context dict for prompt template injection.

        Returns a dict with keys matching the conversation_context_template
        placeholders in the generation prompt config.

        Returns:
            Dict with entities, filings, turn_count, and recent Q&A.
        """
        return {
            "entities": ", ".join(sorted(self.all_entities)) or "none yet",
            "filings": ", ".join(sorted(self.all_filings)) or "none yet",
            "turn_count": str(self.turn_count),
            "recent_qa": self._format_recent_qa(max_turns=3),
        }

    def get_conversation_history(self, max_turns: int = 5) -> list[dict]:
        """Get recent conversation history as message dicts.

        Formats recent turns as a list of user/assistant message pairs
        suitable for passing to the LLM as conversation history.

        Args:
            max_turns: Maximum number of recent turns to include.

        Returns:
            List of message dicts with 'role' and 'content' keys.
        """
        history: list[dict] = []
        recent = self.turns[-max_turns:] if max_turns else self.turns

        for turn in recent:
            history.append({"role": "user", "content": turn.query})
            if turn.answer:
                # Truncate long answers for context window efficiency
                answer_preview = turn.answer[:500]
                if len(turn.answer) > 500:
                    answer_preview += "..."
                history.append({"role": "assistant", "content": answer_preview})

        return history

    def resolve_references(self, query: str) -> str:
        """Resolve ambiguous references in a follow-up query.

        Checks for pronouns and implicit references that can be
        resolved from session context. Returns the query with
        resolved references appended as context.

        Currently handles:
        - "the company" / "it" / "they" → last discussed entity
        - "last quarter" → last discussed period
        - "that filing" → last discussed filing type

        Args:
            query: The user's query with potential references.

        Returns:
            Query with optional context suffix for disambiguation.
        """
        if self.is_first_turn:
            return query

        # Check for ambiguous references
        ambiguous_patterns = [
            (r"\b(the company|the firm|they|their|them|it|its)\b", "entity"),
            (r"\b(last quarter|previous quarter|prior period|that quarter)\b", "period"),
            (r"\b(that filing|the filing|the report|that report)\b", "filing"),
        ]

        context_parts: list[str] = []

        for pattern, ref_type in ambiguous_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                if ref_type == "entity" and self.all_entities:
                    # Use the most recently discussed entity
                    last_entity = self.turns[-1].entities[-1] if self.turns[-1].entities else None
                    if last_entity:
                        context_parts.append(
                            f"(Note: '{ref_type}' likely refers to {last_entity})"
                        )
                elif ref_type == "period" and self.all_periods:
                    last_period = list(self.all_periods)[-1]
                    context_parts.append(
                        f"(Note: time reference likely refers to {last_period})"
                    )
                elif ref_type == "filing" and self.all_filings:
                    last_filing = list(self.all_filings)[-1]
                    context_parts.append(
                        f"(Note: filing reference likely refers to {last_filing})"
                    )

        if context_parts:
            resolved = f"{query}\n\n{'  '.join(context_parts)}"
            logger.info(
                "references_resolved",
                original_query=query[:80],
                context_added=len(context_parts),
            )
            return resolved

        return query

    def _format_recent_qa(self, max_turns: int = 3) -> str:
        """Format recent Q&A pairs as text.

        Args:
            max_turns: Maximum number of recent turns to include.

        Returns:
            Formatted string of recent Q&A pairs.
        """
        if not self.turns:
            return "No prior questions."

        recent = self.turns[-max_turns:]
        parts: list[str] = []

        for i, turn in enumerate(recent, start=1):
            answer_preview = turn.answer[:150] + "..." if len(turn.answer) > 150 else turn.answer
            parts.append(f"  Q{i}: {turn.query}\n  A{i}: {answer_preview}")

        return "\n".join(parts)

    def _summarize_old_turns(self) -> None:
        """Trim old turns, keeping recent ones.

        When session exceeds max_turns, keeps the last max_turns/2
        turns and discards the oldest. Entity/filing accumulators
        are NOT cleared — we keep the full context.
        """
        keep = self.max_turns // 2
        removed = len(self.turns) - keep
        self.turns = self.turns[-keep:]
        logger.info(
            "session_turns_trimmed",
            removed=removed,
            remaining=len(self.turns),
        )

    def to_dict(self) -> dict:
        """Serialize session to dict for API responses.

        Returns:
            Dict representation of the session state.
        """
        return {
            "session_id": self.session_id,
            "turn_count": self.turn_count,
            "entities": sorted(self.all_entities),
            "filings": sorted(self.all_filings),
            "periods": sorted(self.all_periods),
            "cited_chunks": len(self.all_cited_chunks),
            "created_at": self.created_at,
        }


# --------------------------------------------------------------------------- #
# Entity Extraction Helpers
# --------------------------------------------------------------------------- #


def extract_entities(text: str) -> list[str]:
    """Extract ticker-like entities from text.

    Uses regex to find capitalized 1-5 letter words that look
    like stock tickers, filtering out common English words.

    Args:
        text: Text to extract entities from.

    Returns:
        List of unique entity strings.
    """
    matches = TICKER_PATTERN.findall(text)
    entities = [
        m for m in matches
        if m not in TICKER_STOPWORDS and len(m) >= 2
    ]
    return list(dict.fromkeys(entities))  # Deduplicate preserving order


def extract_filings(text: str) -> list[str]:
    """Extract filing type mentions from text.

    Args:
        text: Text to search for filing types.

    Returns:
        List of filing types found.
    """
    found: list[str] = []
    text_lower = text.lower()
    for filing_type in FILING_TYPES:
        if filing_type.lower() in text_lower:
            found.append(filing_type)
    return list(dict.fromkeys(found))


def extract_periods(text: str) -> list[str]:
    """Extract time period references from text.

    Args:
        text: Text to search for period patterns.

    Returns:
        List of period strings found.
    """
    matches = PERIOD_PATTERN.findall(text)
    return list(dict.fromkeys(matches))


# --------------------------------------------------------------------------- #
# Session Store
# --------------------------------------------------------------------------- #


class SessionStore:
    """In-memory store for active conversation sessions.

    Maps session IDs to SessionMemory instances. Used by the
    API layer to maintain per-user sessions.

    Attributes:
        _sessions: Dict of session_id → SessionMemory.
        max_sessions: Maximum concurrent sessions before eviction.
    """

    def __init__(self, max_sessions: int = 1000) -> None:
        """Initialize the session store.

        Args:
            max_sessions: Maximum concurrent sessions.
        """
        self._sessions: dict[str, SessionMemory] = {}
        self.max_sessions = max_sessions

    def get_or_create(self, session_id: str) -> SessionMemory:
        """Get an existing session or create a new one.

        Args:
            session_id: Unique session identifier.

        Returns:
            The SessionMemory for this session.
        """
        if session_id not in self._sessions:
            if len(self._sessions) >= self.max_sessions:
                self._evict_oldest()

            self._sessions[session_id] = SessionMemory(session_id=session_id)
            logger.info("session_created", session_id=session_id)

        return self._sessions[session_id]

    def get(self, session_id: str) -> SessionMemory | None:
        """Get a session by ID, or None if not found.

        Args:
            session_id: Session identifier.

        Returns:
            SessionMemory or None.
        """
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> bool:
        """Delete a session.

        Args:
            session_id: Session to delete.

        Returns:
            True if session existed and was deleted.
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info("session_deleted", session_id=session_id)
            return True
        return False

    @property
    def active_count(self) -> int:
        """Number of active sessions."""
        return len(self._sessions)

    def _evict_oldest(self) -> None:
        """Evict the oldest session to make room."""
        if not self._sessions:
            return

        oldest_id = min(
            self._sessions,
            key=lambda sid: self._sessions[sid].created_at,
        )
        del self._sessions[oldest_id]
        logger.info("session_evicted", session_id=oldest_id)
