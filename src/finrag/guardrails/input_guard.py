"""Input guardrails for the FinRAG pipeline.

Scans user queries BEFORE they enter the retrieval pipeline.
Catches prompt injection attacks, PII in queries, and malformed
inputs before they can corrupt retrieval or waste LLM tokens.

Why input guards matter:
- Prompt injection in the query can override the system prompt,
  causing the LLM to ignore citation rules or leak system details.
- PII in queries gets logged, embedded in vectors, and stored in
  ChromaDB — creating compliance risk (GDPR, SOX).
- Malformed/oversized queries waste compute on retrieval and LLM
  calls that will produce garbage anyway.

Design decisions:
- Pattern-based detection: <1ms per check, deterministic, no false
  negatives on known attack patterns. Regex beats an LLM classifier
  for well-defined threat signatures.
- Severity levels: BLOCK (hard stop, return error) vs WARN (log and
  continue). PII in a query is a WARN — the user might be asking
  "what SSN format does the filing use?" Investment advice is BLOCK.
- Extensible: each guard is a callable. Add new guards by appending
  to the registry, not by modifying existing code.

Debt: DAY-9-001 — Prompt injection patterns are static regex.
      An LLM-based classifier (or fine-tuned distilbert) would
      catch novel attacks. Add on Day 14 as secondary layer.
"""

import re
from dataclasses import dataclass, field
from enum import Enum

import structlog

logger = structlog.get_logger(__name__)


# --------------------------------------------------------------------------- #
# Types
# --------------------------------------------------------------------------- #


class Severity(Enum):
    """Guard check severity level."""

    BLOCK = "block"  # Hard stop — reject the request
    WARN = "warn"  # Log warning but allow through


@dataclass
class GuardResult:
    """Result from a single guard check.

    Attributes:
        passed: Whether the check passed (no issues found).
        guard_name: Name of the guard that produced this result.
        severity: BLOCK or WARN.
        message: Human-readable explanation of the issue.
        details: Additional structured details for logging.
    """

    passed: bool
    guard_name: str
    severity: Severity = Severity.WARN
    message: str = ""
    details: dict = field(default_factory=dict)


@dataclass
class GuardReport:
    """Aggregated report from all input guards.

    Attributes:
        allowed: Whether the query should proceed (no BLOCK results).
        results: Individual guard results.
        blocked_by: Name of the guard that blocked, if any.
        warnings: List of warning messages.
    """

    allowed: bool = True
    results: list[GuardResult] = field(default_factory=list)
    blocked_by: str = ""
    warnings: list[str] = field(default_factory=list)

    def add_result(self, result: GuardResult) -> None:
        """Add a guard result to the report.

        Args:
            result: Individual guard check result.
        """
        self.results.append(result)
        if not result.passed:
            if result.severity == Severity.BLOCK:
                self.allowed = False
                self.blocked_by = result.guard_name
            else:
                self.warnings.append(
                    f"[{result.guard_name}] {result.message}"
                )


# --------------------------------------------------------------------------- #
# Prompt Injection Detection
# --------------------------------------------------------------------------- #

# Patterns that indicate prompt injection attempts.
# Ordered by specificity (most specific first).
# Each tuple: (pattern, description, severity).
INJECTION_PATTERNS: list[tuple[re.Pattern, str, Severity]] = [
    # Direct instruction override
    (
        re.compile(
            r"(ignore|disregard|forget|override)\s+"
            r"(all\s+)?(previous|prior|above|system|original)\s+"
            r"(\w+\s+)?(instructions?|prompts?|rules?|context)",
            re.IGNORECASE,
        ),
        "Instruction override attempt",
        Severity.BLOCK,
    ),
    # System prompt extraction
    (
        re.compile(
            r"(show|reveal|print|display|output|repeat|echo)\s+"
            r"(me\s+)?(the\s+)?(your\s+|system\s+)?(prompt|instructions?|rules?|config)",
            re.IGNORECASE,
        ),
        "System prompt extraction attempt",
        Severity.BLOCK,
    ),
    # Role-play injection
    (
        re.compile(
            r"(you\s+are\s+now|act\s+as|pretend\s+(to\s+be|you\s+are)|"
            r"new\s+persona|switch\s+(to|into)\s+|roleplay\s+as)",
            re.IGNORECASE,
        ),
        "Role-play injection attempt",
        Severity.BLOCK,
    ),
    # Delimiter injection (trying to close system prompt context)
    (
        re.compile(
            r"(```|</?system>|</?user>|</?assistant>|\[INST\]|\[/INST\]|"
            r"<<SYS>>|<</SYS>>|Human:|Assistant:)",
            re.IGNORECASE,
        ),
        "Delimiter injection attempt",
        Severity.BLOCK,
    ),
    # Encoding bypass (base64, hex, rot13 instructions)
    (
        re.compile(
            r"(decode|base64|hex|rot13|ascii|unicode)\s+"
            r"(this|the\s+following|and\s+(execute|run|follow))",
            re.IGNORECASE,
        ),
        "Encoding bypass attempt",
        Severity.BLOCK,
    ),
    # Data exfiltration
    (
        re.compile(
            r"(send|post|fetch|curl|wget|http|request)\s+"
            r"(\w+\s+)*(to|from)?\s*(https?://|ftp://)",
            re.IGNORECASE,
        ),
        "Data exfiltration attempt",
        Severity.BLOCK,
    ),
    # Jailbreak keywords (softer signal, could be false positive)
    (
        re.compile(
            r"\b(jailbreak|DAN|do\s+anything\s+now|evil\s+mode|"
            r"developer\s+mode|unrestricted\s+mode)\b",
            re.IGNORECASE,
        ),
        "Known jailbreak pattern",
        Severity.BLOCK,
    ),
]


def check_prompt_injection(query: str) -> GuardResult:
    """Scan query for prompt injection patterns.

    Runs all injection patterns against the query. Returns on
    first match (fail-fast) since any injection is enough to block.

    Args:
        query: User's input query.

    Returns:
        GuardResult indicating pass/fail with details.
    """
    for pattern, description, severity in INJECTION_PATTERNS:
        match = pattern.search(query)
        if match:
            logger.warning(
                "prompt_injection_detected",
                pattern=description,
                matched_text=match.group()[:50],
                severity=severity.value,
            )
            return GuardResult(
                passed=False,
                guard_name="prompt_injection",
                severity=severity,
                message=f"Potential prompt injection: {description}",
                details={
                    "pattern": description,
                    "matched_text": match.group()[:50],
                },
            )

    return GuardResult(passed=True, guard_name="prompt_injection")


# --------------------------------------------------------------------------- #
# PII Detection
# --------------------------------------------------------------------------- #

# PII patterns for detection in user queries.
# These catch common US-format PII. Extend for other locales.
PII_PATTERNS: list[tuple[re.Pattern, str]] = [
    # SSN: XXX-XX-XXXX or XXXXXXXXX
    (
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "Social Security Number",
    ),
    (
        re.compile(r"\b\d{9}\b"),
        "Potential SSN (9 consecutive digits)",
    ),
    # Credit card: 13-19 digits with optional separators
    (
        re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
        "Credit card number",
    ),
    # Email address
    (
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
        "Email address",
    ),
    # Phone number: various US formats
    (
        re.compile(
            r"\b(?:\+?1[-.\s]?)?\(?[2-9]\d{2}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
        ),
        "Phone number",
    ),
    # Bank account / routing numbers (8-17 digits)
    (
        re.compile(r"\b(account|routing)\s*#?\s*:?\s*\d{8,17}\b", re.IGNORECASE),
        "Bank account/routing number",
    ),
]


def check_pii_in_query(query: str) -> GuardResult:
    """Scan query for personally identifiable information.

    PII in queries is a WARN, not a BLOCK — the user might be
    asking about PII formats in filings. But we log it for
    compliance review.

    Args:
        query: User's input query.

    Returns:
        GuardResult with WARN severity if PII found.
    """
    found_pii: list[str] = []

    for pattern, pii_type in PII_PATTERNS:
        if pattern.search(query):
            found_pii.append(pii_type)

    if found_pii:
        logger.warning(
            "pii_detected_in_query",
            pii_types=found_pii,
        )
        return GuardResult(
            passed=False,
            guard_name="pii_detection",
            severity=Severity.WARN,
            message=f"PII detected in query: {', '.join(found_pii)}",
            details={"pii_types": found_pii},
        )

    return GuardResult(passed=True, guard_name="pii_detection")


# --------------------------------------------------------------------------- #
# Query Sanity Checks
# --------------------------------------------------------------------------- #

# Max query length in characters. Longer queries are suspicious
# (potential prompt injection via padding) and waste tokens.
MAX_QUERY_LENGTH = 2000

# Min query length. Single-character queries produce garbage.
MIN_QUERY_LENGTH = 3


def check_query_sanity(query: str) -> GuardResult:
    """Validate basic query properties.

    Checks:
    - Non-empty
    - Within length bounds (3-2000 chars)
    - Not pure whitespace/punctuation

    Args:
        query: User's input query.

    Returns:
        GuardResult indicating pass/fail.
    """
    stripped = query.strip()

    if not stripped:
        return GuardResult(
            passed=False,
            guard_name="query_sanity",
            severity=Severity.BLOCK,
            message="Empty query",
        )

    if len(stripped) < MIN_QUERY_LENGTH:
        return GuardResult(
            passed=False,
            guard_name="query_sanity",
            severity=Severity.BLOCK,
            message=f"Query too short ({len(stripped)} chars, min {MIN_QUERY_LENGTH})",
        )

    if len(query) > MAX_QUERY_LENGTH:
        return GuardResult(
            passed=False,
            guard_name="query_sanity",
            severity=Severity.BLOCK,
            message=f"Query too long ({len(query)} chars, max {MAX_QUERY_LENGTH})",
        )

    # Check if query has any alphanumeric content
    if not re.search(r"[a-zA-Z0-9]", stripped):
        return GuardResult(
            passed=False,
            guard_name="query_sanity",
            severity=Severity.BLOCK,
            message="Query contains no alphanumeric characters",
        )

    return GuardResult(passed=True, guard_name="query_sanity")


# --------------------------------------------------------------------------- #
# Input Guard Runner
# --------------------------------------------------------------------------- #


def run_input_guards(query: str) -> GuardReport:
    """Run all input guards on a query.

    Executes guards in order of severity:
    1. Query sanity (cheapest, catches garbage early)
    2. Prompt injection (pattern-based, <1ms)
    3. PII detection (regex, <1ms)

    Short-circuits on first BLOCK result — no point running
    remaining guards if we're already rejecting.

    Args:
        query: User's input query.

    Returns:
        GuardReport with aggregated results.
    """
    report = GuardReport()

    guards = [
        check_query_sanity,
        check_prompt_injection,
        check_pii_in_query,
    ]

    for guard_fn in guards:
        result = guard_fn(query)
        report.add_result(result)

        # Short-circuit on BLOCK
        if not result.passed and result.severity == Severity.BLOCK:
            logger.warning(
                "input_blocked",
                guard=result.guard_name,
                message=result.message,
                query_preview=query[:80],
            )
            break

    if report.allowed:
        logger.info(
            "input_guards_passed",
            query_preview=query[:80],
            warnings=len(report.warnings),
        )

    return report
