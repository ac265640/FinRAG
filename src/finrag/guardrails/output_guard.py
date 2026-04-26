"""Output guardrails for the FinRAG pipeline.

Scans generated answers AFTER LLM generation but BEFORE they
reach the user. Catches policy violations that slipped through
the router's decline logic, PII leaking from filings, and
missing safety disclaimers.

Why output guards exist (defense in depth):
- The router catches obvious investment advice queries, but the
  LLM might still generate advice-like language in its answer
  (e.g., "based on these metrics, the company looks undervalued").
- SEC filings contain PII (executive SSNs in proxy statements,
  account numbers in exhibits). The LLM might include these in
  its answer even though the user didn't ask for them.
- Financial answers need disclaimers. Regulatory requirements
  vary by jurisdiction, but a baseline disclaimer is always safer.

Design decisions:
- Output guards run AFTER citation enforcement. They check the
  final answer text, not the raw LLM response. This means they
  see the validated, citation-checked answer.
- PII scrubbing (not just detection): output PII is replaced with
  [REDACTED] markers. Input PII is just warned — we don't modify
  the user's query.
- Disclaimer injection: appended to the answer only when the
  answer contains financial data. Decline responses already have
  their own messaging.
"""

import re
from dataclasses import dataclass, field

import structlog

from finrag.guardrails.input_guard import GuardResult, Severity

logger = structlog.get_logger(__name__)


# --------------------------------------------------------------------------- #
# Investment Advice Detection (Output)
# --------------------------------------------------------------------------- #

# Patterns that indicate the LLM is giving investment advice
# despite the system prompt prohibiting it. Defense in depth.
ADVICE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(
            r"\b(you\s+should|i\s+(would\s+)?recommend|"
            r"consider\s+(buying|selling|investing|holding))\b",
            re.IGNORECASE,
        ),
        "Direct recommendation language",
    ),
    (
        re.compile(
            r"\b(undervalued|overvalued|buy\s+rating|sell\s+rating|"
            r"strong\s+buy|outperform|underperform)\b",
            re.IGNORECASE,
        ),
        "Investment rating language",
    ),
    (
        re.compile(
            r"\b(price\s+target|expected\s+return|upside\s+potential|"
            r"downside\s+risk|fair\s+value\s+estimate)\b",
            re.IGNORECASE,
        ),
        "Price prediction language",
    ),
    (
        re.compile(
            r"\b(guaranteed\s+(return|profit|income)|risk[- ]free|"
            r"can't\s+lose|sure\s+thing|easy\s+money)\b",
            re.IGNORECASE,
        ),
        "Misleading financial promise",
    ),
]


def check_investment_advice_in_output(answer: str) -> GuardResult:
    """Scan answer for investment advice language.

    Defense in depth — the router should have caught advice-seeking
    queries, but the LLM might still produce advice-like language
    when answering factual questions about company performance.

    Severity is WARN, not BLOCK: we flag the answer but still
    deliver it. The disclaimer handles the legal risk. Only
    explicit "you should buy" language is blocked.

    Args:
        answer: Generated answer text.

    Returns:
        GuardResult with details of any advice language found.
    """
    found: list[dict] = []

    for pattern, description in ADVICE_PATTERNS:
        match = pattern.search(answer)
        if match:
            found.append({
                "pattern": description,
                "matched_text": match.group()[:50],
            })

    if found:
        # "Guaranteed return" / "sure thing" is BLOCK-worthy
        has_misleading = any(
            f["pattern"] == "Misleading financial promise" for f in found
        )
        severity = Severity.BLOCK if has_misleading else Severity.WARN

        logger.warning(
            "investment_advice_in_output",
            patterns_found=len(found),
            severity=severity.value,
        )
        return GuardResult(
            passed=False,
            guard_name="output_advice_check",
            severity=severity,
            message=f"Investment advice language detected: "
            f"{', '.join(f['pattern'] for f in found)}",
            details={"matches": found},
        )

    return GuardResult(passed=True, guard_name="output_advice_check")


# --------------------------------------------------------------------------- #
# PII Leakage Detection (Output)
# --------------------------------------------------------------------------- #

# PII patterns for output scanning. These are the same patterns
# as input but applied to the generated answer. If PII from
# filings leaks into the answer, we scrub it.
OUTPUT_PII_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    # SSN: XXX-XX-XXXX
    (
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "SSN",
        "[SSN REDACTED]",
    ),
    # Credit card (13-16 digits with separators)
    (
        re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
        "Credit card",
        "[CARD REDACTED]",
    ),
    # Email address
    (
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
        "Email",
        "[EMAIL REDACTED]",
    ),
    # Bank account numbers (explicit label + digits)
    (
        re.compile(
            r"\b(account|routing)\s*#?\s*:?\s*\d{8,17}\b",
            re.IGNORECASE,
        ),
        "Bank account",
        "[ACCOUNT REDACTED]",
    ),
]


def check_pii_in_output(answer: str) -> GuardResult:
    """Scan answer for PII that leaked from filings.

    Unlike input PII (WARN), output PII is BLOCK severity —
    we cannot send PII from filings to the user without
    scrubbing. The scrub_pii_from_output() function does the
    actual replacement.

    Args:
        answer: Generated answer text.

    Returns:
        GuardResult with PII types found.
    """
    found_pii: list[str] = []

    for pattern, pii_type, _ in OUTPUT_PII_PATTERNS:
        if pattern.search(answer):
            found_pii.append(pii_type)

    if found_pii:
        logger.warning(
            "pii_leakage_in_output",
            pii_types=found_pii,
        )
        return GuardResult(
            passed=False,
            guard_name="output_pii_check",
            severity=Severity.WARN,
            message=f"PII detected in output: {', '.join(found_pii)}",
            details={"pii_types": found_pii},
        )

    return GuardResult(passed=True, guard_name="output_pii_check")


def scrub_pii_from_output(answer: str) -> tuple[str, int]:
    """Replace PII in the answer with redaction markers.

    Unlike detection (which just flags), scrubbing modifies the
    answer text. Each PII type has its own redaction marker for
    transparency.

    Args:
        answer: Generated answer text.

    Returns:
        Tuple of (scrubbed answer, number of redactions made).
    """
    scrubbed = answer
    total_redactions = 0

    for pattern, pii_type, replacement in OUTPUT_PII_PATTERNS:
        scrubbed, count = pattern.subn(replacement, scrubbed)
        total_redactions += count

    if total_redactions > 0:
        logger.info(
            "pii_scrubbed_from_output",
            redactions=total_redactions,
        )

    return scrubbed, total_redactions


# --------------------------------------------------------------------------- #
# Financial Disclaimer
# --------------------------------------------------------------------------- #

FINANCIAL_DISCLAIMER = (
    "\n\n---\n"
    "*Disclaimer: This information is extracted from SEC filings and "
    "public financial documents for research purposes only. It does not "
    "constitute investment advice. Always verify data against original "
    "source documents and consult a licensed financial advisor before "
    "making investment decisions.*"
)

# Signals that the answer contains financial data (needs disclaimer).
FINANCIAL_CONTENT_SIGNALS = re.compile(
    r"\b(revenue|earnings|net\s+income|gross\s+margin|EBITDA|"
    r"free\s+cash\s+flow|EPS|dividend|market\s+cap|"
    r"total\s+assets|operating\s+(income|expense)|"
    r"\$[\d,.]+\s*(billion|million|B|M|K|thousand)|"
    r"\d+(\.\d+)?%|fiscal\s+(year|quarter)|10-[KQ]|"
    r"SEC\s+filing|annual\s+report)\b",
    re.IGNORECASE,
)


def maybe_add_disclaimer(
    answer: str,
    generation_model: str = "",
) -> tuple[str, bool]:
    """Conditionally append financial disclaimer to the answer.

    Only adds disclaimer when:
    - Answer contains financial data signals
    - Not a decline/error response
    - Not a stub response

    Args:
        answer: Generated answer text.
        generation_model: Model that generated the answer.

    Returns:
        Tuple of (answer with optional disclaimer, whether added).
    """
    # Skip for non-answers
    if not answer.strip():
        return answer, False

    # Skip for decline/error/stub responses
    skip_models = {"decline", "error_handler", "stub_v1", "none", "error"}
    if generation_model in skip_models:
        return answer, False

    # Skip if already has disclaimer
    if "does not constitute investment advice" in answer:
        return answer, False

    # Check for financial content
    if FINANCIAL_CONTENT_SIGNALS.search(answer):
        logger.info("financial_disclaimer_added")
        return answer + FINANCIAL_DISCLAIMER, True

    return answer, False


# --------------------------------------------------------------------------- #
# Output Guard Runner
# --------------------------------------------------------------------------- #


@dataclass
class OutputGuardReport:
    """Aggregated report from all output guards.

    Attributes:
        allowed: Whether the answer should be delivered.
        scrubbed_answer: Answer with PII scrubbed and disclaimer added.
        results: Individual guard results.
        warnings: List of warning messages.
        redactions_made: Number of PII redactions.
        disclaimer_added: Whether financial disclaimer was appended.
    """

    allowed: bool = True
    scrubbed_answer: str = ""
    results: list[GuardResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    redactions_made: int = 0
    disclaimer_added: bool = False

    def add_result(self, result: GuardResult) -> None:
        """Add a guard result to the report.

        Args:
            result: Individual guard check result.
        """
        self.results.append(result)
        if not result.passed:
            if result.severity == Severity.BLOCK:
                self.allowed = False
            else:
                self.warnings.append(
                    f"[{result.guard_name}] {result.message}"
                )


def run_output_guards(
    answer: str,
    generation_model: str = "",
) -> OutputGuardReport:
    """Run all output guards on a generated answer.

    Order of operations:
    1. Check for investment advice language
    2. Check for PII leakage
    3. Scrub any PII found
    4. Add financial disclaimer if needed

    Args:
        answer: Generated answer text.
        generation_model: Model that produced the answer.

    Returns:
        OutputGuardReport with scrubbed answer and results.
    """
    report = OutputGuardReport()
    current_answer = answer

    # Check 1: Investment advice
    advice_result = check_investment_advice_in_output(current_answer)
    report.add_result(advice_result)

    # If blocked by advice check, return immediately
    if not advice_result.passed and advice_result.severity == Severity.BLOCK:
        report.scrubbed_answer = (
            "This response was blocked because it contained "
            "misleading financial promises. Please rephrase your "
            "question to focus on factual data from SEC filings."
        )
        logger.warning(
            "output_blocked_advice",
            message=advice_result.message,
        )
        return report

    # Check 2: PII leakage detection
    pii_result = check_pii_in_output(current_answer)
    report.add_result(pii_result)

    # Scrub PII regardless of severity
    if not pii_result.passed:
        current_answer, redactions = scrub_pii_from_output(current_answer)
        report.redactions_made = redactions

    # Step 3: Add disclaimer if needed
    current_answer, disclaimer_added = maybe_add_disclaimer(
        current_answer, generation_model
    )
    report.disclaimer_added = disclaimer_added

    report.scrubbed_answer = current_answer

    logger.info(
        "output_guards_complete",
        allowed=report.allowed,
        warnings=len(report.warnings),
        redactions=report.redactions_made,
        disclaimer_added=report.disclaimer_added,
    )

    return report
