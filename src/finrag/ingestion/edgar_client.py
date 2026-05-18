"""Async client for SEC EDGAR API.

Handles ticker-to-CIK resolution, filing discovery, download,
and basic section parsing. Respects SEC rate limits (10 req/s)
and requires a valid User-Agent header.

Design decisions:
- httpx for async HTTP (I/O-bound calls benefit from async)
- asyncio.Semaphore for client-side rate limiting
- Structured error types for each failure mode
- BeautifulSoup for HTML parsing (regex alone is too fragile)
"""

import asyncio
import json
import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import httpx
import structlog
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from finrag.config import Settings

# Suppress BS4 warning when lxml encounters XML-structured EDGAR filings.
# The parser works correctly regardless — this is a false-positive warning.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

logger = structlog.get_logger(__name__)

# --------------------------------------------------------------------------- #
# Custom exceptions: each failure mode gets its own type
# --------------------------------------------------------------------------- #


class EdgarError(Exception):
    """Base exception for EDGAR client errors."""


class TickerNotFoundError(EdgarError):
    """Raised when a ticker cannot be resolved to a CIK."""


class FilingNotFoundError(EdgarError):
    """Raised when no filings are found for a CIK + filing type."""


class EdgarUnavailableError(EdgarError):
    """Raised when EDGAR API is unreachable after retries."""


class EdgarRateLimitError(EdgarError):
    """Raised when EDGAR returns a 429 rate limit response."""


# --------------------------------------------------------------------------- #
# Data models
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FilingMetadata:
    """Metadata for a single SEC filing.

    Attributes:
        cik: SEC Central Index Key.
        ticker: Stock ticker symbol.
        company_name: Full company name from EDGAR.
        filing_type: Filing form type (10-K, 10-Q, 8-K).
        filing_date: Date the filing was submitted.
        accession_number: Unique filing identifier.
        primary_document_url: URL to the main filing document.
    """

    cik: str
    ticker: str
    company_name: str
    filing_type: str
    filing_date: str
    accession_number: str
    primary_document_url: str


@dataclass
class ParsedFiling:
    """A downloaded and parsed SEC filing.

    Attributes:
        metadata: Filing metadata from EDGAR.
        sections: Dict mapping section name to section text content.
        raw_content_length: Length of the raw HTML content in characters.
    """

    metadata: FilingMetadata
    sections: dict[str, str] = field(default_factory=dict)
    raw_content_length: int = 0


# --------------------------------------------------------------------------- #
# 10-K section patterns
# --------------------------------------------------------------------------- #

# Standard 10-K items. These appear as "Item 1", "Item 1A", etc.
# We look for these in headings and bold text within the filing HTML.
SECTION_10K_ITEMS: dict[str, str] = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Common Equity",
    "6": "Reserved",
    "7": "MD&A",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements",
    "9": "Changes in and Disagreements with Accountants",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "10": "Directors and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership",
    "13": "Certain Relationships",
    "14": "Principal Accountant Fees",
    "15": "Exhibits and Financial Statement Schedules",
}


# --------------------------------------------------------------------------- #
# EDGAR Client
# --------------------------------------------------------------------------- #


class EdgarClient:
    """Async client for SEC EDGAR API.

    Handles all interactions with the SEC EDGAR system including
    ticker resolution, filing discovery, and content download.

    Args:
        settings: Application settings with EDGAR configuration.
    """

    # SEC company tickers endpoint (returns all ticker-to-CIK mappings)
    TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

    # EDGAR filing submissions endpoint
    SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

    # Max retries for transient failures
    MAX_RETRIES = 3

    # Backoff base in seconds
    BACKOFF_BASE = 1.0

    def __init__(self, settings: Settings) -> None:
        """Initialize the EDGAR client.

        Args:
            settings: Application settings containing EDGAR configuration.
        """
        self._settings = settings
        self._headers = {
            "User-Agent": settings.edgar_user_agent,
            "Accept-Encoding": "gzip, deflate",
        }
        # Client-side rate limiting to respect SEC's 10 req/s limit
        self._semaphore = asyncio.Semaphore(settings.edgar_max_rps)
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "EdgarClient":
        """Enter async context manager."""
        self._client = httpx.AsyncClient(
            headers=self._headers,
            timeout=httpx.Timeout(30.0),
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, *args: object) -> None:
        """Exit async context manager."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _request(self, url: str) -> httpx.Response:
        """Make a rate-limited HTTP request with retry logic.

        Args:
            url: The URL to request.

        Returns:
            The HTTP response.

        Raises:
            EdgarUnavailableError: If all retries are exhausted.
            EdgarRateLimitError: If we hit a 429 response.
        """
        if not self._client:
            msg = "Client not initialized. Use 'async with EdgarClient(...)' context manager."
            raise EdgarError(msg)

        for attempt in range(self.MAX_RETRIES):
            async with self._semaphore:
                try:
                    response = await self._client.get(url)

                    if response.status_code == 429:
                        logger.warning(
                            "rate_limit_hit",
                            url=url,
                            attempt=attempt + 1,
                        )
                        if attempt < self.MAX_RETRIES - 1:
                            wait = self.BACKOFF_BASE * (2**attempt)
                            await asyncio.sleep(wait)
                            continue
                        raise EdgarRateLimitError(f"Rate limited by EDGAR after {self.MAX_RETRIES} attempts")

                    if response.status_code == 503:
                        logger.warning(
                            "edgar_unavailable",
                            url=url,
                            attempt=attempt + 1,
                        )
                        if attempt < self.MAX_RETRIES - 1:
                            wait = self.BACKOFF_BASE * (2**attempt)
                            await asyncio.sleep(wait)
                            continue
                        raise EdgarUnavailableError(f"EDGAR unavailable (503) after {self.MAX_RETRIES} attempts")

                    response.raise_for_status()
                    return response

                except httpx.HTTPStatusError:
                    raise
                except httpx.HTTPError as e:
                    logger.warning(
                        "http_error",
                        url=url,
                        error=str(e),
                        attempt=attempt + 1,
                    )
                    if attempt < self.MAX_RETRIES - 1:
                        wait = self.BACKOFF_BASE * (2**attempt)
                        await asyncio.sleep(wait)
                        continue
                    raise EdgarUnavailableError(f"EDGAR unreachable after {self.MAX_RETRIES} attempts: {e}") from e

        raise EdgarUnavailableError("Exhausted all retries")

    async def ticker_to_cik(self, ticker: str) -> tuple[str, str]:
        """Resolve a stock ticker to its SEC CIK number.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL").

        Returns:
            Tuple of (cik_padded, company_name). CIK is zero-padded to 10 digits.

        Raises:
            TickerNotFoundError: If the ticker cannot be found in SEC records.
        """
        ticker_upper = ticker.upper().strip()

        logger.info("resolving_ticker", ticker=ticker_upper)
        response = await self._request(self.TICKERS_URL)
        data = response.json()

        # EDGAR returns format: {"0": {"cik_str": 320193, "ticker": "AAPL", ...}}
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker_upper:
                cik = str(entry["cik_str"]).zfill(10)
                company_name = entry.get("title", "Unknown")
                logger.info(
                    "ticker_resolved",
                    ticker=ticker_upper,
                    cik=cik,
                    company=company_name,
                )
                return cik, company_name

        raise TickerNotFoundError(f"Ticker '{ticker_upper}' not found in SEC EDGAR records.")

    async def get_filing_urls(self, cik: str, filing_type: str, count: int = 5) -> list[FilingMetadata]:
        """Get recent filing URLs for a company.

        Args:
            cik: SEC CIK number (zero-padded to 10 digits).
            filing_type: Filing form type (e.g., "10-K", "10-Q", "8-K").
            count: Maximum number of filings to return.

        Returns:
            List of FilingMetadata objects for matching filings.

        Raises:
            FilingNotFoundError: If no filings match the criteria.
        """
        filing_type_upper = filing_type.upper().strip()

        logger.info(
            "fetching_filings",
            cik=cik,
            filing_type=filing_type_upper,
            count=count,
        )

        url = self.SUBMISSIONS_URL.format(cik=cik)
        response = await self._request(url)
        data = response.json()

        company_name = data.get("name", "Unknown")
        ticker = data.get("tickers", [""])[0] if data.get("tickers") else ""

        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accession_numbers = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])

        results: list[FilingMetadata] = []
        for i, form in enumerate(forms):
            if form.upper() == filing_type_upper and len(results) < count:
                acc_no = accession_numbers[i].replace("-", "")
                doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{acc_no}/{primary_docs[i]}"
                results.append(
                    FilingMetadata(
                        cik=cik,
                        ticker=ticker,
                        company_name=company_name,
                        filing_type=filing_type_upper,
                        filing_date=dates[i],
                        accession_number=accession_numbers[i],
                        primary_document_url=doc_url,
                    )
                )

        if not results:
            raise FilingNotFoundError(f"No {filing_type_upper} filings found for CIK {cik} ({company_name})")

        logger.info(
            "filings_found",
            count=len(results),
            filing_type=filing_type_upper,
            company=company_name,
        )
        return results

    async def download_filing(self, url: str) -> str:
        """Download raw filing content from SEC.

        Args:
            url: URL to the filing document.

        Returns:
            Raw HTML/text content of the filing.

        Raises:
            EdgarUnavailableError: If the filing cannot be downloaded.
        """
        logger.info("downloading_filing", url=url)
        response = await self._request(url)
        content = response.text
        logger.info(
            "filing_downloaded",
            url=url,
            content_length=len(content),
        )
        return content

    def parse_sections(self, raw_content: str, filing_type: str) -> dict[str, str]:
        """Extract named sections from filing HTML.

        Uses BeautifulSoup to find section headings and extract text
        between them. This is a heuristic parser suitable for most
        modern 10-K filings. Older or unusual filings may not parse
        fully. Unparseable sections are logged as warnings, not errors.

        [DEMO-ONLY] This parser handles common patterns well but is
        not production-hardened for all filing variants. Day 2 will
        build a proper section-aware chunker on top of this.

        Args:
            raw_content: Raw HTML content of the filing.
            filing_type: Filing type (e.g., "10-K") to select section patterns.

        Returns:
            Dict mapping section name to extracted text content.
        """
        if filing_type.upper() not in ("10-K", "10-K/A"):
            # For non-10K filings, return the full text as a single section.
            # Note: for 8-K filings the exhibit (Exhibit 99.1) is fetched separately
            # in ingest_filing() and merged in before saving.
            soup = BeautifulSoup(raw_content, "lxml")
            text = soup.get_text(separator="\n", strip=True)
            return {"full_text": text}

        soup = BeautifulSoup(raw_content, "lxml")

        sections: dict[str, str] = {}
        full_text = soup.get_text(separator="\n", strip=True)

        # Build regex patterns for each 10-K item
        # Match patterns like "Item 1.", "Item 1A.", "ITEM 7" etc.
        item_patterns: list[tuple[str, str, re.Pattern[str]]] = []
        for item_num, item_name in SECTION_10K_ITEMS.items():
            pattern = re.compile(
                rf"(?:^|\n)\s*(?:item|ITEM)\s+{re.escape(item_num)}\.?\s*[\.\-\u2014]?\s*"
                rf"(?:{re.escape(item_name)})?",
                re.IGNORECASE | re.MULTILINE,
            )
            item_patterns.append((item_num, item_name, pattern))

        # Find all section start positions
        found_positions: list[tuple[int, str, str]] = []
        for item_num, item_name, pattern in item_patterns:
            matches = list(pattern.finditer(full_text))
            if matches:
                # Use the last match (table of contents often has first match)
                # The actual section content is at the last occurrence
                match = matches[-1] if len(matches) > 1 else matches[0]
                found_positions.append((match.start(), item_num, item_name))

        # Sort by position in document
        found_positions.sort(key=lambda x: x[0])

        # Extract text between consecutive section headings
        for i, (pos, item_num, item_name) in enumerate(found_positions):
            section_key = f"Item {item_num} - {item_name}"
            if i + 1 < len(found_positions):
                end_pos = found_positions[i + 1][0]
            else:
                end_pos = len(full_text)

            section_text = full_text[pos:end_pos].strip()

            # Skip very short sections (likely just the heading)
            min_section_length = 100
            if len(section_text) > min_section_length:
                sections[section_key] = section_text

        if not sections:
            logger.warning(
                "no_sections_parsed",
                content_length=len(full_text),
                msg="Could not identify standard 10-K sections. Returning full text.",
            )
            sections["full_text"] = full_text

        logger.info(
            "sections_parsed",
            section_count=len(sections),
            section_names=list(sections.keys()),
        )
        return sections

    async def get_exhibit_urls(
        self,
        cik: str,
        accession_number: str,
        exhibit_types: list[str] | None = None,
    ) -> list[tuple[str, str]]:
        """Fetch the filing index and return URLs of matching exhibits.

        8-K filings store actual content (earnings results, press releases)
        in Exhibit 99.1 rather than the main document body. This method
        fetches the filing index page and finds exhibit document URLs.

        Args:
            cik: SEC CIK number (zero-padded).
            accession_number: Filing accession number (with dashes).
            exhibit_types: List of exhibit type strings to find, e.g. ["EX-99.1"].
                           If None, defaults to ["EX-99.1", "EX-99"].

        Returns:
            List of (exhibit_type, url) tuples for matching exhibits.
        """
        if exhibit_types is None:
            exhibit_types = ["EX-99.1", "EX-99", "EX-99.2"]

        acc_no_clean = accession_number.replace("-", "")
        index_url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}"
            f"/{acc_no_clean}/{accession_number}-index.htm"
        )

        logger.info("fetching_filing_index", url=index_url)
        try:
            response = await self._request(index_url)
        except Exception as e:
            logger.warning("filing_index_unavailable", url=index_url, error=str(e))
            return []

        soup = BeautifulSoup(response.text, "lxml")
        results: list[tuple[str, str]] = []

        # Parse the EDGAR filing index table
        for row in soup.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 4:
                continue
            
            # Column 3 is the formal document Type (e.g., "EX-99.1", "10-K")
            doc_type = cells[3].get_text(strip=True).upper()
            
            if any(doc_type.startswith(ex.upper()) for ex in exhibit_types):
                link = cells[2].find("a")
                if link and link.get("href"):
                    href = link["href"]
                    if not href.startswith("http"):
                        href = "https://www.sec.gov" + href
                    results.append((doc_type, href))
                    logger.info("exhibit_found", exhibit_type=doc_type, url=href)

        if not results:
            logger.warning(
                "no_exhibits_found",
                accession_number=accession_number,
                exhibit_types=exhibit_types,
            )
        return results

    async def save_filing(
        self,
        parsed: ParsedFiling,
        data_dir: Path,
    ) -> Path:
        """Save a parsed filing to disk with metadata sidecar.

        Creates a directory per company/filing and saves:
        - sections as individual text files
        - metadata as a JSON sidecar

        Args:
            parsed: The parsed filing to save.
            data_dir: Base directory for saved filings.

        Returns:
            Path to the directory where filing was saved.
        """
        meta = parsed.metadata
        safe_date = meta.filing_date.replace("-", "")
        dir_name = f"{meta.ticker}_{meta.filing_type}_{safe_date}"
        filing_dir = data_dir / dir_name
        filing_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata sidecar
        metadata_dict = {
            "cik": meta.cik,
            "ticker": meta.ticker,
            "company_name": meta.company_name,
            "filing_type": meta.filing_type,
            "filing_date": meta.filing_date,
            "accession_number": meta.accession_number,
            "primary_document_url": meta.primary_document_url,
            "raw_content_length": parsed.raw_content_length,
            "sections_found": list(parsed.sections.keys()),
            "saved_at": datetime.now().isoformat(),
        }
        metadata_path = filing_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata_dict, indent=2), encoding="utf-8")

        # Save each section as a separate text file
        for section_name, section_text in parsed.sections.items():
            safe_name = re.sub(r"[^\w\s-]", "", section_name).strip()
            safe_name = re.sub(r"[\s]+", "_", safe_name).lower()
            section_path = filing_dir / f"{safe_name}.txt"
            section_path.write_text(section_text, encoding="utf-8")

        logger.info(
            "filing_saved",
            directory=str(filing_dir),
            sections=len(parsed.sections),
        )
        return filing_dir


async def ingest_filing(
    ticker: str,
    filing_type: str,
    settings: Settings,
    count: int = 1,
) -> list[Path]:
    """High-level ingestion function: fetch, parse, and save filings.

    This is the main entry point for the ingestion pipeline.
    For 8-K filings, also fetches Exhibit 99.1 (press release / earnings
    update) which contains the actual event content.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL").
        filing_type: Filing form type (e.g., "10-K").
        settings: Application settings.
        count: Number of recent filings to fetch.

    Returns:
        List of paths to saved filing directories.
    """
    saved_paths: list[Path] = []

    async with EdgarClient(settings) as client:
        cik, company_name = await client.ticker_to_cik(ticker)
        filings = await client.get_filing_urls(cik, filing_type, count=count)

        for filing_meta in filings:
            logger.info(
                "processing_filing",
                ticker=ticker,
                filing_date=filing_meta.filing_date,
                filing_type=filing_meta.filing_type,
            )

            raw_content = await client.download_filing(filing_meta.primary_document_url)
            sections = client.parse_sections(raw_content, filing_type)

            # For 8-K filings: fetch Exhibit 99.1 which contains the actual
            # event content (press releases, earnings tables, announcements).
            # The main 8-K body is almost always just a wrapper that says
            # "see Exhibit 99.1" — without the exhibit, no useful chunks exist.
            if filing_type.upper() == "8-K":
                exhibits = await client.get_exhibit_urls(
                    cik=cik,
                    accession_number=filing_meta.accession_number,
                )
                for exhibit_type, exhibit_url in exhibits:
                    try:
                        exhibit_raw = await client.download_filing(exhibit_url)
                        exhibit_soup = BeautifulSoup(exhibit_raw, "lxml")
                        exhibit_text = exhibit_soup.get_text(separator="\n", strip=True)
                        if len(exhibit_text) > 200:  # Skip empty/boilerplate exhibits
                            section_key = f"exhibit_{exhibit_type.lower().replace('-', '_').replace('.', '_')}"
                            sections[section_key] = exhibit_text
                            logger.info(
                                "exhibit_ingested",
                                exhibit_type=exhibit_type,
                                text_length=len(exhibit_text),
                            )
                    except Exception as e:
                        logger.warning(
                            "exhibit_download_failed",
                            exhibit_type=exhibit_type,
                            url=exhibit_url,
                            error=str(e),
                        )

            parsed = ParsedFiling(
                metadata=filing_meta,
                sections=sections,
                raw_content_length=len(raw_content),
            )

            path = await client.save_filing(parsed, settings.data_dir)
            saved_paths.append(path)

    logger.info(
        "ingestion_complete",
        ticker=ticker,
        filings_saved=len(saved_paths),
    )
    return saved_paths
