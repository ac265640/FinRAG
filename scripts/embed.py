"""CLI script for embedding chunked filings into ChromaDB.

Reads saved filings from Day 1 ingestion, chunks them (Day 2),
embeds them, and stores in ChromaDB with full metadata.

Usage:
    # Embed a single filing directory
    python scripts/embed.py --filing-dir data/raw/AAPL_10-K_20251031

    # Embed all filings in data/raw/
    python scripts/embed.py --all

    # Show collection stats
    python scripts/embed.py --stats
"""

import argparse
import sys
from pathlib import Path

# Add src to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import structlog

from finrag.ingestion.chunker import chunk_filing_directory
from finrag.vectorstore.chroma_store import ChromaStore

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
)

logger = structlog.get_logger(__name__)

DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_RAW_DIR = DEFAULT_DATA_DIR / "raw"
DEFAULT_CHROMA_DIR = DEFAULT_DATA_DIR / "chroma"


def embed_filing(filing_dir: Path, store: ChromaStore) -> int:
    """Chunk and embed a single filing directory.

    Args:
        filing_dir: Path to a filing directory with metadata.json.
        store: ChromaStore instance for storage.

    Returns:
        Number of chunks embedded.
    """
    logger.info("embedding_filing", directory=str(filing_dir))
    chunks = chunk_filing_directory(filing_dir)
    added = store.add_chunks(chunks)
    logger.info(
        "filing_embedded",
        directory=str(filing_dir),
        chunks_added=added,
    )
    return added


def main() -> None:
    """Parse CLI args and run embedding pipeline."""
    parser = argparse.ArgumentParser(
        description="Embed chunked SEC filings into ChromaDB",
    )
    parser.add_argument(
        "--filing-dir",
        type=Path,
        help="Path to a specific filing directory to embed",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Embed all filing directories in data/raw/",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show collection statistics and exit",
    )
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        default=DEFAULT_CHROMA_DIR,
        help=f"ChromaDB persist directory (default: {DEFAULT_CHROMA_DIR})",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset the collection before embedding",
    )

    args = parser.parse_args()

    store = ChromaStore(persist_dir=args.chroma_dir)

    # Stats mode
    if args.stats:
        stats = store.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        return

    # Reset if requested
    if args.reset:
        logger.warning("resetting_collection")
        store.reset()

    total = 0

    if args.filing_dir:
        # Single filing
        if not args.filing_dir.exists():
            logger.error("directory_not_found", path=str(args.filing_dir))
            sys.exit(1)
        total = embed_filing(args.filing_dir, store)

    elif args.all:
        # All filings in data/raw/
        raw_dir = DEFAULT_RAW_DIR
        if not raw_dir.exists():
            logger.error("raw_directory_not_found", path=str(raw_dir))
            sys.exit(1)

        filing_dirs = [
            d for d in sorted(raw_dir.iterdir())
            if d.is_dir() and (d / "metadata.json").exists()
        ]

        if not filing_dirs:
            logger.warning("no_filings_found", directory=str(raw_dir))
            return

        for filing_dir in filing_dirs:
            total += embed_filing(filing_dir, store)

    else:
        parser.print_help()
        sys.exit(1)

    # Final stats
    stats = store.get_stats()
    logger.info(
        "embedding_complete",
        total_embedded=total,
        collection_total=stats["total_chunks"],
    )


if __name__ == "__main__":
    main()
