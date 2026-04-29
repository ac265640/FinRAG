"""Versioned prompt configuration loader.

Loads prompt templates and model parameters from YAML files
in the configs/prompts/ directory. Enables prompt versioning
without code changes — YAML is committed to git, so every
prompt change is tracked, reviewed, and can trigger CI eval.

Why versioned YAML:
- Prompt changes are the #1 cause of regression in RAG systems.
  A "minor" wording tweak can drop faithfulness by 20%.
- Git history shows exactly when prompts changed, who changed them,
  and which eval results correspond to which prompt version.
- CI can run the golden dataset against each prompt version and
  fail the build if quality drops (Day 15).
- A/B testing: load different versions in production, measure
  citation coverage per version.

Design decisions:
- Pydantic models for config validation: typos in YAML are caught
  at load time, not at runtime when the LLM call fails.
- Default configs directory is configs/prompts/ relative to project
  root. Override via FINRAG_PROMPTS_DIR env var.
- Config caching: loaded once, reused. Call reload() to pick up
  changes without restart (useful for development).
- Fallback to hardcoded defaults if YAML is missing. The system
  should never crash because a config file is absent.
"""

import os
from pathlib import Path

import structlog
import yaml
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


# --------------------------------------------------------------------------- #
# Config Models
# --------------------------------------------------------------------------- #


class ModelConfig(BaseModel):
    """LLM model parameters.

    Attributes:
        name: Model identifier (e.g., "gemini-2.0-flash").
        temperature: Sampling temperature (0-1).
        max_retries: Max retry attempts on enforcement failure.
    """

    name: str = "gemini-2.0-flash"
    temperature: float = 0.1
    max_retries: int = 1


class EnforcementConfig(BaseModel):
    """Citation enforcement thresholds.

    Attributes:
        confidence_threshold: Minimum confidence score to accept.
        min_citations: Minimum number of citations required.
        relevance_floor: Minimum top reranker score to proceed.
    """

    confidence_threshold: float = 0.3
    min_citations: int = 1
    relevance_floor: float = 0.15


class GenerationPromptConfig(BaseModel):
    """Full generation prompt configuration.

    Loaded from configs/prompts/v*_generation.yaml.

    Attributes:
        version: Config version string (e.g., "v1").
        name: Human-readable config name.
        description: Brief description of this config.
        created: Creation date string.
        model: LLM model parameters.
        system_prompt: System prompt for the LLM.
        retry_prompt_suffix: Appended on retry after enforcement failure.
        user_message_template: Template for user message construction.
        chunk_template: Template for formatting individual chunks.
        chunk_separator: Separator between formatted chunks.
        conversation_context_template: Template for injecting conversation
            history into the prompt.
        enforcement: Citation enforcement thresholds.
    """

    version: str = "v1"
    name: str = "citation_grounded_generation"
    description: str = ""
    created: str = ""
    model: ModelConfig = Field(default_factory=ModelConfig)
    system_prompt: str = ""
    retry_prompt_suffix: str = ""
    user_message_template: str = ""
    chunk_template: str = ""
    chunk_separator: str = "\n\n---\n\n"
    conversation_context_template: str = ""
    enforcement: EnforcementConfig = Field(default_factory=EnforcementConfig)


class RerankerConfig(BaseModel):
    """Reranker parameters.

    Attributes:
        model: Cross-encoder model name.
        top_k: Number of top chunks to keep after reranking.
        batch_size: Batch size for reranking.
    """

    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = 5
    batch_size: int = 16


class RetrievalParamsConfig(BaseModel):
    """Retrieval parameters.

    Attributes:
        top_k: Number of candidates to retrieve.
        rrf_k: RRF fusion parameter.
        strategies: List of retrieval strategies to use.
    """

    top_k: int = 20
    rrf_k: int = 60
    strategies: list[str] = Field(default_factory=lambda: ["bm25", "dense"])


class RetrievalPromptConfig(BaseModel):
    """Full retrieval prompt configuration.

    Loaded from configs/prompts/v*_retrieval.yaml.

    Attributes:
        version: Config version string.
        name: Human-readable config name.
        description: Brief description.
        created: Creation date.
        retrieval: Retrieval parameters.
        reranker: Reranker parameters.
        multi_query_prompt: Template for multi-query expansion.
        hyde_prompt: Template for HyDE generation.
    """

    version: str = "v1"
    name: str = "hybrid_retrieval"
    description: str = ""
    created: str = ""
    retrieval: RetrievalParamsConfig = Field(
        default_factory=RetrievalParamsConfig
    )
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    multi_query_prompt: str = ""
    hyde_prompt: str = ""


# --------------------------------------------------------------------------- #
# Config Loader
# --------------------------------------------------------------------------- #

# Module-level cache for loaded configs
_generation_config: GenerationPromptConfig | None = None
_retrieval_config: RetrievalPromptConfig | None = None


def _resolve_configs_dir() -> Path:
    """Resolve the prompt configs directory.

    Priority:
    1. FINRAG_PROMPTS_DIR environment variable
    2. configs/prompts/ relative to project root (3 levels up from this file)

    Returns:
        Path to the prompt configs directory.
    """
    env_dir = os.environ.get("FINRAG_PROMPTS_DIR")
    if env_dir:
        return Path(env_dir)

    # Walk up from this file to find configs/prompts/
    # This file: src/finrag/orchestration/prompt_config.py
    # Project root: ../../../../
    this_file = Path(__file__).resolve()
    project_root = this_file.parent.parent.parent.parent
    return project_root / "configs" / "prompts"


def load_generation_config(
    version: str = "v1",
    configs_dir: Path | None = None,
) -> GenerationPromptConfig:
    """Load a generation prompt config from YAML.

    Args:
        version: Version prefix (e.g., "v1" loads v1_generation.yaml).
        configs_dir: Override configs directory path.

    Returns:
        Validated GenerationPromptConfig.
    """
    global _generation_config

    if _generation_config is not None and _generation_config.version == version:
        return _generation_config

    directory = configs_dir or _resolve_configs_dir()
    config_path = directory / f"{version}_generation.yaml"

    if config_path.exists():
        with open(config_path) as f:
            raw = yaml.safe_load(f)

        config = GenerationPromptConfig(**raw)
        logger.info(
            "generation_config_loaded",
            version=config.version,
            path=str(config_path),
            model=config.model.name,
        )
    else:
        logger.warning(
            "generation_config_not_found",
            path=str(config_path),
            using="defaults",
        )
        config = GenerationPromptConfig(version=version)

    _generation_config = config
    return config


def load_retrieval_config(
    version: str = "v1",
    configs_dir: Path | None = None,
) -> RetrievalPromptConfig:
    """Load a retrieval prompt config from YAML.

    Args:
        version: Version prefix (e.g., "v1" loads v1_retrieval.yaml).
        configs_dir: Override configs directory path.

    Returns:
        Validated RetrievalPromptConfig.
    """
    global _retrieval_config

    if _retrieval_config is not None and _retrieval_config.version == version:
        return _retrieval_config

    directory = configs_dir or _resolve_configs_dir()
    config_path = directory / f"{version}_retrieval.yaml"

    if config_path.exists():
        with open(config_path) as f:
            raw = yaml.safe_load(f)

        config = RetrievalPromptConfig(**raw)
        logger.info(
            "retrieval_config_loaded",
            version=config.version,
            path=str(config_path),
        )
    else:
        logger.warning(
            "retrieval_config_not_found",
            path=str(config_path),
            using="defaults",
        )
        config = RetrievalPromptConfig(version=version)

    _retrieval_config = config
    return config


def reload_configs(version: str = "v1") -> None:
    """Force-reload configs from disk.

    Clears the module-level cache and re-reads YAML files.
    Useful during development when editing configs without restart.

    Args:
        version: Version prefix to reload.
    """
    global _generation_config, _retrieval_config
    _generation_config = None
    _retrieval_config = None
    load_generation_config(version)
    load_retrieval_config(version)
    logger.info("configs_reloaded", version=version)


def get_active_prompt_version() -> dict:
    """Return the currently active prompt versions.

    Returns:
        Dict with generation and retrieval version strings.
    """
    return {
        "generation": _generation_config.version if _generation_config else "not_loaded",
        "retrieval": _retrieval_config.version if _retrieval_config else "not_loaded",
    }
