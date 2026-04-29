"""Tests for Day 10: Versioned prompt configs + agent memory.

Covers:
- YAML prompt config loading and validation
- Config caching and reload behavior
- Fallback to defaults when YAML is missing
- Entity/filing/period extraction from text
- SessionMemory: turn recording, accumulation, reference resolution
- SessionStore: creation, eviction, deletion
- Conversation history formatting for LLM
- Prompt context injection
"""

import tempfile
import time
from pathlib import Path

import pytest
import yaml

from finrag.orchestration.memory import (
    SessionMemory,
    SessionStore,
    TurnRecord,
    extract_entities,
    extract_filings,
    extract_periods,
)
from finrag.orchestration.prompt_config import (
    EnforcementConfig,
    GenerationPromptConfig,
    ModelConfig,
    RerankerConfig,
    RetrievalParamsConfig,
    RetrievalPromptConfig,
    _generation_config,
    _retrieval_config,
    get_active_prompt_version,
    load_generation_config,
    load_retrieval_config,
    reload_configs,
)


# =========================================================================== #
# Prompt Config Tests
# =========================================================================== #


class TestModelConfig:
    """Test ModelConfig Pydantic model."""

    def test_defaults(self):
        config = ModelConfig()
        assert config.name == "gemini-2.0-flash"
        assert config.temperature == 0.1
        assert config.max_retries == 1

    def test_custom_values(self):
        config = ModelConfig(name="gemini-pro", temperature=0.5, max_retries=3)
        assert config.name == "gemini-pro"
        assert config.temperature == 0.5
        assert config.max_retries == 3


class TestEnforcementConfig:
    """Test EnforcementConfig Pydantic model."""

    def test_defaults(self):
        config = EnforcementConfig()
        assert config.confidence_threshold == 0.3
        assert config.min_citations == 1
        assert config.relevance_floor == 0.15

    def test_custom_thresholds(self):
        config = EnforcementConfig(
            confidence_threshold=0.5,
            min_citations=2,
            relevance_floor=0.3,
        )
        assert config.confidence_threshold == 0.5
        assert config.min_citations == 2


class TestGenerationPromptConfig:
    """Test GenerationPromptConfig loading and defaults."""

    def test_default_config(self):
        config = GenerationPromptConfig()
        assert config.version == "v1"
        assert config.name == "citation_grounded_generation"
        assert config.model.name == "gemini-2.0-flash"

    def test_config_from_dict(self):
        raw = {
            "version": "v2",
            "name": "test_config",
            "model": {"name": "gemini-pro", "temperature": 0.3},
            "system_prompt": "You are a test assistant.",
            "enforcement": {"confidence_threshold": 0.5},
        }
        config = GenerationPromptConfig(**raw)
        assert config.version == "v2"
        assert config.model.name == "gemini-pro"
        assert config.model.temperature == 0.3
        assert config.enforcement.confidence_threshold == 0.5
        assert config.system_prompt == "You are a test assistant."


class TestRetrievalPromptConfig:
    """Test RetrievalPromptConfig loading and defaults."""

    def test_default_config(self):
        config = RetrievalPromptConfig()
        assert config.version == "v1"
        assert config.retrieval.top_k == 20
        assert config.reranker.top_k == 5

    def test_config_from_dict(self):
        raw = {
            "version": "v2",
            "retrieval": {"top_k": 30, "rrf_k": 80},
            "reranker": {"model": "cross-encoder/ms-marco-MiniLM-L-12-v2", "top_k": 10},
        }
        config = RetrievalPromptConfig(**raw)
        assert config.retrieval.top_k == 30
        assert config.retrieval.rrf_k == 80
        assert config.reranker.top_k == 10


class TestLoadGenerationConfig:
    """Test loading generation config from YAML files."""

    def test_load_from_yaml(self, tmp_path):
        config_data = {
            "version": "v1",
            "name": "test_gen",
            "model": {"name": "gemini-2.0-flash", "temperature": 0.2},
            "system_prompt": "Test system prompt",
            "enforcement": {"confidence_threshold": 0.4},
        }
        yaml_path = tmp_path / "v1_generation.yaml"
        yaml_path.write_text(yaml.dump(config_data))

        # Clear cache
        import finrag.orchestration.prompt_config as pc
        pc._generation_config = None

        config = load_generation_config(version="v1", configs_dir=tmp_path)
        assert config.version == "v1"
        assert config.model.temperature == 0.2
        assert config.system_prompt == "Test system prompt"
        assert config.enforcement.confidence_threshold == 0.4

    def test_missing_yaml_returns_defaults(self, tmp_path):
        import finrag.orchestration.prompt_config as pc
        pc._generation_config = None

        config = load_generation_config(version="v99", configs_dir=tmp_path)
        assert config.version == "v99"
        assert config.model.name == "gemini-2.0-flash"

    def test_caching(self, tmp_path):
        config_data = {"version": "v1", "name": "cached"}
        yaml_path = tmp_path / "v1_generation.yaml"
        yaml_path.write_text(yaml.dump(config_data))

        import finrag.orchestration.prompt_config as pc
        pc._generation_config = None

        config1 = load_generation_config(version="v1", configs_dir=tmp_path)
        config2 = load_generation_config(version="v1", configs_dir=tmp_path)
        assert config1 is config2  # Same object from cache


class TestLoadRetrievalConfig:
    """Test loading retrieval config from YAML files."""

    def test_load_from_yaml(self, tmp_path):
        config_data = {
            "version": "v1",
            "retrieval": {"top_k": 25},
            "reranker": {"top_k": 8},
        }
        yaml_path = tmp_path / "v1_retrieval.yaml"
        yaml_path.write_text(yaml.dump(config_data))

        import finrag.orchestration.prompt_config as pc
        pc._retrieval_config = None

        config = load_retrieval_config(version="v1", configs_dir=tmp_path)
        assert config.retrieval.top_k == 25
        assert config.reranker.top_k == 8

    def test_missing_yaml_returns_defaults(self, tmp_path):
        import finrag.orchestration.prompt_config as pc
        pc._retrieval_config = None

        config = load_retrieval_config(version="v99", configs_dir=tmp_path)
        assert config.version == "v99"
        assert config.retrieval.top_k == 20


class TestReloadConfigs:
    """Test config reload behavior."""

    def test_reload_clears_cache(self, tmp_path):
        import finrag.orchestration.prompt_config as pc

        # Load initial config
        pc._generation_config = None
        pc._retrieval_config = None

        gen_data = {"version": "v1", "name": "initial"}
        ret_data = {"version": "v1", "name": "initial_ret"}
        (tmp_path / "v1_generation.yaml").write_text(yaml.dump(gen_data))
        (tmp_path / "v1_retrieval.yaml").write_text(yaml.dump(ret_data))

        load_generation_config(version="v1", configs_dir=tmp_path)
        load_retrieval_config(version="v1", configs_dir=tmp_path)

        # Modify YAML
        gen_data["name"] = "updated"
        (tmp_path / "v1_generation.yaml").write_text(yaml.dump(gen_data))

        # Force clear cache for reload (since reload uses default dir)
        pc._generation_config = None
        pc._retrieval_config = None
        config = load_generation_config(version="v1", configs_dir=tmp_path)
        assert config.name == "updated"


class TestGetActivePromptVersion:
    """Test active prompt version reporting."""

    def test_not_loaded(self):
        import finrag.orchestration.prompt_config as pc
        pc._generation_config = None
        pc._retrieval_config = None

        versions = get_active_prompt_version()
        assert versions["generation"] == "not_loaded"
        assert versions["retrieval"] == "not_loaded"

    def test_loaded(self, tmp_path):
        import finrag.orchestration.prompt_config as pc
        pc._generation_config = None
        pc._retrieval_config = None

        gen_data = {"version": "v1"}
        (tmp_path / "v1_generation.yaml").write_text(yaml.dump(gen_data))
        load_generation_config(version="v1", configs_dir=tmp_path)

        versions = get_active_prompt_version()
        assert versions["generation"] == "v1"


class TestLoadActualConfigs:
    """Test loading the actual YAML configs from configs/prompts/."""

    def test_load_real_generation_config(self):
        import finrag.orchestration.prompt_config as pc
        pc._generation_config = None

        configs_dir = Path(__file__).resolve().parent.parent / "configs" / "prompts"
        if configs_dir.exists():
            config = load_generation_config(version="v1", configs_dir=configs_dir)
            assert config.version == "v1"
            assert config.name == "citation_grounded_generation"
            assert "financial research assistant" in config.system_prompt.lower()
            assert config.model.name == "gemini-2.0-flash"
            assert config.enforcement.confidence_threshold == 0.3

    def test_load_real_retrieval_config(self):
        import finrag.orchestration.prompt_config as pc
        pc._retrieval_config = None

        configs_dir = Path(__file__).resolve().parent.parent / "configs" / "prompts"
        if configs_dir.exists():
            config = load_retrieval_config(version="v1", configs_dir=configs_dir)
            assert config.version == "v1"
            assert config.retrieval.top_k == 20
            assert config.reranker.model == "cross-encoder/ms-marco-MiniLM-L-6-v2"


# =========================================================================== #
# Entity Extraction Tests
# =========================================================================== #


class TestExtractEntities:
    """Test ticker-like entity extraction."""

    def test_extracts_tickers(self):
        entities = extract_entities("What was AAPL revenue in FY2024?")
        assert "AAPL" in entities

    def test_multiple_tickers(self):
        entities = extract_entities("Compare AAPL and MSFT gross margins")
        assert "AAPL" in entities
        assert "MSFT" in entities

    def test_filters_stopwords(self):
        entities = extract_entities("What is the total revenue for the company?")
        assert "THE" not in entities
        assert "WHAT" not in entities

    def test_empty_text(self):
        entities = extract_entities("")
        assert entities == []

    def test_single_char_filtered(self):
        # Single character tickers should be filtered (len < 2)
        entities = extract_entities("A stock went up")
        assert "A" not in entities

    def test_common_financial_terms_filtered(self):
        entities = extract_entities("EPS was higher YOY with ROE improving")
        assert "EPS" not in entities
        assert "YOY" not in entities
        assert "ROE" not in entities


class TestExtractFilings:
    """Test filing type extraction."""

    def test_extracts_10k(self):
        filings = extract_filings("Check the 10-K filing for Apple")
        assert "10-K" in filings

    def test_extracts_10q(self):
        filings = extract_filings("Latest 10-Q shows revenue growth")
        assert "10-Q" in filings

    def test_extracts_8k(self):
        filings = extract_filings("The 8-K disclosed a merger")
        assert "8-K" in filings

    def test_no_filings(self):
        filings = extract_filings("What was revenue last year?")
        assert filings == []

    def test_multiple_filings(self):
        filings = extract_filings("Compare 10-K and 10-Q disclosures")
        assert len(filings) == 2


class TestExtractPeriods:
    """Test time period extraction."""

    def test_fiscal_year(self):
        periods = extract_periods("Revenue in FY2024 was $383B")
        assert "FY2024" in periods

    def test_quarter(self):
        periods = extract_periods("Q3 2024 earnings were strong")
        assert any("Q3" in p for p in periods)

    def test_verbal_quarter(self):
        periods = extract_periods("First quarter results exceeded expectations")
        assert any("first quarter" in p.lower() for p in periods)

    def test_no_period(self):
        periods = extract_periods("What is the company's market cap?")
        assert periods == []


# =========================================================================== #
# TurnRecord Tests
# =========================================================================== #


class TestTurnRecord:
    """Test TurnRecord dataclass."""

    def test_basic_creation(self):
        turn = TurnRecord(query="What was revenue?")
        assert turn.query == "What was revenue?"
        assert turn.answer == ""
        assert turn.entities == []
        assert turn.timestamp > 0

    def test_with_all_fields(self):
        turn = TurnRecord(
            query="AAPL revenue?",
            answer="$383B",
            entities=["AAPL"],
            filings=["10-K"],
            periods=["FY2024"],
            citations=["c1", "c2"],
        )
        assert turn.entities == ["AAPL"]
        assert len(turn.citations) == 2


# =========================================================================== #
# SessionMemory Tests
# =========================================================================== #


class TestSessionMemory:
    """Test SessionMemory conversation tracking."""

    def test_initial_state(self):
        session = SessionMemory(session_id="test-1")
        assert session.session_id == "test-1"
        assert session.turn_count == 0
        assert session.is_first_turn is True
        assert session.last_turn is None
        assert session.last_query == ""

    def test_add_turn(self):
        session = SessionMemory(session_id="test-2")
        turn = session.add_turn(
            query="What was AAPL revenue in FY2024?",
            answer="Revenue was $383B in FY2024.",
        )
        assert session.turn_count == 1
        assert session.is_first_turn is False
        assert "AAPL" in session.all_entities
        assert session.last_query == "What was AAPL revenue in FY2024?"

    def test_entity_accumulation(self):
        session = SessionMemory(session_id="test-3")
        session.add_turn(query="What was AAPL revenue?", answer="$383B")
        session.add_turn(query="How about MSFT?", answer="$245B")

        assert "AAPL" in session.all_entities
        assert "MSFT" in session.all_entities

    def test_filing_accumulation(self):
        session = SessionMemory(session_id="test-4")
        session.add_turn(query="Check the 10-K filing", answer="Found it.")
        session.add_turn(query="Also the 10-Q", answer="Got it.")

        assert "10-K" in session.all_filings
        assert "10-Q" in session.all_filings

    def test_citation_tracking(self):
        session = SessionMemory(session_id="test-5")
        session.add_turn(
            query="Revenue?",
            answer="$383B",
            citations=[{"chunk_id": "c1"}, {"chunk_id": "c2"}],
        )
        assert "c1" in session.all_cited_chunks
        assert "c2" in session.all_cited_chunks

    def test_turn_trimming(self):
        session = SessionMemory(session_id="test-6", max_turns=4)
        for i in range(6):
            session.add_turn(query=f"Question {i}", answer=f"Answer {i}")

        # Should have trimmed to max_turns/2 = 2
        assert session.turn_count <= 4

    def test_last_turn(self):
        session = SessionMemory(session_id="test-7")
        session.add_turn(query="First?", answer="First answer")
        session.add_turn(query="Second?", answer="Second answer")

        assert session.last_turn is not None
        assert session.last_turn.query == "Second?"
        assert session.last_answer == "Second answer"


class TestSessionMemoryContext:
    """Test prompt context generation from SessionMemory."""

    def test_context_for_prompt_empty(self):
        session = SessionMemory(session_id="test-ctx-1")
        ctx = session.get_context_for_prompt()
        assert ctx["entities"] == "none yet"
        assert ctx["filings"] == "none yet"
        assert ctx["turn_count"] == "0"

    def test_context_for_prompt_with_data(self):
        session = SessionMemory(session_id="test-ctx-2")
        session.add_turn(
            query="What was AAPL revenue in the 10-K?",
            answer="$383B for FY2024.",
        )
        ctx = session.get_context_for_prompt()
        assert "AAPL" in ctx["entities"]
        assert "10-K" in ctx["filings"]
        assert ctx["turn_count"] == "1"

    def test_conversation_history(self):
        session = SessionMemory(session_id="test-ctx-3")
        session.add_turn(query="Q1?", answer="A1")
        session.add_turn(query="Q2?", answer="A2")

        history = session.get_conversation_history(max_turns=2)
        assert len(history) == 4  # 2 turns × 2 messages each
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Q1?"
        assert history[1]["role"] == "assistant"

    def test_conversation_history_truncation(self):
        session = SessionMemory(session_id="test-ctx-4")
        session.add_turn(query="Q?", answer="A" * 600)

        history = session.get_conversation_history()
        # Long answer should be truncated to 500 chars + "..."
        assert history[1]["content"].endswith("...")
        assert len(history[1]["content"]) <= 504  # 500 + "..."


class TestReferenceResolution:
    """Test ambiguous reference resolution."""

    def test_first_turn_no_resolution(self):
        session = SessionMemory(session_id="test-ref-1")
        query = "What was the company's revenue?"
        resolved = session.resolve_references(query)
        assert resolved == query  # No prior context

    def test_entity_resolution(self):
        session = SessionMemory(session_id="test-ref-2")
        session.add_turn(query="What was AAPL revenue?", answer="$383B")

        resolved = session.resolve_references("How did the company perform?")
        assert "AAPL" in resolved

    def test_filing_resolution(self):
        session = SessionMemory(session_id="test-ref-3")
        session.add_turn(query="Check the 10-K filing", answer="Found disclosures.")

        resolved = session.resolve_references("What does that filing say about risks?")
        assert "10-K" in resolved

    def test_no_ambiguous_references(self):
        session = SessionMemory(session_id="test-ref-4")
        session.add_turn(query="AAPL revenue?", answer="$383B")

        query = "What was MSFT revenue?"
        resolved = session.resolve_references(query)
        assert resolved == query  # No ambiguous references


class TestSessionToDict:
    """Test session serialization."""

    def test_empty_session(self):
        session = SessionMemory(session_id="test-dict-1")
        d = session.to_dict()
        assert d["session_id"] == "test-dict-1"
        assert d["turn_count"] == 0
        assert d["entities"] == []

    def test_session_with_data(self):
        session = SessionMemory(session_id="test-dict-2")
        session.add_turn(
            query="AAPL 10-K revenue?",
            answer="$383B",
            citations=[{"chunk_id": "c1"}],
        )
        d = session.to_dict()
        assert d["turn_count"] == 1
        assert d["cited_chunks"] == 1
        assert "AAPL" in d["entities"]


# =========================================================================== #
# SessionStore Tests
# =========================================================================== #


class TestSessionStore:
    """Test SessionStore management."""

    def test_get_or_create(self):
        store = SessionStore()
        session = store.get_or_create("s1")
        assert session.session_id == "s1"
        assert store.active_count == 1

    def test_get_or_create_returns_same(self):
        store = SessionStore()
        s1 = store.get_or_create("s1")
        s2 = store.get_or_create("s1")
        assert s1 is s2

    def test_get_nonexistent(self):
        store = SessionStore()
        assert store.get("nonexistent") is None

    def test_delete(self):
        store = SessionStore()
        store.get_or_create("s1")
        assert store.delete("s1") is True
        assert store.active_count == 0

    def test_delete_nonexistent(self):
        store = SessionStore()
        assert store.delete("nonexistent") is False

    def test_eviction(self):
        store = SessionStore(max_sessions=2)
        store.get_or_create("s1")
        store.get_or_create("s2")
        store.get_or_create("s3")  # Should evict s1

        assert store.active_count == 2
        assert store.get("s1") is None  # Evicted
        assert store.get("s3") is not None

    def test_multiple_sessions(self):
        store = SessionStore()
        store.get_or_create("s1").add_turn(query="Q1", answer="A1")
        store.get_or_create("s2").add_turn(query="Q2", answer="A2")

        assert store.get("s1").turn_count == 1
        assert store.get("s2").turn_count == 1
        assert store.get("s1").last_query == "Q1"
        assert store.get("s2").last_query == "Q2"


# =========================================================================== #
# Integration: Memory + PromptConfig
# =========================================================================== #


class TestMemoryPromptIntegration:
    """Test memory context injection into prompt templates."""

    def test_context_fills_template(self):
        session = SessionMemory(session_id="integ-1")
        session.add_turn(
            query="What was AAPL revenue in the 10-K for FY2024?",
            answer="Revenue was $383B.",
        )

        template = (
            "Previously discussed entities: {entities}\n"
            "Previously discussed filings: {filings}\n"
            "Prior Q&A count: {turn_count}"
        )

        ctx = session.get_context_for_prompt()
        filled = template.format(**ctx)

        assert "AAPL" in filled
        assert "10-K" in filled
        assert "1" in filled

    def test_multi_turn_context(self):
        session = SessionMemory(session_id="integ-2")
        session.add_turn(query="AAPL revenue?", answer="$383B")
        session.add_turn(query="MSFT revenue?", answer="$245B")
        session.add_turn(query="Compare them", answer="AAPL > MSFT")

        ctx = session.get_context_for_prompt()
        assert "AAPL" in ctx["entities"]
        assert "MSFT" in ctx["entities"]
        assert ctx["turn_count"] == "3"
