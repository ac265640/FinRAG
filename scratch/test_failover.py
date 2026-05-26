import os
import json
import logging
from unittest.mock import patch, MagicMock

# Set up logging to show the fallback process clearly
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Ensure virtual env is used or dependencies are available
from finrag.orchestration.generator import RAGGenerator
from finrag.orchestration.schemas import CitedAnswer

def run_failover_tests():
    print("==================================================")
    print("Starting Multi-Provider LLM Fallback Failover Tests")
    print("==================================================")

    # Some valid context chunks
    context_chunks = [
        {
            "chunk_id": "aapl_rev_001",
            "text": "Apple FY2024 revenue was $383.3 billion.",
            "metadata": {"ticker": "AAPL", "filing": "10-K"},
            "reranker_score": 0.95
        }
    ]
    query = "What was Apple's FY2024 revenue?"

    # Setup generator with fake keys so they are not skipped
    gen = RAGGenerator(
        api_key="fake-primary",
        model_name="gemini-2.5-flash",
        temperature=0.1
    )
    # Populate other keys
    gen._keys["gemini_secondary"] = "fake-secondary"
    gen._keys["cohere"] = "fake-cohere"
    gen._keys["openai"] = "fake-openai"

    print("\n--- Test Case 1: Primary Gemini works perfectly ---")
    good_json = json.dumps({
        "answer_text": "Apple revenue was $383.3B.",
        "citations": [{"chunk_id": "aapl_rev_001", "relevance_score": 0.95}],
        "confidence": 0.95,
        "reasoning": "Extracted from source."
    })

    with patch.object(gen, "_invoke_provider", return_value=good_json) as mock_invoke:
        answer, passed, errors = gen.generate(query, context_chunks)
        print(f"Result: passed={passed}, confidence={answer.confidence}")
        print(f"Answer: {answer.answer_text}")
        assert passed is True
        assert mock_invoke.call_count == 1
        # Check first call was gemini_primary
        args, kwargs = mock_invoke.call_args
        assert kwargs["name"] == "gemini_primary"
        print("✅ Test Case 1 Passed!")

    print("\n--- Test Case 2: Primary Gemini hits 429/quota, falls back to Secondary Gemini ---")
    def side_effect_case_2(name, **kwargs):
        if name == "gemini_primary":
            raise ValueError("RESOURCE_EXHAUSTED: Rate limit exceeded (429)")
        return good_json

    with patch.object(gen, "_invoke_provider", side_effect=side_effect_case_2) as mock_invoke:
        answer, passed, errors = gen.generate(query, context_chunks)
        print(f"Result: passed={passed}, confidence={answer.confidence}")
        print(f"Answer: {answer.answer_text}")
        assert passed is True
        assert mock_invoke.call_count == 2
        # Check second call was gemini_secondary
        calls = mock_invoke.call_args_list
        assert calls[0][1]["name"] == "gemini_primary"
        assert calls[1][1]["name"] == "gemini_secondary"
        print("✅ Test Case 2 Passed!")

    print("\n--- Test Case 3: Both Geminis fail, falls back to Cohere ---")
    def side_effect_case_3(name, **kwargs):
        if name in ("gemini_primary", "gemini_secondary"):
            raise ValueError("RESOURCE_EXHAUSTED: 429 rate limit")
        return good_json

    with patch.object(gen, "_invoke_provider", side_effect=side_effect_case_3) as mock_invoke:
        answer, passed, errors = gen.generate(query, context_chunks)
        print(f"Result: passed={passed}, confidence={answer.confidence}")
        print(f"Answer: {answer.answer_text}")
        assert passed is True
        assert mock_invoke.call_count == 3
        calls = mock_invoke.call_args_list
        assert calls[0][1]["name"] == "gemini_primary"
        assert calls[1][1]["name"] == "gemini_secondary"
        assert calls[2][1]["name"] == "cohere"
        print("✅ Test Case 3 Passed!")

    print("\n--- Test Case 4: Geminis and Cohere fail, falls back to OpenAI ---")
    def side_effect_case_4(name, **kwargs):
        if name in ("gemini_primary", "gemini_secondary"):
            raise ValueError("429 RESOURCE_EXHAUSTED")
        if name == "cohere":
            raise ValueError("Cohere API error: model overloaded")
        return good_json

    with patch.object(gen, "_invoke_provider", side_effect=side_effect_case_4) as mock_invoke:
        answer, passed, errors = gen.generate(query, context_chunks)
        print(f"Result: passed={passed}, confidence={answer.confidence}")
        print(f"Answer: {answer.answer_text}")
        assert passed is True
        assert mock_invoke.call_count == 4
        calls = mock_invoke.call_args_list
        assert calls[0][1]["name"] == "gemini_primary"
        assert calls[1][1]["name"] == "gemini_secondary"
        assert calls[2][1]["name"] == "cohere"
        assert calls[3][1]["name"] == "openai"
        print("✅ Test Case 4 Passed!")

    print("\n--- Test Case 5: All providers fail gracefully ---")
    with patch.object(gen, "_invoke_provider", side_effect=ValueError("Global outage")) as mock_invoke:
        answer, passed, errors = gen.generate(query, context_chunks)
        print(f"Result: passed={passed}, confidence={answer.confidence}")
        print(f"Answer: {answer.answer_text}")
        assert passed is False
        assert mock_invoke.call_count == 8
        assert "temporarily unavailable" in answer.answer_text
        print("✅ Test Case 5 Passed!")

    print("\n==================================================")
    print("All Multi-Provider LLM Fallback Failover Tests PASSED!")
    print("==================================================")

if __name__ == "__main__":
    run_failover_tests()
