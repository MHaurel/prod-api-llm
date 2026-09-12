from pathlib import Path

import pytest

from app.cache import (
    DiskAnswerCache,
    InMemoryAnswerCache,
    SemanticAnswerCache,
    create_answer_cache,
)


class FakeEmbeddingProvider:
    def embed(self, text: str) -> list[float]:
        if "password" in text or "credentials" in text:
            return [1.0, 0.0]
        return [0.0, 1.0]


@pytest.mark.parametrize("cache_type", [InMemoryAnswerCache, DiskAnswerCache])
def test_lexical_caches_match_similar_prompts(
    cache_type: type[InMemoryAnswerCache] | type[DiskAnswerCache],
    tmp_path: Path,
) -> None:
    if cache_type is DiskAnswerCache:
        cache = cache_type(str(tmp_path / "answers.sqlite3"))
    else:
        cache = cache_type()

    cache.put("How do I reset my password?", "Follow these steps", "model-a")

    assert cache.get("How can I reset my password?", "model-a") == "Follow these steps"
    assert cache.get("How can I reset my password?", "model-b") is None


def test_memory_cache_evicts_least_recently_used_entry() -> None:
    cache = InMemoryAnswerCache(similarity_threshold=1, max_entries=2)
    cache.put("first", "one", "model")
    cache.put("second", "two", "model")
    assert cache.get("first", "model") == "one"

    cache.put("third", "three", "model")

    assert cache.get("first", "model") == "one"
    assert cache.get("second", "model") is None
    assert cache.get("third", "model") == "three"


def test_semantic_cache_matches_paraphrases() -> None:
    cache = SemanticAnswerCache(
        FakeEmbeddingProvider(), similarity_threshold=0.8, max_entries=10
    )
    cache.put("How do I reset my password?", "Follow these steps", "model-a")

    assert cache.get("I forgot my login credentials", "model-a") == "Follow these steps"
    assert cache.get("What is the weather?", "model-a") is None
    assert cache.get("I forgot my login credentials", "model-b") is None


@pytest.mark.parametrize(
    ("strategy", "expected_type"),
    [
        ("memory", InMemoryAnswerCache),
        ("disk", DiskAnswerCache),
        ("semantic", SemanticAnswerCache),
    ],
)
def test_cache_strategy_factory(
    strategy: str,
    expected_type: type[InMemoryAnswerCache]
    | type[DiskAnswerCache]
    | type[SemanticAnswerCache],
    tmp_path: Path,
) -> None:
    cache = create_answer_cache(
        strategy,
        path=str(tmp_path / "answers.sqlite3"),
        similarity_threshold=0.9,
        semantic_similarity_threshold=0.8,
        max_entries=10,
        embedding_model_name="unused",
        embedding_provider=FakeEmbeddingProvider(),
    )

    assert isinstance(cache, expected_type)


def test_cache_strategy_factory_rejects_unknown_strategy(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="CACHE_STRATEGY"):
        create_answer_cache(
            "unknown",
            path=str(tmp_path / "answers.sqlite3"),
            similarity_threshold=0.9,
            semantic_similarity_threshold=0.8,
            max_entries=10,
            embedding_model_name="unused",
        )
