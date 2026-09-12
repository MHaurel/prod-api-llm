import re
import sqlite3
import threading
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Protocol


class AnswerCache(Protocol):
    def get(self, prompt: str, model: str) -> str | None: ...

    def put(self, prompt: str, answer: str, model: str) -> None: ...


class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> list[float]: ...


def normalize_prompt(prompt: str) -> str:
    normalized = unicodedata.normalize("NFKC", prompt).casefold()
    return " ".join(re.findall(r"\w+", normalized))


def _validate_settings(similarity_threshold: float, max_entries: int) -> None:
    if not 0 <= similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be between 0 and 1")
    if max_entries < 1:
        raise ValueError("max_entries must be at least 1")


class InMemoryAnswerCache:
    def __init__(
        self,
        similarity_threshold: float = 0.9,
        max_entries: int = 1_000,
    ) -> None:
        _validate_settings(similarity_threshold, max_entries)
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self._entries: OrderedDict[tuple[str, str], str] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, prompt: str, model: str) -> str | None:
        normalized_prompt = normalize_prompt(prompt)
        if not normalized_prompt:
            return None

        with self._lock:
            best_key: tuple[str, str] | None = None
            best_score = 0.0
            for key in self._entries:
                entry_model, entry_prompt = key
                if entry_model != model:
                    continue
                score = SequenceMatcher(
                    None, normalized_prompt, entry_prompt, autojunk=False
                ).ratio()
                if score > best_score:
                    best_key = key
                    best_score = score

            if best_key is None or best_score < self.similarity_threshold:
                return None

            answer = self._entries[best_key]
            self._entries.move_to_end(best_key)
            return answer

    def put(self, prompt: str, answer: str, model: str) -> None:
        normalized_prompt = normalize_prompt(prompt)
        if not normalized_prompt:
            return

        key = (model, normalized_prompt)
        with self._lock:
            self._entries[key] = answer
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)


class DiskAnswerCache:
    def __init__(
        self,
        path: str,
        similarity_threshold: float = 0.9,
        max_entries: int = 1_000,
    ) -> None:
        _validate_settings(similarity_threshold, max_entries)
        self.path = path
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self._lock = threading.Lock()

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS answer_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    normalized_prompt TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    last_accessed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    hit_count INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(model, normalized_prompt)
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS answer_cache_model_idx "
                "ON answer_cache(model)"
            )

    def get(self, prompt: str, model: str) -> str | None:
        normalized_prompt = normalize_prompt(prompt)
        if not normalized_prompt:
            return None

        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT id, normalized_prompt, answer FROM answer_cache WHERE model = ?",
                (model,),
            ).fetchall()

            best_row: sqlite3.Row | None = None
            best_score = 0.0
            for row in rows:
                score = SequenceMatcher(
                    None,
                    normalized_prompt,
                    row["normalized_prompt"],
                    autojunk=False,
                ).ratio()
                if score > best_score:
                    best_row = row
                    best_score = score

            if best_row is None or best_score < self.similarity_threshold:
                return None

            connection.execute(
                """
                UPDATE answer_cache
                SET last_accessed_at = CURRENT_TIMESTAMP, hit_count = hit_count + 1
                WHERE id = ?
                """,
                (best_row["id"],),
            )
            return str(best_row["answer"])

    def put(self, prompt: str, answer: str, model: str) -> None:
        normalized_prompt = normalize_prompt(prompt)
        if not normalized_prompt:
            return

        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO answer_cache (model, prompt, normalized_prompt, answer)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(model, normalized_prompt) DO UPDATE SET
                    prompt = excluded.prompt,
                    answer = excluded.answer,
                    last_accessed_at = CURRENT_TIMESTAMP
                """,
                (model, prompt, normalized_prompt, answer),
            )
            connection.execute(
                """
                DELETE FROM answer_cache
                WHERE id IN (
                    SELECT id FROM answer_cache
                    ORDER BY last_accessed_at ASC, id ASC
                    LIMIT MAX((SELECT COUNT(*) FROM answer_cache) - ?, 0)
                )
                """,
                (self.max_entries,),
            )


class SentenceTransformerEmbeddingProvider:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    @lru_cache(maxsize=2_048)
    def embed(self, text: str) -> list[float]:
        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding.tolist()


@dataclass
class SemanticCacheEntry:
    model: str
    normalized_prompt: str
    answer: str
    embedding: list[float]
    last_accessed: int


class SemanticAnswerCache:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        similarity_threshold: float = 0.8,
        max_entries: int = 1_000,
    ) -> None:
        _validate_settings(similarity_threshold, max_entries)
        self.embedding_provider = embedding_provider
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self._entries: list[SemanticCacheEntry] = []
        self._clock = 0
        self._lock = threading.Lock()

    def get(self, prompt: str, model: str) -> str | None:
        normalized_prompt = normalize_prompt(prompt)
        if not normalized_prompt:
            return None

        with self._lock:
            exact_entry = next(
                (
                    entry
                    for entry in self._entries
                    if entry.model == model
                    and entry.normalized_prompt == normalized_prompt
                ),
                None,
            )
            if exact_entry is not None:
                self._touch(exact_entry)
                return exact_entry.answer
            if not any(entry.model == model for entry in self._entries):
                return None

        embedding = self.embedding_provider.embed(normalized_prompt)
        with self._lock:
            candidates = [entry for entry in self._entries if entry.model == model]
            if not candidates:
                return None

            best_entry = max(candidates, key=lambda entry: _dot(embedding, entry.embedding))
            if _dot(embedding, best_entry.embedding) < self.similarity_threshold:
                return None

            self._touch(best_entry)
            return best_entry.answer

    def put(self, prompt: str, answer: str, model: str) -> None:
        normalized_prompt = normalize_prompt(prompt)
        if not normalized_prompt:
            return

        embedding = self.embedding_provider.embed(normalized_prompt)
        with self._lock:
            existing_entry = next(
                (
                    entry
                    for entry in self._entries
                    if entry.model == model
                    and entry.normalized_prompt == normalized_prompt
                ),
                None,
            )
            if existing_entry is not None:
                existing_entry.answer = answer
                existing_entry.embedding = embedding
                self._touch(existing_entry)
                return

            self._clock += 1
            self._entries.append(
                SemanticCacheEntry(
                    model=model,
                    normalized_prompt=normalized_prompt,
                    answer=answer,
                    embedding=embedding,
                    last_accessed=self._clock,
                )
            )
            if len(self._entries) > self.max_entries:
                least_recently_used = min(
                    self._entries, key=lambda entry: entry.last_accessed
                )
                self._entries.remove(least_recently_used)

    def _touch(self, entry: SemanticCacheEntry) -> None:
        self._clock += 1
        entry.last_accessed = self._clock


def _dot(left: list[float], right: list[float]) -> float:
    return sum(left_value * right_value for left_value, right_value in zip(left, right))


def create_answer_cache(
    strategy: str,
    *,
    path: str,
    similarity_threshold: float,
    semantic_similarity_threshold: float,
    max_entries: int,
    embedding_model_name: str,
    embedding_provider: EmbeddingProvider | None = None,
) -> AnswerCache:
    if strategy == "memory":
        return InMemoryAnswerCache(similarity_threshold, max_entries)
    if strategy == "disk":
        return DiskAnswerCache(path, similarity_threshold, max_entries)
    if strategy == "semantic":
        provider = embedding_provider or SentenceTransformerEmbeddingProvider(
            embedding_model_name
        )
        return SemanticAnswerCache(
            provider,
            semantic_similarity_threshold,
            max_entries,
        )
    raise ValueError("CACHE_STRATEGY must be one of: memory, disk, semantic")
