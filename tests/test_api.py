from collections.abc import AsyncIterator, Iterator

from fastapi.testclient import TestClient
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

import app.main as main
from app.main import app, get_model


class FakeModel:
    async def ainvoke(self, messages: list[tuple[str, str]]) -> AIMessage:
        prompt = messages[1][1]
        return AIMessage(content=f'{{"answer": "Answer to: {prompt}"}}')

    async def astream(self, prompt: str) -> AsyncIterator[AIMessageChunk]:
        for content in ["Answer ", f"to: {prompt}"]:
            yield AIMessageChunk(content=content)


@pytest.fixture(autouse=True)
def fake_model() -> Iterator[None]:
    app.dependency_overrides[get_model] = lambda: FakeModel()
    yield
    app.dependency_overrides.clear()


def test_answer() -> None:
    with TestClient(app) as client:
        response = client.post("/v1/answers", json={"prompt": "Hello"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Answer to: Hello",
        "model": "openai/gpt-5.6-luna",
    }


def test_empty_prompt_is_rejected() -> None:
    with TestClient(app) as client:
        response = client.post("/v1/answers", json={"prompt": ""})

    assert response.status_code == 422


def test_answer_retries_invalid_json() -> None:
    class RetryModel:
        attempts = 0

        async def ainvoke(self, messages: list[tuple[str, str]]) -> AIMessage:
            self.attempts += 1
            if self.attempts == 1:
                return AIMessage(content="not json")
            return AIMessage(content='{"answer": "Valid answer"}')

    model = RetryModel()
    app.dependency_overrides[get_model] = lambda: model

    with TestClient(app) as client:
        response = client.post("/v1/answers", json={"prompt": "Hello"})

    assert response.status_code == 200
    assert response.json()["answer"] == "Valid answer"
    assert model.attempts == 2


def test_answer_fails_after_configured_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    class InvalidModel:
        attempts = 0

        async def ainvoke(self, messages: list[tuple[str, str]]) -> AIMessage:
            self.attempts += 1
            return AIMessage(content='{"wrong_field": "Invalid answer"}')

    model = InvalidModel()
    monkeypatch.setattr(main, "ANSWER_MAX_RETRIES", 1)
    app.dependency_overrides[get_model] = lambda: model

    with TestClient(app) as client:
        response = client.post("/v1/answers", json={"prompt": "Hello"})

    assert response.status_code == 502
    assert response.json() == {
        "detail": "The model did not return valid JSON after 2 attempts"
    }
    assert model.attempts == 2


def test_stream_answer() -> None:
    with TestClient(app) as client:
        response = client.post("/v1/answers/stream", json={"prompt": "Hello"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text == (
        'event: token\ndata: {"content": "Answer "}\n\n'
        'event: token\ndata: {"content": "to: Hello"}\n\n'
        'event: done\ndata: {"model": "openai/gpt-5.6-luna"}\n\n'
    )
