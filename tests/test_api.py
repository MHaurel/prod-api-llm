from collections.abc import AsyncIterator, Iterator

from fastapi.testclient import TestClient
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

import app.main as main
from app.main import app, get_fallback_model, get_model


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
    app.dependency_overrides[get_fallback_model] = lambda: FakeModel()
    yield
    app.dependency_overrides.clear()


def test_answer() -> None:
    with TestClient(app) as client:
        response = client.post("/v1/answers", json={"prompt": "Hello"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Answer to: Hello",
        "model": main.PRIMARY_MODEL_NAME,
    }


def test_empty_prompt_is_rejected() -> None:
    with TestClient(app) as client:
        response = client.post("/v1/answers", json={"prompt": ""})

    assert response.status_code == 422


def test_answer_retries_invalid_json() -> None:
    class InvalidModel:
        attempts = 0

        async def ainvoke(self, messages: list[tuple[str, str]]) -> AIMessage:
            self.attempts += 1
            return AIMessage(content="not json")

    class FallbackModel:
        attempts = 0

        async def ainvoke(self, messages: list[tuple[str, str]]) -> AIMessage:
            self.attempts += 1
            return AIMessage(content='{"answer": "Valid answer"}')

    primary_model = InvalidModel()
    fallback_model = FallbackModel()
    app.dependency_overrides[get_model] = lambda: primary_model
    app.dependency_overrides[get_fallback_model] = lambda: fallback_model

    with TestClient(app) as client:
        response = client.post("/v1/answers", json={"prompt": "Hello"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Valid answer",
        "model": main.FALLBACK_MODEL_NAME,
    }
    assert primary_model.attempts == 1
    assert fallback_model.attempts == 1


def test_answer_uses_fallback_after_primary_request_fails() -> None:
    class FailingModel:
        async def ainvoke(self, messages: list[tuple[str, str]]) -> AIMessage:
            raise RuntimeError("Model unavailable")

    app.dependency_overrides[get_model] = lambda: FailingModel()

    with TestClient(app) as client:
        response = client.post("/v1/answers", json={"prompt": "Hello"})

    assert response.status_code == 200
    assert response.json()["model"] == main.FALLBACK_MODEL_NAME


def test_answer_fails_after_configured_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    class InvalidModel:
        attempts = 0

        async def ainvoke(self, messages: list[tuple[str, str]]) -> AIMessage:
            self.attempts += 1
            return AIMessage(content='{"wrong_field": "Invalid answer"}')

    model = InvalidModel()
    monkeypatch.setattr(main, "ANSWER_MAX_RETRIES", 1)
    app.dependency_overrides[get_model] = lambda: model
    app.dependency_overrides[get_fallback_model] = lambda: model

    with TestClient(app) as client:
        response = client.post("/v1/answers", json={"prompt": "Hello"})

    assert response.status_code == 502
    assert response.json() == {
        "detail": "The models did not return valid JSON after 2 attempts"
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
        f'event: done\ndata: {{"model": "{main.PRIMARY_MODEL_NAME}"}}\n\n'
    )
