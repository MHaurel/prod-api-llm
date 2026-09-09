from collections.abc import AsyncIterator, Iterator

from fastapi.testclient import TestClient
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from app.main import app, get_model


class FakeModel:
    async def ainvoke(self, prompt: str) -> AIMessage:
        return AIMessage(content=f"Answer to: {prompt}")

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
