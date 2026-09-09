from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
import pytest

from app.main import app, get_model


class FakeModel:
    async def ainvoke(self, prompt: str) -> AIMessage:
        return AIMessage(content=f"Answer to: {prompt}")


@pytest.fixture(autouse=True)
def fake_model() -> None:
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
