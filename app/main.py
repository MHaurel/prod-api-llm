import os
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "openai/gpt-5.6-luna"


class AnswerRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=20_000)


class AnswerResponse(BaseModel):
    answer: str
    model: str


@lru_cache
def get_model() -> ChatOpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    return ChatOpenAI(
        model=MODEL_NAME,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
    )


ModelDependency = Annotated[ChatOpenAI, Depends(get_model)]

app = FastAPI(title="Production LLM API", version="0.1.0")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/answers", response_model=AnswerResponse)
async def answer(request: AnswerRequest, model: ModelDependency) -> AnswerResponse:
    try:
        response = await model.ainvoke(request.prompt)
    except Exception as exc:
        raise HTTPException(status_code=502, detail="The model request failed") from exc

    if not isinstance(response.content, str):
        raise HTTPException(status_code=502, detail="The model returned an unsupported response")

    return AnswerResponse(answer=response.content, model=MODEL_NAME)

