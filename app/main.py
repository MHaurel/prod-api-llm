import json
import os
from functools import lru_cache
from typing import Annotated, AsyncIterator

from fastapi import Depends, FastAPI, HTTPException
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.responses import StreamingResponse

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
PRIMARY_MODEL_NAME = os.getenv("PRIMARY_MODEL_NAME", "openai/gpt-5.6-luna")
FALLBACK_MODEL_NAME = os.getenv("FALLBACK_MODEL_NAME", "openai/gpt-5.6-sol")
ANSWER_MAX_RETRIES = int(os.getenv("ANSWER_MAX_RETRIES", "2"))

if ANSWER_MAX_RETRIES < 0:
    raise ValueError("ANSWER_MAX_RETRIES must be zero or greater")


class AnswerRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=20_000)


class AnswerResponse(BaseModel):
    answer: str
    model: str


class GeneratedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)


ANSWER_FORMAT_PROMPT = f"""Return only valid JSON matching this JSON Schema:
{json.dumps(GeneratedAnswer.model_json_schema())}
Do not wrap the JSON in Markdown code fences or include any other text."""


@lru_cache
def get_model() -> ChatOpenAI:
    return create_model(PRIMARY_MODEL_NAME)


@lru_cache
def get_fallback_model() -> ChatOpenAI:
    return create_model(FALLBACK_MODEL_NAME)


def create_model(model_name: str) -> ChatOpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
    )


ModelDependency = Annotated[ChatOpenAI, Depends(get_model)]
FallbackModelDependency = Annotated[ChatOpenAI, Depends(get_fallback_model)]

app = FastAPI(title="Production LLM API", version="0.1.0")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/answers", response_model=AnswerResponse)
async def answer(
    request: AnswerRequest,
    model: ModelDependency,
    fallback_model: FallbackModelDependency,
) -> AnswerResponse:
    max_attempts = ANSWER_MAX_RETRIES + 1
    messages = [
        ("system", ANSWER_FORMAT_PROMPT),
        ("human", request.prompt),
    ]

    for attempt in range(max_attempts):
        current_model = model if attempt == 0 else fallback_model
        current_model_name = PRIMARY_MODEL_NAME if attempt == 0 else FALLBACK_MODEL_NAME

        try:
            response = await current_model.ainvoke(messages)
        except Exception:
            response = None

        if response is not None and isinstance(response.content, str):
            try:
                generated_answer = GeneratedAnswer.model_validate_json(response.content)
                return AnswerResponse(
                    answer=generated_answer.answer,
                    model=current_model_name,
                )
            except ValidationError:
                pass

        if attempt < max_attempts - 1:
            messages.append(
                (
                    "human",
                    "The previous generation attempt was unsuccessful. "
                    "Try again and return only the JSON object.",
                )
            )

    raise HTTPException(
        status_code=502,
        detail=f"The models did not return valid JSON after {max_attempts} attempts",
    )


async def stream_answer(prompt: str, model: ChatOpenAI) -> AsyncIterator[str]:
    try:
        async for chunk in model.astream(prompt):
            if isinstance(chunk.content, str) and chunk.content:
                data = json.dumps({"content": chunk.content})
                yield f"event: token\ndata: {data}\n\n"
    except Exception:
        data = json.dumps({"message": "The model request failed"})
        yield f"event: error\ndata: {data}\n\n"
        return

    data = json.dumps({"model": PRIMARY_MODEL_NAME})
    yield f"event: done\ndata: {data}\n\n"


@app.post("/v1/answers/stream")
async def answer_stream(request: AnswerRequest, model: ModelDependency) -> StreamingResponse:
    return StreamingResponse(
        stream_answer(request.prompt, model),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
