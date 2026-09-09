import json
import os
from functools import lru_cache
from typing import Annotated, AsyncIterator

from fastapi import Depends, FastAPI, HTTPException
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.responses import StreamingResponse

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "openai/gpt-5.6-luna"
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
    max_attempts = ANSWER_MAX_RETRIES + 1
    messages = [
        ("system", ANSWER_FORMAT_PROMPT),
        ("human", request.prompt),
    ]

    for attempt in range(max_attempts):
        try:
            response = await model.ainvoke(messages)
        except Exception as exc:
            raise HTTPException(status_code=502, detail="The model request failed") from exc

        if isinstance(response.content, str):
            try:
                generated_answer = GeneratedAnswer.model_validate_json(response.content)
                return AnswerResponse(answer=generated_answer.answer, model=MODEL_NAME)
            except ValidationError:
                pass

        if attempt < max_attempts - 1:
            messages.append(
                (
                    "human",
                    "Your previous response did not match the required JSON schema. "
                    "Try again and return only the JSON object.",
                )
            )

    raise HTTPException(
        status_code=502,
        detail=f"The model did not return valid JSON after {max_attempts} attempts",
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

    data = json.dumps({"model": MODEL_NAME})
    yield f"event: done\ndata: {data}\n\n"


@app.post("/v1/answers/stream")
async def answer_stream(request: AnswerRequest, model: ModelDependency) -> StreamingResponse:
    return StreamingResponse(
        stream_answer(request.prompt, model),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
