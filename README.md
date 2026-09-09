# Production LLM API

A small FastAPI service that sends prompts to GPT-5.6 Luna through OpenRouter using LangChain.

## Run locally

Create an OpenRouter API key, then run:

```bash
cp .env.example .env
# Add your key to .env and adjust ANSWER_MAX_RETRIES if needed, then export it.
set -a
source .env
set +a

uv sync
uv run uvicorn app.main:app --reload
```

Send a prompt:

```bash
curl http://127.0.0.1:8000/v1/answers \
  --header 'Content-Type: application/json' \
  --data '{"prompt":"Why is the sky blue?"}'
```

The response has this shape:

```json
{
  "answer": "...",
  "model": "openai/gpt-5.6-luna"
}
```

Internally, the model is instructed to return this JSON format:

```json
{
  "answer": "..."
}
```

The API validates that format and retries generation according to
`ANSWER_MAX_RETRIES` in `.env` (default: `2`, in addition to the initial attempt).
After all attempts return invalid JSON, it returns HTTP 502.
The format can be changed in the `GeneratedAnswer` model in `app/main.py`.

Stream an answer with server-sent events (SSE):

```bash
curl --no-buffer http://127.0.0.1:8000/v1/answers/stream \
  --header 'Content-Type: application/json' \
  --data '{"prompt":"Why is the sky blue?"}'
```

The stream emits `token` events as text arrives and ends with a `done` event:

```text
event: token
data: {"content": "The sky"}

event: token
data: {"content": " is blue..."}

event: done
data: {"model": "openai/gpt-5.6-luna"}
```

To see the answer appear token by token in your terminal, run:

```bash
uv run python scripts/stream_answer.py "Why is the sky blue?"
```

Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

## Test

```bash
uv run pytest
```
