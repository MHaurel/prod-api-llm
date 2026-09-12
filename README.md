# Production LLM API

A small FastAPI service that sends prompts to configurable OpenRouter models using LangChain.

## Run locally

Create an OpenRouter API key, then run:

```bash
cp .env.example .env
# Add your key to .env and adjust the model or retry settings if needed, then export it.
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
The first attempt uses `PRIMARY_MODEL_NAME`; retries use `FALLBACK_MODEL_NAME`.
Failures include both model request errors and responses that do not match the JSON
format. After all attempts fail, the API returns HTTP 502.
The format can be changed in the `GeneratedAnswer` model in `app/main.py`.

Successful JSON answers can use one of three model-specific cache strategies. Select
one with `CACHE_STRATEGY`:

- `memory`: fuzzy prompt matching in process memory; cleared when the process stops.
- `disk` (default): the same fuzzy matching persisted in a local SQLite database.
- `semantic`: cosine similarity over local sentence-transformer embeddings, stored
  in process memory. The embedding model is downloaded the first time it is used.

The fuzzy strategies are case-insensitive and ignore punctuation and repeated
whitespace. Additional settings are:

- `CACHE_DB_PATH` (default: `.cache/answers.sqlite3`; used by `disk`)
- `CACHE_SIMILARITY_THRESHOLD` (default: `0.9`; used by `memory` and `disk`)
- `CACHE_SEMANTIC_SIMILARITY_THRESHOLD` (default: `0.8`)
- `CACHE_MAX_ENTRIES` (default: `1000`; least-recently-used entries are removed first)
- `EMBEDDING_MODEL_NAME` (default: `sentence-transformers/all-MiniLM-L6-v2`)

The streaming endpoint is not cached because it has a different output contract.

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

Streaming uses the primary model only and does not retry because its tokens are sent
to the client immediately.

To see the answer appear token by token in your terminal, run:

```bash
uv run python scripts/stream_answer.py "Why is the sky blue?"
```

Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

## Test

```bash
uv run pytest
```
