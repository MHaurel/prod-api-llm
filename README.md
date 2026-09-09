# Production LLM API

A small FastAPI service that sends prompts to GPT-5.6 Luna through OpenRouter using LangChain.

## Run locally

Create an OpenRouter API key, then run:

```bash
cp .env.example .env
# Add your key to .env, then export it into the current shell.
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

Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

## Test

```bash
uv run pytest
```
