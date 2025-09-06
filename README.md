## small-search-agent

A minimal agentic demo showing how small language models can perform web-grounded QA using Structure-Guided Generation (SGG) with strict JSON schemas. With SGG, even a ~1B model (runnable on a laptop) decides when to search the web and then composes a grounded answer from retrieved sources.

### Key features
- Schema-guided decision: the model strictly outputs JSON (Pydantic schemas) to choose between “search” or “answer”.
- Web search grounding: integrates Tavily to retrieve fresh, relevant sources.
- Small model friendly: works with local Ollama models (e.g., `gemma3:1b`).
- Environment config via `.env` (optional).

### Requirements
- Python 3.12+
- [uv](https://astral.sh) installed
- [Ollama](https://ollama.ai) running locally with a small model available (default `gemma3:1b`)
- Tavily API key (`TAVILY_API_KEY`)

### Quickstart
1) Create and sync the environment (project already defines dependencies in `pyproject.toml`):
```bash
uv venv
uv sync
```

2) Provide your Tavily key:
```bash
export TAVILY_API_KEY="your_key_here"
# or put it into .env (python-dotenv is supported)
```

3) Run
- Recommended:
```bash
uv run python -m main "Tell me three quirky facts about wolves"
```
- Select model (optional, defaults to env `OLLAMA_MODEL` or `gemma3:1b`):
```bash
uv run python -m main --model gemma3:1b "What is CRDT in one paragraph?"
```

### How it works (high level)
1) Decision step: the SLM returns strict JSON selecting either:
   - `search` with 1–3 web queries, or
   - `answer` directly.
2) If `search`, the app uses Tavily to fetch results, builds a compact sources block, and asks the model to synthesize a grounded answer from those sources only.

### Configuration
- `TAVILY_API_KEY`: required for web search.
- `OLLAMA_MODEL`: optional, e.g. `gemma3:1b`. You must have the model pulled in Ollama.

### Troubleshooting
- Ensure Ollama is running and the model exists: `ollama run gemma3:1b`.
- Use `uv tree` to inspect dependency resolution.
- Use quotes around multi-word prompts to avoid shell tokenization issues.

### License
MIT