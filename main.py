#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agentic demo: schema-guided decision (search | answer) + Tavily search + grounded answer

Usage:
  export TAVILY_API_KEY="..."
  python agent_tavily_demo.py "Tell me three quirky facts about wolves."
  # Или без аргументов — будет интерактивный режим.

Requires:
  - ollama
  - pydantic>=2
  - tavily-python
  - python-dotenv (optional, for .env)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Literal, Union

from pydantic import BaseModel, Field, ValidationError
from tavily import TavilyClient

import ollama


# =========================
# 1) Pydantic схемы
# =========================

class SearchRequest(BaseModel):
    reasoning: str = Field(..., description="Why search this (1 short sentence)")
    request: str = Field(..., description="Full search query for the web")

class Search(BaseModel):
    name: Literal['search']
    request: List[SearchRequest]

class Answer(BaseModel):
    name: Literal['answer']
    reasoning: str = Field(..., description="Brief decision rationale")
    answer: str

class ModelResponse(BaseModel):
    model_response: Union[Search, Answer]


# =========================
# 2) System-промпты
# =========================

DECISION_SYS_TEMPLATE = """
Role:
You decide whether to call the web_search tool based on the user's question and produce a structured response.

Tools:
- web_search: the only available tool.

Decision rule:
- If the question needs up-to-date, factual, niche, or verifiable information, return a tool call ("search").
- If the question is generic knowledge you can answer reliably without searching, return a direct answer ("answer").
- When uncertain, prefer "search".

Hard requirements:
- OUTPUT ONLY VALID JSON that matches the provided JSON Schema.
- No extra keys, no markdown, no prose.
- Keep "reasoning" to one short sentence.
- If you select "search", produce 1–3 SearchRequest items covering different angles.

Schema (return EXACTLY one of the following via the top-level object):
{schema}

Valid examples (STRICT SHAPE):

EXAMPLE A (answer):
{{
  "model_response": {{
    "name": "answer",
    "reasoning": "This can be answered without external facts.",
    "answer": "Domestic cats are crepuscular and have a strong righting reflex..."
  }}
}}

EXAMPLE B (search):
{{
  "model_response": {{
    "name": "search",
    "request": [
      {{
        "reasoning": "Find reputable sources with recent facts.",
        "request": "site:nationalgeographic.com OR site:smithsonianmag.com surprising facts about wolves 2023..2025"
      }},
      {{
        "reasoning": "Add behavioral research.",
        "request": "wolf pack dynamics peer-reviewed 2020..2025"
      }}
    ]
  }}
}}
""".strip()

ANSWER_SYS_TEMPLATE = """
Role:
You are a careful, grounded answerer. You MUST synthesize an answer to the user's question
STRICTLY from the provided search results. If the results are insufficient, say so explicitly.
Keep the reasoning concise and the final answer clear and specific.

Hard requirements:
- Base your answer ONLY on "SOURCES" content below.
- Be concise and concrete. Avoid filler text.
- If helpful, include a short bullet list.
- Optionally include a "Sources:" section with the most relevant URLs (3–6).
- OUTPUT ONLY VALID JSON matching the provided schema.

Schema:
{schema}
""".strip()


# =========================
# 3) Инициализация клиентов
# =========================

def get_tavily_client() -> TavilyClient:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("[ERROR] TAVILY_API_KEY is not set. Export it or put into .env", file=sys.stderr)
        sys.exit(1)
    return TavilyClient(api_key=api_key)

def call_ollama_chat(messages, model: str, format_schema: dict | None = None, temperature: float = 0.2) -> str:
    kwargs = {
        "model": model,
        "messages": messages,
        "options": {"temperature": temperature},
    }
    if format_schema is not None:
        kwargs["format"] = format_schema
    resp = ollama.chat(**kwargs)
    return resp["message"]["content"]


# =========================
# 4) Логика решения + поиск + финальный ответ
# =========================

def decide_search_or_answer(question: str, model: str = "gemma3:1b") -> ModelResponse:
    sys_prompt = DECISION_SYS_TEMPLATE.replace("{schema}", json.dumps(ModelResponse.model_json_schema(), ensure_ascii=False))
    content = call_ollama_chat(
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": question},
        ],
        model=model,
        format_schema=ModelResponse.model_json_schema(),
        temperature=0.2,
    )
    try:
        return ModelResponse.model_validate_json(content)
    except ValidationError as e:
        print("\n[ERROR] Decision JSON failed validation. Raw content:\n", content, file=sys.stderr)
        raise e


def tavily_search_many(queries: List[str], max_results_per_query: int = 5, search_depth: str = "advanced") -> List[dict]:
    """
    Выполняем Tavily по каждому запросу и агрегируем результаты.
    Возвращаем список объектов вида:
    { "query": str, "results": [ {"title":..., "url":..., "content":...}, ... ] }
    """
    client = get_tavily_client()
    aggregated = []
    for q in queries:
        print(f"[TAVILY] {q}")
        try:
            res = client.search(
                query=q,
                search_depth=search_depth,
                max_results=max_results_per_query,
                include_domains=None,              # можно ограничить домены при желании
                exclude_domains=None,
                include_answer=False,              # берём «сырьё», без автосаммари
                include_images=False,
                include_image_descriptions=False,
            )
            # API возвращает: {"results": [{"title","url","content",...}, ...], ...}
            results = res.get("results", [])
        except Exception as e:
            print(f"[WARN] Tavily error for query: {q} → {e}", file=sys.stderr)
            results = []
        aggregated.append({"query": q, "results": results})
    return aggregated


def build_sources_context(aggregated: List[dict], top_k: int = 8) -> tuple[str, list[dict]]:
    """
    Готовим компактный контекст «источников» для LLM.
    Дедуп по URL, отбор top_k по порядку появления.
    Возвращает:
      - text_context: нумерованный список источников (для system+user)
      - sources_list: список словарей {index, title, url, content}
    """
    seen = set()
    flat = []
    for bucket in aggregated:
        for item in bucket["results"]:
            url = item.get("url", "")
            if not url or url in seen:
                continue
            seen.add(url)
            flat.append({
                "title": item.get("title") or "",
                "url": url,
                "content": item.get("content") or "",
            })
            if len(flat) >= top_k:
                break
        if len(flat) >= top_k:
            break

    # Формируем нумерованный блок
    lines = []
    sources_list = []
    for i, it in enumerate(flat, start=1):
        title = it["title"].strip()[:160]
        url = it["url"]
        content = (it["content"] or "").strip().replace("\n", " ")
        # Подрежем контент, чтобы не перегружать микро-модель
        snippet = content[:800]
        lines.append(f"[{i}] {title}\nURL: {url}\nSnippet: {snippet}\n")
        sources_list.append({"index": i, **it})

    text_context = "\n".join(lines)
    return text_context, sources_list


def answer_with_sources(question: str, sources_block: str, model: str = "gemma3:1b") -> Answer:
    sys_prompt = ANSWER_SYS_TEMPLATE.replace("{schema}", json.dumps(Answer.model_json_schema(), ensure_ascii=False))
    user_prompt = f"""USER_QUESTION:
    {question}

    SOURCES:
    {sources_block}

    Return valid JSON only.
    """
    content = call_ollama_chat(
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=model,
        format_schema=Answer.model_json_schema(),
        temperature=0.2,
    )
    try:
        return Answer.model_validate_json(content)
    except ValidationError as e:
        print("\n[ERROR] Final answer JSON failed validation. Raw content:\n", content, file=sys.stderr)
        raise e


# =========================
# 5) CLI
# =========================

def run_once(question: str, model: str = "gemma3:1b") -> None:
    print(f"\n=== USER: {question}")
    decision = decide_search_or_answer(question, model=model)

    mr = decision.model_response
    if isinstance(mr, Answer) or getattr(mr, "name", None) == "answer":
        print("\n--- DECISION: ANSWER ---")
        print(decision.model_response.model_dump_json(indent=2))
        print("\nFinal answer:\n")
        print(mr.answer)
        return

    # Иначе: SEARCH
    print("\n--- DECISION: SEARCH ---")
    for i, req in enumerate(mr.request, 1):
        print(f"  [{i}] {req.reasoning} -> {req.request}")

    queries = [r.request for r in mr.request]
    aggregated = tavily_search_many(queries, max_results_per_query=5, search_depth="advanced")
    sources_block, sources_list = build_sources_context(aggregated, top_k=8)

    print("\n--- SOURCES (compact) ---")
    print(sources_block)

    final = answer_with_sources(question, sources_block, model=model)

    print("\n--- GROUNDED ANSWER (JSON) ---")
    print(final.model_dump_json(indent=2))

    print("\nFinal answer:\n")
    print(final.answer)


def main():
    try:
        from dotenv import load_dotenv  # optional
        load_dotenv()
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Schema-guided (search|answer) with Tavily grounding")
    parser.add_argument("question", nargs="*", help="User question")
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "gemma3:1b"), help="Ollama model name")
    args = parser.parse_args()

    if args.question:
        question = " ".join(args.question).strip()
        run_once(question, model=args.model)
        return

    # Interactive
    print("Interactive mode. Ctrl+C to exit.")
    while True:
        try:
            q = input("\n> ") 
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q.strip():
            continue
        run_once(q.strip(), model=args.model)


if __name__ == "__main__":
    main()
