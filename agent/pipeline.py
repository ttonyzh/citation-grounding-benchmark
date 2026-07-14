"""Orchestrates the search -> fetch -> answer loop."""

import re

from agent.fetch import fetch_and_extract
from agent.llm import answer_with_citations
from agent.search import search


def run(
    question: str,
    k: int = 5,
    model: str = "llama-3.3-70b-versatile",
    fetch_pages: bool = True,
) -> dict:
    """Run the full pipeline for a question and return the answer, citations, and sources.

    When fetch_pages is False, the LLM sees only search-snippet text instead of
    full extracted pages — useful for comparing grounding quality with and
    without full-page fetch.
    """
    results = search(question, k=k)

    sources = []
    for r in results:
        if fetch_pages:
            fetched = fetch_and_extract(r["url"])
        else:
            fetched = {"url": r["url"], "status": "ok", "error": None, "text": r["snippet"]}
        fetched["title"] = r["title"]
        sources.append(fetched)

    answer = answer_with_citations(question, sources, model=model)

    cited_indices = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})
    citations = [
        {
            "index": i,
            "url": sources[i - 1]["url"],
            "title": sources[i - 1]["title"],
            "status": sources[i - 1]["status"],
        }
        for i in cited_indices
        if 1 <= i <= len(sources)
    ]

    return {
        "question": question,
        "answer": answer,
        "citations": citations,
        "sources": sources,
    }
