"""Tavily web search wrapper."""

import os

from tavily import TavilyClient

_client: TavilyClient | None = None


def _get_client() -> TavilyClient:
    global _client
    if _client is None:
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError(
                "TAVILY_API_KEY not set. Copy .env.example to .env and add your key."
            )
        _client = TavilyClient(api_key=api_key)
    return _client


def search(query: str, k: int = 5) -> list[dict]:
    """Run a web search and return up to k results as {title, url, snippet}."""
    client = _get_client()
    response = client.search(query=query, max_results=k)
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", ""),
        }
        for r in response.get("results", [])
    ]
