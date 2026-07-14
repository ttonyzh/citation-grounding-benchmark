"""Fetch URLs and extract clean article text, tolerating dead links and junk pages."""

import requests
import trafilatura

TIMEOUT = 10
MAX_CHARS = 6000
USER_AGENT = (
    "Mozilla/5.0 (compatible; citation-benchmark-agent/0.1; "
    "research agent for a citation-grounding benchmark)"
)


def fetch_and_extract(url: str) -> dict:
    """Fetch a URL and extract clean text. Never raises — always returns a status dict."""
    try:
        resp = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
    except requests.RequestException as exc:
        return {"url": url, "status": "fetch_failed", "error": str(exc), "text": None}

    text = trafilatura.extract(resp.text, include_comments=False, include_tables=False)
    if not text or not text.strip():
        return {
            "url": url,
            "status": "extract_failed",
            "error": "no extractable content",
            "text": None,
        }

    return {"url": url, "status": "ok", "error": None, "text": text[:MAX_CHARS]}
