"""Groq API call and citation-grounding prompt."""

import os
import threading
import time

from groq import Groq

_client: Groq | None = None

# Groq's free tier is 30 requests/min, 1,000 requests/day for most models —
# throttle conservatively below that (agent generation and, from
# eval/grader.py, judge grading share this budget).
_MIN_CALL_INTERVAL_SECONDS = 2.5
_last_call_time = 0.0
_throttle_lock = threading.Lock()


def throttle() -> None:
    """Block until it's safe to make another Groq API call, to respect rate limits."""
    global _last_call_time
    with _throttle_lock:
        wait = _MIN_CALL_INTERVAL_SECONDS - (time.time() - _last_call_time)
        if wait > 0:
            time.sleep(wait)
        _last_call_time = time.time()


SYSTEM_PROMPT = """You are a research assistant. You are given a question and a \
numbered list of sources retrieved from the web. Write a direct, well-supported \
answer to the question.

Rules:
- Cite every factual claim inline using the matching source number in square \
brackets, e.g. [1] or [2][3]. Place the citation immediately after the claim it \
supports.
- Only cite a source for a claim if that source's text actually contains support \
for the claim. Do not cite a source just because it is topically related.
- Some sources are marked UNAVAILABLE (the page failed to load or had no \
extractable content). Never cite an UNAVAILABLE source. If the available sources \
do not cover part of the question, say so explicitly instead of guessing.
- If the sources disagree or the topic is genuinely contested, say so and \
attribute each position to its source rather than picking one answer.
- Do not state anything as fact that is not supported by the sources. If none of \
the sources answer the question, say so plainly rather than answering from prior \
knowledge.
- Keep the answer focused and avoid restating the sources verbatim.
"""


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Copy .env.example to .env and add your key."
            )
        _client = Groq(api_key=api_key)
    return _client


def build_sources_block(sources: list[dict]) -> str:
    lines = []
    for i, s in enumerate(sources, start=1):
        if s["status"] == "ok":
            lines.append(f"[{i}] {s['url']}\n{s['text']}\n")
        else:
            lines.append(f"[{i}] {s['url']}\nUNAVAILABLE ({s['status']}: {s['error']})\n")
    return "\n".join(lines)


def answer_with_citations(
    question: str,
    sources: list[dict],
    model: str = "llama-3.3-70b-versatile",
) -> str:
    """Ask the model to answer the question, citing the given sources inline."""
    client = _get_client()
    sources_block = build_sources_block(sources)
    user_message = (
        f"Question: {question}\n\nSources:\n{sources_block}\n\n"
        "Write an answer to the question using inline citations [n]."
    )
    throttle()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content
