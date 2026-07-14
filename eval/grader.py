"""LLM-as-judge grading pipeline.

For each question in eval/questions.json: run the agent, then ask a judge
model to grade (a) overall answer accuracy against ground truth, (b) whether
each individual citation is actually grounded in its source's text, and
(c) whether the answer hallucinated instead of flagging uncertainty.

Usage:
    python -m eval.grader
    python -m eval.grader --limit 3          # smoke-test on a few questions
    python -m eval.grader --k 3 --no-fetch    # tune the agent's own settings
"""

import argparse
import csv
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

from groq import Groq  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from agent.llm import build_sources_block, throttle  # noqa: E402
from agent.pipeline import run as run_agent  # noqa: E402

QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.json"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RUNS_CHECKPOINT_PATH = RESULTS_DIR / "_checkpoint_runs.jsonl"
GRADES_CHECKPOINT_PATH = RESULTS_DIR / "_checkpoint_grades.jsonl"
SPOT_CHECK_FRACTION = 0.20
JUDGE_MODEL = "openai/gpt-oss-120b"
MAX_RETRIES = 6

_client: Groq | None = None


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


def _with_retry(fn, *args, **kwargs):
    """Retry a flaky API call with exponential backoff (handles free-tier rate limits)."""
    delay = 15
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if attempt == MAX_RETRIES - 1:
                raise
            print(f"    retrying after error ({exc.__class__.__name__}); waiting {delay}s...")
            time.sleep(delay)
            delay = min(delay * 2, 90)


def _load_checkpoint(path: Path) -> dict[str, dict]:
    """Load a JSONL checkpoint file into a dict keyed by question_id."""
    if not path.exists():
        return {}
    items = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                items[obj["question_id"]] = obj
    return items


def _append_checkpoint(path: Path, item: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(item) + "\n")
        f.flush()


class AnswerAccuracy(BaseModel):
    verdict: Literal["correct", "partially_correct", "incorrect"]
    reasoning: str


class CitationCheck(BaseModel):
    citation_index: int
    claim: str
    verdict: Literal[
        "grounded",
        "partially_grounded",
        "ungrounded",
        "cited_unavailable_source",
        "acknowledged_unavailable",
    ]
    reasoning: str


def strict_json_schema(schema: dict) -> dict:
    """Recursively add additionalProperties: false to every object in the schema.
    Groq's strict structured-output mode requires this on every nested object
    (including $defs), but Pydantic's model_json_schema() only sets it at the
    top level."""
    if isinstance(schema, dict):
        if schema.get("type") == "object" or "properties" in schema:
            schema.setdefault("additionalProperties", False)
        for value in schema.values():
            if isinstance(value, dict):
                strict_json_schema(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        strict_json_schema(item)
    return schema


class GradeResult(BaseModel):
    answer_accuracy: AnswerAccuracy
    citation_checks: list[CitationCheck]
    hallucinated: bool
    hallucination_reasoning: str
    flagged_uncertainty_appropriately: bool
    uncertainty_reasoning: str


GRADER_SYSTEM_PROMPT = """You are a strict, careful fact-checker grading a research \
agent's answer for a citation-grounding benchmark. You will be given a question, \
ground truth, what good behavior looks like for this specific question, the \
agent's full answer (with inline citations like [1], [2]), and the full numbered \
list of sources the agent had access to (including their fetched text, or \
UNAVAILABLE if a source could not be fetched).

Grade three things:

1. ANSWER ACCURACY: Compare the agent's answer to the ground truth. "correct" \
means all key facts are present with no factual errors. "partially_correct" \
means some key facts are present but something is missing, wrong, or a false \
premise was accepted instead of corrected. "incorrect" means the core answer is \
wrong.

2. CITATION GROUNDING: For EVERY citation marker used in the answer, identify \
the specific claim it is attached to, then check that claim against that \
source's actual text. List every distinct citation instance separately, even if \
the same source number is cited more than once for different claims.
   - "grounded": the source text clearly supports the claim.
   - "partially_grounded": the source text supports part of the claim, or \
supports it more weakly or generally than the claim implies.
   - "ungrounded": the source text does not support the claim, or contradicts it.
   - "cited_unavailable_source": the citation is used as if an UNAVAILABLE source \
supports a substantive claim (e.g. "Company X's revenue grew 20% [3]" where [3] is \
UNAVAILABLE). This is always an error.
   - "acknowledged_unavailable": the citation appears only where the agent \
explicitly says that source is unavailable, insufficient, or doesn't address the \
question (e.g. "sources [2] and [3] did not cover this"). This is transparent, \
correct behavior, not an error — do not mark it as grounded or ungrounded either.

3. HALLUCINATION AND UNCERTAINTY:
   - hallucinated: true if the agent stated something as established fact that \
is not supported by any cited source and is not common knowledge (a fabricated \
statistic, invented date, or invented finding).
   - flagged_uncertainty_appropriately: true if, given that some sources were \
unavailable, contested, or insufficient, the agent explicitly communicated that \
limitation rather than answering as if the available sources fully settled the \
question. If there was nothing to be uncertain about, this should be true by \
default.

Be strict and specific — quote or closely paraphrase the exact claim text you're \
checking rather than summarizing the whole answer. Keep every "reasoning" field to \
one short sentence — you are working under a tight output budget."""


def load_questions(limit: int | None = None) -> list[dict]:
    with open(QUESTIONS_PATH) as f:
        questions = json.load(f)
    return questions[:limit] if limit else questions


def generate_agent_runs(
    questions: list[dict], k: int = 5, model: str = "llama-3.3-70b-versatile", fetch_pages: bool = True
) -> list[dict]:
    """Run the agent pipeline on every question, checkpointing after each one so a
    crash (e.g. a free-tier rate-limit wall) doesn't lose completed work. Re-running
    skips any question already present in the checkpoint file."""
    done = _load_checkpoint(RUNS_CHECKPOINT_PATH)
    runs = []
    for i, q in enumerate(questions, start=1):
        if q["id"] in done:
            print(f"[{i}/{len(questions)}] Skipping {q['id']} (already in checkpoint)")
            runs.append(done[q["id"]])
            continue
        print(f"[{i}/{len(questions)}] Running agent on {q['id']}: {q['question']}")
        result = _with_retry(run_agent, q["question"], k=k, model=model, fetch_pages=fetch_pages)
        run = {"question_id": q["id"], **result}
        _append_checkpoint(RUNS_CHECKPOINT_PATH, run)
        runs.append(run)
    return runs


def grade_run(question_data: dict, agent_result: dict) -> GradeResult:
    """Ask the judge model to grade one agent run against its ground truth."""
    client = _get_client()
    sources_block = build_sources_block(agent_result["sources"])
    gt = question_data["ground_truth"]

    prompt = f"""Question: {question_data['question']}

Ground truth summary: {gt['summary']}
Key facts that should appear in a correct answer: {gt['key_facts']}
Is this topic contested? {gt['contested']}
Expected agent behavior: {question_data['expected_behavior']}

Agent's answer:
{agent_result['answer']}

Sources the agent had access to:
{sources_block}
"""

    def _call():
        throttle()
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            max_completion_tokens=4096,
            messages=[
                {"role": "system", "content": GRADER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "grade_result",
                    "strict": True,
                    "schema": strict_json_schema(GradeResult.model_json_schema()),
                },
            },
        )
        # Parsing inside the retryable call: a model that returns malformed JSON
        # on one attempt often succeeds on the next, so treat a validation
        # failure the same as a transient API error.
        return GradeResult.model_validate_json(response.choices[0].message.content)

    return _with_retry(_call)


def grade_all(questions: list[dict], runs: list[dict]) -> list[dict]:
    """Grade every agent run, checkpointing after each one (see generate_agent_runs)."""
    runs_by_id = {r["question_id"]: r for r in runs}
    done = _load_checkpoint(GRADES_CHECKPOINT_PATH)
    graded = []
    for i, q in enumerate(questions, start=1):
        if q["id"] in done:
            print(f"[{i}/{len(questions)}] Skipping {q['id']} (already in checkpoint)")
            graded.append(done[q["id"]])
            continue
        run = runs_by_id[q["id"]]
        print(f"[{i}/{len(questions)}] Grading {q['id']}")
        grade = grade_run(q, run)
        record = {
            "question_id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "answer": run["answer"],
            "sources": run["sources"],
            "grade": grade.model_dump(),
        }
        _append_checkpoint(GRADES_CHECKPOINT_PATH, record)
        graded.append(record)
    return graded


def build_spot_check_sample(
    graded: list[dict], fraction: float = SPOT_CHECK_FRACTION, seed: int = 42
) -> list[dict]:
    """Flatten every individual judge grade into rows and sample a fraction for manual review."""
    items = []
    for g in graded:
        qid, question, grade = g["question_id"], g["question"], g["grade"]

        items.append(
            {
                "question_id": qid,
                "question": question,
                "check_type": "answer_accuracy",
                "detail": "",
                "context": g["answer"][:2000],
                "judge_verdict": grade["answer_accuracy"]["verdict"],
                "judge_reasoning": grade["answer_accuracy"]["reasoning"],
                "human_verdict": "",
                "human_notes": "",
            }
        )
        items.append(
            {
                "question_id": qid,
                "question": question,
                "check_type": "hallucination",
                "detail": "",
                "context": g["answer"][:2000],
                "judge_verdict": "hallucinated" if grade["hallucinated"] else "no_hallucination",
                "judge_reasoning": grade["hallucination_reasoning"],
                "human_verdict": "",
                "human_notes": "",
            }
        )
        items.append(
            {
                "question_id": qid,
                "question": question,
                "check_type": "uncertainty_flagging",
                "detail": "",
                "context": g["answer"][:2000],
                "judge_verdict": "flagged" if grade["flagged_uncertainty_appropriately"] else "did_not_flag",
                "judge_reasoning": grade["uncertainty_reasoning"],
                "human_verdict": "",
                "human_notes": "",
            }
        )

        sources_by_index = {i: s for i, s in enumerate(g["sources"], start=1)}
        for c in grade["citation_checks"]:
            src = sources_by_index.get(c["citation_index"], {})
            excerpt = (
                (src.get("text") or "")[:2000]
                if src.get("status") == "ok"
                else f"UNAVAILABLE ({src.get('status')})"
            )
            items.append(
                {
                    "question_id": qid,
                    "question": question,
                    "check_type": "citation",
                    "detail": f"[{c['citation_index']}] {src.get('url', '')}",
                    "context": f"CLAIM: {c['claim']}\n\nSOURCE TEXT: {excerpt}",
                    "judge_verdict": c["verdict"],
                    "judge_reasoning": c["reasoning"],
                    "human_verdict": "",
                    "human_notes": "",
                }
            )

    rng = random.Random(seed)
    sample_size = max(1, round(len(items) * fraction))
    return rng.sample(items, sample_size)


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the citation-grounding grading pipeline.")
    parser.add_argument("--limit", type=int, default=None, help="Only grade the first N questions (for smoke-testing).")
    parser.add_argument("--k", type=int, default=5, help="Number of search results per question (default: 5).")
    parser.add_argument("--model", default="llama-3.3-70b-versatile", help="Model used by the agent under test.")
    parser.add_argument("--no-fetch", action="store_true", help="Run the agent in snippet-only mode.")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore any existing checkpoint files and start over (use if --k/--model/--no-fetch changed).",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    if args.fresh:
        for path in (RUNS_CHECKPOINT_PATH, GRADES_CHECKPOINT_PATH):
            path.unlink(missing_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    questions = load_questions(limit=args.limit)

    print(f"Running agent on {len(questions)} questions...")
    runs = generate_agent_runs(questions, k=args.k, model=args.model, fetch_pages=not args.no_fetch)
    raw_path = RESULTS_DIR / f"raw_runs_{timestamp}.json"
    with open(raw_path, "w") as f:
        json.dump(runs, f, indent=2)
    print(f"Saved raw agent runs to {raw_path}")

    print("Grading agent runs...")
    graded = grade_all(questions, runs)
    graded_path = RESULTS_DIR / f"graded_{timestamp}.json"
    with open(graded_path, "w") as f:
        json.dump(graded, f, indent=2)
    print(f"Saved graded results to {graded_path}")

    spot_check_rows = build_spot_check_sample(graded)
    spot_check_path = RESULTS_DIR / f"spot_check_{timestamp}.csv"
    write_csv(spot_check_rows, spot_check_path)
    print(
        f"Saved {len(spot_check_rows)} spot-check rows "
        f"({SPOT_CHECK_FRACTION:.0%} sample) to {spot_check_path}"
    )


if __name__ == "__main__":
    main()
