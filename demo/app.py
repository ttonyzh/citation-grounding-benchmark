"""Streamlit demo for the citation-grounding research agent.

Run with: streamlit run demo/app.py
"""

import os
import sys
from pathlib import Path
from typing import Literal

import streamlit as st
from dotenv import load_dotenv

# `streamlit run` doesn't add the project root to sys.path, so the sibling
# agent/ and eval/ packages aren't importable without this.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

load_dotenv()

from groq import Groq  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from agent.llm import build_sources_block, throttle  # noqa: E402
from agent.pipeline import run as run_agent  # noqa: E402
from eval.grader import strict_json_schema  # noqa: E402

st.set_page_config(page_title="Citation-Grounding Agent Demo", page_icon="🔍", layout="wide")

EXAMPLE_QUESTIONS = {
    "When did Elon Musk found Tesla?": "False-premise trap",
    "Who is the mayor of Springfield?": "Name-collision trap",
    "What is the current price of GPT-4 access through the OpenAI API?": "Stale/deprecated product trap",
    "Who is the current CEO of OpenAI?": "Time-sensitive (straightforward)",
    "Is nuclear power safer than solar power?": "Ambiguous / contested",
}

VERDICT_ICON = {
    "grounded": "🟢",
    "partially_grounded": "🟡",
    "ungrounded": "🔴",
    "cited_unavailable_source": "🔴",
    "acknowledged_unavailable": "⚪",
}

DEMO_JUDGE_SYSTEM_PROMPT = """You are a strict, careful fact-checker. You will be given a \
question, an agent's answer with inline citations like [1], [2], and the full numbered list \
of sources the agent had access to (including their fetched text, or UNAVAILABLE if a source \
could not be fetched).

For EVERY citation marker used in the answer, identify the specific claim it is attached to, \
then check that claim against that source's actual text. List every distinct citation instance \
separately, even if the same source number is cited more than once for different claims.
- "grounded": the source text clearly supports the claim.
- "partially_grounded": the source text supports part of the claim, or supports it more weakly \
or generally than the claim implies.
- "ungrounded": the source text does not support the claim, or contradicts it.
- "cited_unavailable_source": the citation is used as if an UNAVAILABLE source supports a \
substantive claim. This is always an error.
- "acknowledged_unavailable": the citation appears only where the agent explicitly says that \
source is unavailable or insufficient. This is transparent, correct behavior, not an error.

Also decide: hallucinated = true if the agent stated something as established fact that is not \
supported by any cited source and is not common knowledge.

Be strict and specific — quote or closely paraphrase the exact claim text you're checking. Keep \
every "reasoning" field to one short sentence — you are working under a tight output budget."""


class DemoCitationCheck(BaseModel):
    citation_index: int
    claim: str
    verdict: Literal[
        "grounded", "partially_grounded", "ungrounded", "cited_unavailable_source", "acknowledged_unavailable"
    ]
    reasoning: str


class DemoGradeResult(BaseModel):
    citation_checks: list[DemoCitationCheck]
    hallucinated: bool
    hallucination_reasoning: str


# Groq's free tier caps openai/gpt-oss-120b at 8,000 tokens/minute total
# (input + output combined). Full-fetch sources can be several thousand
# tokens on their own with k=5, so trim what the judge sees here rather
# than relying on output budget alone to stay under the ceiling.
JUDGE_SOURCE_CHAR_LIMIT = 1500


def check_grounding(question: str, answer: str, sources: list[dict], judge_model: str) -> DemoGradeResult:
    """Ask a judge model whether each citation in the answer is actually grounded in its source."""
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    trimmed_sources = [
        {**s, "text": s["text"][:JUDGE_SOURCE_CHAR_LIMIT] if s.get("text") else s.get("text")}
        for s in sources
    ]
    sources_block = build_sources_block(trimmed_sources)
    prompt = (
        f"Question: {question}\n\nAgent's answer:\n{answer}\n\n"
        f"Sources the agent had access to:\n{sources_block}"
    )
    throttle()
    response = client.chat.completions.create(
        model=judge_model,
        max_completion_tokens=2048,
        messages=[
            {"role": "system", "content": DEMO_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "demo_grade_result",
                "strict": True,
                "schema": strict_json_schema(DemoGradeResult.model_json_schema()),
            },
        },
    )
    return DemoGradeResult.model_validate_json(response.choices[0].message.content)


st.title("🔍 Citation-Grounding Research Agent")
st.caption(
    "A deliberately simple research agent, built to test one question: "
    "when it cites a source, does the source actually say that?"
)

missing_keys = [k for k in ("GROQ_API_KEY", "TAVILY_API_KEY") if not os.environ.get(k)]
if missing_keys:
    st.error(
        f"Missing API key(s): {', '.join(missing_keys)}. "
        "Copy .env.example to .env and add your keys, then restart the app."
    )
    st.stop()

with st.sidebar:
    st.header("Settings")
    k = st.slider("Search results (k)", 1, 8, 5)
    fetch_pages = not st.checkbox(
        "Snippets only (--no-fetch)",
        value=False,
        help="Skip full-page fetch and use only Tavily's search snippets. "
        "See the README's ablation section for how this changes grounding and accuracy.",
    )
    agent_model = st.text_input("Agent model", value="llama-3.3-70b-versatile")

    st.divider()
    st.subheader("Try a trap question")
    for q, label in EXAMPLE_QUESTIONS.items():
        if st.button(q, help=label, use_container_width=True, key=f"example_{q}"):
            st.session_state["question_input"] = q
            st.rerun()

    st.divider()
    st.caption("Full methodology and benchmark results: see the project README.")

if "question_input" not in st.session_state:
    st.session_state["question_input"] = ""

question = st.text_input(
    "Ask a research question",
    key="question_input",
    placeholder="e.g. Who is the current CEO of OpenAI?",
)
run_clicked = st.button("Run agent", type="primary")

if run_clicked and question:
    with st.spinner("Searching, fetching, and generating answer..."):
        try:
            result = run_agent(question, k=k, model=agent_model, fetch_pages=fetch_pages)
        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()
    st.session_state["last_result"] = result
    st.session_state.pop("last_grade", None)

result = st.session_state.get("last_result")
if result:
    st.subheader("Answer")
    st.markdown(result["answer"])

    st.subheader("Citations")
    if result["citations"]:
        st.dataframe(
            [
                {"#": c["index"], "URL": c["url"], "Title": c["title"], "Fetch status": c["status"]}
                for c in result["citations"]
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("No citations were used in this answer.")

    unavailable = [s for s in result["sources"] if s["status"] != "ok"]
    if unavailable:
        st.caption(
            f"{len(unavailable)} of {len(result['sources'])} fetched sources were unavailable "
            "(dead link / no extractable content)."
        )

    st.divider()
    st.markdown(
        "**This is the actual point of the project.** The answer above has citations — "
        "but does each source really say what's claimed?"
    )
    if st.button("Check citation grounding"):
        with st.spinner("Checking each citation against its actual source text..."):
            try:
                grade = check_grounding(
                    result["question"], result["answer"], result["sources"], "openai/gpt-oss-120b"
                )
                st.session_state["last_grade"] = grade
            except Exception as exc:
                st.error(
                    "Grounding check failed — this is usually the Groq free tier's "
                    f"token-per-minute limit on a question with many/long sources. "
                    f"Try again in a moment, or use a smaller k. ({exc.__class__.__name__})"
                )

    grade = st.session_state.get("last_grade")
    if grade:
        st.subheader("Citation grounding check")
        for c in grade.citation_checks:
            icon = VERDICT_ICON.get(c.verdict, "")
            with st.expander(f"{icon} [{c.citation_index}] {c.verdict} — {c.claim[:80]}"):
                st.write("**Claim:**", c.claim)
                st.write("**Verdict:**", c.verdict)
                st.write("**Judge reasoning:**", c.reasoning)

        if grade.hallucinated:
            st.error(f"⚠️ Possible hallucination: {grade.hallucination_reasoning}")
        else:
            st.success("No hallucination detected in this answer.")
