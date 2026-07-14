# A Benchmark for Citation-Grounding Failures in LLM Research Agents

A rigorous evaluation of how often a simple LLM research agent's citations
actually support the claims they're attached to — and what conditions cause
citation-grounding to fail.

## Research Question

> When a research agent cites a source, how often does that source actually
> support the cited claim? What conditions — stale sources, dead links,
> ambiguous questions, contested topics — drive grounding failures?

## Architecture

The agent is deliberately simple. The complexity is in the evaluation.

```
Question
   │
   ▼
Tavily Search  (top-k results)
   │
   ▼
Fetch & Extract  (trafilatura strips boilerplate)
   │
   ▼
Groq / Llama (cite sources inline as [1], [2], …)
   │
   ▼
Answer + citation list  →  Grading pipeline
```

## Project Structure

```
/agent
    pipeline.py   orchestrates the search → fetch → LLM loop
    search.py     Tavily API wrapper
    fetch.py      URL fetching + text extraction (trafilatura)
    llm.py        Groq API call + citation prompt

/eval
    questions.json   benchmark question set with ground truth
    grader.py        LLM-as-judge grading pipeline

/demo
    app.py           Streamlit interface for live testing

/results             output from eval runs (gitignored)

run_agent.py         CLI entry point
```

## Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API keys
cp .env.example .env
# Edit .env with your Groq and Tavily keys
```

Get API keys (both have free tiers — no payment required):
- Groq: https://console.groq.com/ (free tier: 30 requests/min, 1,000 requests/day on most models — no credit card required)
- Tavily: https://app.tavily.com/ (free tier: 1,000 searches/month)

## Running the Agent

```bash
python run_agent.py "What is the Fermi paradox?"
python run_agent.py "Who currently leads the WHO?" --k 3
python run_agent.py "What is the current federal funds rate?" --k 5
```

## Running the Benchmark

```bash
# Grade all questions in eval/questions.json
python -m eval.grader

# Results written to results/
```

## Failure Categories

| Category | Description |
|---|---|
| **Straightforward factual** | Stable facts with clear sourcing |
| **Time-sensitive** | Answers that change frequently; tests stale-source failures |
| **Ambiguous / contested** | Topics where sources legitimately disagree |
| **Trap questions** | Queries where top search results are likely irrelevant or dead |

## Results

Full run: 26 questions, `llama-3.3-70b-versatile` as the agent, `openai/gpt-oss-120b`
as the LLM judge, `k=5` search results per question, full-page fetch enabled.
Raw output in `/results` (gitignored — re-run `python -m eval.grader` to reproduce).

### Citation Grounding Rate by Category

Each row is every individual citation the agent made in that category, checked
against the actual fetched text of the source it pointed to.

| Category | n citations | Grounded | Partial | Ungrounded | Acknowledged unavailable |
|---|---|---|---|---|---|
| Straightforward | 42 | 35 (83%) | 1 (2%) | 1 (2%) | 5 (12%) |
| Time-sensitive | 48 | 41 (85%) | 2 (4%) | 5 (10%) | 0 (0%) |
| Ambiguous / contested | 59 | 53 (90%) | 2 (3%) | 1 (2%) | 3 (5%) |
| Trap | 53 | 42 (79%) | 4 (8%) | 4 (8%) | 3 (6%) |
| **All categories** | **202** | **171 (85%)** | **9 (4%)** | **11 (5%)** | **11 (5%)** |

Notably, **zero** citations landed in the worst failure mode — citing an
unavailable source as if it supported a claim. Every source the agent
couldn't fetch was either not cited, or explicitly flagged as unavailable in
the answer text ("acknowledged unavailable" above).

### Answer Accuracy by Category

| Category | n | Correct | Partially correct | Incorrect |
|---|---|---|---|---|
| Straightforward | 6 | 6 (100%) | 0 | 0 |
| Time-sensitive | 7 | 2 (29%) | 3 (43%) | 2 (29%) |
| Ambiguous / contested | 6 | 1 (17%) | 5 (83%) | 0 |
| Trap | 7 | 2 (29%) | 4 (57%) | 1 (14%) |

Straightforward facts were perfect. Everything else degraded — which is the
expected shape of the result, not noise: **time-sensitive questions were the
single worst category** for both accuracy and grounding, more so than the
purpose-built trap questions.

### Other findings

- **Hallucination rate: 2/26 (8%)** — both examples below.
- **Appropriate uncertainty flagging: 21/26 (81%)** — the agent explicitly
  called out unavailable, contested, or insufficient sources in most cases
  where it should have.

### Judge Validation (Human Spot-Check)

An LLM-as-judge pipeline is only as trustworthy as its judge. To check that,
I manually reviewed 20 of the 56 rows in the 20% spot-check sample
(`results/spot_check_*.csv`) — reading each claim and its source text myself
and forming an independent verdict *before* seeing the judge's, then
comparing.

**Agreement rate: 89% (17/19 resolved rows).** One row was set aside rather
than forced to a verdict — a claim about what *other* sources didn't say,
where the judge's own stated reasoning contradicted its own verdict, and the
negative-polarity phrasing made it too ambiguous to call cleanly.

The two real disagreements:

- **A confirmed judge bug** (`ts-07`, iPhone release date): the judge marked a
  citation to a T-Mobile marketing page as "grounded," but that page mentions
  no date anywhere — its own stated reasoning cited two entirely different
  sources instead of the one actually being checked.
- **A category-precision slip** (`ts-01`, OpenAI CEO): the judge labeled a
  citation to a dead Instagram login page "grounded" rather than the more
  specific "acknowledged unavailable" category. Not factually wrong, just the
  less precise of two valid labels.

89% is a reasonable trust level for this pipeline — not a rubber stamp, and
not so low that the headline numbers above are meaningless. Finding a real
judge bug via spot-checking is the validation working as intended, not a
mark against it.

### Concrete Failure Examples

**1. Citation directly contradicts its own source** (`amb-01`,
*"Is nuclear power safer than solar power?"*)

> **Agent wrote:** "...the safety of nuclear power compared to solar power is
> not directly addressed in the available sources [5]."
>
> **Source [5]** (ourworldindata.org/safest-sources-of-energy) **actually says:**
> "...nuclear and modern renewable energy sources are vastly safer and
> cleaner [than fossil fuels]..." — the article's whole thesis is a direct
> comparison of nuclear vs. renewables (including solar), and states they're
> comparably safe.

The agent cited the one source that most directly answers the question, then
claimed that source doesn't address the question — the opposite of what it
says. The judge flagged this as a hallucination rather than a citation error,
since the fabrication is in the *characterization* of the source, not a
factual claim standing alone.

**2. A topically-relevant source doesn't support the specific claim**
(`ts-02`, *"Who is the current Secretary-General of the United Nations?"*)

> **Agent wrote:** "The current Secretary-General of the United Nations is
> António Guterres [2]..."
>
> **Source [2]** (un.org/sg/en/content/sg/biography) **actually says:**
> "Prior to his appointment as Secretary-General, Mr. Guterres served as
> United Nations High Commissioner for Refugees..." — the entire excerpt
> describes his *prior* roles (Portuguese PM, UNHCR chief). It never actually
> states that he currently holds the position.

The claim happens to be true, and the source is genuinely about the right
person — but the specific fetched excerpt never confirms current officeholder
status. A citation can be topically correct and still not ground the specific
claim it's attached to.

**3. Every individual citation checks out, but the framing is stale**
(`trap-05`, *"What is the current price of GPT-4 access through the OpenAI API?"*)

> **Agent wrote:** "The current price of GPT-4 access through the OpenAI API
> is $30.00 per million input tokens and $60.00 per million output tokens [1]."
>
> **Source [1]** (pricepertoken.com) **actually says exactly that** — $30/$60
> per million tokens is genuinely what the page states. But the same page
> also shows GPT-4's 8K-token context window and bottom-20th-percentile
> intelligence benchmark score — hallmarks of a page describing a model that
> was current in 2023, not 2026. GPT-4 proper is a deprecated legacy model by
> the time this question was asked.

This is the most important failure in the set: **every single citation in
this answer independently passed grounding review** — the judge marked all
six as "grounded." The answer still hallucinated, because "the source says
X" and "X is still true today" are different claims, and per-citation text
matching alone can't catch the gap between them. This is why the grading
pipeline scores hallucination as a separate dimension from citation grounding
rather than deriving one from the other.
