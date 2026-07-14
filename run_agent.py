#!/usr/bin/env python3
"""
CLI entry point for the citation-grounding research agent.

Usage:
    python run_agent.py "What is the Fermi paradox?"
    python run_agent.py "Who currently leads the WHO?" --k 3
    python run_agent.py "What is the current federal funds rate?" --k 5 --model llama-3.1-8b-instant
"""

import argparse
import sys

from dotenv import load_dotenv

load_dotenv()

from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.table import Table  # noqa: E402

from agent.pipeline import run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the citation-grounding research agent on a question.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("question", help="The research question to answer.")
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        metavar="K",
        help="Number of search results to fetch and pass to the LLM (default: 5).",
    )
    parser.add_argument(
        "--model",
        default="llama-3.3-70b-versatile",
        help="Groq model to use for answer generation (default: llama-3.3-70b-versatile).",
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Skip fetching full page content; use Tavily snippets only.",
    )
    args = parser.parse_args()

    console = Console()

    try:
        with console.status("[bold cyan]Searching, fetching, and generating answer..."):
            result = run(
                args.question,
                k=args.k,
                model=args.model,
                fetch_pages=not args.no_fetch,
            )
    except RuntimeError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)

    console.print(Panel(result["answer"], title="Answer", border_style="cyan"))

    table = Table(title="Citations")
    table.add_column("#", justify="right")
    table.add_column("URL", overflow="fold")
    table.add_column("Title", overflow="fold")
    table.add_column("Fetch status")

    for c in result["citations"]:
        status_style = "green" if c["status"] == "ok" else "red"
        table.add_row(
            str(c["index"]),
            c["url"],
            c["title"],
            f"[{status_style}]{c['status']}[/{status_style}]",
        )
    console.print(table)

    if not result["citations"]:
        console.print("[dim]No sources were cited in the answer.[/dim]")

    unavailable = [s for s in result["sources"] if s["status"] != "ok"]
    if unavailable:
        console.print(
            f"[dim]{len(unavailable)} of {len(result['sources'])} fetched sources "
            "were unavailable (dead link / no extractable content).[/dim]"
        )


if __name__ == "__main__":
    main()
