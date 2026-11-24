"""CLI interface for the Multi-Agent Deep Researcher using Rich."""
import warnings
warnings.filterwarnings("ignore")

import asyncio
import argparse
import sys
import os
import re
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

from deep_researcher.orchestrator import DeepResearcherOrchestrator


def slugify(text: str) -> str:
    """Converts a topic string into a clean filename slug."""
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[-\s]+", "_", text)[:50]


def print_banner(console: Console):
    """Displays the application header."""
    banner = (
        "[bold cyan]🔬 Multi-Agent Deep Researcher[/bold cyan]\n"
        "[dim]Team: [bold]Searcher[/bold] (Discovery & Scraping) | "
        "[bold]Fact-Checker[/bold] (Claims & Conflicts) | "
        "[bold]Writer[/bold] (Synthesis & Citations)[/dim]"
    )
    console.print(Panel(banner, border_style="cyan"))


async def async_main(args: argparse.Namespace):
    console = Console()
    print_banner(console)

    topic = args.topic.strip()
    if not topic:
        console.print("[bold red]Error: No research topic specified.[/bold red]")
        sys.exit(1)

    # Check for GEMINI_API_KEY
    if not os.getenv("GEMINI_API_KEY"):
        console.print(
            "[bold red]Error: GEMINI_API_KEY is not set.[/bold red]\n"
            "Please set your Gemini API key in your environment or in a .env file:\n"
            "[yellow]export GEMINI_API_KEY=\"your-api-key\"[/yellow] or add to [yellow].env[/yellow]"
        )
        sys.exit(1)

    output_path = args.output
    if not output_path:
        slug = slugify(topic)
        output_path = f"deep_research_brief_{slug}.md"

    console.print(f"[bold]Target Topic:[/bold] {topic}")
    console.print(f"[bold]Max Sources:[/bold] {args.max_sources} | [bold]Max Search Loops:[/bold] {args.max_search_loops}")
    console.print(f"[bold]Target File:[/bold] [green]{output_path}[/green]\n")

    orchestrator = DeepResearcherOrchestrator(
        search_model=args.model_fast,
        fact_check_extraction_model=args.model_fast,
        fact_check_reasoning_model=args.model_pro,
        writer_model=args.model_pro,
    )

    try:
        brief = await orchestrator.run_research(
            topic=topic,
            max_sources=args.max_sources,
            max_search_loops=args.max_search_loops,
            on_status=lambda msg: console.print(msg),
        )

        # Write output markdown
        Path(output_path).write_text(brief.full_markdown, encoding="utf-8")
        console.print(f"\n[bold green]✔ Research Brief successfully saved to:[/bold green] [underline]{output_path}[/underline]")

        # If JSON dump requested
        if args.save_json:
            json_path = output_path.replace(".md", ".json")
            Path(json_path).write_text(brief.model_dump_json(indent=2), encoding="utf-8")
            console.print(f"[bold green]✔ Full Claim Matrix & Metadata saved to:[/bold green] [underline]{json_path}[/underline]")

        # Print Executive Summary in Rich Panel
        console.print("\n")
        console.print(
            Panel(
                Markdown(brief.executive_summary),
                title="[bold green]Executive Summary[/bold green]",
                border_style="green",
            )
        )

        # Print Bibliography Summary Table
        table = Table(title="Primary & Consulted Sources", border_style="dim")
        table.add_column("#", justify="center", style="bold cyan")
        table.add_column("Type", justify="center")
        table.add_column("Source Domain", style="yellow")
        table.add_column("Title", style="white")

        for b in brief.bibliography:
            badge = "[bold green]PRIMARY[/bold green]" if b.is_primary else "[dim]SECONDARY[/dim]"
            table.add_row(str(b.citation_number), badge, b.domain, b.title[:50])

        console.print(table)

    except Exception as e:
        console.print(f"\n[bold red]Fatal Research Pipeline Error:[/bold red] {str(e)}")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Deep Researcher: Scrapes primary sources, verifies conflicts, and compiles cited briefs."
    )
    parser.add_argument(
        "topic",
        type=str,
        nargs="?",
        default="",
        help="The complex topic, hypothesis, or question to deeply research.",
    )
    parser.add_argument(
        "--topic",
        dest="topic_flag",
        type=str,
        default=None,
        help="Alternative flag for topic string.",
    )
    parser.add_argument(
        "-s",
        "--max-sources",
        type=int,
        default=10,
        help="Maximum primary and secondary sources to scrape and analyze (default: 10).",
    )
    parser.add_argument(
        "-l",
        "--max-search-loops",
        type=int,
        default=2,
        help="Maximum targeted re-search iterations to resolve detected conflicts (default: 2).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output markdown file path (defaults to deep_research_brief_<topic>.md).",
    )
    parser.add_argument(
        "--model-fast",
        type=str,
        default="gemini-2.5-flash",
        help="Gemini model for search query generation & raw claim extraction (default: gemini-2.5-flash).",
    )
    parser.add_argument(
        "--model-pro",
        type=str,
        default="gemini-3.1-pro-preview",
        help="Gemini model for conflict arbitration & brief synthesis (default: gemini-3.1-pro-preview).",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Export raw structured claim matrix and bibliography to JSON alongside the markdown.",
    )

    args = parser.parse_args()
    if args.topic_flag:
        args.topic = args.topic_flag

    if not args.topic:
        parser.print_help()
        sys.exit(1)

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
