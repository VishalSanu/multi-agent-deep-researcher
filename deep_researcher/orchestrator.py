"""Orchestrator coordinating Searcher, Fact-Checker, and Writer agents."""
import asyncio
from typing import Optional, Callable, List
from deep_researcher.models.schemas import (
    ResearchBrief,
    SourceDocument,
    FactCheckReport,
)
from deep_researcher.agents.searcher import SearcherAgent
from deep_researcher.agents.fact_checker import FactCheckerAgent
from deep_researcher.agents.writer import WriterAgent


class DeepResearcherOrchestrator:
    """Coordinates the multi-agent deep research cycle."""

    def __init__(
        self,
        search_model: str = "gemini-2.5-flash",
        fact_check_extraction_model: str = "gemini-2.5-flash",
        fact_check_reasoning_model: str = "gemini-3.1-pro-preview",
        writer_model: str = "gemini-3.1-pro-preview",
    ):
        self.searcher = SearcherAgent(model=search_model)
        self.fact_checker = FactCheckerAgent(
            extraction_model=fact_check_extraction_model,
            reasoning_model=fact_check_reasoning_model,
        )
        self.writer = WriterAgent(model=writer_model)

    async def run_research(
        self,
        topic: str,
        max_sources: int = 10,
        max_search_loops: int = 2,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> ResearchBrief:
        """
        Executes the end-to-end multi-agent research lifecycle:
        1. Query formulation targeting primary sources & controversies
        2. Web search & high-fidelity content extraction (Trafilatura/httpx)
        3. Atomic claim extraction & cross-document audit (Fact-Checker)
        4. Targeted follow-up research loops for contested facts
        5. Publication-grade synthesis & citation mapping (Writer)
        """
        def notify(msg: str):
            if on_status:
                on_status(msg)

        notify(f"[bold blue]Starting Deep Research Investigation on:[/bold blue] [bold yellow]{topic}[/bold yellow]")

        # Phase 1: Planning Queries
        notify("[cyan]Phase 1: Searcher Agent formulating query matrix...[/cyan]")
        queries = await self.searcher.plan_queries(topic)
        for q in queries:
            notify(f"  • [dim]Planned Query:[/dim] {q}")

        # Phase 2: Primary Source Search & Extraction
        notify(f"[cyan]Phase 2: Searching and scraping up to {max_sources} primary/authoritative sources...[/cyan]")
        documents = await self.searcher.execute_search_and_scrape(
            queries=queries,
            max_sources=max_sources,
            on_progress=notify,
        )

        if not documents:
            raise RuntimeError("No valid source documents could be retrieved or scraped for the topic.")

        # Phase 3: Fact-Checking & Conflict Audit
        notify("[cyan]Phase 3: Fact-Checker Agent auditing sources & constructing atomic claim matrix...[/cyan]")
        fact_report = await self.fact_checker.audit_and_cross_reference(
            topic=topic,
            documents=documents,
            on_progress=notify,
        )

        # Phase 4: Targeted Re-search Loop for Contested Claims
        loop_count = 0
        while (
            loop_count < max_search_loops
            and fact_report.follow_up_queries
        ):
            loop_count += 1
            notify(
                f"[magenta]Phase 4 [Loop {loop_count}/{max_search_loops}]: "
                f"Resolving {len(fact_report.follow_up_queries)} critical contradictions via targeted re-search...[/magenta]"
            )
            targeted_docs = await self.searcher.targeted_research(
                conflict_queries=fact_report.follow_up_queries[:3],
                max_sources_per_query=2,
                on_progress=notify,
            )

            if not targeted_docs:
                notify("[dim]No additional documents found for targeted queries. Proceeding with existing evidence.[/dim]")
                break

            # Filter out already seen URLs
            existing_urls = {d.url for d in documents}
            new_unique_docs = [d for d in targeted_docs if d.url not in existing_urls]

            if not new_unique_docs:
                notify("[dim]No new unique sources discovered. Concluding targeted search loop.[/dim]")
                break

            documents.extend(new_unique_docs)
            notify(f"[green]Added {len(new_unique_docs)} new primary documents. Re-auditing claim matrix...[/green]")

            # Re-audit with expanded document set
            fact_report = await self.fact_checker.audit_and_cross_reference(
                topic=topic,
                documents=documents,
                on_progress=notify,
            )

        # Phase 5: Synthesis & Citation Brief Generation
        notify(
            f"[cyan]Phase 5: Writer Agent synthesizing final brief across {len(documents)} sources and "
            f"{len(fact_report.claims)} verified claims...[/cyan]"
        )
        brief = await self.writer.compile_brief(
            topic=topic,
            documents=documents,
            fact_report=fact_report,
            on_progress=notify,
        )

        notify("[bold green]✔ Multi-Agent Deep Research Complete! Brief successfully generated.[/bold green]")
        return brief
