"""Searcher Agent: generates targeted sub-queries, searches, and scrapes primary sources."""
import asyncio
from typing import List, Optional
from pydantic import BaseModel, Field

from deep_researcher.agents.base import BaseAgent
from deep_researcher.models.schemas import SourceDocument
from deep_researcher.tools.search import search_web
from deep_researcher.tools.scraper import scrape_urls


class QueryPlan(BaseModel):
    """Sub-queries generated to uncover primary sources and opposing viewpoints."""
    queries: List[str] = Field(
        ...,
        description=(
            "List of 3 to 5 distinct, high-precision search queries designed to find "
            "primary documents, peer-reviewed research, regulatory filings, and diverse viewpoints."
        ),
    )
    focus_areas: List[str] = Field(
        ..., description="Key technical, statistical, or controversy angles being investigated"
    )


class SearcherAgent(BaseAgent):
    """Specialized agent responsible for query planning, web search, and content extraction."""

    def __init__(self, model: str = "gemini-2.5-flash"):
        super().__init__(name="Searcher", default_model=model)

    async def plan_queries(self, topic: str, context: Optional[str] = None) -> List[str]:
        """Deconstructs the user topic into search queries targeting primary sources and controversies."""
        system_prompt = (
            "You are an elite research librarian and intelligence analyst. "
            "Given a research topic, your objective is to generate 3-5 distinct, precise web search queries. "
            "You must include queries aimed at:\n"
            "1. Primary data, official reports, institutional papers, or government documents.\n"
            "2. Critical perspectives, known controversies, and dissenting technical opinions.\n"
            "3. Recent empirical findings and verifiable figures.\n"
            "Avoid generic buzzword queries. Focus on terms that appear in primary documentation."
        )

        user_prompt = f"Topic to investigate: {topic}"
        if context:
            user_prompt += f"\nAdditional Context / Focus: {context}"

        plan: QueryPlan = await self.generate_structured(
            prompt=user_prompt,
            response_schema=QueryPlan,
            system_instruction=system_prompt,
        )
        return plan.queries

    async def execute_search_and_scrape(
        self,
        queries: List[str],
        max_sources: int = 10,
        per_query_results: int = 5,
        on_progress=None,
    ) -> List[SourceDocument]:
        """
        Executes search queries in parallel, collects unique candidate URLs,
        and scrapes them asynchronously.
        """
        if on_progress:
            on_progress(f"[cyan]Dispatching {len(queries)} search queries across primary databases...[/cyan]")

        # Run searches concurrently
        search_tasks = [search_web(q, max_results=per_query_results) for q in queries]
        search_results_nested = await asyncio.gather(*search_tasks)

        candidate_urls = []
        seen = set()

        for res_list in search_results_nested:
            for item in res_list:
                url = item["url"]
                if url not in seen:
                    seen.add(url)
                    candidate_urls.append(url)

        if on_progress:
            on_progress(f"[cyan]Discovered {len(candidate_urls)} candidate URLs. Initiating scraping...[/cyan]")

        # Scrape top candidates up to 2x max_sources to ensure we get enough quality documents
        urls_to_scrape = candidate_urls[: max_sources * 2]
        documents = await scrape_urls(urls_to_scrape, max_concurrency=6)

        # Slice to desired max_sources
        selected_docs = documents[:max_sources]
        if on_progress:
            primary_count = sum(1 for d in selected_docs if d.is_primary)
            on_progress(
                f"[green]Successfully scraped {len(selected_docs)} documents "
                f"({primary_count} primary/authoritative sources).[/green]"
            )

        return selected_docs

    async def targeted_research(
        self, conflict_queries: List[str], max_sources_per_query: int = 3, on_progress=None
    ) -> List[SourceDocument]:
        """Runs targeted searches to resolve specific contradictions or missing claims."""
        if not conflict_queries:
            return []
        if on_progress:
            on_progress(f"[yellow]Executing targeted research on {len(conflict_queries)} contested queries...[/yellow]")

        return await self.execute_search_and_scrape(
            conflict_queries,
            max_sources=len(conflict_queries) * max_sources_per_query,
            per_query_results=max_sources_per_query,
            on_progress=on_progress,
        )
