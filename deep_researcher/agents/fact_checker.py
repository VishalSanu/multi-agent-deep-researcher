"""Fact-Checker Agent: extracts atomic claims, evaluates credibility, detects contradictions, and arbitrates disputes."""
import asyncio
from typing import List, Optional
from pydantic import BaseModel, Field

from deep_researcher.agents.base import BaseAgent
from deep_researcher.models.schemas import (
    SourceDocument,
    AtomicClaim,
    ConflictGroup,
    FactCheckReport,
)


class ExtractedClaimsList(BaseModel):
    """Container for atomic claims extracted from a single document."""
    claims: List[AtomicClaim] = Field(
        ...,
        description=(
            "List of 3 to 8 atomic, verifiable claims extracted from the text. "
            "Each must have a direct verbatim quote from the source supporting it."
        ),
    )


class CrossReferenceAnalysis(BaseModel):
    """Analysis of tensions, disagreements, and contradictions between claims."""
    conflicts: List[ConflictGroup] = Field(
        default_factory=list,
        description="List of detected conflicts, contradictions, or significant nuances between claims",
    )
    overall_verdict: str = Field(
        ..., description="Executive summary of the empirical consensus and main points of dispute"
    )
    recommended_re_search_queries: List[str] = Field(
        default_factory=list,
        description="Targeted search queries recommended to resolve critical disputed facts",
    )


class FactCheckerAgent(BaseAgent):
    """Specialized agent that audits sources, constructs claim matrices, and detects factual conflicts."""

    def __init__(self, extraction_model: str = "gemini-2.5-flash", reasoning_model: str = "gemini-3.1-pro-preview"):
        super().__init__(name="FactChecker", default_model=reasoning_model)
        self.extraction_model = extraction_model
        self.reasoning_model = reasoning_model

    async def extract_claims_from_doc(self, doc: SourceDocument, doc_idx: int) -> List[AtomicClaim]:
        """Extracts atomic claims and verbatim quotes from a single scraped document."""
        system_prompt = (
            "You are a rigorous evidentiary fact-checker. "
            "Given a source document, extract 3 to 8 atomic, discrete factual claims. "
            "For every claim:\n"
            "- It must be an objective, testable assertion (statistics, timeline, mechanism, finding).\n"
            "- You must provide an exact verbatim quote from the text.\n"
            "- Provide a subtopic tag (e.g. 'Safety', 'Economic Impact', 'Technical Performance').\n"
            "- Determine stance ('Supports', 'Contradicts', 'Nuances', or 'Neutral').\n"
            "- Mark credibility and primary source status according to the document metadata."
        )

        user_prompt = (
            f"Source URL: {doc.url}\n"
            f"Title: {doc.title}\n"
            f"Domain: {doc.domain}\n"
            f"Is Primary Source: {doc.is_primary}\n"
            f"Base Credibility: {doc.credibility_score}\n\n"
            f"Document Text:\n{doc.cleaned_markdown[:9000]}"
        )

        try:
            result = await self.generate_structured(
                prompt=user_prompt,
                response_schema=ExtractedClaimsList,
                system_instruction=system_prompt,
                model=self.extraction_model,
                temperature=0.1,
            )
            # Ensure unique IDs and link back to document
            claims = []
            for i, c in enumerate(result.claims):
                c.id = f"C{doc_idx}_{i+1}"
                c.source_url = doc.url
                c.source_title = doc.title
                c.source_domain = doc.domain
                c.is_primary_source = doc.is_primary
                if doc.is_primary:
                    c.credibility_score = max(c.credibility_score, 0.85)
                claims.append(c)
            return claims
        except Exception:
            return []

    async def audit_and_cross_reference(
        self,
        topic: str,
        documents: List[SourceDocument],
        on_progress=None,
    ) -> FactCheckReport:
        """
        Orchestrates extraction of claims across all documents, performs cross-referencing,
        and identifies direct contradictions and nuances.
        """
        if on_progress:
            on_progress(f"[cyan]Extracting atomic claims from {len(documents)} source documents...[/cyan]")

        # Extract claims concurrently
        tasks = [self.extract_claims_from_doc(doc, i + 1) for i, doc in enumerate(documents)]
        claim_lists = await asyncio.gather(*tasks)

        all_claims: List[AtomicClaim] = []
        for cl in claim_lists:
            all_claims.extend(cl)

        if on_progress:
            on_progress(f"[green]Extracted {len(all_claims)} atomic claims. Performing cross-document audit...[/green]")

        if not all_claims:
            return FactCheckReport(
                claims=[],
                conflicts=[],
                summary_verdict="Insufficient claims could be extracted from the retrieved sources.",
                follow_up_queries=[],
            )

        # Build cross-reference prompt
        claims_summary = []
        for c in all_claims:
            primary_badge = "[PRIMARY SOURCE]" if c.is_primary_source else "[SECONDARY]"
            claims_summary.append(
                f"- ID: {c.id} | Subtopic: {c.subtopic} | Stance: {c.stance} {primary_badge}\n"
                f"  Claim: {c.text}\n"
                f"  Source: {c.source_title} ({c.source_domain})\n"
                f"  Quote: \"{c.quote}\"\n"
            )
        claims_text = "\n".join(claims_summary)

        system_prompt = (
            "You are a Senior Fact-Checker and Intelligence Arbiter. "
            "You have received a collection of atomic claims extracted from primary and secondary sources. "
            "Your tasks:\n"
            "1. Scrutinize all claims for direct empirical contradictions, conflicting statistics, divergent timelines, "
            "or conflicting causal interpretations.\n"
            "2. For each conflict, identify Claim A and Claim B, explain why they conflict, and provide a balanced analytical "
            "synthesis (e.g., differences in methodology, time period, sample size, or corporate/ideological bias).\n"
            "3. If a conflict is critical to the topic and cannot be resolved with the current claims, set "
            "needs_targeted_search=True and propose a targeted search query.\n"
            "4. Summarize the overall consensus versus dispute landscape."
        )

        user_prompt = (
            f"Topic: {topic}\n\n"
            f"Claims Extracted from Sources:\n{claims_text}\n\n"
            "Conduct the cross-reference audit and report conflicts and required re-search queries."
        )

        analysis = await self.generate_structured(
            prompt=user_prompt,
            response_schema=CrossReferenceAnalysis,
            system_instruction=system_prompt,
            model=self.reasoning_model,
            temperature=0.2,
        )

        report = FactCheckReport(
            claims=all_claims,
            conflicts=analysis.conflicts,
            summary_verdict=analysis.overall_verdict,
            follow_up_queries=analysis.recommended_re_search_queries,
        )

        if on_progress:
            unresolved = sum(1 for c in report.conflicts if c.needs_targeted_search)
            on_progress(
                f"[yellow]Audit complete: {len(report.conflicts)} conflicts detected "
                f"({unresolved} requiring targeted follow-up re-search).[/yellow]"
            )

        return report
