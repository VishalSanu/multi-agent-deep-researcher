"""Writer Agent: compiles the verified claims into an authoritative, cited research brief."""
from typing import List, Dict
from pydantic import BaseModel, Field

from deep_researcher.agents.base import BaseAgent
from deep_researcher.models.schemas import (
    SourceDocument,
    FactCheckReport,
    ResearchBrief,
    SynthesisSection,
    CitationReference,
)


class DraftSections(BaseModel):
    """Thematic sections of the narrative synthesis."""
    executive_summary: str = Field(
        ..., description="Crisp executive summary (2-3 paragraphs) synthesizing the entire topic"
    )
    sections: List[SynthesisSection] = Field(
        ...,
        description=(
            "List of 3 to 6 thematic sections. "
            "Must use inline numbered citations like [1], [2] referencing the source bibliography."
        ),
    )


class WriterAgent(BaseAgent):
    """Specialized agent that synthesizes evidence and formats publication-ready briefs."""

    def __init__(self, model: str = "gemini-3.1-pro-preview"):
        super().__init__(name="Writer", default_model=model)

    def _build_bibliography(self, documents: List[SourceDocument]) -> List[CitationReference]:
        """Creates a deterministic citation map [1], [2], etc., for all sources."""
        bib = []
        for i, doc in enumerate(documents):
            # Extract a key excerpt snippet
            snippet = doc.cleaned_markdown[:300].replace("\n", " ").strip()
            bib.append(
                CitationReference(
                    citation_number=i + 1,
                    title=doc.title,
                    url=doc.url,
                    domain=doc.domain,
                    is_primary=doc.is_primary,
                    key_excerpt=f"{snippet}...",
                )
            )
        return bib

    def _format_conflict_table(self, report: FactCheckReport, bib_map: Dict[str, int]) -> str:
        """Formats the detected conflicting claims into a clean Markdown table."""
        if not report.conflicts:
            return "_No significant empirical contradictions were detected across the analyzed sources._\n"

        rows = [
            "| Topic / Dispute | Perspective A (Source) | Perspective B (Source) | Synthesis / Resolution |",
            "| :--- | :--- | :--- | :--- |",
        ]

        for conf in report.conflicts:
            src_a_num = bib_map.get(conf.claim_a.source_url, "")
            cite_a = f"[{src_a_num}]" if src_a_num else conf.claim_a.source_domain
            src_b_num = bib_map.get(conf.claim_b.source_url, "")
            cite_b = f"[{src_b_num}]" if src_b_num else conf.claim_b.source_domain

            persp_a = f"**{conf.claim_a.text}**<br>_Quote: \"{conf.claim_a.quote[:100]}...\"_ {cite_a}"
            persp_b = f"**{conf.claim_b.text}**<br>_Quote: \"{conf.claim_b.quote[:100]}...\"_ {cite_b}"
            synth = conf.resolved_synthesis or conf.nature_of_conflict

            rows.append(f"| **{conf.topic_focus}** | {persp_a} | {persp_b} | {synth} |")

        return "\n".join(rows) + "\n"

    async def compile_brief(
        self,
        topic: str,
        documents: List[SourceDocument],
        fact_report: FactCheckReport,
        on_progress=None,
    ) -> ResearchBrief:
        """Compiles all verified evidence, claims, and bibliography into the final markdown brief."""
        if on_progress:
            on_progress("[cyan]Writer Agent structuring bibliography and drafting synthesis sections...[/cyan]")

        bibliography = self._build_bibliography(documents)
        bib_url_to_num = {b.url: b.citation_number for b in bibliography}

        # Build context prompt for LLM
        bib_context_lines = []
        for b in bibliography:
            primary_tag = "[PRIMARY SOURCE]" if b.is_primary else "[SECONDARY]"
            bib_context_lines.append(f"[{b.citation_number}] {primary_tag} {b.title} ({b.domain}) - {b.url}")
        bib_context = "\n".join(bib_context_lines)

        claims_context_lines = []
        for c in fact_report.claims:
            cite_num = bib_url_to_num.get(c.source_url, "?")
            claims_context_lines.append(
                f"- [Cite [{cite_num}]] Claim: {c.text} (Quote: \"{c.quote[:120]}\") | Subtopic: {c.subtopic}"
            )
        claims_context = "\n".join(claims_context_lines)

        conflict_context_lines = []
        for cf in fact_report.conflicts:
            c_a = bib_url_to_num.get(cf.claim_a.source_url, "?")
            c_b = bib_url_to_num.get(cf.claim_b.source_url, "?")
            conflict_context_lines.append(
                f"- Area: {cf.topic_focus} | Stance A [{c_a}]: {cf.claim_a.text} vs Stance B [{c_b}]: {cf.claim_b.text} | Resolution: {cf.resolved_synthesis}"
            )
        conflict_context = "\n".join(conflict_context_lines)

        system_prompt = (
            "You are a Principal Policy & Scientific Writer. "
            "You have been provided with a thoroughly fact-checked evidence matrix, source bibliography, and conflict audit. "
            "Your task is to write an authoritative, deep-dive research brief.\n"
            "Requirements:\n"
            "1. Write a comprehensive, nuanced Executive Summary (2-3 paragraphs).\n"
            "2. Organize the brief into 3-5 distinct thematic sections.\n"
            "3. EVERY factual assertion must be attributed with numbered bracketed citations matching the provided bibliography, "
            "e.g. 'Recent data indicates a 40% increase in efficiency [1][3].'\n"
            "4. Clearly articulate nuance, trade-offs, and empirical limits where evidence diverges.\n"
            "5. Maintain an objective, rigorous, publication-grade analytical tone."
        )

        user_prompt = (
            f"Topic: {topic}\n\n"
            f"Fact-Checker Consensus Verdict:\n{fact_report.summary_verdict}\n\n"
            f"Available Bibliography Sources (Use these exact numbers [N] for inline citations):\n{bib_context}\n\n"
            f"Verified Claims & Quotes:\n{claims_context}\n\n"
            f"Conflict & Nuance Log:\n{conflict_context}\n\n"
            "Draft the executive summary and detailed thematic sections with bracketed citations."
        )

        draft = await self.generate_structured(
            prompt=user_prompt,
            response_schema=DraftSections,
            system_instruction=system_prompt,
            temperature=0.25,
        )

        conflict_markdown = self._format_conflict_table(fact_report, bib_url_to_num)

        # Assemble the full complete markdown document
        md_parts = [
            f"# Deep Research Brief: {topic}\n",
            "## Executive Summary\n",
            f"{draft.executive_summary}\n",
            "## Evidence Synthesis\n",
        ]

        for sec in draft.sections:
            md_parts.append(f"### {sec.heading}\n\n{sec.content}\n")

        md_parts.append("## Conflicting Claims & Nuance Analysis\n")
        md_parts.append(conflict_markdown)

        md_parts.append("## Bibliography & Primary Sources\n")
        for b in bibliography:
            primary_label = "*(Primary Source)*" if b.is_primary else "*(Secondary Source)*"
            md_parts.append(
                f"- **[{b.citation_number}]** [{b.title}]({b.url}) {primary_label}\n"
                f"  - **Domain:** `{b.domain}`\n"
                f"  - **Archival Excerpt:** > \"{b.key_excerpt}\"\n"
            )

        md_parts.append("\n## Appendix: Claim Verification Audit Trail\n")
        md_parts.append(f"**Overall Audit Verdict:** {fact_report.summary_verdict}\n\n")
        md_parts.append(f"Total Atomic Claims Audited: **{len(fact_report.claims)}** | Conflicts Analyzed: **{len(fact_report.conflicts)}**\n")

        full_md = "\n".join(md_parts)

        return ResearchBrief(
            topic=topic,
            title=f"Deep Research Brief: {topic}",
            executive_summary=draft.executive_summary,
            sections=draft.sections,
            conflict_analysis_markdown=conflict_markdown,
            bibliography=bibliography,
            claims=fact_report.claims,
            conflicts=fact_report.conflicts,
            full_markdown=full_md,
        )
