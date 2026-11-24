"""Pydantic schemas for the Multi-Agent Deep Researcher."""
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class AtomicClaim(BaseModel):
    """A discrete, verifiable factual assertion extracted from a primary or secondary source."""
    id: str = Field(..., description="Unique claim identifier, e.g. C1, C2")
    text: str = Field(..., description="The concise factual claim")
    subtopic: str = Field(..., description="Categorical subtopic or thematic tag")
    source_url: str = Field(..., description="URL of the source where claim was found")
    source_title: str = Field(..., description="Title or publication name")
    source_domain: str = Field(..., description="Clean domain name of the source")
    quote: str = Field(..., description="Direct quote from the source supporting the claim")
    credibility_score: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Heuristic or LLM credibility score (0-1)"
    )
    is_primary_source: bool = Field(
        default=False, description="Whether the source is considered primary (gov, edu, arXiv, official doc)"
    )
    stance: Literal["Supports", "Contradicts", "Nuances", "Neutral"] = Field(
        default="Neutral", description="Stance relative to the core topic or related claims"
    )


class SourceDocument(BaseModel):
    """Scraped and parsed web source document."""
    url: str
    title: str
    domain: str
    raw_content: str
    cleaned_markdown: str
    is_primary: bool = False
    credibility_score: float = 0.5


class ConflictGroup(BaseModel):
    """Represents a direct conflict, tension, or nuance difference between claims."""
    topic_focus: str = Field(..., description="Specific subtopic or question where the conflict occurs")
    claim_a: AtomicClaim = Field(..., description="First claim asserting stance A")
    claim_b: AtomicClaim = Field(..., description="Second claim asserting contradictory or nuanced stance B")
    nature_of_conflict: str = Field(..., description="Detailed explanation of what makes these claims clash")
    resolved_synthesis: Optional[str] = Field(
        default=None, description="Analytical resolution or balanced consensus explaining the divergence"
    )
    resolution_confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Confidence in the resolution (0-1)"
    )
    needs_targeted_search: bool = Field(
        default=False, description="True if conflict cannot be arbitrated without more evidence"
    )
    suggested_query: Optional[str] = Field(
        default=None, description="Targeted query to resolve the conflict"
    )


class FactCheckReport(BaseModel):
    """Aggregate audit report produced by the Fact-Checker agent."""
    claims: List[AtomicClaim] = Field(default_factory=list)
    conflicts: List[ConflictGroup] = Field(default_factory=list)
    summary_verdict: str = Field(..., description="Overview of factual consensus and dispute landscape")
    follow_up_queries: List[str] = Field(
        default_factory=list, description="Targeted queries needed for unresolved contradictions"
    )


class CitationReference(BaseModel):
    """A bibliography item mapped to a bracketed citation [N]."""
    citation_number: int
    title: str
    url: str
    domain: str
    is_primary: bool
    key_excerpt: str


class SynthesisSection(BaseModel):
    """A thematic section of the final research brief."""
    heading: str
    content: str = Field(..., description="Markdown paragraph(s) with inline citations [1], [2], etc.")
    cited_numbers: List[int] = Field(default_factory=list)


class ResearchBrief(BaseModel):
    """The final compiled and cited research brief."""
    topic: str
    title: str
    executive_summary: str
    sections: List[SynthesisSection]
    conflict_analysis_markdown: str
    bibliography: List[CitationReference]
    claims: List[AtomicClaim] = Field(default_factory=list)
    conflicts: List[ConflictGroup] = Field(default_factory=list)
    full_markdown: str
