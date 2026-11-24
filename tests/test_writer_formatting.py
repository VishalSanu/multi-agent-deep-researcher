"""Tests for Writer agent citation formatting and conflict table assembly."""
from deep_researcher.models.schemas import (
    SourceDocument,
    AtomicClaim,
    ConflictGroup,
    FactCheckReport,
)
from deep_researcher.agents.writer import WriterAgent


def test_writer_table_and_bibliography():
    writer = WriterAgent.__new__(WriterAgent)

    doc1 = SourceDocument(
        url="https://nature.com/paper1",
        title="High Energy Density Batteries",
        domain="nature.com",
        raw_content="...",
        cleaned_markdown="Empirical tests indicate 450 Wh/kg achievable in pouch format.",
        is_primary=True,
        credibility_score=0.95,
    )
    doc2 = SourceDocument(
        url="https://ieee.org/paper2",
        title="Grid Scale Economics",
        domain="ieee.org",
        raw_content="...",
        cleaned_markdown="Current manufacturing yields limit mass deployment until 2028.",
        is_primary=True,
        credibility_score=0.92,
    )

    bib = writer._build_bibliography([doc1, doc2])
    assert len(bib) == 2
    assert bib[0].citation_number == 1
    assert bib[0].domain == "nature.com"
    assert bib[1].citation_number == 2

    c1 = AtomicClaim(
        id="C1_1",
        text="Commercial rollout expected by 2025.",
        subtopic="Timeline",
        source_url="https://nature.com/paper1",
        source_title="High Energy Density Batteries",
        source_domain="nature.com",
        quote="Projected rollout in 2025.",
        credibility_score=0.95,
        is_primary_source=True,
        stance="Supports",
    )
    c2 = AtomicClaim(
        id="C2_1",
        text="Mass deployment infeasible before 2028 due to yield limits.",
        subtopic="Timeline",
        source_url="https://ieee.org/paper2",
        source_title="Grid Scale Economics",
        source_domain="ieee.org",
        quote="Yield constraints push scaled availability to 2028.",
        credibility_score=0.92,
        is_primary_source=True,
        stance="Contradicts",
    )
    conflict = ConflictGroup(
        topic_focus="Commercial Deployment Timeline",
        claim_a=c1,
        claim_b=c2,
        nature_of_conflict="Disagreement between lab pilot readiness vs industrial scale yield ramp.",
        resolved_synthesis="Pilot units deploy in 2025-2026, but gigawatt-scale production likely reaches parity only by 2028.",
        resolution_confidence=0.85,
        needs_targeted_search=False,
    )
    report = FactCheckReport(
        claims=[c1, c2],
        conflicts=[conflict],
        summary_verdict="Technological viability demonstrated; industrialization timelines vary.",
        follow_up_queries=[],
    )

    bib_map = {b.url: b.citation_number for b in bib}
    table = writer._format_conflict_table(report, bib_map)

    assert "Commercial Deployment Timeline" in table
    assert "[1]" in table
    assert "[2]" in table
    assert "Disagreement between lab pilot readiness" in table or "Pilot units deploy" in table
