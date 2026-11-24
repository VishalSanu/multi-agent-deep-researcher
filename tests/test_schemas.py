"""Tests for Pydantic data models and serialization."""
from deep_researcher.models.schemas import (
    AtomicClaim,
    ConflictGroup,
    FactCheckReport,
    CitationReference,
    SynthesisSection,
    ResearchBrief,
)


def test_atomic_claim_creation():
    claim = AtomicClaim(
        id="C1_1",
        text="Solid-state batteries demonstrate 95% capacity retention after 1000 cycles.",
        subtopic="Cycle Life",
        source_url="https://nature.com/articles/s41586-024-00000",
        source_title="Nature Energy Study",
        source_domain="nature.com",
        quote="The pouch cell maintained 95% capacity at 1000 full depth-of-discharge cycles.",
        credibility_score=0.92,
        is_primary_source=True,
        stance="Supports",
    )
    assert claim.id == "C1_1"
    assert claim.is_primary_source is True
    assert claim.stance == "Supports"


def test_conflict_group_and_report():
    c1 = AtomicClaim(
        id="C1_1",
        text="LFP cells cost $65 per kWh at pack level in 2024.",
        subtopic="Cost",
        source_url="https://bnef.com/report",
        source_title="BloombergNEF",
        source_domain="bnef.com",
        quote="Pack-level pricing fell to $65/kWh.",
        credibility_score=0.88,
        is_primary_source=False,
        stance="Supports",
    )
    c2 = AtomicClaim(
        id="C2_1",
        text="LFP pack production costs remain above $90 per kWh in North America.",
        subtopic="Cost",
        source_url="https://energy.gov/eere",
        source_title="US DOE Report",
        source_domain="energy.gov",
        quote="North American manufacturing costs average $92/kWh.",
        credibility_score=0.95,
        is_primary_source=True,
        stance="Contradicts",
    )
    conflict = ConflictGroup(
        topic_focus="Pack Level LFP Cost",
        claim_a=c1,
        claim_b=c2,
        nature_of_conflict="Discrepancy due to geographic market segmentation (China vs North America).",
        resolved_synthesis="Global blended prices in China reached $65/kWh, while North American localized manufacturing carries a $90+/kWh premium.",
        resolution_confidence=0.9,
        needs_targeted_search=False,
    )
    report = FactCheckReport(
        claims=[c1, c2],
        conflicts=[conflict],
        summary_verdict="Consensus exists on downward price trajectory, but substantial regional cost variance remains.",
        follow_up_queries=[],
    )

    data = report.model_dump()
    assert len(data["conflicts"]) == 1
    assert data["conflicts"][0]["resolution_confidence"] == 0.9
