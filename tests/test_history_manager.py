"""Tests for history manager persistence and retrieval."""
import shutil
from pathlib import Path
from deep_researcher.models.schemas import ResearchBrief, CitationReference, SynthesisSection
from deep_researcher.history_manager import save_session, list_sessions, load_session, HISTORY_DIR


def test_history_save_and_load(tmp_path, monkeypatch):
    test_dir = tmp_path / "test_history"
    monkeypatch.setattr("deep_researcher.history_manager.HISTORY_DIR", test_dir)

    brief = ResearchBrief(
        topic="Test Topic on Clean Energy",
        title="Deep Research Brief: Test Topic on Clean Energy",
        executive_summary="This is a test summary.",
        sections=[
            SynthesisSection(
                heading="Key Findings",
                content="Empirical data shows rapid growth [1].",
                cited_numbers=[1],
            )
        ],
        conflict_analysis_markdown="None",
        bibliography=[
            CitationReference(
                citation_number=1,
                title="Energy Report",
                url="https://energy.gov/report",
                domain="energy.gov",
                is_primary=True,
                key_excerpt="Rapid growth documented.",
            )
        ],
        full_markdown="# Deep Research Brief: Test Topic on Clean Energy\n\nThis is a test summary.",
    )

    result = save_session(brief)
    assert Path(result["md_path"]).exists()
    assert Path(result["json_path"]).exists()

    sessions = list_sessions()
    assert len(sessions) == 1
    assert "Test Topic on Clean Energy" in sessions[0]["topic"]

    loaded = load_session(result["json_path"])
    assert loaded is not None
    assert loaded["topic"] == "Test Topic on Clean Energy"
    assert len(loaded["bibliography"]) == 1
