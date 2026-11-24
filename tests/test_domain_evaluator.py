"""Tests for domain authority and primary source evaluation."""
from deep_researcher.tools.scraper import evaluate_domain


def test_gov_domain():
    is_primary, cred, domain = evaluate_domain("https://www.cdc.gov/flu/weekly/index.htm")
    assert is_primary is True
    assert cred >= 0.90
    assert domain == "cdc.gov"


def test_arxiv_domain():
    is_primary, cred, domain = evaluate_domain("https://arxiv.org/abs/2301.00000")
    assert is_primary is True
    assert cred >= 0.90
    assert domain == "arxiv.org"


def test_edu_domain():
    is_primary, cred, domain = evaluate_domain("https://news.mit.edu/2024/research-announcement")
    assert is_primary is True
    assert cred >= 0.90
    assert domain == "news.mit.edu"


def test_news_wire():
    is_primary, cred, domain = evaluate_domain("https://www.reuters.com/business/finance/article")
    assert is_primary is False
    assert cred >= 0.80
    assert domain == "reuters.com"


def test_general_domain():
    is_primary, cred, domain = evaluate_domain("https://some-tech-blog.io/post/123")
    assert is_primary is False
    assert cred <= 0.70
