"""Web scraper module using Trafilatura, BeautifulSoup, and httpx."""
import asyncio
import re
from urllib.parse import urlparse
from typing import List, Tuple, Optional
import httpx
from bs4 import BeautifulSoup

try:
    import trafilatura
except ImportError:
    trafilatura = None

from deep_researcher.models.schemas import SourceDocument

PRIMARY_DOMAINS_SUFFIXES = {
    ".gov",
    ".mil",
    ".edu",
    ".ac.uk",
    ".europa.eu",
    ".int",
}

KNOWN_PRIMARY_DOMAINS = {
    "arxiv.org",
    "biorxiv.org",
    "medrxiv.org",
    "nature.com",
    "science.org",
    "nih.gov",
    "ncbi.nlm.nih.gov",
    "cdc.gov",
    "who.int",
    "sec.gov",
    "ieee.org",
    "acm.org",
    "github.com",
    "pypi.org",
    "courthousenews.com",
    "supremecourt.gov",
    "whitehouse.gov",
    "federalregister.gov",
    "un.org",
}

KNOWN_HIGH_CREDIBILITY_NEWS = {
    "reuters.com",
    "apnews.com",
    "bloomberg.com",
    "wsj.com",
    "ft.com",
    "bbc.com",
    "bbc.co.uk",
    "economist.com",
}


def evaluate_domain(url: str) -> Tuple[bool, float, str]:
    """
    Evaluates whether a domain is a primary or high-credibility source and assigns a score (0.0 to 1.0).
    Returns (is_primary, credibility_score, clean_domain).
    """
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]

    # Check top-level suffix matches
    is_primary = False
    credibility = 0.65

    for suffix in PRIMARY_DOMAINS_SUFFIXES:
        if domain.endswith(suffix):
            is_primary = True
            credibility = 0.95
            return is_primary, credibility, domain

    # Check known primary repositories & institutions
    for known in KNOWN_PRIMARY_DOMAINS:
        if domain == known or domain.endswith("." + known):
            is_primary = True
            credibility = 0.92
            return is_primary, credibility, domain

    # Check known high-credibility news/wire services
    for news in KNOWN_HIGH_CREDIBILITY_NEWS:
        if domain == news or domain.endswith("." + news):
            is_primary = False
            credibility = 0.85
            return is_primary, credibility, domain

    # Heuristic for official docs / research portals
    if "docs." in domain or "research." in domain or "press." in domain:
        is_primary = True
        credibility = 0.88
        return is_primary, credibility, domain

    return is_primary, credibility, domain


def _extract_text_fallback(html: str) -> Tuple[str, str]:
    """Fallback text extractor using BeautifulSoup."""
    soup = BeautifulSoup(html, "html.parser")

    # Extract title
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else "Untitled Page"

    # Remove script, style, navigation, footer tags
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "aside"]):
        tag.decompose()

    # Get body text
    body = soup.find("body")
    text = body.get_text(separator="\n", strip=True) if body else soup.get_text(separator="\n", strip=True)
    # Collapse multiple blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return title, text[:15000]


def extract_content_from_html(html: str, url: str) -> Tuple[str, str]:
    """Extracts title and clean text/markdown from raw HTML."""
    if trafilatura:
        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            output_format="markdown",
            url=url,
        )
        metadata = trafilatura.extract_metadata(html)
        title = metadata.title if (metadata and metadata.title) else ""

        if extracted and len(extracted.strip()) > 150:
            if not title:
                # Try simple title scrape
                soup = BeautifulSoup(html, "html.parser")
                t_tag = soup.find("title")
                title = t_tag.get_text(strip=True) if t_tag else "Source Document"
            return title, extracted[:20000]

    # Fallback to BeautifulSoup
    return _extract_text_fallback(html)


async def scrape_url(
    client: httpx.AsyncClient, url: str, semaphore: asyncio.Semaphore
) -> Optional[SourceDocument]:
    """Scrapes a single URL with rate limiting, timeouts, and headers."""
    async with semaphore:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        try:
            resp = await client.get(url, headers=headers, follow_redirects=True, timeout=12.0)
            if resp.status_code >= 400:
                return None

            html = resp.text
            title, clean_text = extract_content_from_html(html, url)
            if not clean_text or len(clean_text.strip()) < 100:
                return None

            is_primary, credibility, domain = evaluate_domain(url)

            return SourceDocument(
                url=url,
                title=title or domain,
                domain=domain,
                raw_content=html[:4000],
                cleaned_markdown=clean_text,
                is_primary=is_primary,
                credibility_score=credibility,
            )
        except Exception:
            return None


async def scrape_urls(urls: List[str], max_concurrency: int = 5) -> List[SourceDocument]:
    """Asynchronously scrapes multiple URLs and returns parsed SourceDocument objects."""
    semaphore = asyncio.Semaphore(max_concurrency)
    limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
    async with httpx.AsyncClient(limits=limits, verify=False) as client:
        tasks = [scrape_url(client, url, semaphore) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    valid_docs = [r for r in results if isinstance(r, SourceDocument)]
    # Sort docs by credibility and primary source preference
    valid_docs.sort(key=lambda d: (d.is_primary, d.credibility_score), reverse=True)
    return valid_docs
