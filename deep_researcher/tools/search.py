"""Search tool module providing Tavily and DuckDuckGo search integration."""
import os
import asyncio
from typing import List, Dict, Any
from urllib.parse import urlparse
import httpx

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None

from deep_researcher.tools.scraper import evaluate_domain

EXCLUDED_DOMAINS = {
    "pinterest.com",
    "facebook.com",
    "instagram.com",
    "tiktok.com",
    "twitter.com",
    "x.com",
    "youtube.com",
    "quora.com",
    "reddit.com",
}


async def _search_tavily(query: str, api_key: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Searches Tavily API for rich structured results."""
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "advanced",
        "include_answer": False,
        "max_results": max_results,
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                results = []
                for item in data.get("results", []):
                    results.append({
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "snippet": item.get("content", ""),
                    })
                return results
        except Exception:
            pass
    return []


def _search_ddg_sync(query: str, max_results: int = 7) -> List[Dict[str, Any]]:
    """Synchronous DuckDuckGo search using duckduckgo_search library."""
    if not DDGS:
        return []
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            formatted = []
            for item in results:
                formatted.append({
                    "title": item.get("title", ""),
                    "url": item.get("href", ""),
                    "snippet": item.get("body", ""),
                })
            return formatted
    except Exception:
        return []


async def search_web(query: str, max_results: int = 7) -> List[Dict[str, Any]]:
    """
    Executes a web search with Tavily (if API key available) or DuckDuckGo fallback.
    Prioritizes primary and authoritative sources.
    """
    tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
    raw_results = []

    if tavily_key:
        raw_results = await _search_tavily(query, tavily_key, max_results=max_results)

    # Fallback to DuckDuckGo if Tavily failed or key not provided
    if not raw_results:
        raw_results = await asyncio.to_thread(_search_ddg_sync, query, max_results=max_results)

    # Deduplicate and filter out unwanted domains
    seen_urls = set()
    cleaned_results = []

    for item in raw_results:
        url = item.get("url", "").strip()
        if not url or url in seen_urls:
            continue
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if any(ex in domain for ex in EXCLUDED_DOMAINS):
            continue

        seen_urls.add(url)
        is_primary, credibility, clean_domain = evaluate_domain(url)
        cleaned_results.append({
            "title": item.get("title", clean_domain),
            "url": url,
            "snippet": item.get("snippet", ""),
            "domain": clean_domain,
            "is_primary": is_primary,
            "credibility_score": credibility,
        })

    # Sort so primary sources and high-credibility domains appear first
    cleaned_results.sort(key=lambda x: (x["is_primary"], x["credibility_score"]), reverse=True)
    return cleaned_results[:max_results]
