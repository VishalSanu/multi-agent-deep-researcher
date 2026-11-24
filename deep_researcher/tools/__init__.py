"""Tools for searching and scraping primary sources."""
from deep_researcher.tools.search import search_web
from deep_researcher.tools.scraper import scrape_urls, evaluate_domain

__all__ = ["search_web", "scrape_urls", "evaluate_domain"]
