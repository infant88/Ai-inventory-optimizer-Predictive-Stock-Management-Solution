"""
Data Ingestion Module - Web Scraping (BeautifulSoup)
-----------------------------------------------------
Extracts product metadata (name, price, rating) from an online source
and returns it as a list of dicts, ready to merge into the catalog.

This is the general-purpose scraping utility described in the report
(Section 3.2 / 6.2). Point `scrape_product_listing` at any public
listing page whose HTML structure you know, and adjust the CSS
selectors in `SELECTORS` to match that site.

Note: outbound network access is sandboxed in this build/dev
environment, so live scraping may not run everywhere this code is
executed. The simulation engine (simulation.py) is used as the
fallback data source so the rest of the app works without live scraping.
"""

from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup

# Adjust these selectors to match the target site's HTML structure.
SELECTORS = {
    "item": "div.product-card",
    "name": "h2.product-title",
    "price": "span.price",
    "rating": "span.rating",
}

HEADERS = {"User-Agent": "Mozilla/5.0 (AI-Inventory-Optimizer data-collector)"}


@dataclass
class ScrapedProduct:
    name: str
    price: float | None
    rating: float | None


def _parse_price(text: str) -> float | None:
    digits = "".join(ch for ch in text if ch.isdigit() or ch == ".")
    try:
        return float(digits) if digits else None
    except ValueError:
        return None


def _parse_rating(text: str) -> float | None:
    try:
        return float(text.strip().split("/")[0])
    except (ValueError, IndexError):
        return None


def scrape_product_listing(url: str, timeout: int = 10) -> list[ScrapedProduct]:
    """
    Fetches `url` and extracts product cards using the CSS selectors in
    SELECTORS. Returns a list of ScrapedProduct. Raises requests.RequestException
    on network errors so the caller can fall back to simulated data.
    """
    response = requests.get(url, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    results = []
    for card in soup.select(SELECTORS["item"]):
        name_el = card.select_one(SELECTORS["name"])
        price_el = card.select_one(SELECTORS["price"])
        rating_el = card.select_one(SELECTORS["rating"])

        results.append(
            ScrapedProduct(
                name=name_el.get_text(strip=True) if name_el else "Unknown",
                price=_parse_price(price_el.get_text()) if price_el else None,
                rating=_parse_rating(rating_el.get_text()) if rating_el else None,
            )
        )
    return results


def scrape_or_fallback(url: str, fallback_rows: list[dict]) -> list[dict]:
    """
    Tries to scrape `url`; on any failure (offline, blocked, selector
    mismatch) returns `fallback_rows` untouched so the app keeps working.
    """
    try:
        scraped = scrape_product_listing(url)
        return [{"name": p.name, "price": p.price, "rating": p.rating} for p in scraped]
    except Exception:
        return fallback_rows
