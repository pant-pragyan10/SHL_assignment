"""Small runner to scrape the SHL catalog and persist cleaned JSON."""
from typing import Optional
import argparse
import logging
import os

from shl_agent.ingestion.scraper import CatalogScraper

logger = logging.getLogger(__name__)


def run(output_path: str, base_url: Optional[str] = None, max_pages: int = 50) -> None:
    """Run the scraper and save results to `output_path`.

    Args:
        output_path: where to write JSON
        base_url: optional base URL to scrape (defaults to SHL home)
        max_pages: maximum candidate pages to process
    """
    scraper = CatalogScraper(base_url=base_url) if base_url else CatalogScraper()
    records = scraper.scrape(max_pages=max_pages)
    logger.info("Scraped %d records", len(records))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    scraper.save(records, output_path)
    logger.info("Saved catalog to %s", output_path)


def _cli():
    p = argparse.ArgumentParser(description="Run SHL catalog scraper")
    p.add_argument("--out", default="data/processed/catalog.json")
    p.add_argument("--base-url", default=None)
    p.add_argument("--max-pages", type=int, default=50)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    run(args.out, base_url=args.base_url, max_pages=args.max_pages)


if __name__ == "__main__":
    _cli()
