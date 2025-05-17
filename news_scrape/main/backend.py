import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mongo.mongo_client import MongoService

logger = logging.getLogger(__name__)

mongo_client = MongoService()


def parse_datetime(time_str: str) -> datetime | None:
    """
    Convert a string timestamp to a datetime object with UTC timezone.

    Supports multiple formats:
    - ISO format (2025-05-14T12:00:00Z, 2025-05-14T12:00:00+00:00)
    - Simple date (2025-05-14)
    - Date with time (2025-05-14 12:00:00)

    Parameters:
        time_str (str): Timestamp string to convert

    Returns:
        datetime: Datetime object with UTC timezone, or None if parsing fails
    """
    if not time_str:
        return None

    formats = [
        "%Y-%m-%dT%H:%M:%S.%fZ",  # ISO format with microseconds
        "%Y-%m-%dT%H:%M:%SZ",  # ISO format without microseconds
        "%Y-%m-%dT%H:%M:%S.%f%z",  # ISO format with timezone and microseconds
        "%Y-%m-%dT%H:%M:%S%z",  # ISO format with timezone
        "%Y-%m-%d %H:%M:%S",  # Simple datetime
        "%Y-%m-%d %H:%M",  # Simple datetime without seconds
        "%Y-%m-%d",  # Just date
    ]

    # Try each format until one works
    for fmt in formats:
        try:
            dt = datetime.strptime(time_str, fmt)
            # Add UTC timezone if not present
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue

    # If we get here, none of the formats worked
    logger.warning(f"Could not parse datetime string: {time_str}")
    return None


@dataclass
class ScrapedNews:
    docs_count: int = field(default=0)
    urls: list[Any] = field(default_factory=list)


async def get_scraped_urls_by_time_interval(start_time: str, end_time: str):
    """
    Get scraped URLs from MongoDB within a specified time interval.

    Parameters:
    start_time (str): Start time in ISO format.
    end_time (str): End time in ISO format.

    Returns:
    list: List of URLs scraped within the specified time interval.
    """
    start_time_iso = parse_datetime(start_time)
    end_time_iso = parse_datetime(end_time)
    if not start_time_iso or not end_time_iso:
        logger.error("Invalid start or end time format.")
        raise ValueError("Invalid start or end time format.")
    scraped_news = await mongo_client.get_news_metadata_by_time_interval(
        start_time=start_time_iso, end_time=end_time_iso
    )
    if scraped_news:
        urls = [news["url"] for news in scraped_news]
        return ScrapedNews(
            docs_count=len(scraped_news),
            urls=urls,
        )
    else:
        logger.info("No URLs found in the specified time interval.")
        return ScrapedNews()


async def get_total_scraped_urls():
    """
    Get all scraped URLs from MongoDB.

    Returns:
    list: List of all scraped URLs.
    """
    scraped_news = await mongo_client.get_all_news_metadata()
    if scraped_news:
        logger.info(f"Found {len(scraped_news)} URLs in the database.")
        return len(scraped_news)
    else:
        logger.info("No URLs found in the database.")
        return 0


async def get_ner_count_by_time_interval(start_time: str, end_time: str) -> list[Any]:
    """
    Get NER counts from MongoDB within a specified time interval.

    Parameters:
    start_time (str): Start time in ISO format.
    end_time (str): End time in ISO format.

    Returns:
    list: List of NER counts within the specified time interval.
    """
    start_time_iso = parse_datetime(start_time)
    end_time_iso = parse_datetime(end_time)
    if not start_time_iso or not end_time_iso:
        logger.error("Invalid start or end time format.")
        raise ValueError("Invalid start or end time format.")
    ner_counts = await mongo_client.get_count_ner_results_by_time_intervals(
        start=start_time_iso, end=end_time_iso
    )
    res = []
    if ner_counts:
        for count in ner_counts:
            res.append(
                {
                    "hour": count["hour"],
                    "mentioned_frequency": count["mentioned_frequency"],
                    "time": count["time"],
                }
            )
        logger.info(f"Found {len(res)} NER counts in the specified time interval.")
    else:
        logger.info("No NER counts found in the specified time interval.")
    return res
