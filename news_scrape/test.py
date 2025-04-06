
from typing import Any
import os
import sys
import asyncio
import uuid
from datetime import datetime
from urllib.parse import urlparse

import trafilatura
from bs4 import BeautifulSoup

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import timedelta

from news_scrape.mongo_db import AsyncMongoDB as AsyncMongoClient
from news_scrape.mongo_db import MongoDB
from news_scrape.timestamp_utils import Timestamp, TimestampClient
from news_scrape.scraper import AsyncWebScraper


async def urls_from_webpage() -> list[str]:
    """
    Get all urls from rss feeds that are not existed in the database.

    Output:
        urls: a list of urls from rss feeds
    """
    import logging

    logger = logging.getLogger("rss_producer")
    from bs4 import BeautifulSoup

    urls = []

    feeds = [
        "https://feeds.bloomberg.com/crypto/news.rss",
        "https://news.bitcoin.com/rss",
        "https://cryptodnes.bg/en/feed",
        "https://ambcrypto.com/feed/",
        "https://coingape.com/feed/",
        "https://www.cryptotimes.io/feed/",
        "https://blockworks.co/feed",
        "https://cryptoslate.com/feed/",
        "https://decrypt.co/feed",
        "https://zycrypto.com/feed/",
        "https://dailyhodl.com/feed/",
        "https://bitcoinist.com/feed/",
        "https://crypto.news/feed/",
        "https://finbold.com/feed/",
        "https://u.today/rss",
        "https://www.newsbtc.com/feed/",
        "https://insidebitcoins.com/feed",
        "https://cryptomufasa.com/feed/",
        "https://cryptobriefing.com/feed/",
        "https://www.coindesk.com/arc/outboundfeeds/rss",
        "https://www.thecoinrepublic.com/feed/",
        "https://en.bitcoinsistemi.com/feed/",
        "https://crypto-economy.com/feed/",
        "https://www.cointribune.com/en/feed/",
        "https://cryptonews.com/rss/",
        "https://dailycoin.com/feed/",
        "https://en.cryptonomist.ch/feed/",
        "https://coinpaprika.com/news/feed/",
        "https://cryptodaily.co.uk/feed",
        "https://beincrypto.com/feed/",
        "https://www.cryptopolitan.com/rss",
        ""
    ]

    for url in feeds:
        scraper = AsyncWebScraper()
        try:
            print("Fetching URL: " + url)
            res = await scraper.fetch(url)
            soup = BeautifulSoup(res, "xml")
            for item in soup.find_all("item"):
                links = item.findAll("link")
                for link in links:
                    print(link.text)
                    if (
                        "/videos/" not in link.text
                    ):  # Handling cases when it is a url to a video or a podcast for some websites
                        urls.append(link.text)

        except RuntimeError:
            continue
    logger.info("Number of new urls in this session: " + str(len(urls)))
    return urls


if __name__ == "__main__":
    urls =  asyncio.run(urls_from_webpage())
    mongo_db = MongoDB()
    timestamp = TimestampClient().now.timestamp()
    _=mongo_db.insert(
        database_name="news",
        collection_name="urls",
        data={"urls": urls, "timestamp": timestamp},
    )
    print(urls)
