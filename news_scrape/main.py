from math import e
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
from news_scrape.timestamp_utils import Timestamp
from news_scrape.scraper import AsyncWebScraper

class ContentProcessor:
    """
    Class for content processing: check for sponsored content, remove noise texts,...
    """

    def __init__(self, content: str):
        self.content = content

    def remove_noise(self):
        """Remove all noise texts -> texts that are irrelevant the main content, including disclaimer, advertisement,..."""
        text = self.content
        while "--" in text:
            text = text[text.index("--") + 2 :]
        if ". Visit" in text:
            text = text[: text.index(". Visit")]
        if ". visit" in text:
            text = text[: text.index(". visit")]
        while "-0-" in text:
            text = text[: text.index("-0-")]
        while "References\n" in text:
            text = text[: text.index("References\n")]
        while "Featured image from" in text:
            text = text[: text.index("Featured image from")]
        while "Subscribe to" in text:
            text = text[: text.index("Subscribe to")]
        while "To learn more" in text:
            text = text[: text.index("To learn more")]
        while "Learn more about" in text:
            text = text[: text.index("Learn more about")]
        while "NOTE:" in text:
            text = text[: text.index("NOTE:")]
        while "More information can be found" in text:
            text = text[: text.index("More information can be found")]
        while "For more information" in text:
            text = text[: text.index("For more information")]
        while "View source version" in text:
            text = text[: text.index("View source version")]
        while "Check price action" in text:
            text = text[: text.index("Check price action")]
        while "Don't Miss a Beat" in text:
            text = text[: text.index("Don't Miss a Beat")]
        while "Read more:" in text:
            text = text[: text.index("Read more:")]
        while "Read Also" in text:
            text = text[: text.index("Read Also")]
        while "Also Read:" in text:
            text = text[: text.index("Also Read:")]
        while "Follow Us on" in text:
            text = text[: text.index("Follow Us on")]
        while "Disclosure" in text[1000:]:
            text = text[: text.index("Disclosure")]
        while "Related" in text:
            text = text[: text.index("Related")]
        while "Read More" in text:
            text = text[: text.index("Read More")]
        while "Join Our" in text:
            text = text[: text.index("Join Our")]
        while "PlayDoge" in text:
            text = text[: text.index("PlayDoge")]
        while "Frequently Asked Questions" in text:
            text = text[: text.index("Frequently Asked Questions")]
        while "Recommended Articles" in text:
            text = text[: text.index("Recommended Articles")]
        while "More" in text:
            text = text[: text.index("More")]
        while "Trusted" in text:
            text = text[: text.index("Trusted")]
        while "Advertisement\n" in text:
            text = text[: text.index("Advertisement\n")]
        while "Maximize your Cointribune" in text:
            text = text[: text.index("Maximize your Cointribune")]
        while "Follow us on Google News" in text:
            text = text[: text.index("Follow us on Google News")]
        return text

    def is_sponsored_content(self):
        """
        Filtering out sponsored posts from the content.

        Args:
            txt: content as string
        Returns:
            boolean: True if the content is a sponsored post.
        """
        content = self.content
        if "This content is sponsored" in content:
            return True
        elif "The above article is sponsored content" in content:
            return True
        elif "This is a paid release" in content:
            return True
        elif "This article is sponsored content" in content:
            return True
        elif "This is a paid post" in content:
            return True
        elif "This is a sponsored press release" in content:
            return True
        elif "This is a sponsored article" in content:
            return True
        elif "Press release sponsored by our commercial partners." in content:
            return True
        elif "This content is provided by a third party" in content:
            return True
        elif "This article is a SPONSORED" in content:
            return True
        elif "For more information" in content:
            return True
        elif "This is a sponsored post" in content:
            return True
        elif "This content is a sponsored article" in content:
            return True
        elif "The below article is Sponsored Content" in content:
            return True
        else:
            return False

    def is_sponsored_tags(self, tag_list: str):
        """
        Check if the content contains any tags indicating that this is a sponsored post
        Args:
            tag_list (str): The tags

        Returns:
            bool: True if the content contains any tags indicating that this is a sponsored post, False otherwise
        """
        if "sponsor" in tag_list.lower():
            return True
        elif "advertisement" in tag_list.lower():
            return True
        elif "partnered" in tag_list.lower():
            return True
        else:
            return False


async def get_existing_urls():

    #TODO: get existing crawled urls from DB
    """
    Get a list of existing urls in DB to prevent duplicated urls when scraping

    Output:
    list of urls
    """
    url_list = []

    return url_list


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

    existing_urls = [] # storing all existing urls into a list
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
    ]

    for url in feeds:
        scraper = AsyncWebScraper()
        try:
            res = await scraper.fetch(url)
            soup = BeautifulSoup(res, "xml")
            for item in soup.find_all("item"):
                links = item.findAll("link")
                for link in links:
                    if (
                        link.text not in existing_urls and "/videos/" not in link.text
                    ):  # Handling cases when it is a url to a video or a podcast for some websites
                        urls.append(link.text)
        except RuntimeError:
            continue
    logger.info("Number of new urls in this session: " + str(len(urls)))
    return urls





def deep_find(dictionary, target_key) -> Any:
    """
    Recursively searches for a key in a nested dictionary and returns its value.
    Used for any json_ld dicts from any websites to get the value of a specific key.

    Args:
        dictionary (dict): The dictionary to search in.
        target_key (str): The key to search for.

    Returns:
        The value of the key if found, otherwise None.
    """
    if isinstance(dictionary, dict):
        for key, value in dictionary.items():
            if key == target_key:
                return value
            found = deep_find(value, target_key)
            if found is not None:
                return found
    elif isinstance(dictionary, list):
        for item in dictionary:
            found = deep_find(item, target_key)
            if found is not None:
                return found
    return []


async def extract_meta_content(soup, property_name, attr="property", default="") -> Any:
    """Async helper function to extract value from meta tag and a specific attribute"""
    try:
        return soup.find("meta", attrs={attr: property_name})["content"]
    except Exception:
        return default


async def extract_json_ld_data(soup) -> list[dict]:
    """Extract JSON-LD data from the page. Several websites use this format to add structured data to their pages. See: https://json-ld.org/"""
    import json

    try:
        return [
            json.loads(x.string, strict=False)
            for x in soup.find_all("script", type="application/ld+json")
        ]
    except Exception:
        return []



async def async_extract_data_from_url(url):


    scraper = AsyncWebScraper()
    doc = await scraper.fetch(url)

    # Ensure response processing is non-blocking
    response_text = doc.split("CONTENT END", 1)[0]

    parsed = urlparse(url)
    namespace = uuid.NAMESPACE_URL
    uuid_v5 = uuid.uuid5(namespace, url)
    domain_name = parsed.hostname

    soup = BeautifulSoup(response_text, "html.parser")
    for div in soup.find_all("div", attrs={"class": "news-tab-content"}):
        div.decompose()

    extract = trafilatura.bare_extraction(soup.prettify())
    json_ld_data = await extract_json_ld_data(soup)

    # Metadata extraction
    try:
        upload_date = (
            await extract_meta_content(soup, "article:published_time")
            or await extract_meta_content(soup, "published", "name")
            or json_ld_data[0].get("@graph", [{}])[0].get("datePublished", "")
            or json_ld_data[0].get("publishedAt", "")
        )
    except Exception as e:
        upload_date = ""
    try:
        description = await extract_meta_content(
            soup, "og:description"
        ) or json_ld_data[2].get("description", "")
    except Exception as e:
        description = ""
    image_url = await extract_meta_content(soup, "og:image")
    last_modified = await extract_meta_content(soup, "article:modified_time")
    author = await extract_meta_content(soup, "author") or await extract_meta_content(
        soup, "author", "name"
    )
    try:
        tags = deep_find(json_ld_data[0], "articleSection")
        article_keywords = deep_find(json_ld_data[0], "keywords")
    except Exception as e:
        tags = []
        article_keywords = []

    if isinstance(extract, dict):
        title = extract.get("title", "")
        print(title)
        content = extract.get("text", "")
    else:
        title = getattr(extract, 'title', '')
        print(title)
        content = getattr(extract, 'text', '')

    # if not ContentProcessor().is_sponsored_tags(
    #     str(tags)
    # ) and not ContentProcessor().is_sponsored_content(str(article_keywords)):
    #     multipartite_keywords, bert_keywords, doc_embeddings = await extract_keywords(
    #         title + ". " + content
    #     )
    # else:
    #     multipartite_keywords, bert_keywords, doc_embeddings = [], [], None

    crawl_date = datetime.now().strftime("%Y-%m-%dT%H:%M:%S%z")

    # tokens_mentioned = await async_gliner_extractor.retrieve_tokens_found(content)
    # gliner_results = await async_gliner_extractor.retrieve_entities_found(content)

    return {
        "_id": str(uuid_v5),
        "url": url,
        "domain_name": domain_name,
        "title": title,
        "content": content,
        "published_time": upload_date,
        "description": description,
        "tags": tags,
        "article_keywords": article_keywords,
        "last_modified": last_modified,
        "author": author,
        "crawl_date": crawl_date,
        # "multipartite_keywords": multipartite_keywords,
        # "bert_keywords": bert_keywords,
        "image_url": image_url,
        # "doc_embeddings": doc_embeddings,
        # "tokens_mentioned": tokens_mentioned,
        # "gliner_results": str(gliner_results),
    }

if __name__ == "__main__":
    import asyncio


    # Example usage
    url = "https://beincrypto.com/bitcoin-whales-reach-high-market-uncertainty/"
    data = asyncio.run(async_extract_data_from_url(url))
    print(data)