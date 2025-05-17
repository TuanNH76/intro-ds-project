import asyncio
import logging
import os
import random
import sys
from typing import Any

import cloudscraper
import httpx
from aiohttp import ClientSession
from curl_cffi import requests as curl_requests
from curl_cffi.requests import AsyncSession

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logger = logging.getLogger("services")


class WebScraper:
    def __init__(
        self, strategies: list[str] | None = None, user_agent: str | None = None
    ):
        self.strategies: list[str] = strategies or [
            "curl_cffi",
            "cloudscraper",
            "selenium",
            "urllib",
        ]
        self.user_agent = (
            user_agent
            or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
        self.status_code: int | str = 200

    def _fetch_with_curl(
        self, url: str, headers: dict[str, Any] | None = None, timeout: int = 10
    ) -> str:
        """Fetch a URL using curl_cffi."""
        headers = headers or {"User-Agent": self.user_agent}
        response = curl_requests.get(
            url,
            headers=headers,
            timeout=timeout,
            impersonate="chrome",
            thread=None,
            curl_options=None,
            debug=None,
        )
        self.status_code = response.status_code
        return response.text

    def _fetch_with_cloudscraper(
        self, url: str, headers: dict[str, Any] | None = None, timeout: int = 10
    ) -> str:
        """Fetch a URL using cloudscraper."""
        scraper = cloudscraper.create_scraper()  # pyright: ignore[reportUnknownMemberType]
        headers = headers or {"User-Agent": self.user_agent}
        response = scraper.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        self.status_code = response.status_code
        return response.text

    def _fetch_with_urllib(self, url: str):
        """Fetch a URL using urllib."""
        import urllib.error
        import urllib.request

        headers = {"User-Agent": self.user_agent}
        request = urllib.request.Request(url, headers=headers)

        try:
            with urllib.request.urlopen(request) as response:
                content = response.read().decode("utf-8")
                self.status_code = response.getcode()
                return content
        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to fetch URL {url}: {e}")

    def fetch(self, url: str):
        """Fetch a URL using the available strategies."""
        for strategy in self.strategies:
            try:
                # Sequentially try each strategy until one succeeds, or raise an exception if all strategies fail
                if strategy == "curl_cffi":
                    content = self._fetch_with_curl(url)
                elif strategy == "cloudscraper":
                    content = self._fetch_with_cloudscraper(url)
                elif strategy == "urllib":
                    content = self._fetch_with_urllib(url)
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")

                if self.status_code == 200:
                    return content
                else:
                    logger.info(
                        f"Strategy {strategy} failed with status code {self.status_code}. Switching to next strategy."
                    )
            except Exception as e:
                logger.info(
                    f"Strategy {strategy} encountered an error: {e}. Switching to next strategy."
                )

        raise RuntimeError(f"All strategies failed to fetch the URL: {url}")


class AsyncWebScraper:
    def __init__(
        self,
        strategies: list[str] | None = None,
        user_agent: str | None = None,
    ):
        self.strategies = strategies or [
            "curl_cffi",
            "cloudscraper",
            "httpx",
            "aiohttp",
            # "patchright",
            # "ulixee",
        ]
        self.user_agent = (
            user_agent
            or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
        self.status_code: int | str | None = None
        self.delay_range = (1, 5)

    async def _add_delay(self):
        """Introduce a random delay to prevent bot detection."""
        delay = random.uniform(*self.delay_range)
        await asyncio.sleep(delay)

    async def _fetch_with_curl_cffi(
        self, url: str, headers: dict[str, Any] | None = None, timeout: int = 10
    ) -> str:
        """Fetch a URL using curl_cffi."""
        headers = headers or {"User-Agent": self.user_agent}
        await self._add_delay()  # Add delay before making the request
        async with AsyncSession() as session:
            response = await session.get(
                url,
                headers=headers,
                timeout=timeout,
                impersonate="chrome",
            )
            self.status_code = response.status_code
            return response.text

    async def _fetch_with_httpx(
        self, url: str, headers: dict[str, Any] | None = None, timeout: int = 10
    ) -> str:
        """Fetch a URL using httpx."""
        headers = headers or {"User-Agent": self.user_agent}
        await self._add_delay()  # Add delay before making the request
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            self.status_code = response.status_code
            return response.text

    async def _fetch_with_aiohttp(
        self, url: str, headers: dict[str, Any] | None = None, timeout: int = 10
    ):
        """Fetch a URL using aiohttp."""
        headers = headers or {"User-Agent": self.user_agent}
        await self._add_delay()
        async with ClientSession(headers=headers) as session:
            async with session.get(url, timeout=timeout) as response:
                if response.status != 200:
                    raise RuntimeError(f"Failed with status code {response.status}")
                self.status_code = response.status
                return await response.text()

    async def _fetch_with_cloudscraper(
        self, url: str, headers: dict[str, Any] | None = None
    ):
        """Fetch a URL using Cloudscraper."""
        headers = headers or {"User-Agent": self.user_agent}
        scraper: cloudscraper.CloudScraper = cloudscraper.create_scraper()  # pyright: ignore[reportUnknownMemberType]

        def fetch_in_thread():
            """Run Cloudscraper in a separate thread."""
            response = scraper.get(url, headers=headers)
            response.raise_for_status()
            self.status_code = response.status_code
            return response.text

        await self._add_delay()
        return await asyncio.to_thread(fetch_in_thread)

    async def fetch(self, url: str) -> str:
        """Fetch a URL using the available strategies asynchronously."""
        for strategy in self.strategies:
            try:
                # Sequentially try each strategy until one succeeds, or raise an exception if all strategies fail
                if strategy == "curl_cffi":
                    content = await self._fetch_with_curl_cffi(url)
                elif strategy == "aiohttp":
                    content = await self._fetch_with_aiohttp(url)
                elif strategy == "cloudscraper":
                    content = await self._fetch_with_cloudscraper(url)
                elif strategy == "httpx":
                    content = await self._fetch_with_httpx(url)
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")

                if self.status_code == 200:
                    return content
                else:
                    logger.info(
                        f"Strategy {strategy} failed with status code {self.status_code}. Switching to next strategy."
                    )
            except Exception as e:
                logger.info(
                    f"Strategy {strategy} encountered an error: {e}. Switching to next strategy."
                )
        self.status_code = 404
        raise RuntimeError(f"All strategies failed to fetch the URL: {url}")
