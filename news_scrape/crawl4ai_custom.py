import asyncio

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig


async def extract_structured_data_using_css_extractor(url):
    browser_config = BrowserConfig(headless=False, java_script_enabled=True)

    js_scroll_to_bottom = """
(async () => {
    let lastHeight = 0;
    while (true) {
        window.scrollBy(0, 500);  // scroll xuống thêm 500px
        await new Promise(r => setTimeout(r, 800));  // đợi nội dung tải
        
        let newHeight = document.body.scrollHeight;
        if (newHeight === lastHeight) break;
        lastHeight = newHeight;
    }
})();
"""

    crawler_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        scan_full_page=True,
        scroll_delay=3,
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url=url, config=crawler_config)
        with open("result.txt", "w") as f:
            f.write(str(result))
        print(result)


async def main(url):
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url)
        print(result.markdown)


if __name__ == "__main__":
    url = "https://cointelegraph.com/"
    asyncio.run(extract_structured_data_using_css_extractor(url))
