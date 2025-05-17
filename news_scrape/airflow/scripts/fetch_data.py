from producer.coin_price_producer import fetch_coin
from producer.ner_producer import process_mentions
from producer.rss_producer import stream_data


async def fetch_new_data():
    """
    Fetch data from various sources asynchronously.
    """
    # Fetch data from the RSS feed
    await stream_data()


async def fetch_price_coin():
    """
    Fetch coin data asynchronously.
    """

    await fetch_coin()


async def process_mentions_data(
    execution_date, start_time_delta: int = 12, end_time_delta: int = 6
):
    """
    Process mentions data asynchronously.
    """
    res = await process_mentions(execution_date, start_time_delta, end_time_delta)
    return res


if __name__ == "__main__":
    import asyncio

    asyncio.run(fetch_new_data())
