import asyncio
import os
import sys
from datetime import datetime, timedelta

import httpx
import pandas as pd
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mongo.mongo_client import MongoService

_ = load_dotenv()
API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY")

LIMIT = 2000  # max hours per API call


COIN_SYMBOLS = [
    "BTC",
    "ETH",
    "BNB",
    "XRP",
    "ADA",
    "SOL",
    "DOT",
    "DOGE",
    "AVAX",
    "MATIC",
    "LTC",
    "UNI",
    "LINK",
    "ATOM",
    "XLM",
]

mongo_client = MongoService()


async def get_ohlcv(client: httpx.AsyncClient, coin: str, to_ts: int) -> list[dict]:
    """Async get OHLCV data from CryptoCompare"""
    url = "https://min-api.cryptocompare.com/data/v2/histohour"
    params = {
        "fsym": coin,
        "tsym": "USD",
        "limit": LIMIT - 1,
        "toTs": to_ts,
        "api_key": API_KEY,
    }
    try:
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get("Data", {}).get("Data", [])
    except Exception as e:
        print(f"❌ Error fetching {coin} @ {to_ts}: {e}")
        return []


async def update_coin(client: httpx.AsyncClient, coin: str):
    collection_name = f"{coin}_raw"
    last_ts = await mongo_client.get_latest_timestamp(collection_name)
    print(f"\n📥 Fetching {coin} | Last timestamp: {last_ts}")

    start_time = last_ts + timedelta(hours=1) if last_ts else datetime(2025, 5, 14)
    end_time = datetime.utcnow()

    total_hours = int((end_time - start_time).total_seconds() // 3600)
    num_chunks = (total_hours + LIMIT - 1) // LIMIT

    for chunk_id in range(num_chunks):
        chunk_start = start_time + timedelta(hours=chunk_id * LIMIT)
        chunk_end = min(chunk_start + timedelta(hours=LIMIT), end_time)
        to_ts = int((chunk_start + timedelta(hours=LIMIT - 1)).timestamp())

        print(
            f"🧩 [{coin}] Chunk {chunk_id + 1}/{num_chunks}: {chunk_start} → {chunk_end}"
        )
        data = await get_ohlcv(client, coin, to_ts)

        if data:
            df = pd.DataFrame(data)
            df["datetime"] = df["time"].apply(lambda ts: datetime.utcfromtimestamp(ts))
            df = df[(df["datetime"] >= chunk_start) & (df["datetime"] <= chunk_end)]
            df["coin"] = coin

            if "_id" in df.columns:
                df = df.drop(columns=["_id"])

            if not df.empty:
                _ = await mongo_client.save_dataframe_to_collection(df, collection_name)
                print(f"✅ Inserted {len(df)} records for {coin}")

        await asyncio.sleep(0.2)  # optional: rate-limit between chunks


async def fetch_coin():
    async with httpx.AsyncClient(timeout=15.0) as client:
        tasks = [update_coin(client, coin) for coin in COIN_SYMBOLS]
        _ = await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(fetch_coin())
