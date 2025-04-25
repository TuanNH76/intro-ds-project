import os
import time
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta

from coins import COIN_SYMBOLS
from utils.mongodb import insert_many_documents, get_latest_timestamp

load_dotenv()
API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY")
MONGO_DB = "mydatabase"  # or whatever your database name is

LIMIT = 2000  # max API hours per call

def get_ohlcv(coin, to_ts):
    url = "https://min-api.cryptocompare.com/data/v2/histohour"
    params = {
        "fsym": coin,
        "tsym": "USD",
        "limit": LIMIT - 1,
        "toTs": to_ts,
        "api_key": API_KEY
    }
    response = requests.get(url, params=params).json()
    return response.get("Data", {}).get("Data", [])

# Loop over all coins
for coin in COIN_SYMBOLS:
    collection_name = f"raw_{coin}"
    # Get last timestamp for this coin
    last_ts = get_latest_timestamp(MONGO_DB, collection_name)
    print(f"Last timestamp for {coin}: {last_ts}")
    START_DATE = last_ts + timedelta(hours=1) if last_ts else datetime(2020, 1, 1)
    END_DATE = datetime.utcnow()

    total_hours = int((END_DATE - START_DATE).total_seconds() // 3600)
    num_chunks = (total_hours + LIMIT - 1) // LIMIT

    print(f"\n🪙 Updating {coin} from {START_DATE} to {END_DATE} ({total_hours} hours → {num_chunks} chunks)")

    for chunk_id in range(num_chunks):
        chunk_start = START_DATE + timedelta(hours=chunk_id * LIMIT)
        chunk_end = chunk_start + timedelta(hours=LIMIT)

        if chunk_end > END_DATE:
            chunk_end = END_DATE

        print(f"🧩 Chunk {chunk_id + 1}: {chunk_start} → {chunk_end}")
        to_ts = int((chunk_start + timedelta(hours=LIMIT - 1)).timestamp())
        data = get_ohlcv(coin, to_ts)
        documents = []
        for entry in data:
            ts = entry["time"]
            dt = datetime.utcfromtimestamp(ts)
            if dt < chunk_start or dt > chunk_end:
                continue

            entry["coin"] = coin
            entry["datetime"] = dt
            entry.pop("_id", None)  # remove _id to avoid dup error
            documents.append(entry)
        
        if documents:
            insert_many_documents(collection_name, documents)
            print(f"✅ {coin} inserted {len(documents)} docs")

        time.sleep(0.2)  # prevent rate limit
