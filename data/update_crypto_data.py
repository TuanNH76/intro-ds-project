import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta

from coins import COIN_SYMBOLS
from utils.mongodb import get_latest_timestamp, save_dataframe_to_collection

# Force reload environment variables
load_dotenv(override=True)
API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY")

LIMIT = 2000  # max API hours per call

def get_ohlcv(coin, to_ts):
    """Get OHLCV data from CryptoCompare API"""
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
    collection_name = f"{coin}_raw"  # Using {coin}_raw format
    
    # Get last timestamp for this coin
    last_ts = get_latest_timestamp(collection_name)
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
        
        # Convert to DataFrame
        if data:
            df = pd.DataFrame(data)
            
            # Convert timestamp to datetime
            df['datetime'] = df['time'].apply(lambda ts: datetime.utcfromtimestamp(ts))
            
            # Filter records within the chunk timeframe
            df = df[(df['datetime'] >= chunk_start) & (df['datetime'] <= chunk_end)]
            
            # Add coin symbol
            df['coin'] = coin
            
            # Drop _id if exists
            if '_id' in df.columns:
                df = df.drop(columns=['_id'])
            
            # Use save_dataframe_to_collection which handles creating timeseries collection
            if not df.empty:
                save_dataframe_to_collection(df, collection_name)
                print(f"✅ {coin} inserted {len(df)} docs")
        
        time.sleep(0.2)  # prevent rate limit