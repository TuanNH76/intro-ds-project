import requests
import os
import csv
from dotenv import load_dotenv
from datetime import datetime

# Load API key
load_dotenv()
API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY")

coins = ["BTC", "ETH", "BNB", "XRP", "ADA", "SOL", "DOT", "DOGE", "AVAX", "MATIC", "LTC", "UNI", "LINK", "ATOM", "XLM"]

for coin in coins:
    print(f"Fetching: {coin}")

    url = "https://min-api.cryptocompare.com/data/v2/histoday"
    params = {
        "fsym": coin,
        "tsym": "USD",
        "limit": 1460,  # 4 years of daily data
        "api_key": API_KEY
    }

    response = requests.get(url, params=params)
    data = response.json()

    if data["Response"] != "Success":
        print(f"❌ Failed for {coin}: {data.get('Message')}")
        continue

    entries = data["Data"]["Data"]
    filename = f"{coin}_price_volume_4y.csv"

    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Date", "Close Price (USD)", "Volume To (USD)"])
        for entry in entries:
            date = datetime.utcfromtimestamp(entry["time"]).strftime('%Y-%m-%d')
            close_price = entry["close"]
            volume = entry["volumeto"]
            writer.writerow([date, close_price, volume])

    print(f"✅ Saved {len(entries)} rows to {filename}")
