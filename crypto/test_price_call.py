import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv('COINRANKING_API_KEY')

HEADERS = {
    'x-access-token': API_KEY
}

# Base URLs
STATS_URL = 'https://api.coinranking.com/v2/stats/coins'
HISTORY_BASE_URL = 'https://api.coinranking.com/v2/coin/{uuid}/history'

# Bitcoin UUID and coin name
btc_uuid = 'Qwsogvtv82FCd'
coin_name = 'Bitcoin'

# Parameters for 5-year time series
PARAMS = {
    'timePeriod': '1y'
}

def get_coin_stats():
    try:
        response = requests.get(STATS_URL, headers=HEADERS)
        response.raise_for_status()
        data = response.json()
        print("Coin Stats:")
        print(data)
    except requests.exceptions.RequestException as e:
        print("Error fetching coin stats:", e)

def get_btc_history():
    try:
        url = HISTORY_BASE_URL.format(uuid=btc_uuid)
        response = requests.get(url, headers=HEADERS, params=PARAMS)
        response.raise_for_status()
        data = response.json()
        print(f"{coin_name} Historical Data (5 years):")
        print(data)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {coin_name} history:", e)

if __name__ == "__main__":
    get_coin_stats()
    get_btc_history()
