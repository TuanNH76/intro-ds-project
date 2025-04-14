import time
import requests
from datetime import datetime

coin_id = "bitcoin"
vs_currency = "usd"

from_date = int(time.mktime(datetime(2021, 4, 14).timetuple()))
to_date = int(time.mktime(datetime(2022, 4, 10).timetuple()))

url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range"
params = {
    "vs_currency": vs_currency,
    "from": from_date,
    "to": to_date
}

response = requests.get(url, params=params)
data = response.json()

print(f"Status Code: {response.status_code}")
if "prices" in data:
    print(data["prices"][:5])
else:
    print("❌ No 'prices' found. Response:")
    print(data)
