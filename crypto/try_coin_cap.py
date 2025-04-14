import requests
import time
import datetime
import pandas as pd
import os
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
API_KEY = os.getenv("COIN_CAP_API_KEY")

def to_milliseconds(dt):
    return int(time.mktime(dt.timetuple()) * 1000)

# Set date range: 1 year
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=365)

start_ms = to_milliseconds(start_date)
end_ms = to_milliseconds(end_date)

# API call
url = "https://api.coincap.io/v2/assets/bitcoin/history"
params = {
    "interval": "d1",
    "start": start_ms,
    "end": end_ms
}

headers = {}
if API_KEY:
    headers["Authorization"] = f"Bearer {API_KEY}"

response = requests.get(url, params=params, headers=headers)

# Check for success
if response.status_code != 200:
    print("Error:", response.status_code, response.text)
    exit()

json_data = response.json()

# Validate the structure
if "data" not in json_data or not isinstance(json_data["data"], list):
    print("Invalid data format:", json_data)
    exit()

# Convert to DataFrame
df = pd.DataFrame(json_data["data"])

# Convert numeric fields
for col in df.columns:
    if col != "date":
        df[col] = pd.to_numeric(df[col], errors="coerce")

df["date"] = pd.to_datetime(df["date"])

print(df.head())

# Save to CSV
df.to_csv("bitcoin_coincap_1year_all_fields.csv", index=False)
