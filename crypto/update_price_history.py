import pandas as pd
import requests
from dotenv import load_dotenv
import os
from tqdm import tqdm
from datetime import datetime, timedelta

# Load API key
load_dotenv()
API_KEY = os.getenv('COINRANKING_API_KEY')

HEADERS = {
    'x-access-token': API_KEY
}

HISTORY_BASE_URL = 'https://api.coinranking.com/v2/coin/{uuid}/history'

# Load existing CSV
csv_file = 'crypto_price_history_top100.csv'
df_existing = pd.read_csv(csv_file, parse_dates=['timestamp'])
df_existing.set_index('timestamp', inplace=True)

# Determine the last date in the file and yesterday (UTC)
last_date = df_existing.index.max().normalize()
yesterday = datetime.utcnow().date() - timedelta(days=1)

# If already up-to-date
if last_date >= pd.Timestamp(yesterday):
    print("CSV is already up to date.")
    exit()

# Choose timePeriod based on gap size
delta_days = (yesterday - last_date.date()).days
if delta_days <= 1:
    time_period = '24h'
elif delta_days <= 7:
    time_period = '7d'
elif delta_days <= 30:
    time_period = '30d'
elif delta_days <= 90:
    time_period = '3m'
elif delta_days <= 365:
    time_period = '1y'
elif delta_days <= 1825:
    time_period = '5y'
else:
    time_period = 'all'

print(f"Last date: {last_date.date()}, Yesterday: {yesterday}, Fetching with timePeriod: {time_period}")

# Read coin list
coin_df = pd.read_csv('coin_list.csv')
new_data_frames = {}

# Loop through coins to get new data
for _, row in tqdm(coin_df.iterrows(), total=len(coin_df), desc="Updating coins"):
    uuid = row['uuid']
    name = row['name']

    url = HISTORY_BASE_URL.format(uuid=uuid)
    params = {'timePeriod': time_period}
    response = requests.get(url, headers=HEADERS, params=params)

    if response.status_code == 200:
        history = response.json()['data']['history']
        df = pd.DataFrame(history)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('timestamp', inplace=True)

        df = df[df.index.time == pd.Timestamp("00:00:00").time()]
        df.index = df.index.normalize()
        df = df.rename(columns={'price': name})
        df[name] = df[name].astype(float)
        new_data_frames[name] = df[[name]]
    else:
        print(f"Failed for {name} ({uuid}) - {response.status_code}: {response.text}")

# Merge new data and filter only newer rows
if not new_data_frames:
    print("No new data retrieved.")
    exit()

df_new = pd.concat(new_data_frames.values(), axis=1)
df_new.sort_index(inplace=True)

# Remove overlapping dates
df_new_filtered = df_new[df_new.index > last_date]

# Append and save
df_updated = pd.concat([df_existing, df_new_filtered])
df_updated.to_csv(csv_file)
print(f"Updated {csv_file} with {len(df_new_filtered)} new rows.")
