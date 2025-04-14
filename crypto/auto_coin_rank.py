import pandas as pd
import requests
from dotenv import load_dotenv
import os
from tqdm import tqdm

# Load API key
load_dotenv()
API_KEY = os.getenv('COINRANKING_API_KEY')

HEADERS = {
    'x-access-token': API_KEY
}

HISTORY_BASE_URL = 'https://api.coinranking.com/v2/coin/{uuid}/history'
PARAMS = {
    'timePeriod': '5y'
}

# Read coin list
coin_df = pd.read_csv('coin_list.csv')
data_frames = {}

# Loop through each coin and get historical price
for _, row in tqdm(coin_df.iterrows(), total=len(coin_df), desc="Fetching histories"):
    uuid = row['uuid']
    name = row['name']
    
    url = HISTORY_BASE_URL.format(uuid=uuid)
    response = requests.get(url, headers=HEADERS, params=PARAMS)
    
    if response.status_code == 200:
        history = response.json()['data']['history']
        df = pd.DataFrame(history)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('timestamp', inplace=True)

        df = df[df.index.time == pd.Timestamp('00:00:00').time()] 
        df = df.rename(columns={'price': name})
        df[name] = df[name].astype(float)
        data_frames[name] = df[[name]]
    else:
        print(f"Failed for {name} ({uuid}) - {response.status_code}: {response.text}")

# Merge all histories into one CSV
combined_df = pd.concat(data_frames.values(), axis=1)
combined_df.sort_index(inplace=True)
combined_df.to_csv('crypto_price_history_top100.csv')
print("Saved to crypto_price_history_top100.csv")
