import requests
import os
import pandas as pd
from dotenv import load_dotenv

# Load API key
load_dotenv()
API_KEY = os.getenv('COINRANKING_API_KEY')

# Get top 100 coins
url = 'https://api.coinranking.com/v2/coins'
headers = {'x-access-token': API_KEY}
params = {'limit': 100, 'orderBy': 'marketCap'}

response = requests.get(url, headers=headers, params=params)

if response.status_code == 200:
    coins = response.json()['data']['coins']
    df_coin_list = pd.DataFrame([{
        'rank': coin['rank'],
        'name': coin['name'],
        'symbol': coin['symbol'],
        'uuid': coin['uuid']
    } for coin in coins])
    
    df_coin_list.to_csv('coin_list.csv', index=False)
    print("Saved top 100 coins to coin_list.csv")
else:
    print(f"Error: {response.status_code}, {response.text}")
