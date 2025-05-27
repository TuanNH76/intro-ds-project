import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta

from coins import COIN_SYMBOLS
from utils.mongodb import load_collection_to_dataframe, save_dataframe_to_collection

# Force reload environment variables
load_dotenv(override=True)

def get_ohlcv_binance(coin, target_datetime):
    """Get OHLCV data from Binance API"""
    symbol_mapping = {
        'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT', 'BNB': 'BNBUSDT',
        'XRP': 'XRPUSDT', 'ADA': 'ADAUSDT', 'SOL': 'SOLUSDT',
        'DOT': 'DOTUSDT', 'DOGE': 'DOGEUSDT', 'AVAX': 'AVAXUSDT',
        'MATIC': 'MATICUSDT', 'LTC': 'LTCUSDT', 'LINK': 'LINKUSDT',
        'UNI': 'UNIUSDT', 'ATOM': 'ATOMUSDT', 'XLM': 'XLMUSDT'
    }
    
    if coin not in symbol_mapping:
        return None
    
    symbol = symbol_mapping[coin]
    url = "https://api.binance.com/api/v3/klines"
    
    start_time = int(target_datetime.timestamp() * 1000)
    end_time = int((target_datetime + timedelta(hours=1)).timestamp() * 1000)
    
    params = {
        'symbol': symbol,
        'interval': '1h',
        'startTime': start_time,
        'endTime': end_time,
        'limit': 1
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data and len(data) > 0:
            kline = data[0]
            return {
                'time': int(target_datetime.timestamp()),
                'open': float(kline[1]),
                'high': float(kline[2]),
                'low': float(kline[3]),
                'close': float(kline[4]),
                'volumefrom': float(kline[5]),
                'volumeto': float(kline[7]),
                'conversionType': 'direct',
                'conversionSymbol': ''
            }
        return None
    except Exception as e:
        return None

def get_ohlcv_coinbase(coin, target_datetime):
    """Get OHLCV data from Coinbase Pro API"""
    product_mapping = {
        'BTC': 'BTC-USD', 'ETH': 'ETH-USD', 'LTC': 'LTC-USD',
        'LINK': 'LINK-USD', 'ADA': 'ADA-USD', 'DOT': 'DOT-USD',
        'UNI': 'UNI-USD', 'ATOM': 'ATOM-USD', 'XLM': 'XLM-USD',
        'DOGE': 'DOGE-USD', 'SOL': 'SOL-USD', 'AVAX': 'AVAX-USD',
        'MATIC': 'MATIC-USD'
    }
    
    if coin not in product_mapping:
        return None
    
    product_id = product_mapping[coin]
    url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"
    
    start_time = target_datetime.isoformat()
    end_time = (target_datetime + timedelta(hours=1)).isoformat()
    
    params = {
        'start': start_time,
        'end': end_time,
        'granularity': 3600
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data and len(data) > 0:
            candle = data[0]
            return {
                'time': int(target_datetime.timestamp()),
                'open': float(candle[3]),
                'high': float(candle[2]),
                'low': float(candle[1]),
                'close': float(candle[4]),
                'volumefrom': float(candle[5]),
                'volumeto': float(candle[5]) * float(candle[4]),
                'conversionType': 'direct',
                'conversionSymbol': ''
            }
        return None
    except Exception as e:
        return None

def get_ohlcv_coingecko(coin, target_datetime):
    """Get OHLCV data from CoinGecko API"""
    coin_mapping = {
        'BTC': 'bitcoin', 'ETH': 'ethereum', 'BNB': 'binancecoin',
        'XRP': 'ripple', 'ADA': 'cardano', 'SOL': 'solana',
        'DOT': 'polkadot', 'DOGE': 'dogecoin', 'AVAX': 'avalanche-2',
        'MATIC': 'polygon', 'LTC': 'litecoin', 'LINK': 'chainlink',
        'UNI': 'uniswap', 'ATOM': 'cosmos', 'XLM': 'stellar'
    }
    
    if coin not in coin_mapping:
        return None
    
    coin_id = coin_mapping[coin]
    target_date = target_datetime.date()
    from_ts = int(datetime.combine(target_date, datetime.min.time()).timestamp())
    to_ts = int(datetime.combine(target_date, datetime.max.time()).timestamp())
    
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range"
    params = {
        'vs_currency': 'usd',
        'from': from_ts,
        'to': to_ts
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        prices = data.get('prices', [])
        volumes = data.get('total_volumes', [])
        
        if not prices:
            return None
        
        target_ts = int(target_datetime.timestamp() * 1000)
        closest_price = None
        closest_volume = 0
        
        for price_point in prices:
            if abs(price_point[0] - target_ts) < 3600000:
                closest_price = price_point[1]
                break
        
        for volume_point in volumes:
            if abs(volume_point[0] - target_ts) < 3600000:
                closest_volume = volume_point[1]
                break
        
        if closest_price:
            return {
                'time': int(target_datetime.timestamp()),
                'open': closest_price,
                'high': closest_price,
                'low': closest_price,
                'close': closest_price,
                'volumefrom': closest_volume,
                'volumeto': closest_volume,
                'conversionType': 'direct',
                'conversionSymbol': ''
            }
        return None
    except Exception as e:
        return None

def get_ohlcv_multi_source(coin, target_datetime):
    """Try multiple data sources until one succeeds"""
    # Try Binance first (most reliable)
    data = get_ohlcv_binance(coin, target_datetime)
    if data:
        return data, "Binance"
    
    # Try Coinbase second
    data = get_ohlcv_coinbase(coin, target_datetime)
    if data:
        return data, "Coinbase"
    
    # Try CoinGecko last
    data = get_ohlcv_coingecko(coin, target_datetime)
    if data:
        return data, "CoinGecko"
    
    return None, "Failed"

def find_missing_hours(df):
    """Find missing hours in a sorted datetime dataframe"""
    if df.empty:
        return []
    
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Create expected hourly sequence
    start_time = df['datetime'].min().replace(minute=0, second=0, microsecond=0)
    end_time = df['datetime'].max().replace(minute=0, second=0, microsecond=0)
    
    expected_hours = pd.date_range(start=start_time, end=end_time, freq='H')
    existing_hours = set(df['datetime'].dt.floor('H'))
    
    missing_hours = [dt for dt in expected_hours if dt not in existing_hours]
    return sorted(missing_hours)

def backfill_coin_data(coin):
    """Backfill missing hours for a specific coin"""
    collection_name = f"{coin}_raw"
    
    print(f"\n🪙 Processing {coin}...")
    
    # Load existing data
    df = load_collection_to_dataframe(collection_name)
    
    if df.empty:
        print(f"   ❌ No data found for {coin}")
        return 0, 0
    
    # Convert datetime and sort
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    print(f"   📊 Existing records: {len(df):,}")
    print(f"   📅 Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    # Find missing hours
    missing_hours = find_missing_hours(df)
    
    if not missing_hours:
        print(f"   ✅ No missing hours found!")
        return 0, 0
    
    print(f"   ⚠️  Found {len(missing_hours)} missing hours")
    
    # Show sample of missing hours
    sample_size = min(5, len(missing_hours))
    print(f"   📋 Sample missing hours: {[dt.strftime('%Y-%m-%d %H:%M') for dt in missing_hours[:sample_size]]}")
    if len(missing_hours) > sample_size:
        print(f"        ... and {len(missing_hours) - sample_size} more")
    
    # Backfill missing hours
    backfilled_data = []
    success_count = 0
    
    for i, missing_hour in enumerate(missing_hours):
        print(f"   🕐 [{i+1:3d}/{len(missing_hours)}] {missing_hour.strftime('%Y-%m-%d %H:%M')} ", end="")
        
        # Try to get data for this hour
        data, source = get_ohlcv_multi_source(coin, missing_hour)
        
        if data:
            backfilled_data.append(data)
            success_count += 1
            print(f"✅ {source}")
        else:
            print("❌ All sources failed")
        
        # Rate limiting
        time.sleep(0.3)
        
        # Progress update every 20 requests
        if (i + 1) % 20 == 0:
            print(f"   📈 Progress: {i+1}/{len(missing_hours)} ({success_count} successful)")
    
    # Save backfilled data
    if backfilled_data:
        df_backfill = pd.DataFrame(backfilled_data)
        df_backfill['datetime'] = df_backfill['time'].apply(lambda ts: datetime.utcfromtimestamp(ts))
        df_backfill['coin'] = coin
        
        if '_id' in df_backfill.columns:
            df_backfill = df_backfill.drop(columns=['_id'])
        
        df_backfill = df_backfill.drop_duplicates(subset=['datetime', 'coin'])
        
        print(f"   💾 Saving {len(df_backfill)} backfilled records...")
        save_dataframe_to_collection(df_backfill, collection_name)
        print(f"   ✅ {coin} backfill completed!")
    
    print(f"   📊 Results: {success_count}/{len(missing_hours)} hours recovered ({success_count/len(missing_hours)*100:.1f}%)")
    
    return success_count, len(missing_hours)

def verify_completeness(coin):
    """Verify data completeness after backfill"""
    collection_name = f"{coin}_raw"
    df = load_collection_to_dataframe(collection_name)
    
    if df.empty:
        return 0
    
    df['datetime'] = pd.to_datetime(df['datetime'])
    missing_hours = find_missing_hours(df)
    
    return len(missing_hours)

def main():
    """Main backfill function"""
    print("🚀 Comprehensive Crypto Data Backfill")
    print("=" * 60)
    print("📡 Using: Binance → Coinbase → CoinGecko APIs")
    print(f"🪙 Processing {len(COIN_SYMBOLS)} coins")
    print()
    
    total_success = 0
    total_attempts = 0
    results = []
    
    for coin_idx, coin in enumerate(COIN_SYMBOLS):
        print(f"[{coin_idx + 1:2d}/{len(COIN_SYMBOLS)}]", end=" ")
        
        try:
            success, attempts = backfill_coin_data(coin)
            total_success += success
            total_attempts += attempts
            
            # Verify completeness
            remaining_gaps = verify_completeness(coin)
            
            results.append({
                'coin': coin,
                'attempted': attempts,
                'recovered': success,
                'remaining_gaps': remaining_gaps,
                'success_rate': success/attempts*100 if attempts > 0 else 100
            })
            
            print(f"   🎯 Final: {remaining_gaps} gaps remaining")
            
        except Exception as e:
            print(f"   ❌ Error processing {coin}: {str(e)}")
            results.append({
                'coin': coin,
                'attempted': 0,
                'recovered': 0,
                'remaining_gaps': -1,
                'success_rate': 0
            })
        
        # Pause between coins
        if coin_idx < len(COIN_SYMBOLS) - 1:
            print(f"   ⏸️ Pausing 2 seconds...")
            time.sleep(2)
    
    # Final summary
    print(f"\n{'='*60}")
    print("📋 FINAL SUMMARY")
    print(f"{'='*60}")
    
    print(f"🌐 Overall Results:")
    print(f"   • Total missing hours found: {total_attempts:,}")
    print(f"   • Successfully recovered: {total_success:,}")
    print(f"   • Overall success rate: {total_success/total_attempts*100:.1f}%" if total_attempts > 0 else "   • No missing hours found")
    
    print(f"\n🏆 Results by Coin:")
    print(f"{'Coin':>4} {'Attempted':>9} {'Recovered':>9} {'Remaining':>9} {'Success%':>8}")
    print("-" * 50)
    
    for result in results:
        if result['attempted'] > 0:
            print(f"{result['coin']:>4} {result['attempted']:>9} {result['recovered']:>9} "
                  f"{result['remaining_gaps']:>9} {result['success_rate']:>7.1f}%")
        else:
            status = "Complete" if result['remaining_gaps'] == 0 else "Error"
            print(f"{result['coin']:>4} {status:>9}")
    
    # Recommendations
    perfect_coins = [r for r in results if r['remaining_gaps'] == 0]
    problematic_coins = [r for r in results if r['remaining_gaps'] > 0]
    
    print(f"\n💡 Recommendations:")
    print(f"   ✅ {len(perfect_coins)} coins have complete data")
    
    if problematic_coins:
        print(f"   ⚠️  {len(problematic_coins)} coins still have gaps:")
        for coin_result in problematic_coins[:5]:  # Show top 5
            print(f"      • {coin_result['coin']}: {coin_result['remaining_gaps']} missing hours")
        print(f"   💡 Consider:")
        print(f"      1. Re-running this script (some API failures may be temporary)")
        print(f"      2. Using interpolation for remaining gaps")
        print(f"      3. Accepting gaps if they represent market downtime")
    
    print(f"\n🎉 Backfill process completed!")

if __name__ == "__main__":
    main()