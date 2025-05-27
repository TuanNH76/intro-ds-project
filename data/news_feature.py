import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from coins import COIN_SYMBOLS
from utils.mongodb import (
    get_latest_timestamp, 
    load_collection_to_dataframe, 
    save_dataframe_to_collection
)

def calculate_news_indicators(df, coin_symbol):
    """Calculate news indicators for a specific coin"""
    df = df.copy()
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Get the mentioned count for this specific coin
    mentioned_col = f'mentioned_frequency.{coin_symbol.lower()}'
    
    if mentioned_col not in df.columns:
        print(f"Warning: {mentioned_col} not found in data")
        return pd.DataFrame()
    
    # Create base dataframe
    result_df = pd.DataFrame({
        'datetime': df['datetime'],
        'coin': coin_symbol.upper(),
        'mentioned': df[mentioned_col]
    })
    
    # Rolling averages
    result_df['mentioned_6h'] = result_df['mentioned'].rolling(6, min_periods=1).mean()
    result_df['mentioned_24h'] = result_df['mentioned'].rolling(24, min_periods=1).mean()
    
    # Momentum indicators
    result_df['momentum'] = result_df['mentioned'].pct_change().fillna(0)
    result_df['momentum_6h'] = (result_df['mentioned'] / result_df['mentioned'].shift(6) - 1).fillna(0)
    
    # Volatility and relative strength
    result_df['volatility_6h'] = result_df['mentioned'].rolling(6, min_periods=1).std().fillna(0)
    result_df['relative_strength'] = result_df['mentioned'] / result_df['mentioned'].mean()
    
    return result_df

def process_news_features():
    """Process news feature engineering for all coins"""
    
    print("🚀 Starting news feature engineering pipeline...")
    
    # Load frequency data
    print("📥 Loading frequency collection data...")
    frequency_df = load_collection_to_dataframe("frequency")
    
    if frequency_df.empty:
        print("❌ No frequency data found")
        return
    
    # Convert hour field to datetime and rename
    if 'hour' in frequency_df.columns:
        frequency_df['datetime'] = pd.to_datetime(frequency_df['hour'])
        frequency_df = frequency_df.drop('hour', axis=1)
    
    # Flatten the mentioned_frequency nested structure
    if 'mentioned_frequency' in frequency_df.columns:
        # Extract nested mentioned_frequency data
        mentioned_data = pd.json_normalize(frequency_df['mentioned_frequency'])
        # Add prefix to column names
        mentioned_data.columns = ['mentioned_frequency.' + col for col in mentioned_data.columns]
        # Combine with datetime
        frequency_df = pd.concat([frequency_df[['datetime']], mentioned_data], axis=1)
    
    # Sort by datetime
    frequency_df = frequency_df.sort_values('datetime').reset_index(drop=True)
    
    print(f"📊 Loaded {len(frequency_df)} frequency records")
    print(f"📅 Date range: {frequency_df['datetime'].min()} to {frequency_df['datetime'].max()}")
    
    # Create complete hourly timeline to fill gaps
    start_time = frequency_df['datetime'].min()
    end_time = frequency_df['datetime'].max()
    
    # Generate complete hourly range
    complete_timeline = pd.date_range(start=start_time, end=end_time, freq='h')
    
    # Create complete dataframe with all hours
    complete_df = pd.DataFrame({'datetime': complete_timeline})
    
    # Merge with existing data and fill missing values with 0
    frequency_df = complete_df.merge(frequency_df, on='datetime', how='left')
    
    # Fill missing mention counts with 0 for all coins
    mention_cols = [col for col in frequency_df.columns if col.startswith('mentioned_frequency.')]
    frequency_df[mention_cols] = frequency_df[mention_cols].fillna(0)
    
    print(f"📈 Extended to complete hourly timeline: {len(frequency_df)} records")
    
    # Process each coin
    for coin in COIN_SYMBOLS:
        try:
            print(f"\n🪙 Processing news features for {coin}...")
            
            news_collection = f"{coin}_news"
            
            # Get the latest timestamp from news collection to avoid reprocessing
            last_news_timestamp = get_latest_timestamp(news_collection)
            
            # Filter data to process
            process_df = frequency_df.copy()
            
            if last_news_timestamp:
                print(f"📅 Last news timestamp: {last_news_timestamp}")
                # Process data from 24 hours before last timestamp to ensure context
                start_from = last_news_timestamp - timedelta(hours=24)
                process_df = process_df[process_df['datetime'] >= start_from]
                print(f"🔄 Processing {len(process_df)} records from {start_from}")
            else:
                print(f"🆕 First time processing - processing all {len(process_df)} records")
            
            if process_df.empty:
                print(f"✅ {coin} news features are up to date")
                continue
            
            # Calculate news indicators for this coin
            news_df = calculate_news_indicators(process_df, coin)
            
            if news_df.empty:
                print(f"❌ No news data generated for {coin}")
                continue
            
            # Remove NaN values
            initial_rows = len(news_df)
            news_df = news_df.dropna()
            removed_rows = initial_rows - len(news_df)
            if removed_rows > 0:
                print(f"🧹 Removed {removed_rows} rows with NaN values")
            
            # If we have a last timestamp, only save new records
            if last_news_timestamp:
                news_df = news_df[news_df['datetime'] > last_news_timestamp]
                print(f"💾 Saving {len(news_df)} new records to {news_collection}")
            else:
                print(f"💾 Saving {len(news_df)} records to {news_collection}")
            
            if not news_df.empty:
                # Format datetime to match your specified format
                news_df['datetime'] = news_df['datetime'].dt.strftime('%Y-%m-%dT%H:%M:%S.000+00:00')
                news_df['datetime'] = pd.to_datetime(news_df['datetime'])
                
                # Save to MongoDB
                save_dataframe_to_collection(news_df, news_collection)
                print(f"✅ {coin} news feature engineering completed!")
                
                # Display summary
                print(f"📊 News feature summary for {coin}:")
                print(f"   - Date range: {news_df['datetime'].min()} to {news_df['datetime'].max()}")
                print(f"   - Records: {len(news_df)}")
                print(f"   - Average mentions/hour: {news_df['mentioned'].mean():.2f}")
                print(f"   - Max mentions/hour: {news_df['mentioned'].max()}")
                print(f"   - Features: mentioned, mentioned_6h, mentioned_24h, momentum, momentum_6h, volatility_6h, relative_strength")
            else:
                print(f"ℹ️ No new news data to process for {coin}")
                
        except Exception as e:
            print(f"❌ Error processing news features for {coin}: {str(e)}")
            continue
    
    print("\n🎉 News feature engineering pipeline completed!")

def get_news_feature_summary():
    """Get summary of news features across all coins"""
    print("\n📈 News Feature Summary Across All Coins:")
    print("=" * 50)
    
    for coin in COIN_SYMBOLS:
        try:
            news_collection = f"{coin}_news"
            news_df = load_collection_to_dataframe(news_collection)
            
            if not news_df.empty:
                news_df['datetime'] = pd.to_datetime(news_df['datetime'])
                print(f"\n{coin}:")
                print(f"  Records: {len(news_df)}")
                print(f"  Date range: {news_df['datetime'].min()} to {news_df['datetime'].max()}")
                print(f"  Avg mentions: {news_df['mentioned'].mean():.2f}")
                print(f"  Max mentions: {news_df['mentioned'].max()}")
                print(f"  Total mentions: {news_df['mentioned'].sum()}")
            else:
                print(f"\n{coin}: No data")
                
        except Exception as e:
            print(f"\n{coin}: Error - {str(e)}")

if __name__ == "__main__":
    # Run the news feature engineering
    process_news_features()
    
    # Show summary
    get_news_feature_summary()