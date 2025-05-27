import pandas as pd
from datetime import datetime
from coins import COIN_SYMBOLS
from utils.mongodb import (
    load_collection_to_dataframe, 
    save_dataframe_to_collection,
    get_collection
)

def integrate_coin_data(coin_symbol):
    """
    Integrate news and features data for a specific coin based on datetime overlap
    """
    news_collection = f"{coin_symbol}_news"
    features_collection = f"{coin_symbol}_features"
    integrated_collection = f"{coin_symbol}_all"
    
    print(f"\n🔗 Integrating data for {coin_symbol}...")
    
    # Load news data
    print(f"📰 Loading news data from {news_collection}...")
    news_df = load_collection_to_dataframe(news_collection)
    
    if news_df.empty:
        print(f"❌ No news data found for {coin_symbol}")
        return
    
    # Load features data
    print(f"📊 Loading features data from {features_collection}...")
    features_df = load_collection_to_dataframe(features_collection)
    
    if features_df.empty:
        print(f"❌ No features data found for {coin_symbol}")
        return
    
    # Convert datetime columns to pandas datetime
    news_df['datetime'] = pd.to_datetime(news_df['datetime'])
    features_df['datetime'] = pd.to_datetime(features_df['datetime'])
    
    # Sort by datetime
    news_df = news_df.sort_values('datetime').reset_index(drop=True)
    features_df = features_df.sort_values('datetime').reset_index(drop=True)
    
    print(f"📅 News data range: {news_df['datetime'].min()} to {news_df['datetime'].max()}")
    print(f"📅 Features data range: {features_df['datetime'].min()} to {features_df['datetime'].max()}")
    
    # Find overlapping time period
    overlap_start = max(news_df['datetime'].min(), features_df['datetime'].min())
    overlap_end = min(news_df['datetime'].max(), features_df['datetime'].max())
    
    print(f"🔄 Overlapping period: {overlap_start} to {overlap_end}")
    
    # Filter both datasets to overlap period only
    news_overlap = news_df[
        (news_df['datetime'] >= overlap_start) & 
        (news_df['datetime'] <= overlap_end)
    ].copy()
    
    features_overlap = features_df[
        (features_df['datetime'] >= overlap_start) & 
        (features_df['datetime'] <= overlap_end)
    ].copy()
    
    print(f"📰 News records in overlap: {len(news_overlap)}")
    print(f"📊 Features records in overlap: {len(features_overlap)}")
    
    if news_overlap.empty or features_overlap.empty:
        print(f"❌ No overlapping data found for {coin_symbol}")
        return
    
    # Remove unnecessary columns
    news_columns_to_remove = ['_id']
    features_columns_to_remove = ['_id', 'time']  # 'time' is unix timestamp, we have datetime
    
    news_clean = news_overlap.drop(columns=[col for col in news_columns_to_remove if col in news_overlap.columns])
    features_clean = features_overlap.drop(columns=[col for col in features_columns_to_remove if col in features_overlap.columns])
    
    # Merge on datetime (inner join to keep only matching timestamps)
    print("🔗 Merging data on datetime...")
    integrated_df = pd.merge(
        features_clean, 
        news_clean, 
        on='datetime', 
        how='inner',
        suffixes=('_features', '_news')
    )
    
    # Handle duplicate coin columns if they exist
    if 'coin_features' in integrated_df.columns and 'coin_news' in integrated_df.columns:
        # Keep one coin column and drop the other
        integrated_df['coin'] = integrated_df['coin_features']
        integrated_df = integrated_df.drop(columns=['coin_features', 'coin_news'])
    elif 'coin_features' in integrated_df.columns:
        integrated_df['coin'] = integrated_df['coin_features']
        integrated_df = integrated_df.drop(columns=['coin_features'])
    elif 'coin_news' in integrated_df.columns:
        integrated_df['coin'] = integrated_df['coin_news']
        integrated_df = integrated_df.drop(columns=['coin_news'])
    
    print(f"✅ Successfully integrated {len(integrated_df)} records")
    
    if not integrated_df.empty:
        # Save to new collection
        print(f"💾 Saving integrated data to {integrated_collection}...")
        save_dataframe_to_collection(integrated_df, integrated_collection)
        
        print(f"🎉 Integration completed for {coin_symbol}!")
        print(f"📊 Final dataset summary:")
        print(f"   - Records: {len(integrated_df)}")
        print(f"   - Date range: {integrated_df['datetime'].min()} to {integrated_df['datetime'].max()}")
        print(f"   - Features: {len(integrated_df.columns) - 2}")  # -2 for datetime and coin
        print(f"   - News features: {len([col for col in integrated_df.columns if any(news_col in col for news_col in ['mentioned', 'momentum', 'relative_strength'])])}")
        print(f"   - Technical features: {len([col for col in integrated_df.columns if any(tech_col in col for tech_col in ['rsi', 'macd', 'bb_', 'ema_', 'sma_'])])}")
        
        # Display sample of integrated data
        print("\n📋 Sample of integrated data:")
        print(integrated_df[['datetime', 'coin', 'close_price', 'mentioned', 'rsi', 'momentum']].head())
        
    else:
        print(f"❌ No data to save for {coin_symbol}")

def integrate_all_coins():
    """
    Integrate data for all coins in COIN_SYMBOLS
    """
    print("🚀 Starting crypto data integration pipeline...")
    
    successful_integrations = 0
    failed_integrations = []
    
    for coin in COIN_SYMBOLS:
        try:
            integrate_coin_data(coin)
            successful_integrations += 1
        except Exception as e:
            print(f"❌ Error integrating {coin}: {str(e)}")
            failed_integrations.append(coin)
            continue
    
    print(f"\n🎉 Data integration pipeline completed!")
    print(f"✅ Successfully integrated: {successful_integrations} coins")
    if failed_integrations:
        print(f"❌ Failed integrations: {failed_integrations}")

def check_integration_status():
    """
    Check the status of integrated collections
    """
    print("🔍 Checking integration status...")
    
    for coin in COIN_SYMBOLS:
        try:
            integrated_collection = f"{coin}_all"
            df = load_collection_to_dataframe(integrated_collection)
            
            if not df.empty:
                df['datetime'] = pd.to_datetime(df['datetime'])
                print(f"✅ {coin}_all: {len(df)} records, {df['datetime'].min()} to {df['datetime'].max()}")
            else:
                print(f"❌ {coin}_all: No data")
                
        except Exception as e:
            print(f"❌ {coin}_all: Error - {str(e)}")

if __name__ == "__main__":
    # You can run individual functions or all of them
    
    # Integrate all coins
    integrate_all_coins()
    
    # Check status after integration
    print("\n" + "="*50)
    check_integration_status()
    
    # Or integrate a specific coin only:
    # integrate_coin_data("BTC")