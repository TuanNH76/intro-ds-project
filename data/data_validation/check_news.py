import pandas as pd
from datetime import datetime, timedelta
from utils.mongodb import load_collection_to_dataframe

def check_btc_news_detailed():
    """Detailed analysis of BTC_news collection gaps"""
    print("🔍 Analyzing BTC_news collection in detail...")
    
    # Load BTC news data
    btc_df = load_collection_to_dataframe("BTC_news")
    
    if btc_df.empty:
        print("❌ No BTC_news data found")
        return
    
    # Convert datetime and sort
    btc_df['datetime'] = pd.to_datetime(btc_df['datetime'])
    btc_df = btc_df.sort_values('datetime').reset_index(drop=True)
    
    print(f"📊 Total BTC_news records: {len(btc_df)}")
    print(f"📅 Date range: {btc_df['datetime'].min()} to {btc_df['datetime'].max()}")
    
    # Calculate time differences
    btc_df['time_diff'] = btc_df['datetime'].diff()
    btc_df['gap_hours'] = btc_df['time_diff'].dt.total_seconds() / 3600
    
    # Find gaps longer than 1 hour
    gaps = btc_df[btc_df['gap_hours'] > 1].copy()
    
    print(f"\n⚠️ Found {len(gaps)} continuity gaps longer than 1 hour:")
    print("=" * 60)
    
    for idx, row in gaps.iterrows():
        gap_start = btc_df.loc[idx-1, 'datetime'] if idx > 0 else None
        gap_end = row['datetime']
        gap_hours = row['gap_hours']
        
        print(f"Gap #{idx}: {gap_hours:.1f} hours")
        print(f"   From: {gap_start}")
        print(f"   To:   {gap_end}")
        print(f"   Missing hours: {int(gap_hours - 1)}")
        print()
    
    return gaps

def analyze_gap_patterns():
    """Analyze patterns in the gaps"""
    print("🔍 Analyzing gap patterns...")
    
    btc_df = load_collection_to_dataframe("BTC_news")
    btc_df['datetime'] = pd.to_datetime(btc_df['datetime'])
    btc_df = btc_df.sort_values('datetime')
    
    # Calculate gaps
    btc_df['time_diff'] = btc_df['datetime'].diff()
    btc_df['gap_hours'] = btc_df['time_diff'].dt.total_seconds() / 3600
    
    gaps = btc_df[btc_df['gap_hours'] > 1].copy()
    
    if gaps.empty:
        print("✅ No gaps found")
        return
    
    # Add day of week and hour analysis
    gaps['day_of_week'] = gaps['datetime'].dt.day_name()
    gaps['hour_of_day'] = gaps['datetime'].dt.hour
    gaps['date'] = gaps['datetime'].dt.date
    
    print(f"\n📊 Gap Statistics:")
    print(f"   Total gaps: {len(gaps)}")
    print(f"   Average gap size: {gaps['gap_hours'].mean():.1f} hours")
    print(f"   Largest gap: {gaps['gap_hours'].max():.1f} hours")
    print(f"   Smallest gap: {gaps['gap_hours'].min():.1f} hours")
    
    print(f"\n📅 Gaps by day of week:")
    day_counts = gaps['day_of_week'].value_counts()
    for day, count in day_counts.items():
        print(f"   {day}: {count} gaps")
    
    print(f"\n⏰ Gaps by hour of day:")
    hour_counts = gaps['hour_of_day'].value_counts().sort_index()
    for hour, count in hour_counts.items():
        print(f"   {hour:02d}:00: {count} gaps")
    
    print(f"\n📆 Gaps by date:")
    date_counts = gaps['date'].value_counts().sort_index()
    for date, count in date_counts.items():
        print(f"   {date}: {count} gaps")

def find_missing_hours_in_gaps():
    """Generate list of all missing hours within gaps"""
    print("\n🔍 Generating complete list of missing hours...")
    
    btc_df = load_collection_to_dataframe("BTC_news")
    btc_df['datetime'] = pd.to_datetime(btc_df['datetime'])
    btc_df = btc_df.sort_values('datetime')
    
    # Get overall date range
    start_time = btc_df['datetime'].min()
    end_time = btc_df['datetime'].max()
    
    # Generate complete hourly timeline
    complete_timeline = pd.date_range(start=start_time, end=end_time, freq='h')
    
    # Find missing hours
    existing_hours = set(btc_df['datetime'])
    missing_hours = [hour for hour in complete_timeline if hour not in existing_hours]
    
    print(f"📊 Timeline Analysis:")
    print(f"   Expected hours: {len(complete_timeline)}")
    print(f"   Existing hours: {len(existing_hours)}")
    print(f"   Missing hours: {len(missing_hours)}")
    print(f"   Coverage: {len(existing_hours)/len(complete_timeline)*100:.1f}%")
    
    if missing_hours:
        print(f"\n❌ First 20 missing hours:")
        for i, missing_hour in enumerate(missing_hours[:20]):
            print(f"   {missing_hour}")
        
        if len(missing_hours) > 20:
            print(f"   ... and {len(missing_hours) - 20} more")
    
    return missing_hours

def check_data_quality():
    """Check data quality in BTC_news"""
    print("\n🔍 Checking BTC_news data quality...")
    
    btc_df = load_collection_to_dataframe("BTC_news")
    btc_df['datetime'] = pd.to_datetime(btc_df['datetime'])
    
    print(f"📊 Data Quality Report:")
    print(f"   Total records: {len(btc_df)}")
    
    # Check for duplicates
    duplicates = btc_df['datetime'].duplicated().sum()
    print(f"   Duplicate timestamps: {duplicates}")
    
    # Check for null values
    null_counts = btc_df.isnull().sum()
    print(f"   Null values:")
    for col, count in null_counts.items():
        if count > 0:
            print(f"      {col}: {count}")
    
    # Check feature ranges
    numeric_cols = ['mentioned', 'mentioned_6h', 'mentioned_24h', 'momentum', 'momentum_6h', 'volatility_6h', 'relative_strength']
    existing_cols = [col for col in numeric_cols if col in btc_df.columns]
    
    print(f"\n📈 Feature Statistics:")
    for col in existing_cols:
        print(f"   {col}:")
        print(f"      Min: {btc_df[col].min():.2f}")
        print(f"      Max: {btc_df[col].max():.2f}")
        print(f"      Mean: {btc_df[col].mean():.2f}")
        print(f"      Std: {btc_df[col].std():.2f}")

def main():
    """Main analysis function"""
    print("🚀 Starting detailed BTC_news analysis...")
    
    # Detailed gap analysis
    gaps = check_btc_news_detailed()
    
    # Pattern analysis
    analyze_gap_patterns()
    
    # Missing hours analysis
    missing_hours = find_missing_hours_in_gaps()
    
    # Data quality check
    check_data_quality()
    
    print("\n🎉 BTC_news analysis completed!")
    
    return gaps, missing_hours

if __name__ == "__main__":
    gaps, missing_hours = main()