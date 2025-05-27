import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
from utils.mongodb import load_collection_to_dataframe
from coins import COIN_SYMBOLS

def check_missing_hours_for_coin(coin_symbol, collection_suffix="_all"):
    """
    Check for missing hours in a specific coin's data collection.
    
    Args:
        coin_symbol (str): The coin symbol (e.g., 'BTC', 'ETH')
        collection_suffix (str): The suffix for the collection name (default: '_all')
    
    Returns:
        dict: Summary of missing hours analysis
    """
    collection_name = f"{coin_symbol}{collection_suffix}"
    
    print(f"\n🔍 Checking missing hours for {coin_symbol} in collection: {collection_name}")
    
    try:
        # Load data from MongoDB
        df = load_collection_to_dataframe(collection_name)
        
        if df.empty:
            print(f"❌ No data found for {coin_symbol}")
            return {
                'coin': coin_symbol,
                'total_records': 0,
                'missing_hours': [],
                'missing_count': 0,
                'data_range': None,
                'status': 'no_data'
            }
        
        # Convert datetime column to pandas datetime
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Sort by datetime
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Get data range
        start_date = df['datetime'].min()
        end_date = df['datetime'].max()
        
        print(f"📅 Data range: {start_date} to {end_date}")
        print(f"📊 Total records: {len(df)}")
        
        # Create complete hourly range
        expected_hours = pd.date_range(start=start_date, end=end_date, freq='H')
        
        # Get actual hours from data
        actual_hours = set(df['datetime'])
        
        # Find missing hours
        missing_hours = []
        for expected_hour in expected_hours:
            if expected_hour not in actual_hours:
                missing_hours.append(expected_hour)
        
        missing_count = len(missing_hours)
        expected_count = len(expected_hours)
        
        print(f"⏰ Expected hours: {expected_count}")
        print(f"❌ Missing hours: {missing_count}")
        
        if missing_count > 0:
            print(f"📉 Data completeness: {((expected_count - missing_count) / expected_count * 100):.2f}%")
            
            # Show first few missing hours as examples
            if missing_count <= 10:
                print(f"🕐 Missing hours: {[dt.strftime('%Y-%m-%d %H:%M') for dt in missing_hours]}")
            else:
                print(f"🕐 First 10 missing hours: {[dt.strftime('%Y-%m-%d %H:%M') for dt in missing_hours[:10]]}")
                print(f"   ... and {missing_count - 10} more")
        else:
            print("✅ No missing hours found - data is complete!")
        
        return {
            'coin': coin_symbol,
            'collection': collection_name,
            'total_records': len(df),
            'expected_records': expected_count,
            'missing_hours': missing_hours,
            'missing_count': missing_count,
            'data_range': (start_date, end_date),
            'completeness_percentage': ((expected_count - missing_count) / expected_count * 100) if expected_count > 0 else 0,
            'status': 'complete' if missing_count == 0 else 'incomplete'
        }
        
    except Exception as e:
        print(f"❌ Error processing {coin_symbol}: {str(e)}")
        return {
            'coin': coin_symbol,
            'status': 'error',
            'error': str(e)
        }

def check_missing_hours_all_coins(collection_suffix="_all"):
    """
    Check missing hours for all coins and provide a summary report.
    
    Args:
        collection_suffix (str): The suffix for collection names (default: '_all')
    
    Returns:
        dict: Comprehensive report of missing hours across all coins
    """
    print("🚀 Starting missing hours analysis for all coins...")
    
    results = {}
    summary_stats = {
        'total_coins': 0,
        'coins_with_data': 0,
        'complete_coins': 0,
        'incomplete_coins': 0,
        'coins_with_errors': 0,
        'total_missing_hours': 0
    }
    
    for coin in COIN_SYMBOLS:
        result = check_missing_hours_for_coin(coin, collection_suffix)
        results[coin] = result
        
        summary_stats['total_coins'] += 1
        
        if result['status'] == 'error':
            summary_stats['coins_with_errors'] += 1
        elif result['status'] == 'no_data':
            continue
        else:
            summary_stats['coins_with_data'] += 1
            summary_stats['total_missing_hours'] += result['missing_count']
            
            if result['status'] == 'complete':
                summary_stats['complete_coins'] += 1
            else:
                summary_stats['incomplete_coins'] += 1
    
    # Print summary report
    print("\n" + "="*60)
    print("📊 MISSING HOURS ANALYSIS SUMMARY")
    print("="*60)
    print(f"🪙 Total coins analyzed: {summary_stats['total_coins']}")
    print(f"📈 Coins with data: {summary_stats['coins_with_data']}")
    print(f"✅ Complete coins (no missing hours): {summary_stats['complete_coins']}")
    print(f"⚠️  Incomplete coins (missing hours): {summary_stats['incomplete_coins']}")
    print(f"❌ Coins with errors: {summary_stats['coins_with_errors']}")
    print(f"⏰ Total missing hours across all coins: {summary_stats['total_missing_hours']}")
    
    # Show most problematic coins
    if summary_stats['incomplete_coins'] > 0:
        print(f"\n🔍 COINS WITH MOST MISSING DATA:")
        incomplete_coins = [(coin, data) for coin, data in results.items() 
                           if data['status'] == 'incomplete']
        incomplete_coins.sort(key=lambda x: x[1]['missing_count'], reverse=True)
        
        for coin, data in incomplete_coins[:5]:  # Top 5 most problematic
            completeness = data['completeness_percentage']
            print(f"   {coin}: {data['missing_count']} missing hours ({completeness:.1f}% complete)")
    
    return results, summary_stats

def find_missing_hour_gaps(coin_symbol, collection_suffix="_all", min_gap_hours=1):
    """
    Find continuous gaps of missing hours (useful for identifying data outages).
    
    Args:
        coin_symbol (str): The coin symbol
        collection_suffix (str): Collection name suffix
        min_gap_hours (int): Minimum gap size to report (default: 1)
    
    Returns:
        list: List of gap periods with start, end, and duration
    """
    result = check_missing_hours_for_coin(coin_symbol, collection_suffix)
    
    if result['status'] != 'incomplete' or not result['missing_hours']:
        return []
    
    missing_hours = sorted(result['missing_hours'])
    gaps = []
    
    if not missing_hours:
        return gaps
    
    # Find continuous gaps
    gap_start = missing_hours[0]
    gap_end = missing_hours[0]
    
    for i in range(1, len(missing_hours)):
        current_hour = missing_hours[i]
        expected_next = gap_end + timedelta(hours=1)
        
        if current_hour == expected_next:
            # Continue current gap
            gap_end = current_hour
        else:
            # End current gap and start new one
            gap_duration = int((gap_end - gap_start).total_seconds() / 3600) + 1
            if gap_duration >= min_gap_hours:
                gaps.append({
                    'start': gap_start,
                    'end': gap_end,
                    'duration_hours': gap_duration
                })
            
            gap_start = current_hour
            gap_end = current_hour
    
    # Don't forget the last gap
    gap_duration = int((gap_end - gap_start).total_seconds() / 3600) + 1
    if gap_duration >= min_gap_hours:
        gaps.append({
            'start': gap_start,
            'end': gap_end,
            'duration_hours': gap_duration
        })
    
    return gaps

def main():
    """Main function to run the missing hours analysis"""
    
    # Check all coins
    results, summary = check_missing_hours_all_coins()
    
    # Optionally, analyze gaps for coins with missing data
    print(f"\n🔍 ANALYZING DATA GAPS...")
    for coin in COIN_SYMBOLS[:3]:  # Analyze first 3 coins as example
        gaps = find_missing_hour_gaps(coin)
        if gaps:
            print(f"\n📉 {coin} - Found {len(gaps)} data gaps:")
            for gap in gaps[:5]:  # Show first 5 gaps
                print(f"   {gap['start'].strftime('%Y-%m-%d %H:%M')} to "
                      f"{gap['end'].strftime('%Y-%m-%d %H:%M')} "
                      f"({gap['duration_hours']} hours)")

if __name__ == "__main__":
    main()