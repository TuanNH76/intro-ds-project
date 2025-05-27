import pandas as pd
from datetime import datetime, timedelta
from utils.mongodb import load_collection_to_dataframe

def find_missing_hours():
    """Find the exact missing hour periods across all coins"""
    
    # Use BTC as reference since all coins have the same pattern
    coin = "BTC"
    collection_name = f"{coin}_raw"
    
    print(f"🔍 Analyzing missing hours using {coin} as reference...")
    
    # Load data
    df = load_collection_to_dataframe(collection_name)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    print(f"📊 Data range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"📊 Total records: {len(df):,}")
    
    # Find gaps
    time_diffs = df['datetime'].diff().dropna()
    gaps = time_diffs[time_diffs > timedelta(hours=1)]
    
    print(f"\n⚠️  Found {len(gaps)} gaps totaling {sum(gap.total_seconds()/3600 for gap in gaps):.0f} missing hours")
    print(f"\n📋 Missing Time Periods:")
    print(f"{'#':>2} {'Start':>19} {'End':>19} {'Hours':>6} {'Days':>5}")
    print("-" * 60)
    
    total_missing = 0
    gap_details = []
    
    for i, (idx, gap) in enumerate(gaps.items()):
        gap_start = df.loc[idx-1, 'datetime']
        gap_end = df.loc[idx, 'datetime']
        missing_hours = int(gap.total_seconds() / 3600)
        missing_days = missing_hours / 24
        
        total_missing += missing_hours
        
        gap_details.append({
            'start': gap_start,
            'end': gap_end,
            'hours': missing_hours
        })
        
        print(f"{i+1:>2} {gap_start.strftime('%Y-%m-%d %H:%M'):>19} "
              f"{gap_end.strftime('%Y-%m-%d %H:%M'):>19} "
              f"{missing_hours:>6} {missing_days:>5.1f}")
    
    print("-" * 60)
    print(f"{'TOTAL':>2} {' '*39} {total_missing:>6} {total_missing/24:>5.1f}")
    
    # Group by patterns
    print(f"\n📈 Gap Size Distribution:")
    gap_sizes = [gap['hours'] for gap in gap_details]
    unique_sizes = sorted(set(gap_sizes))
    
    for size in unique_sizes:
        count = gap_sizes.count(size)
        print(f"   • {size:>3} hours: {count:>2} occurrences")
    
    # Show the actual missing hour timestamps
    print(f"\n🕐 All Missing Hour Timestamps:")
    print("=" * 50)
    
    all_missing_hours = []
    for gap in gap_details:
        current = gap['start'] + timedelta(hours=1)
        while current < gap['end']:
            all_missing_hours.append(current)
            current += timedelta(hours=1)
    
    print(f"Total missing timestamps: {len(all_missing_hours)}")
    
    # Group by date to see daily patterns
    missing_by_date = {}
    for dt in all_missing_hours:
        date_key = dt.date()
        if date_key not in missing_by_date:
            missing_by_date[date_key] = []
        missing_by_date[date_key].append(dt.hour)
    
    print(f"\n📅 Missing Hours by Date:")
    for date, hours in sorted(missing_by_date.items()):
        hours_str = ', '.join(map(str, sorted(hours)))
        print(f"   {date}: {len(hours):>2} hours missing ({hours_str})")
    
    # Save to file for reference
    with open('missing_hours_report.txt', 'w') as f:
        f.write("Missing Hours Report\n")
        f.write("===================\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Reference Coin: {coin}\n")
        f.write(f"Total Gaps: {len(gaps)}\n")
        f.write(f"Total Missing Hours: {total_missing}\n\n")
        
        f.write("Gap Details:\n")
        for i, gap in enumerate(gap_details):
            f.write(f"{i+1:2d}. {gap['start'].strftime('%Y-%m-%d %H:%M')} to "
                   f"{gap['end'].strftime('%Y-%m-%d %H:%M')} ({gap['hours']} hours)\n")
        
        f.write(f"\nAll Missing Timestamps:\n")
        for dt in sorted(all_missing_hours):
            f.write(f"{dt.strftime('%Y-%m-%d %H:%M')}\n")
    
    print(f"\n💾 Detailed report saved to: missing_hours_report.txt")
    
    return gap_details, all_missing_hours

if __name__ == "__main__":
    gap_details, missing_hours = find_missing_hours()
    
    print(f"\n🎯 Summary:")
    print(f"   • These missing hours affect ALL coins identically")
    print(f"   • Likely caused by API downtime or data collection issues")
    print(f"   • Consider backfilling from alternative data sources")
    print(f"   • Or use interpolation/forward-fill for missing hours")