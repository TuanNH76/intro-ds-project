#!/usr/bin/env python
# run_integration.py

import sys
import os
import argparse
from datetime import datetime

# Add the parent directory and current directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_feature_integration import integrate_all_coins, integrate_coin_data
from coins import COIN_SYMBOLS
from ml_data_utilities import list_available_ml_feature_coins, get_feature_stats

def main():
    parser = argparse.ArgumentParser(description='Run ML Feature Integration from September 1, 2024')
    parser.add_argument('--coin', help='Process a specific coin symbol (e.g., BTC)')
    parser.add_argument('--stats', action='store_true', help='Show statistics for existing ML feature collections')
    args = parser.parse_args()
    
    # Set default start date to September 1, 2024
    from_date = datetime(2024, 9, 1)
    
    if args.stats:
        # Show statistics for existing ML feature collections
        available_coins = list_available_ml_feature_coins()
        
        if not available_coins:
            print("No ML feature collections found.")
            return
        
        print(f"Found {len(available_coins)} coins with ML feature collections:")
        
        # Print header
        print(f"{'Coin':<6} {'Records':<10} {'Date Range':<30} {'Features':<10}")
        print("-" * 60)
        
        # Print stats for each coin
        for coin in available_coins:
            stats = get_feature_stats(coin)
            date_range = f"{stats['earliest_date'].strftime('%Y-%m-%d')} to {stats['latest_date'].strftime('%Y-%m-%d')}"
            print(f"{coin:<6} {stats['record_count']:<10} {date_range:<30} {stats['feature_count']:<10}")
        
        return
    
    # Run integration
    if args.coin:
        if args.coin in COIN_SYMBOLS:
            print(f"Processing {args.coin} from {from_date.strftime('%Y-%m-%d')}...")
            integrate_coin_data(args.coin, from_date)
        else:
            print(f"Error: {args.coin} is not in the list of supported coins.")
            print(f"Supported coins: {', '.join(COIN_SYMBOLS)}")
    else:
        print(f"Processing all {len(COIN_SYMBOLS)} coins from {from_date.strftime('%Y-%m-%d')}...")
        integrate_all_coins(from_date)
    
    print("Integration complete!")

if __name__ == "__main__":
    main()