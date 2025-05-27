# crypto_target_creation.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from utils.mongodb import (
    load_collection_to_dataframe, 
    save_dataframe_to_collection,
    get_latest_timestamp,
    find_documents
)
from coins import COIN_SYMBOLS

def filter_data_from_2023(df):
    """Filter data to include only from 2023 onwards"""
    if 'datetime' not in df.columns:
        return df
    
    # Convert datetime column if it's not already datetime
    if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
        df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Filter from 2023-01-01 onwards
    start_date = datetime(2023, 1, 1)
    filtered_df = df[df['datetime'] >= start_date].copy()
    
    print(f"Filtered data from {len(df)} to {len(filtered_df)} records (2023 onwards)")
    return filtered_df.reset_index(drop=True)

def create_target_variables(df):
    """Create prediction targets for 1-12 hour horizons"""
    result_df = df.copy()
    
    # Sort by datetime to ensure correct shifts
    result_df = result_df.sort_values('datetime').reset_index(drop=True)
    
    # Create targets for each hour from 1 to 12
    for hour in range(1, 13):
        future_price = result_df['close_price'].shift(-hour)
        
        # Main target: percentage price change
        result_df[f'target_{hour}h'] = (
            (future_price - result_df['close_price']) / result_df['close_price'] * 100
        )
        
        # Binary direction (1 = up, 0 = down)
        result_df[f'direction_{hour}h'] = (result_df[f'target_{hour}h'] > 0).astype(int)
    
    # IMPORTANT: Remove last 12 rows since they can't have future targets
    # This ensures we never have NaN targets due to shift operations
    if len(result_df) > 12:
        result_df = result_df.iloc[:-12].copy()
        print(f"Removed last 12 rows (no future data available for targets)")
    else:
        print(f"Warning: Dataset has only {len(result_df)} rows, less than 12 hours of data")
        return pd.DataFrame()  # Return empty if insufficient data
    
    return result_df

def process_coin_targets(coin_symbol, update_only=False):
    """
    Create target variables for a single cryptocurrency
    """
    print(f"Creating targets for {coin_symbol}...")
    
    # Load from features collection (which has clean OHLCV data)
    features_collection = f'{coin_symbol}_features'
    targets_collection = f'{coin_symbol}_targets'
    
    # Load data based on update mode
    if update_only:
        latest_timestamp = get_latest_timestamp(targets_collection)
        if latest_timestamp:
            # Need extra buffer since we'll remove last 12 hours
            query_start = latest_timestamp - timedelta(hours=24)
            query = {"datetime": {"$gt": query_start}}
            new_docs = find_documents(features_collection, query)
            
            if not new_docs:
                print(f"No new data for {coin_symbol} targets")
                return None
            
            features_df = pd.DataFrame(new_docs)
        else:
            features_df = load_collection_to_dataframe(features_collection)
    else:
        features_df = load_collection_to_dataframe(features_collection)
    
    if features_df.empty:
        print(f"No feature data available for {coin_symbol}")
        return None
    
    # Filter to 2023+ data
    features_df = filter_data_from_2023(features_df)
    
    # We only need basic columns for target calculation
    required_cols = ['datetime', 'close_price']
    
    # Check if we have required columns
    missing_cols = [col for col in required_cols if col not in features_df.columns]
    if missing_cols:
        print(f"Missing required columns for {coin_symbol}: {missing_cols}")
        return None
    
    # Keep only essential columns for target calculation
    target_base_df = features_df[required_cols].copy()
    
    # Create target variables
    targets_df = create_target_variables(target_base_df)
    
    if targets_df.empty:
        print(f"No valid target data for {coin_symbol}")
        return None
    
    # Filter new records if updating
    if update_only and 'latest_timestamp' in locals() and latest_timestamp:
        targets_df = targets_df[targets_df['datetime'] > latest_timestamp]
    
    # Remove any remaining NaN values
    targets_df = targets_df.dropna().reset_index(drop=True)
    
    if targets_df.empty:
        print(f"No valid targets after processing for {coin_symbol}")
        return None
    
    # Save targets to separate collection
    save_dataframe_to_collection(targets_df, targets_collection)
    
    latest_target_time = targets_df['datetime'].max()
    print(f"Saved {len(targets_df)} target records for {coin_symbol}")
    print(f"Latest target timestamp: {latest_target_time}")
    
    return targets_df

def verify_target_integrity(coin_symbol):
    """Verify that targets don't contain future data that shouldn't exist"""
    targets_collection = f'{coin_symbol}_targets'
    
    try:
        targets_df = load_collection_to_dataframe(targets_collection)
        if targets_df.empty:
            return True
        
        # Check if any target columns have all NaN values (which shouldn't happen)
        target_cols = [col for col in targets_df.columns if col.startswith('target_') or col.startswith('direction_')]
        
        for col in target_cols:
            if targets_df[col].isna().all():
                print(f"WARNING: {coin_symbol} - Column {col} has all NaN values")
                return False
            
            nan_count = targets_df[col].isna().sum()
            if nan_count > 0:
                print(f"WARNING: {coin_symbol} - Column {col} has {nan_count} NaN values")
        
        print(f"✓ Target integrity check passed for {coin_symbol}")
        return True
        
    except Exception as e:
        print(f"Error verifying targets for {coin_symbol}: {str(e)}")
        return False

def main():
    """Main execution function for target creation only"""
    print(f"Starting TARGET CREATION for {len(COIN_SYMBOLS)} cryptocurrencies...")
    print("Note: Latest 12 hours will be excluded (no future data available)")
    
    results = {}
    
    for coin in COIN_SYMBOLS:
        try:
            targets_collection = f'{coin}_targets'
            latest_timestamp = get_latest_timestamp(targets_collection)
            
            if latest_timestamp:
                print(f"Updating targets for {coin} (last: {latest_timestamp})")
                result_df = process_coin_targets(coin, update_only=True)
                status = f"Updated with {len(result_df)} records" if result_df is not None else "No new data"
            else:
                print(f"Full target processing for {coin}")
                result_df = process_coin_targets(coin, update_only=False)
                status = f"Processed {len(result_df)} records" if result_df is not None else "No data available"
            
            # Verify target integrity
            if result_df is not None:
                verify_target_integrity(coin)
            
            results[coin] = status
            
        except Exception as e:
            print(f"Error processing targets for {coin}: {str(e)}")
            results[coin] = f"ERROR: {str(e)}"
    
    # Summary
    print("\n" + "="*60)
    print("TARGET CREATION SUMMARY")
    print("="*60)
    for coin, result in results.items():
        print(f"{coin:8}: {result}")
    
    print(f"\nCompleted target creation for {len(COIN_SYMBOLS)} cryptocurrencies!")
    print("\nIMPORTANT NOTES:")
    print("- Target collections exclude the latest 12 hours (no future data)")
    print("- Use {coin}_features for model features")
    print("- Use {coin}_targets for model targets")
    print("- Join on 'datetime' column for training data")

if __name__ == "__main__":
    main()