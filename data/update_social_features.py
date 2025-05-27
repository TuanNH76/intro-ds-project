# update_social_features.py

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add the parent directory to sys.path to properly import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.mongodb import (
    get_collection,
    load_collection_to_dataframe,
    save_dataframe_to_collection
)
from coins import COIN_SYMBOLS

def update_social_features():
    """
    Create social features for all coins using timestamps from the mentioned_frequency collection,
    including time since last mention. Ensures all hours are covered by generating a continuous
    time series of hourly data.
    """
    print("Starting social features update for all coins...")
    
    # Load the entire mentioned_frequency collection
    raw_df = load_collection_to_dataframe("mentioned_frequency")
    
    if raw_df.empty:
        print("No data found in mentioned_frequency collection.")
        return
    
    # Ensure we have datetime format
    if 'time' in raw_df.columns:
        raw_df['datetime'] = pd.to_datetime(raw_df['time'])
    elif 'hour' in raw_df.columns:
        raw_df['datetime'] = pd.to_datetime(raw_df['hour'])
    
    # Sort by datetime for consistent processing
    raw_df = raw_df.sort_values('datetime')
    
    # Find min and max datetime to create a complete hourly series
    min_datetime = raw_df['datetime'].min().floor('h')
    max_datetime = raw_df['datetime'].max().floor('h') + timedelta(hours=1)
    
    # Create a complete hourly datetime range
    all_hours = pd.date_range(start=min_datetime, end=max_datetime, freq='h')
    
    print(f"Creating continuous hourly data from {min_datetime} to {max_datetime}")
    print(f"Total hours to process: {len(all_hours)}")
    print(f"Processing social features for {len(COIN_SYMBOLS)} coins...")
    
    # Process each coin
    for coin in COIN_SYMBOLS:
        print(f"Processing {coin}...")
        
        # Create a DataFrame with continuous hours
        continuous_df = pd.DataFrame({'datetime': all_hours})
        
        # Create a dictionary to store features for this coin at each hour
        coin_features = []
        
        # Track the last time this coin was mentioned
        last_mentioned_time = None
        
        # Process each hour in the continuous time series
        for current_time in all_hours:
            # Find the row in raw_df for this timestamp, if it exists
            matching_rows = raw_df[raw_df['datetime'].dt.floor('h') == current_time]
            
            if not matching_rows.empty:
                # Get the first matching row (there should usually be just one per hour)
                exact_row = matching_rows.iloc[0]
                # Get mentioned count for this coin at this time
                mentioned = exact_row['mentioned_frequency'].get(coin, 0)
            else:
                # No data for this hour, set mentioned to 0
                mentioned = 0
            
            # Update last_mentioned_time if this coin is mentioned
            if mentioned > 0:
                last_mentioned_time = current_time
                hours_since_last_mention = 0
            else:
                # Calculate hours since last mention
                if last_mentioned_time is not None:
                    hours_since_last_mention = (current_time - last_mentioned_time).total_seconds() / 3600
                else:
                    hours_since_last_mention = -1  # No previous mention, use -1 instead of None
            
            # Calculate 6-hour lookback (or however many hours are available)
            lookback_start = current_time - timedelta(hours=6)
            lookback_rows = raw_df[
                (raw_df['datetime'] > lookback_start) &
                (raw_df['datetime'] <= current_time)
            ]
            
            # Calculate 24-hour lookback
            day_lookback_start = current_time - timedelta(hours=24)
            day_lookback_rows = raw_df[
                (raw_df['datetime'] > day_lookback_start) &
                (raw_df['datetime'] <= current_time)
            ]
            
            # Calculate features - handle empty DataFrames
            if not lookback_rows.empty:
                mentioned_6h = sum([r.get('mentioned_frequency', {}).get(coin, 0) for _, r in lookback_rows.iterrows()])
            else:
                mentioned_6h = 0
                
            if not day_lookback_rows.empty:
                mentioned_24h = sum([r.get('mentioned_frequency', {}).get(coin, 0) for _, r in day_lookback_rows.iterrows()])
            else:
                mentioned_24h = 0
            
            # Calculate momentum (rate of change)
            momentum = 0
            if len(lookback_rows) > 1:
                earliest = lookback_rows.iloc[0].get('mentioned_frequency', {}).get(coin, 0)
                latest = lookback_rows.iloc[-1].get('mentioned_frequency', {}).get(coin, 0)
                momentum = latest - earliest
            
            # Create feature document
            feature_doc = {
                'datetime': current_time,
                'mentioned': mentioned,
                'mentioned_6h': mentioned_6h,
                'mentioned_24h': mentioned_24h,
                'momentum': momentum,
                'hours_since_last_mention': hours_since_last_mention,
                # Store last_mentioned_time as a string or None to avoid NaT issues
                'last_mentioned_time': last_mentioned_time.isoformat() if last_mentioned_time is not None else None
            }
            
            coin_features.append(feature_doc)
        
        # Create DataFrame from features
        coin_df = pd.DataFrame(coin_features)
        
        # Save to MongoDB collection with updated naming convention
        collection_name = f"{coin}_Social_Features"
        
        # Drop the existing collection if it exists to ensure a fresh start
        collection = get_collection(collection_name)
        if collection.count_documents({}) > 0:
            print(f"Dropping existing collection: {collection_name}")
            collection.drop()
        
        # Make sure we're not sending NaT values to MongoDB
        # This replaces NaT values with None
        for col in coin_df.select_dtypes(include=['datetime64']).columns:
            coin_df[col] = coin_df[col].astype(object).where(~coin_df[col].isna(), None)
        
        # Save to MongoDB time series collection
        save_dataframe_to_collection(coin_df, collection_name)
        
        print(f"Created collection: {collection_name} with {len(coin_df)} records")
    
    print("Social features update completed for all coins.")

if __name__ == "__main__":
    update_social_features()