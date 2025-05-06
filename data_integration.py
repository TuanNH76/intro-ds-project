# data_integration.py
# Integrate cryptocurrency data from multiple sources for ML feature preparation

import pandas as pd
from datetime import datetime, timedelta
import numpy as np

from utils.mongodb import (
    load_collection_to_dataframe,
    save_dataframe_to_collection,
    get_collection,
    find_documents
)
from coins import COIN_SYMBOLS

def integrate_coin_data(coin_symbol):
    """
    Integrates data from technical indicators and social features for a specific coin
    and saves the result to MLFeature_{coin_symbol} collection.
    
    Parameters:
    coin_symbol (str): The symbol of the cryptocurrency to process
    
    Returns:
    pandas.DataFrame: The integrated feature dataset ready for ML
    """
    print(f"Integrating data for {coin_symbol}...")
    
    # Collection names
    technical_collection = f"{coin_symbol}_technical_indicators"
    social_collection = f"social_feature_{coin_symbol.lower()}"
    output_collection = f"MLFeature_{coin_symbol}"
    
    # Load technical indicators data
    try:
        technical_df = load_collection_to_dataframe(technical_collection)
        if technical_df.empty:
            print(f"No technical indicator data found for {coin_symbol}")
            return None
        print(f"Loaded {len(technical_df)} technical indicator records")
    except Exception as e:
        print(f"Error loading technical data for {coin_symbol}: {str(e)}")
        return None
    
    # Load social features data
    try:
        social_df = load_collection_to_dataframe(social_collection)
        if social_df.empty:
            print(f"No social feature data found for {coin_symbol}")
            social_df = None
        else:
            print(f"Loaded {len(social_df)} social feature records")
    except Exception as e:
        print(f"No social data available for {coin_symbol}: {str(e)}")
        social_df = None
    
    # Ensure datetime column is properly formatted
    technical_df['datetime'] = pd.to_datetime(technical_df['datetime'])
    if social_df is not None:
        social_df['datetime'] = pd.to_datetime(social_df['datetime'])
    
    # -------------------------
    # Feature engineering
    # -------------------------
    
    # 1. Create price-based target variables (for regression or classification)
    technical_df = technical_df.sort_values('datetime')
    
    # Calculate future price changes for different time horizons
    for hours in [1, 3, 6, 12, 24]:
        # Shift close price to get future price
        technical_df[f'future_price_{hours}h'] = technical_df['close_price'].shift(-hours)
        
        # Calculate percent change
        technical_df[f'price_change_{hours}h'] = (
            (technical_df[f'future_price_{hours}h'] - technical_df['close_price']) / 
            technical_df['close_price'] * 100
        )
        
        # Create binary classification targets (1 if price goes up, 0 if down)
        technical_df[f'price_direction_{hours}h'] = (
            technical_df[f'price_change_{hours}h'] > 0
        ).astype(int)
        
        # Create multi-class classification targets
        # -1: significant down, 0: sideways, 1: significant up
        threshold = 1.0  # 1% threshold for significant movement
        conditions = [
            technical_df[f'price_change_{hours}h'] <= -threshold,
            technical_df[f'price_change_{hours}h'] >= threshold,
        ]
        choices = [-1, 1]
        technical_df[f'price_movement_{hours}h'] = np.select(
            conditions, choices, default=0
        )
    
    # 2. Calculate additional derived features
    
    # Volatility features
    technical_df['price_range'] = technical_df['high_price'] - technical_df['low_price']
    technical_df['price_range_pct'] = (
        technical_df['price_range'] / technical_df['close_price'] * 100
    )
    
    # Volume features
    technical_df['volume_change_pct'] = technical_df['volume_to'].pct_change() * 100
    
    # 3. Merge with social data if available
    if social_df is not None:
        # Round datetimes to the nearest hour for proper joining
        technical_df['datetime_hour'] = technical_df['datetime'].dt.floor('H')
        social_df['datetime_hour'] = social_df['datetime'].dt.floor('H')
        
        # Merge dataframes on datetime_hour
        merged_df = pd.merge(
            technical_df, 
            social_df, 
            left_on='datetime_hour', 
            right_on='datetime_hour',
            how='left', 
            suffixes=('', '_social')
        )
        
        # Drop the extra datetime column from social df
        if 'datetime_social' in merged_df.columns:
            merged_df = merged_df.drop(columns=['datetime_social'])
        
        # Forward fill missing social data
        social_columns = ['mentioned', 'mentioned_6h', 'mentioned_24h', 'momentum']
        for col in social_columns:
            if col in merged_df.columns:
                merged_df[col] = merged_df[col].ffill()
    else:
        merged_df = technical_df.copy()
        merged_df['datetime_hour'] = merged_df['datetime'].dt.floor('H')
    
    # 4. Calculate cross-data features if social data is available
    if social_df is not None:
        # Calculate price/volume to social mention ratios
        if 'mentioned' in merged_df.columns and merged_df['mentioned'].sum() > 0:
            merged_df['price_to_mention'] = merged_df['close_price'] / merged_df['mentioned'].replace(0, 0.1)
            merged_df['volume_to_mention'] = merged_df['volume_to'] / merged_df['mentioned'].replace(0, 0.1)
    
    # 5. Clean up and prepare final dataset
    
    # Drop rows with NaN values (from calculations like pct_change)
    merged_df = merged_df.dropna(subset=[f'price_change_{hours}h' for hours in [1, 3, 6, 12, 24]])
    
    # Drop unneeded columns
    columns_to_drop = [
        'future_price_1h', 'future_price_3h', 'future_price_6h', 
        'future_price_12h', 'future_price_24h'
    ]
    columns_to_drop = [col for col in columns_to_drop if col in merged_df.columns]
    
    if columns_to_drop:
        merged_df = merged_df.drop(columns=columns_to_drop)
    
    # Keep the original datetime and drop datetime_hour
    if 'datetime_hour' in merged_df.columns:
        merged_df = merged_df.drop(columns=['datetime_hour'])
    
    # Sort by datetime for time series consistency
    merged_df = merged_df.sort_values('datetime')
    
    # Save the integrated dataset to MongoDB
    save_dataframe_to_collection(merged_df, output_collection)
    
    print(f"Successfully integrated data for {coin_symbol}. {len(merged_df)} records saved to '{output_collection}' collection.")
    
    return merged_df

def integrate_all_coins():
    """
    Integrate data for all cryptocurrencies in the COIN_SYMBOLS list.
    """
    print(f"Starting data integration for {len(COIN_SYMBOLS)} cryptocurrencies...")
    
    results = {}
    
    for coin in COIN_SYMBOLS:
        try:
            df = integrate_coin_data(coin)
            if df is not None:
                results[coin] = len(df)
            else:
                results[coin] = "No data integrated"
        except Exception as e:
            print(f"Error integrating data for {coin}: {str(e)}")
            results[coin] = f"ERROR: {str(e)}"
    
    # Print summary
    print("\nIntegration Summary:")
    for coin, result in results.items():
        print(f"{coin}: {result}")
    
    print("Data integration complete for all cryptocurrencies!")
    return results

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Integrate cryptocurrency data for ML feature preparation')
    parser.add_argument('--coin', help='Process a specific coin symbol (e.g., BTC)')
    args = parser.parse_args()
    
    if args.coin:
        if args.coin in COIN_SYMBOLS:
            integrate_coin_data(args.coin)
        else:
            print(f"Error: {args.coin} is not in the list of supported coins.")
            print(f"Supported coins: {', '.join(COIN_SYMBOLS)}")
    else:
        integrate_all_coins()