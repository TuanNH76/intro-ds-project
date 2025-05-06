# feature_engineering.py

import pandas as pd
from ta.trend import SMAIndicator, MACD, ADXIndicator, AroonIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from datetime import datetime, timedelta

from utils.mongodb import (
    load_collection_to_dataframe, 
    save_dataframe_to_collection,
    get_latest_timestamp,
    find_documents,
    get_collection
)
from coins import COIN_SYMBOLS

def clean_raw_data(df):
    """
    This function removes unnecessary columns from the raw data and renames columns for clarity.

    Parameters:
    df (pandas.DataFrame): The raw DataFrame with OHLCV data.

    Returns:
    pandas.DataFrame: The cleaned DataFrame with unnecessary columns removed and renamed columns.
    """
    # Remove unnecessary columns if they exist
    cols_to_drop = ['coin', 'conversionType', 'conversionSymbol']
    drop_cols = [col for col in cols_to_drop if col in df.columns]
    
    if drop_cols:
        df_cleaned = df.drop(columns=drop_cols)
    else:
        df_cleaned = df.copy()
    
    # Rename columns for clarity
    column_mapping = {
        'high': 'high_price',
        'low': 'low_price',
        'open': 'open_price',
        'close': 'close_price',
        'volumeto': 'volume_to',
        'volumefrom': 'volume_from'
    }
    
    # Only rename columns that exist
    rename_cols = {k: v for k, v in column_mapping.items() if k in df_cleaned.columns}
    df_cleaned = df_cleaned.rename(columns=rename_cols)
    
    return df_cleaned

def calculate_technical_indicators(df):
    """
    Calculate various technical indicators for the given DataFrame.
    
    Parameters:
    df (pandas.DataFrame): DataFrame with OHLCV data
    
    Returns:
    pandas.DataFrame: DataFrame with added technical indicators
    """
    # Make a copy to avoid modifying the original DataFrame
    result_df = df.copy()
    
    # Ensure DataFrame is sorted by datetime
    if 'datetime' in result_df.columns:
        result_df = result_df.sort_values('datetime')
    
    # 1. Calculate On-Balance Volume (OBV)
    def calculate_obv(df):
        """Manually calculate On-Balance Volume (OBV)"""
        obv = [0]
        for i in range(1, len(df)):
            if df['close_price'].iloc[i] > df['close_price'].iloc[i - 1]:  # price increase
                obv.append(obv[-1] + df['volume_to'].iloc[i])
            elif df['close_price'].iloc[i] < df['close_price'].iloc[i - 1]:  # price decrease
                obv.append(obv[-1] - df['volume_to'].iloc[i])
            else:  # no change
                obv.append(obv[-1])
        return obv

    result_df['obv'] = calculate_obv(result_df)

    # 2. Calculate Accumulation/Distribution Line (A/D)
    def calculate_ad_line(df):
        """Manually calculate the Accumulation/Distribution (A/D) line"""
        ad_line = [0]  # Start AD at 0
        for i in range(1, len(df)):
            # Check if high and low are the same to avoid division by zero
            if df['high_price'].iloc[i] == df['low_price'].iloc[i]:
                # If prices didn't change, use 0 for the multiplier
                mf_multiplier = 0
            else:
                mf_multiplier = ((df['close_price'].iloc[i] - df['low_price'].iloc[i]) - 
                                (df['high_price'].iloc[i] - df['close_price'].iloc[i])) / \
                                (df['high_price'].iloc[i] - df['low_price'].iloc[i])
            
            mf_volume = mf_multiplier * df['volume_to'].iloc[i]
            ad_line.append(ad_line[-1] + mf_volume)
        return ad_line

    result_df['acc_dist'] = calculate_ad_line(result_df)

    # 3. Calculate Average Directional Index (ADX)
    result_df['adx'] = ADXIndicator(result_df['high_price'], result_df['low_price'], 
                                    result_df['close_price'], window=14).adx()

    # 4. Calculate Aroon Indicator (Aroon Up, Aroon Down)
    aroon = AroonIndicator(high=result_df['high_price'], low=result_df['low_price'], window=14, fillna=False)
    result_df['aroon_up'] = aroon.aroon_up()
    result_df['aroon_down'] = aroon.aroon_down()

    # 5. Calculate MACD (Moving Average Convergence Divergence)
    macd_indicator = MACD(result_df['close_price'])
    result_df['macd'] = macd_indicator.macd()
    result_df['macd_signal'] = macd_indicator.macd_signal()
    result_df['macd_diff'] = macd_indicator.macd_diff()

    # 6. Calculate Relative Strength Index (RSI)
    result_df['rsi'] = RSIIndicator(result_df['close_price'], window=14).rsi()

    # 7. Calculate Stochastic Oscillator
    stoch = StochasticOscillator(result_df['high_price'], result_df['low_price'], 
                                result_df['close_price'], window=14)
    result_df['stoch_k'] = stoch.stoch()
    result_df['stoch_d'] = stoch.stoch_signal()

    # 8. Calculate Bollinger Bands
    bollinger = BollingerBands(result_df['close_price'], window=20)
    result_df['bb_upper'] = bollinger.bollinger_hband()
    result_df['bb_middle'] = bollinger.bollinger_mavg()
    result_df['bb_lower'] = bollinger.bollinger_lband()
    result_df['bb_width'] = (result_df['bb_upper'] - result_df['bb_lower']) / result_df['bb_middle']

    # 9. Calculate Simple Moving Averages
    result_df['sma_7'] = SMAIndicator(result_df['close_price'], window=7).sma_indicator()
    result_df['sma_20'] = SMAIndicator(result_df['close_price'], window=20).sma_indicator()
    result_df['sma_50'] = SMAIndicator(result_df['close_price'], window=50).sma_indicator()
    result_df['sma_200'] = SMAIndicator(result_df['close_price'], window=200).sma_indicator()

    return result_df

def process_coin(coin_symbol, update_only=False):
    """
    Process a single cryptocurrency by loading data, calculating indicators, and saving to MongoDB.
    
    Parameters:
    coin_symbol (str): The symbol of the cryptocurrency to process
    update_only (bool): If True, only process new data since the last update
    
    Returns:
    pandas.DataFrame: The processed DataFrame with technical indicators
    """
    print(f"Processing {coin_symbol}{'(update only)' if update_only else ''}...")
    
    # Raw collection name
    raw_collection_name = f'{coin_symbol}_raw'
    # Output collection name
    output_collection_name = f'{coin_symbol}_technical_indicators'
    
    if update_only:
        # Get latest timestamp from the indicators collection
        latest_timestamp = get_latest_timestamp(output_collection_name)
        
        if latest_timestamp:
            # Add a small buffer to avoid duplicate entries
            query_start_time = latest_timestamp - timedelta(hours=1)
            
            # Query for documents newer than the latest timestamp
            query = {"datetime": {"$gt": query_start_time}}
            
            # Get the collection
            collection = get_collection(raw_collection_name)
            
            # Execute query
            new_docs = find_documents(raw_collection_name, query)
            
            if not new_docs:
                print(f"No new data for {coin_symbol} since {latest_timestamp}")
                return None
            
            # Convert to DataFrame
            raw_df = pd.DataFrame(new_docs)
        else:
            # If no data exists in the output collection, process all data
            print(f"No existing data found for {coin_symbol} in {output_collection_name}. Processing all data.")
            raw_df = load_collection_to_dataframe(raw_collection_name)
    else:
        # Load all raw data
        raw_df = load_collection_to_dataframe(raw_collection_name)
    
    # Ensure we have data to process
    if raw_df.empty:
        print(f"No data available for {coin_symbol}")
        return None
    
    # Clean the raw data
    cleaned_df = clean_raw_data(raw_df)
    
    # If updating, we need to include some historical data for accurate indicator calculation
    if update_only and latest_timestamp:
        # Load enough historical data for the longest indicator window (200 for SMA)
        historical_query = {
            "datetime": {
                "$lte": query_start_time,
                "$gte": query_start_time - timedelta(days=30)  # Approximately 30 days should be enough
            }
        }
        
        historical_docs = find_documents(raw_collection_name, historical_query)
        
        if historical_docs:
            historical_df = pd.DataFrame(historical_docs)
            historical_df = clean_raw_data(historical_df)
            
            # Combine historical and new data
            combined_df = pd.concat([historical_df, cleaned_df]).drop_duplicates(subset=['datetime']).reset_index(drop=True)
            
            # Calculate indicators on the combined data
            processed_df = calculate_technical_indicators(combined_df)
            
            # Filter to only keep the new data after calculation
            processed_df = processed_df[processed_df['datetime'] > latest_timestamp]
        else:
            # Fall back to calculating on just the new data if no historical data is available
            processed_df = calculate_technical_indicators(cleaned_df)
    else:
        # Calculate technical indicators on all data
        processed_df = calculate_technical_indicators(cleaned_df)
    
    # Drop NaN values (from rolling calculations)
    processed_df = processed_df.dropna().reset_index(drop=True)
    
    # Ensure we have data to save
    if processed_df.empty:
        print(f"No valid data after processing for {coin_symbol}")
        return None
    
    # Save to MongoDB
    save_dataframe_to_collection(processed_df, output_collection_name)
    
    print(f"Completed processing {coin_symbol}. {len(processed_df)} records saved to '{output_collection_name}' collection.")
    
    return processed_df

def update_feature_engineering():
    """
    Update technical indicators for all cryptocurrencies with new data since the last update.
    This function should be called after new raw data is added to the database.
    """
    print(f"Starting update for {len(COIN_SYMBOLS)} cryptocurrencies...")
    
    update_results = {}
    
    for coin in COIN_SYMBOLS:
        try:
            result_df = process_coin(coin, update_only=True)
            if result_df is not None:
                update_results[coin] = len(result_df)
        except Exception as e:
            print(f"Error updating {coin}: {str(e)}")
            update_results[coin] = f"ERROR: {str(e)}"
    
    # Print summary
    print("\nUpdate Summary:")
    for coin, result in update_results.items():
        print(f"{coin}: {result} new records processed" if isinstance(result, (int, float)) else result)
    
    print("Technical indicators update complete!")
    return update_results

def process_all_coins():
    """
    Process all cryptocurrencies in the COIN_SYMBOLS list from scratch.
    """
    print(f"Starting full processing for {len(COIN_SYMBOLS)} cryptocurrencies...")
    
    for coin in COIN_SYMBOLS:
        try:
            process_coin(coin, update_only=False)
        except Exception as e:
            print(f"Error processing {coin}: {str(e)}")
    
    print("Full processing complete for all cryptocurrencies!")

if __name__ == "__main__":
    import sys
    
    # Check if we should update or do full processing
    if len(sys.argv) > 1 and sys.argv[1] == "--update":
        update_feature_engineering()
    else:
        process_all_coins()