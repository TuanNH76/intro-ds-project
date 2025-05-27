# crypto_feature_engineering.py

import pandas as pd
import numpy as np
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

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

def clean_and_prepare_data(df):
    """Clean and prepare cryptocurrency data"""
    # Filter to 2023+ data first
    df = filter_data_from_2023(df)
    
    # Drop unnecessary columns
    cols_to_drop = ['coin', 'conversionType', 'conversionSymbol']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # Standardize column names
    column_mapping = {
        'high': 'high_price',
        'low': 'low_price', 
        'open': 'open_price',
        'close': 'close_price',
        'volumeto': 'volume_to',
        'volumefrom': 'volume_from'
    }
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
    
    # Sort by datetime
    df = df.sort_values('datetime').reset_index(drop=True)
    
    return df

def create_basic_features(df):
    """Create basic price and volume features"""
    result_df = df.copy()
    
    # Price-based features
    result_df['price_change_1h'] = result_df['close_price'].pct_change(1) * 100
    result_df['price_change_6h'] = result_df['close_price'].pct_change(6) * 100  
    result_df['price_change_24h'] = result_df['close_price'].pct_change(24) * 100
    
    # Candlestick features
    result_df['body_size'] = abs(result_df['close_price'] - result_df['open_price'])
    result_df['body_pct'] = result_df['body_size'] / result_df['open_price'] * 100
    result_df['hl_range'] = result_df['high_price'] - result_df['low_price']
    result_df['hl_range_pct'] = result_df['hl_range'] / result_df['close_price'] * 100
    
    # Price position within range
    result_df['price_position'] = np.where(
        result_df['hl_range'] > 0,
        (result_df['close_price'] - result_df['low_price']) / result_df['hl_range'],
        0.5
    )
    
    # Volume features (if available)
    if 'volume_from' in result_df.columns:
        result_df['volume_change_24h'] = result_df['volume_from'].pct_change(24) * 100
        result_df['price_volume_trend'] = (
            result_df['price_change_24h'] * np.sign(result_df['volume_change_24h'])
        )
    
    return result_df

def create_technical_indicators(df):
    """Create essential technical indicators with proper bounds checking"""
    result_df = df.copy()
    
    # Moving Averages
    result_df['sma_12'] = SMAIndicator(result_df['close_price'], window=12).sma_indicator()
    result_df['sma_24'] = SMAIndicator(result_df['close_price'], window=24).sma_indicator()
    result_df['ema_12'] = EMAIndicator(result_df['close_price'], window=12).ema_indicator()
    result_df['ema_24'] = EMAIndicator(result_df['close_price'], window=24).ema_indicator()
    
    # Price distance from MAs (as percentages)
    result_df['price_sma12_dist'] = ((result_df['close_price'] - result_df['sma_12']) / result_df['sma_12'] * 100)
    result_df['price_ema12_dist'] = ((result_df['close_price'] - result_df['ema_12']) / result_df['ema_12'] * 100)
    result_df['sma_crossover'] = (result_df['sma_12'] > result_df['sma_24']).astype(int)
    
    # MACD - Keep original values, don't normalize these
    macd = MACD(result_df['close_price'])
    result_df['macd'] = macd.macd()
    result_df['macd_signal'] = macd.macd_signal()
    result_df['macd_histogram'] = macd.macd_diff()
    result_df['macd_bullish'] = (result_df['macd'] > result_df['macd_signal']).astype(int)
    
    # RSI - Should be 0-100, don't normalize
    result_df['rsi'] = RSIIndicator(result_df['close_price'], window=14).rsi()
    # Clip RSI to valid range just in case
    result_df['rsi'] = np.clip(result_df['rsi'], 0, 100)
    result_df['rsi_overbought'] = (result_df['rsi'] > 70).astype(int)
    result_df['rsi_oversold'] = (result_df['rsi'] < 30).astype(int)
    
    # Stochastic - Should be 0-100, don't normalize  
    stoch = StochasticOscillator(result_df['high_price'], result_df['low_price'], result_df['close_price'])
    result_df['stoch_k'] = stoch.stoch()
    result_df['stoch_d'] = stoch.stoch_signal()
    # Clip to valid range
    result_df['stoch_k'] = np.clip(result_df['stoch_k'], 0, 100)
    result_df['stoch_d'] = np.clip(result_df['stoch_d'], 0, 100)
    
    # Bollinger Bands
    bb = BollingerBands(result_df['close_price'], window=20)
    result_df['bb_upper'] = bb.bollinger_hband()
    result_df['bb_middle'] = bb.bollinger_mavg()
    result_df['bb_lower'] = bb.bollinger_lband()
    
    # BB Position (0-1 range)
    bb_range = result_df['bb_upper'] - result_df['bb_lower']
    result_df['bb_position'] = np.where(
        bb_range > 0,
        (result_df['close_price'] - result_df['bb_lower']) / bb_range,
        0.5  # Default to middle if range is 0
    )
    result_df['bb_position'] = np.clip(result_df['bb_position'], 0, 1)
    
    # BB Squeeze indicator
    result_df['bb_width'] = (bb_range / result_df['bb_middle'] * 100)
    result_df['bb_squeeze'] = (result_df['bb_width'] < 10).astype(int)
    
    return result_df

def create_advanced_features(df):
    """Create advanced feature engineering with proper bounds"""
    result_df = df.copy()
    
    # Volatility features - use absolute values
    price_returns = result_df['close_price'].pct_change()
    result_df['volatility_12h'] = price_returns.rolling(12).std() * 100
    result_df['volatility_24h'] = price_returns.rolling(24).std() * 100
    
    # Volatility ratio (avoid division by zero)
    result_df['volatility_ratio'] = np.where(
        result_df['volatility_24h'] > 0,
        result_df['volatility_12h'] / result_df['volatility_24h'],
        1.0
    )
    
    # Momentum features
    result_df['momentum_12h'] = (result_df['close_price'] / result_df['close_price'].shift(12) - 1) * 100
    result_df['momentum_24h'] = (result_df['close_price'] / result_df['close_price'].shift(24) - 1) * 100
    
    # Support/Resistance levels
    result_df['support_24h'] = result_df['low_price'].rolling(24).min()
    result_df['resistance_24h'] = result_df['high_price'].rolling(24).max()
    result_df['support_distance'] = (result_df['close_price'] - result_df['support_24h']) / result_df['close_price']
    result_df['resistance_distance'] = (result_df['resistance_24h'] - result_df['close_price']) / result_df['close_price']
    
    # Trend strength (use absolute momentum and RSI deviation)
    rsi_deviation = np.where(
        result_df['rsi'].notna(),
        abs(result_df['rsi'] - 50) / 50,
        0
    )
    result_df['trend_strength'] = abs(result_df['momentum_24h']) * (1 + rsi_deviation)
    
    # Time-based features
    if 'datetime' in result_df.columns:
        result_df['hour'] = result_df['datetime'].dt.hour
        result_df['day_of_week'] = result_df['datetime'].dt.dayofweek
        result_df['is_weekend'] = (result_df['day_of_week'] >= 5).astype(int)
        
        # Cyclical encoding for time features
        result_df['hour_sin'] = np.sin(2 * np.pi * result_df['hour'] / 24)
        result_df['hour_cos'] = np.cos(2 * np.pi * result_df['hour'] / 24)
        result_df['dow_sin'] = np.sin(2 * np.pi * result_df['day_of_week'] / 7)
        result_df['dow_cos'] = np.cos(2 * np.pi * result_df['day_of_week'] / 7)
    
    return result_df

def normalize_features(df):
    """Selectively normalize only specific price/volume features, keep indicators intact"""
    result_df = df.copy()
    
    # Only normalize basic price/volume features - NOT technical indicators
    safe_to_normalize = [
        'price_change_1h', 'price_change_6h', 'price_change_24h',
        'body_pct', 'hl_range_pct', 'momentum_12h', 'momentum_24h',
        'price_sma12_dist', 'price_ema12_dist', 'trend_strength'
    ]
    
    # Only include features that exist and are safe to normalize
    normalize_cols = [col for col in safe_to_normalize if col in result_df.columns]
    
    if normalize_cols:
        scaler = StandardScaler()
        result_df[normalize_cols] = scaler.fit_transform(result_df[normalize_cols])
        print(f"Normalized {len(normalize_cols)} features: {normalize_cols}")
    
    # Keep technical indicators (RSI, MACD, Stochastic, etc.) in their original ranges
    print("Technical indicators (RSI, MACD, Stochastic, BB) kept in original ranges")
    
    return result_df

def process_coin_features(coin_symbol, update_only=False):
    """
    Process features for a single cryptocurrency (NO TARGET VARIABLES)
    """
    print(f"Processing features for {coin_symbol}...")
    
    raw_collection = f'{coin_symbol}_raw'
    features_collection = f'{coin_symbol}_features'
    
    # Load data
    if update_only:
        latest_timestamp = get_latest_timestamp(features_collection)
        if latest_timestamp:
            query_start = latest_timestamp - timedelta(hours=48)  # Buffer for calculations
            query = {"datetime": {"$gt": query_start}}
            new_docs = find_documents(raw_collection, query)
            
            if not new_docs:
                print(f"No new data for {coin_symbol}")
                return None
            
            raw_df = pd.DataFrame(new_docs)
        else:
            raw_df = load_collection_to_dataframe(raw_collection)
    else:
        raw_df = load_collection_to_dataframe(raw_collection)
    
    if raw_df.empty:
        print(f"No data available for {coin_symbol}")
        return None
    
    # Feature engineering pipeline (NO TARGETS)
    cleaned_df = clean_and_prepare_data(raw_df)
    basic_features_df = create_basic_features(cleaned_df)
    technical_df = create_technical_indicators(basic_features_df)
    advanced_df = create_advanced_features(technical_df)
    final_df = normalize_features(advanced_df)
    
    # Filter new records if updating
    if update_only and 'latest_timestamp' in locals() and latest_timestamp:
        final_df = final_df[final_df['datetime'] > latest_timestamp]
    
    # Remove rows with NaN values
    final_df = final_df.dropna().reset_index(drop=True)
    
    if final_df.empty:
        print(f"No valid data after processing for {coin_symbol}")
        return None
    
    # Save features to MongoDB
    save_dataframe_to_collection(final_df, features_collection)
    print(f"Saved {len(final_df)} feature records for {coin_symbol}")
    
    return final_df

def main():
    """Main execution function for feature engineering only"""
    print(f"Starting FEATURE ENGINEERING for {len(COIN_SYMBOLS)} cryptocurrencies (2023+ data)...")
    
    results = {}
    
    for coin in COIN_SYMBOLS:
        try:
            features_collection = f'{coin}_features'
            latest_timestamp = get_latest_timestamp(features_collection)
            
            if latest_timestamp:
                print(f"Updating features for {coin} (last: {latest_timestamp})")
                result_df = process_coin_features(coin, update_only=True)
                status = f"Updated with {len(result_df)} records" if result_df is not None else "No new data"
            else:
                print(f"Full feature processing for {coin}")
                result_df = process_coin_features(coin, update_only=False)
                status = f"Processed {len(result_df)} records" if result_df is not None else "No data available"
            
            results[coin] = status
            
        except Exception as e:
            print(f"Error processing {coin}: {str(e)}")
            results[coin] = f"ERROR: {str(e)}"
    
    # Summary
    print("\n" + "="*60)
    print("FEATURE ENGINEERING SUMMARY")
    print("="*60)
    for coin, result in results.items():
        print(f"{coin:8}: {result}")
    
    print(f"\nCompleted feature engineering for {len(COIN_SYMBOLS)} cryptocurrencies!")
    print("Next step: Run crypto_target_creation.py to create target variables")

if __name__ == "__main__":
    main()