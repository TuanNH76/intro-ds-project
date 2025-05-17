# update_social_features.py
import os
import sys
from datetime import timedelta

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mongo.mongo_client import MongoService
from producer.coin_price_producer import COIN_SYMBOLS

mongo_client = MongoService()


async def update_social_features():
    """
    Create social features for all coins using timestamps from the mentioned_frequency collection,
    including time since last mention.
    """
    print("Starting social features update for all coins...")

    # Load the entire mentioned_frequency collection
    df = await mongo_client.load_collection_to_dataframe("mentioned_frequency")

    # Ensure we have datetime format
    if "time" in df.columns:
        df["datetime"] = pd.to_datetime(df["time"])
    elif "hour" in df.columns:
        df["datetime"] = pd.to_datetime(df["hour"])

    # Sort by datetime for consistent processing
    df = df.sort_values("datetime")

    # Get unique datetime values from the collection
    unique_datetimes = df["datetime"].unique()

    print(
        f"Processing social features for {len(unique_datetimes)} timestamps and {len(COIN_SYMBOLS)} coins..."
    )

    # Process each coin
    for coin in COIN_SYMBOLS:
        print(f"Processing {coin}...")

        # Create a list to store features for this coin
        coin_features = []

        # Track the last time this coin was mentioned
        last_mentioned_time = None

        # Process each timestamp from the mentioned_frequency collection
        for current_time in unique_datetimes:
            # Get the exact row for this timestamp
            exact_row = df[df["datetime"] == current_time].iloc[0]

            # Get mentioned count for this coin at this time
            mentioned = exact_row["mentioned_frequency"].get(coin, 0)

            # Update last_mentioned_time if this coin is mentioned
            if mentioned > 0:
                last_mentioned_time = current_time
                hours_since_last_mention = 0
            else:
                # Calculate hours since last mention
                if last_mentioned_time is not None:
                    hours_since_last_mention = (
                        current_time - last_mentioned_time
                    ).total_seconds() / 3600
                else:
                    hours_since_last_mention = (
                        -1
                    )  # No previous mention, use -1 instead of None

            # Calculate 6-hour lookback (or however many hours are available)
            lookback_rows = df[
                (df["datetime"] > current_time - timedelta(hours=6))
                & (df["datetime"] <= current_time)
            ]

            # Calculate 24-hour lookback
            day_lookback_rows = df[
                (df["datetime"] > current_time - timedelta(hours=24))
                & (df["datetime"] <= current_time)
            ]

            # Calculate features
            mentioned_6h = sum(
                [
                    r["mentioned_frequency"].get(coin, 0)
                    for _, r in lookback_rows.iterrows()
                ]
            )
            mentioned_24h = sum(
                [
                    r["mentioned_frequency"].get(coin, 0)
                    for _, r in day_lookback_rows.iterrows()
                ]
            )

            # Calculate momentum (rate of change)
            momentum = 0
            if len(lookback_rows) > 1:
                earliest = lookback_rows.iloc[0]["mentioned_frequency"].get(coin, 0)
                latest = lookback_rows.iloc[-1]["mentioned_frequency"].get(coin, 0)
                momentum = latest - earliest

            # Create feature document
            feature_doc = {
                "datetime": current_time,
                "mentioned": mentioned,
                "mentioned_6h": mentioned_6h,
                "mentioned_24h": mentioned_24h,
                "momentum": momentum,
                "hours_since_last_mention": hours_since_last_mention,
                # Store last_mentioned_time as a string or None to avoid NaT issues
                "last_mentioned_time": last_mentioned_time.isoformat()
                if last_mentioned_time is not None
                else None,
            }

            coin_features.append(feature_doc)

        # Create DataFrame from features
        coin_df = pd.DataFrame(coin_features)

        # Save to MongoDB collection with updated naming convention
        collection_name = f"{coin}_Social_Features"

        # Drop the existing collection if it exists to ensure a fresh start
        collection = mongo_client.get_collection(collection_name)
        if await collection.count_documents({}) > 0:
            print(f"Dropping existing collection: {collection_name}")
            await collection.drop()

        # Make sure we're not sending NaT values to MongoDB
        # This replaces NaT values with None
        for col in coin_df.select_dtypes(include=["datetime64"]).columns:
            coin_df[col] = coin_df[col].astype(object).where(~coin_df[col].isna(), None)

        # Save to MongoDB time series collection
        await mongo_client.save_dataframe_to_collection(coin_df, collection_name)

        print(f"Created collection: {collection_name} with {len(coin_df)} records")

    print("Social features update completed for all coins.")


if __name__ == "__main__":
    import asyncio

    asyncio.run(update_social_features())
