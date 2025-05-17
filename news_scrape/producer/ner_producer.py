import asyncio
import os
import sys
from datetime import datetime, timedelta

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mongo.mongo_client import MongoService
from utils.count_ner import count_crypto_mentions_hourly


async def process_mentions(
    execution_date: datetime, start_time_delta: int = 24, end_time_delta: int = 6
) -> str:
    """
    Process and analyze news articles for cryptocurrency mentions.

    Args:
        execution_date: The execution datetime from Airflow

    Returns:
        str: Summary of processing results
    """
    mongo_client = MongoService()
    try:
        # Define time range for scraping (last 3 hours)
        start_time = execution_date - timedelta(hours=start_time_delta)
        end_time = execution_date + timedelta(hours=end_time_delta)

        # Get news articles within that time range
        news_data = await mongo_client.get_news_metadata_for_ner_by_time_interval(
            start_time, end_time
        )

        if not news_data:
            print("No news data found for processing.")
            return "No data to process."

        print(f"Processing {len(news_data)} articles for NER counts...")

        # Count crypto mentions
        counts = count_crypto_mentions_hourly(news_data)

        # Use upsert logic to aggregate mentions by hour
        upsert_stat = await mongo_client.upsert_ner_counts(counts)
        if upsert_stat:
            print(f"\nUpserted {upsert_stat} hourly mention records.")
            article_ids = [doc["_id"] for doc in news_data]

            _ = await mongo_client.news_metadata_collection.update_many(
                {"_id": {"$in": article_ids}},
                {
                    "$set": {
                        "ner_counted": True,
                        "ner_counted_at": datetime.utcnow(),  # Optional: lưu thời điểm đã xử lý
                    }
                },
            )

        return f"Processed {len(news_data)} articles. Upserted {len(counts)} hourly mention records."
    finally:
        await mongo_client.close()


if __name__ == "__main__":
    # Example execution date for testing
    execution_date = datetime(2025, 5, 15, 5, 0, 0)  # May 14, 2025, 12:00 PM
    print(f"Testing with execution date: {execution_date}")
    result = asyncio.run(process_mentions(execution_date))
    print(result)
