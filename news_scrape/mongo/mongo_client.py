import logging
import os
import sys
from datetime import datetime, timezone

import pandas as pd
from motor.core import AgnosticCollection as AsyncCollection
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING, DESCENDING

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Any

from dotenv import load_dotenv

_ = load_dotenv()

logger = logging.getLogger("mongodb")


def _build_mongo_uri() -> str:
    username = os.getenv("MONGO_INITDB_ROOT_USERNAME")
    password = os.getenv("MONGO_INITDB_ROOT_PASSWORD")
    host = os.getenv("MONGO_HOST", "news-mongo")
    port = os.getenv("MONGO_INTERNAL_PORT", "27017")
    return f"mongodb://{username}:{password}@{host}:{port}/?authSource=admin"


# Current time in UTC (recommended for databases)
def get_current_utc_timestamp() -> datetime:
    return datetime.now(timezone.utc)


# Get normalized timestamp for the start of day (00:00:00)
def get_normalized_day_timestamp(dt: datetime | None = None) -> datetime:
    if dt is None:
        dt = datetime.now(timezone.utc)
    return datetime(dt.year, dt.month, dt.day, 0, 0, 0, tzinfo=timezone.utc)


# Convert datetime to string format
def timestamp_to_str(dt: datetime, fmt: str = "%Y-%m-%d") -> str:
    return dt.strftime(fmt)


# Convert string to datetime
def str_to_timestamp(date_str: str, fmt: str = "%Y-%m-%d") -> datetime:
    return datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)


def normalize_timestamp(ts: datetime) -> datetime:
    """Normalize timestamp to start of day in UTC."""
    return datetime(ts.year, ts.month, ts.day, 0, 0, 0, tzinfo=timezone.utc)


class MongoService:
    def __init__(
        self,
        uri: str | None = None,
        db_name: str | None = None,
        news_url_collection_name: str = "news_urls",
        news_metadata_collection_name: str = "news_metadata",
        news_content_collection_name: str = "news_content",
        ner_results_collection_name: str = "ner_results",
        count_ner_collection_name: str = "count_ner",
    ):
        self.uri: str = uri or _build_mongo_uri()

        self.db_name: str = db_name or os.getenv("MONGO_DB", "url_data")

        self.client: AsyncIOMotorClient[dict[str, Any]] = AsyncIOMotorClient(self.uri)
        self.news_url_collection: AsyncCollection[dict[str, Any]] = self.client[
            self.db_name
        ][news_url_collection_name]
        self.news_metadata_collection: AsyncCollection[dict[str, Any]] = self.client[
            self.db_name
        ][news_metadata_collection_name]
        self.news_content_collection: AsyncCollection[dict[str, Any]] = self.client[
            self.db_name
        ][news_content_collection_name]
        self.ner_results_collection: AsyncCollection[dict[str, Any]] = self.client[
            self.db_name
        ][ner_results_collection_name]
        self.count_ner_collection: AsyncCollection[dict[str, Any]] = self.client[
            self.db_name
        ][count_ner_collection_name]

    def get_collection(self, collection_name: str):
        """
        Returns a MongoDB collection by name.
        """
        return self.client[self.db_name][collection_name]

    async def find_documents(
        self,
        collection_name: str,
        query: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Find documents in a MongoDB collection based on a query.

        Args:
            collection_name (str): The name of the collection to search.
            query (dict): The query to filter documents.
            projection (dict): Optional projection to limit fields returned.

        Returns:
            list[dict]: A list of documents matching the query.
        """
        collection = self.client[self.db_name][collection_name]
        cursor = collection.find(query)
        results = []

        async for document in cursor:
            results.append(document)

        return results

    async def get_existing_news_urls(self) -> list[str]:
        """
        Return a list of all existing news URLs from the database.

        Returns:
            list[str]: A list of URL strings stored in the news_url_collection.
        """
        cursor = self.news_metadata_collection.find({}, {"url": 1, "_id": 0})
        urls = []

        async for document in cursor:
            if "url" in document and document["url"]:
                urls.append(document["url"])

        return urls

    async def insert_news_url(
        self, list_url: list[str], timestamp: datetime | None = None
    ) -> None:
        """
        Insert a list of news URLs into the database with their corresponding timestamp.

        Args:
            list_url (list[str]): A list of URL strings to be inserted.
            timestamp (datetime): The timestamp indicating when these URLs were scraped.

        Returns:
            None
        """
        if not list_url:
            logger.info("No URLs to insert")
            return

        # Get existing URLs to avoid duplicates
        existing_urls = await self.get_existing_news_urls()
        new_urls = [url for url in list_url if url not in existing_urls]

        if not new_urls:
            logger.info("All URLs already exist in the database")
            return

        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # Normalize to start of day for consistent date comparisons
        day_start = normalize_timestamp(timestamp)
        date_str = day_start.strftime("%Y-%m-%d")

        # Rest of your existing code...
        documents: list[Any] = []
        for url in list_url:
            documents.append(
                {
                    "url": url,
                    "scraped_date": date_str,
                    "created_at": timestamp,  # Original timestamp with time
                    "day_timestamp": day_start,  # Normalized for easy day-based queries
                    "processed": False,
                }
            )

        if documents:
            try:
                result = await self.news_url_collection.insert_many(documents)
                logger.info(
                    f"Inserted {len(result.inserted_ids)} new news URLs into database"
                )
            except Exception as e:
                logger.error(f"Error inserting URLs: {e}")

    async def update_news_url(
        self, url: str, timestamp: datetime | None = None
    ) -> None:
        """
        Update the processed status of a news URL in the database.

        Args:
            url (str): The URL to be updated.
            timestamp (datetime): The timestamp indicating when this URL was scraped.

        Returns:
            None
        """
        if not url:
            logger.info("No URL to update")
            return

        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # Normalize to start of day for consistent date comparisons
        day_start = normalize_timestamp(timestamp)

        # Rest of your existing code...
        result = await self.news_url_collection.update_one(
            {"url": url},
            {
                "$set": {
                    "processed": True,
                    "updated_at": timestamp,
                    "day_timestamp": day_start,
                }
            },
        )
        if result.modified_count > 0:
            logger.info(f"Updated news URL: {url}")
        else:
            logger.info(f"No update needed for URL: {url}")

    async def insert_news_metadata(
        self, metadata: dict[str, Any], timestamp: datetime | None = None
    ) -> None:
        """
        Insert news metadata into the database.

        Args:
            metadata (dict[str, Any]): A dictionary containing news metadata.
            timestamp (datetime): The timestamp indicating when this metadata was scraped.

        Returns:
            None
        """
        if not metadata:
            logger.info("No metadata to insert")
            return

        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # Rest of your existing code...
        metadata["created_at"] = timestamp
        _ = await self.news_metadata_collection.insert_one(metadata)

    async def get_news_metadata(self, url: str) -> dict[str, Any] | None:
        """
        Retrieve news metadata from the database based on the URL.

        Args:
            url (str): The URL for which to retrieve metadata.

        Returns:
            dict[str, Any]: The metadata document associated with the URL.
        """
        if not url:
            logger.info("No URL provided for metadata retrieval")
            return None

        metadata = await self.news_metadata_collection.find_one({"url": url})
        if metadata:
            return metadata
        else:
            logger.info(f"No metadata found for URL: {url}")
            return None

    async def get_news_metadata_by_id(self, metadata_id: str) -> dict[str, Any] | None:
        """
        Retrieve news metadata from the database based on the metadata ID.

        Args:
            metadata_id (str): The ID of the metadata document to retrieve.

        Returns:
            dict[str, Any]: The metadata document associated with the ID.
        """
        if not metadata_id:
            logger.info("No metadata ID provided for retrieval")
            return None

        metadata = await self.news_metadata_collection.find_one({"_id": metadata_id})
        if metadata:
            return metadata
        else:
            logger.info(f"No metadata found for ID: {metadata_id}")
            return None

    async def get_all_news_metadata(self) -> list[dict[str, Any]]:
        """
        Retrieve all news metadata from the database.

        Returns:
            list[dict[str, Any]]: A list of all metadata documents.
        """
        cursor = self.news_metadata_collection.find({})
        metadata_list = []

        async for document in cursor:
            metadata_list.append(document)

        return metadata_list

    async def get_news_metadata_by_normalized_time_interval(
        self, start_time: datetime, end_time: datetime
    ) -> list[dict[str, Any]]:
        """
        Retrieve news metadata from the database based on a time interval.
        Args:
            start_time (datetime): The start of the time interval.
            end_time (datetime): The end of the time interval.
        Returns:
            list[dict[str, Any]]: A list of metadata documents within the specified time interval.
        """
        # Normalize to start of day for consistent date comparisons
        start_day = normalize_timestamp(start_time)
        end_day = normalize_timestamp(end_time)

        cursor = self.news_metadata_collection.find(
            {"day_timestamp": {"$gte": start_day, "$lt": end_day}}
        )
        metadata_list = []

        async for document in cursor:
            metadata_list.append(document)

        return metadata_list

    async def get_news_metadata_for_ner_by_time_interval(
        self, start_time: datetime, end_time: datetime
    ) -> list[dict[str, Any]]:
        """
        Retrieve news metadata from the database based on a precise time interval.

        Args:
            start_time (datetime): Start of the interval (inclusive).
            end_time (datetime): End of the interval (exclusive).

        Returns:
            list[dict[str, Any]]: News documents within the time range.
        """
        cursor = self.news_metadata_collection.find(
            {
                "published_time": {"$gte": start_time, "$lt": end_time},
                "ner_counted": {"$ne": True},
            }
        )

        results = []
        async for doc in cursor:
            results.append(doc)

        return results

    async def get_news_metadata_by_time_interval(
        self, start_time: datetime, end_time: datetime
    ) -> list[dict[str, Any]]:
        """
        Retrieve news metadata from the database based on a precise time interval.

        Args:
            start_time (datetime): Start of the interval (inclusive).
            end_time (datetime): End of the interval (exclusive).

        Returns:
            list[dict[str, Any]]: News documents within the time range.
        """
        cursor = self.news_metadata_collection.find(
            {
                "published_time": {"$gte": start_time, "$lt": end_time},
            }
        )

        results = []
        async for doc in cursor:
            results.append(doc)

        return results

    async def insert_count_ner_results(
        self, results: list[dict[str, Any]], timestamp: datetime
    ) -> None:
        """
        Insert count of NER results into the database.
        Args:
            results (dict[str, Any]): A dictionary containing count of NER results.
            metadata_id (str): Optional ID of existing metadata to link with.
        Returns:
            None
        """
        if not results:
            logger.info("No count NER results to insert")
            return

        # Insert count NER results document
        try:
            result = await self.count_ner_collection.insert_one(
                {"results": results, "timestamp": timestamp}
            )
            logger.info(
                f"Inserted count NER results with ID: {len(result.inserted_id)}"
            )
        except Exception as e:
            logger.error(f"Error inserting count NER results: {e}")

    async def upsert_ner_counts(self, results: list[dict[str, Any]]):
        if not results:
            return None

        stats = {"matched": 0, "modified": 0, "upserted": 0}
        for record in results:
            hour = record["hour"]
            freq = record["mentioned_frequency"]
            time = record["time"]

            # Chuyển dict nested thành dict flat để dùng $inc
            inc_fields = {
                f"mentioned_frequency.{token}": count for token, count in freq.items()
            }

            update_res = await self.count_ner_collection.update_one(
                {"hour": hour},
                {
                    "$inc": inc_fields,
                    "$set": {"time": time, "updated_at": datetime.utcnow()},
                },
                upsert=True,
            )
            stats["matched"] += update_res.matched_count
            stats["modified"] += update_res.modified_count
            stats["upserted"] += 1 if update_res.upserted_id is not None else 0

            return stats

    async def get_count_ner_results_by_time_intervals(
        self, start: datetime, end: datetime
    ) -> list[dict[str, Any]] | None:
        """
        Retrieve count NER results from the database based on the time intervals.

        Args:
            start (datetime): The start of the time interval.
            end (datetime): The end of the time interval.

        Returns:
            list[dict[str, Any]]: A list of count NER results documents within the specified time interval.
        """
        # Normalize to start of day for consistent date comparisons

        cursor = self.count_ner_collection.find({"time": {"$gte": start, "$lt": end}})
        results_list = []

        async for document in cursor:
            results_list.append(document)

        return results_list

    async def get_coin_timeseries(
        self,
        collection_name: str,
        coin_symbol: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """
        Fetch time series data for a specific coin from a processed collection.
        Returns list of dicts sorted by datetime.
        """
        collection = self.client[self.db_name][collection_name]
        query = {f"data.{coin_symbol}": {"$exists": True}}

        if start_time or end_time:
            query["datetime"] = {}
            if start_time:
                query["datetime"]["$gte"] = start_time
            if end_time:
                query["datetime"]["$lte"] = end_time

        projection = {"datetime": 1, f"data.{coin_symbol}": 1, "_id": 0}

        cursor = collection.find(query, projection).sort("datetime", ASCENDING)

        results = []
        async for doc in cursor:
            coin_data = doc["data"].get(coin_symbol, {})
            results.append(
                {
                    "datetime": doc["datetime"],
                    "close": coin_data.get("close"),
                    "volume_usd": coin_data.get("volume_usd"),
                }
            )

        return results

    async def get_latest_timestamp(self, collection_name: str) -> datetime | None:
        """
        Get the most recent datetime value from a collection.
        """
        collection = self.client[self.db_name][collection_name]
        doc = await collection.find_one(sort=[("datetime", DESCENDING)])
        return doc.get("datetime") if doc and "datetime" in doc else None

    async def load_collection_to_dataframe(self, collection_name: str) -> pd.DataFrame:
        """
        Load a collection into a pandas DataFrame.
        """
        collection = self.client[self.db_name][collection_name]
        cursor = collection.find()
        docs = [doc async for doc in cursor]

        df = pd.DataFrame(docs)
        if "_id" in df.columns:
            df.drop(columns=["_id"], inplace=True)

        return df

    async def save_dataframe_to_collection(
        self, df: pd.DataFrame, collection_name: str, time_field: str = "datetime"
    ) -> None:
        """
        Save a DataFrame to a MongoDB time-series collection.
        """
        df[time_field] = pd.to_datetime(df[time_field])

        if (
            collection_name
            not in await self.client[self.db_name].list_collection_names()
        ):
            _ = await self.client[self.db_name].create_collection(
                collection_name,
                timeseries={"timeField": time_field, "granularity": "hours"},
            )

        collection = self.client[self.db_name][collection_name]
        records = df.to_dict(orient="records")

        if records:
            _ = await collection.insert_many(records)

    async def close(self) -> None:
        """
        Close the MongoDB client connection.
        Returns:
            None
        """
        self.client.close()
        logger.info("MongoDB connection closed.")
