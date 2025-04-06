import os
import sys
import logging
from datetime import datetime, timedelta
from motor.motor_asyncio import AsyncIOMotorClient
from aiohttp import ClientSession
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel
from pymongo import MongoClient

_ = load_dotenv()

mongo_uri = os.getenv("MONGO_URI")
logger = logging.getLogger("mongodb")

class MongoDB(BaseModel):
    client: Any = None

    def __init__(self, **data):
        super().__init__(**data)
        self.client = MongoClient(mongo_uri)

    def insert(self, database_name: str, collection_name: str, data:dict[str,Any]) -> str|Any:
        db = self.client[database_name]
        collection = db[collection_name]
        result = collection.insert_one(data)
        return result.inserted_id
    def get_existing_urls(self, database_name: str, collection_name: str) -> list[str]:
        db = self.client[database_name]
        collection = db[collection_name]
        urls = collection.distinct("url")
        return urls


class AsyncMongoDB:
    def __init__(self, host: str, database: str, collection: str):
        """
        Initializes the MongoDB client with the provided host.

        Args:
            host (str): The host address of the MongoDB server.
        """
        self.client = AsyncIOMotorClient(host)
        self.database = database
        self.db = self.client[database]
        self.collection = self.db[collection]

    async def count(self):
        return await self.collection.count_documents({})

    async def get_data_from_id(self, id: str) -> dict:
        """
        Retrieves a document from the specified collection with the given ID.
        Args:
            id (str): The ID of the document to retrieve.
        Returns:
            dict: The document with the specified ID.
        """
        if await self.collection.find_one({"_id": id}) is not None:
            return await self.collection.find_one({"_id": id})
        elif await self.collection.find_one({"id": id}) is not None:
            return await self.collection.find_one({"id": id})
        else:
            return {}

    async def get_data_from_timestamp_interval(
        self, timestamp: datetime, interval: str
    ) -> dict:
        search_result = await self.collection.find_one(
            {"timestamp": timestamp, "time_interval": interval}
        )
        if isinstance(search_result, dict):
            return search_result
        else:
            return {}

    async def upsert_document_by_timestamp_interval(
        self, timestamp: datetime, interval: str, document: dict
    ):
        """
        Upsert a document into the specified collection with the given ID.
        Args:

            timestamp (timestamp): The ID of the document to insert.
            interval (str): the time interval of the document.
            document (dict): The document to insert.
        """
        await self.collection.update_one(
            {"timestamp": timestamp, "time_interval": interval},
            {"$set": document},
            upsert=True,
        )

    async def update_top_keywords(
        self, timestamp: datetime, interval: str, top_keywords: list
    ):
        await self.collection.update_one(
            {"timestamp": timestamp, "time_interval": interval},
            {"$set": {"top_keywords": top_keywords}},
        )

    async def get_data_from_timestamp(self, timestamp: datetime) -> dict:
        """
        Retrieves documents from the specified collection that were inserted or updated after the given timestamp.

        Args:
            timestamp (datetime): The timestamp to filter documents.

        Returns:
            data: The document with the specified timestamp. Returns to the previous timestamp if the current timestamp is not found.
        """
        data = await self.collection.find_one({"timestamp": timestamp})
        if data is None:
            data = await self.collection.find_one(
                {"timestamp": timestamp - timedelta(days=1)}
            )
            if data is None:
                return {}
        return data

    async def get_data_from_key(self, key: str, value: str) -> dict:
        search_result = await self.collection.find_one({key: value})
        if isinstance(search_result, dict):
            return search_result
        else:
            return {}

    async def upsert_document_by_id(self, id: str, document: dict):
        """
        Upsert a document into the specified collection with the given ID.
        Args:

            id (str): The ID of the document to insert.
            document (dict): The document to insert.
        """
        await self.collection.update_one({"_id": id}, {"$set": document}, upsert=True)

    async def upsert_document_by_keyword(self, document: dict):
        await self.collection.update_one(
            {"keyword": document["keyword"]}, {"$set": document}, upsert=True
        )

    async def retrieve_all_documents(self) -> list:
        search_result = await self.collection.find({})
        return list(search_result)

    async def close(self):
        """
        Closes the MongoDB client connection gracefully.
        """
        self.client.close()
        logger.info("MongoDB client connection closed.")

