# mongodb.py

import os
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime
import pandas as pd

def get_mongo_connection():
    """
    Creates a new MongoDB connection with freshly loaded environment variables.
    """
    # Force reload environment variables
    load_dotenv(override=True)
    
    # Get env vars
    MONGO_URI = os.getenv("MONGO_URI")
    MONGO_DB = os.getenv("MONGO_DB")
    
    print(f"Connecting to database: {MONGO_DB}")
    
    # Connect to MongoDB
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    
    return client, db

# Initialize connection
client, db = get_mongo_connection()

def refresh_connection():
    """
    Refreshes the MongoDB connection with updated environment variables.
    Call this function after updating your .env file.
    """
    global client, db
    client.close()  # Close existing connection
    client, db = get_mongo_connection()
    return "Connection refreshed with updated environment variables."

def get_collection(collection_name):
    """
    Returns a MongoDB collection by name.
    """
    return db[collection_name]

def insert_document(collection_name, doc):
    """
    Inserts a document into the specified collection.
    """
    collection = get_collection(collection_name)
    return collection.insert_one(doc)

def insert_many_documents(collection_name, docs):
    """
    Inserts multiple documents into the specified collection.
    """
    collection = get_collection(collection_name)
    return collection.insert_many(docs)

def find_documents(collection_name, query={}):
    """
    Finds documents in the specified collection matching the query.
    """
    collection = get_collection(collection_name)
    return list(collection.find(query))

def delete_documents(collection_name, query={}):
    """
    Deletes documents from the specified collection matching the query.
    """
    collection = get_collection(collection_name)
    return collection.delete_many(query)

def get_coin_timeseries(collection_name, coin_symbol, start_time=None, end_time=None):
    """
    Fetch time series data for a specific coin from a processed collection.
    - coin_symbol: str, e.g., "BTC"
    - start_time, end_time: datetime.datetime (optional)
    Returns list of dicts sorted by datetime.
    
    Example:
    data = get_coin_timeseries(
        collection_name="processed_coin_data",
        coin_symbol="BTC",
        start_time=datetime(2021, 1, 1),
        end_time=datetime(2023, 12, 31)
    )
    """
    collection = db[collection_name]

    query = {f"data.{coin_symbol}": {"$exists": True}}

    if start_time or end_time:
        query["datetime"] = {}
        if start_time:
            query["datetime"]["$gte"] = start_time
        if end_time:
            query["datetime"]["$lte"] = end_time

    projection = {
        "datetime": 1,
        f"data.{coin_symbol}": 1,
        "_id": 0
    }

    cursor = collection.find(query, projection).sort("datetime", 1)

    result = []
    for doc in cursor:
        result.append({
            "datetime": doc["datetime"],
            "close": doc["data"][coin_symbol].get("close"),
            "volume_usd": doc["data"][coin_symbol].get("volume_usd")
        })

    return result

def get_latest_timestamp(collection_name: str) -> datetime | None:
    """
    Get the most recent datetime value from a collection.

    Args:
        database_name (str): The name of the database.
        collection_name (str): The name of the collection.

    Returns:
        datetime | None: The latest timestamp found in the collection, or None if empty.
    """
    collection = db[collection_name]
    doc = collection.find_one(sort=[("datetime", -1)])
    if doc and "datetime" in doc:
        return doc["datetime"]
    return None

def load_collection_to_dataframe(collection_name: str) -> pd.DataFrame:
    # db = client[database_name]
    collection = db[collection_name]

    # Fetch all documents and sort by datetime ascending
    # cursor = collection.find().sort("datetime", 1)
    cursor = collection.find()
    
    # Convert to DataFrame
    df = pd.DataFrame(list(cursor))

    # Optional cleanup: remove MongoDB internal _id
    if "_id" in df.columns:
        df.drop(columns=["_id"], inplace=True)

    return df

def save_dataframe_to_collection(df: pd.DataFrame, collection_name: str, time_field: str = "datetime") -> None:
    df[time_field] = pd.to_datetime(df[time_field])

    # Create time series collection if not exists
    if collection_name not in db.list_collection_names():
        db.create_collection(
            collection_name,
            timeseries={
                "timeField": time_field,
                "granularity": "hours"  # Can be "minutes" or "seconds" if your data is finer
            }
        )

    collection = db[collection_name]

    # Convert to dict and insert
    data = df.to_dict(orient="records")
    if data:
        collection.insert_many(data)