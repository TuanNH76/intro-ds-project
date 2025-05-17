import ast
from collections import defaultdict
from datetime import datetime, timezone
from typing import List

top10_crypto = {
    "btc": ["btc", "bitcoin"],
    "eth": ["eth", "ethereum"],
    "xrp": ["xrp", "ripple"],
    "ltc": ["ltc", "litecoin"],
    "doge": ["doge", "dogecoin"],
    "sol": ["sol", "solana"],
    "dot": ["dot", "polkadot"],
    "avax": ["avax", "avalanche"],
    "link": ["link", "chainlink"],
    "bnb": ["bnb", "binance"],
    "ada": ["ada", "cardano"],
    "xlm": ["xlm", "stellar"],
    "matic": ["matic", "polygon"],
    "uni": ["uni", "uniswap"],
    "atom": ["atom", "cosmos"],
}

import re


def is_entity_mentioning_crypto(entity_text: str, crypto_names: List[str]) -> bool:
    entity_text = entity_text.lower()
    entity_tokens = set(re.findall(r"\b\w+\b", entity_text))

    for name in crypto_names:
        if name.lower() in entity_tokens:
            return True
    return False


def parse_published_time(value) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:
            return None
    return None


def count_crypto_mentions_hourly(documents, top10_crypto=top10_crypto):
    """
    Đếm số lần nhắc đến top crypto tokens theo từng giờ của mỗi ngày.

    Args:
        documents: Danh sách bài báo, mỗi bài chứa 'entities' và 'published_time'.
        top10_crypto: Dict mapping từ token symbol -> [symbol, fullname].

    Returns:
        Danh sách dict gồm 'hour' (YYYY-MM-DD HH), 'mentioned_frequency', và 'time' (datetime UTC chuẩn hóa theo giờ).
    """
    hourly_stats = defaultdict(lambda: {token: 0 for token in top10_crypto.keys()})
    hour_times = {}  # Map from hour_str to datetime (UTC, floored)
    processed_count = 0

    for doc in documents:
        published_time = doc.get("published_time")
        if not published_time:
            continue

        # Parse nếu là string
        if isinstance(published_time, str):
            try:
                ts = parse_published_time(published_time)
            except Exception:
                continue
        else:
            ts = published_time

        # Đảm bảo timezone UTC
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        else:
            ts = ts.astimezone(timezone.utc)

        # Làm tròn về đầu giờ
        ts_hour = ts.replace(minute=0, second=0, microsecond=0)
        hour_str = ts_hour.strftime("%Y-%m-%d %H")

        # Lưu datetime chuẩn để gắn vào kết quả
        hour_times[hour_str] = ts_hour

        # Parse NER
        entities = doc.get("ner", "")
        if entities:
            try:
                ner_list = ast.literal_eval(entities)
            except Exception:
                ner_list = []
        else:
            ner_list = []

        for entity in ner_list:
            entity_text = entity.get("text", "")
            for token, names in top10_crypto.items():
                if is_entity_mentioning_crypto(entity_text, names):
                    hourly_stats[hour_str][token] += 1

        processed_count += 1

    # Kết quả
    results = []
    for hour_str, freq in sorted(hourly_stats.items()):
        results.append(
            {
                "hour": hour_str,
                "mentioned_frequency": freq,
                "time": hour_times[hour_str],  # datetime object chuẩn UTC + rounded
            }
        )

    print(f"Processed {processed_count} documents.")
    return results
