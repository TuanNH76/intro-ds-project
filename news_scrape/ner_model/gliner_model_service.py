import logging
from enum import Enum
from typing import Any

import httpx
import spacy

logger = logging.getLogger(__name__)
gliner_service_url = "https://dev.loomix.ai/v1/api/model"
GLINER_SUPPORTED_LABELS: list[str] = [
    "Token Cryptocurrency",
    "Partially Algorithmic Stablecoin",
    "Lending",
    "Uncollateralized Lending",
    "Lending Pool",
    "NFT Lending",
    "RWA Lending",
    "Collateralized debt position CDP",
    "CDP Manager",
    "Liquidity Automation",
    "Liquidity manager",
    "Staking",
    "Staking Pool",
    "Restaking",
    "Liquid Staking",
    "Liquid Restaking",
    "Yield",
    "Yield Aggregator",
    "RWA",
    "Launchpad",
    "Leveraged Farming",
    "Farm",
    "Reserve Currency",
    "Indexes",
    "Synthetics",
    "Derivatives",
    "Liquidations",
    "Basis Trading",
    "Exchange",
    "Cexes",
    "Dexes",
    "Prediction Market",
    "Trading App",
    "NFT Marketplace",
    "DEX Aggregator",
    "Chain",
    "Bridge",
    "Cross Chain",
    "Wallet Address",
    "Contract Address",
    "DeFi Project",
    "NftFi",
    "Algo-Stables",
    "Governance Incentives",
    "Telegram Bot",
    "Normal Entity",
    "Business",
    "People",
    "Technology",
    "Concept",
]
NER_THRESHOLD: float = 0.5


class Endpoint(str, Enum):
    NER = "/gliner"


labels = [
    "Token Cryptocurrency",
    "Lending",
    "Chain",
    "Exchange",
    "Bridge",
    "Staking",
    "Yield Aggregator",
    "Launchpad",
    "Normal Entity",
    "Business",
    "Wallet Address",
    "DeFi Project",
    "Concept",
    "NftFi",
    "Prediction Market",
    "Leveraged Farming",
    "Staking Pool",
    "Restaking",
    "NFT Marketplace",
    "Dexes",
    "Farm",
    "DEX Aggregator",
    "Cross Chain",
    "Uncollateralized Lending",
    "Partially Algorithmic Stablecoin",
    "Synthetics",
    "Derivatives",
    "Liquidations",
    "Basis Trading",
    "NFT Lending",
    "Cexes",
    "Gaming",
    "Trading App",
    "Liquidity manager",
    "Liquid Staking",
    "Yield",
    "Liquid Restaking",
    "RWA",
    "RWA Lending",
    "People",
    "Contract Address",
    "NFT Collection",
    "Decentralized Exchange",
]


nlp = spacy.load("en_core_web_sm")


def split_text_into_chunks(text, max_tokens=100):
    """
    Chia văn bản thành các chunk sao cho mỗi chunk không vượt quá max_tokens token.
    Văn bản được tách thành từng câu bằng spaCy, sau đó cộng dồn các câu cho đến khi đạt giới hạn token.

    Args:
        text (str): Văn bản đầu vào.
        max_tokens (int): Số token tối đa trong mỗi chunk.

    Returns:
        List[str]: Danh sách các chunk đã chia.
    """

    doc = nlp(text)

    # Tách câu giữ nguyên khoảng trắng cuối câu
    sentences = [sent.text_with_ws for sent in doc.sents]

    chunks = []
    current_chunk = ""
    current_token_count = 0
    current_offset = 0
    for sentence in sentences:
        num_tokens = len(nlp(sentence))  # Đếm số token trong câu

        # Nếu thêm câu này vào mà vượt quá max_tokens, lưu chunk hiện tại và tạo chunk mới
        if current_token_count + num_tokens > max_tokens and current_chunk:
            chunks.append((current_chunk, current_offset))
            current_offset += len(current_chunk)
            current_chunk = ""
            current_token_count = 0

        # Thêm câu vào chunk hiện tại
        current_chunk += sentence
        current_token_count += num_tokens

    # Thêm chunk cuối nếu còn dữ liệu
    if current_chunk:
        chunks.append((current_chunk, current_offset))

    return chunks


def update_entity_positions(entities, chunk_offset):
    """
    Cập nhật vị trí thực thể từ chunk về vị trí trong văn bản gốc.

    Args:
        entities (List[Dict]): Danh sách thực thể từ mô hình.
        chunk_offset (int): Offset của chunk trong văn bản gốc.

    Returns:
        List[Dict]: Danh sách thực thể với vị trí chính xác trong văn bản gốc.
    """
    updated_entities = []
    for entity in entities:
        entity["start"] += chunk_offset
        entity["end"] += chunk_offset
        updated_entities.append(entity)
    return updated_entities


class GlinerModelService:
    def __init__(self, url: str = gliner_service_url):
        self.url: str = url

    async def post(
        self, endpoint: str, payload: dict[str, Any]
    ) -> list[dict[str, Any]]:
        data: list[dict[str, Any]] = []
        logger.info(f"POST request to {endpoint}")
        # print(f"POST request to {endpoint} with payload: {payload}")
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.url}/{endpoint.lstrip('/')}", json=payload
                )
                _ = response.raise_for_status()
                data = response.json()
        except httpx.RequestError as e:
            logger.error(f"Request error at {endpoint}: {e}")
            print(f"Request error at {endpoint}: {e}")
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error at {endpoint}: {e}")
            print(f"HTTP error at {endpoint}: {e}")
        except ValueError as e:
            logger.error(f"Failed to decode JSON from {endpoint}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error at {endpoint}: {e}")
            print(f"Unexpected error at {endpoint}: {e}")
        return data

    async def _ner(self, text: str) -> list[dict[str, Any]]:
        """
        Call the NER endpoint of the Gliner model service.
        :param payload: The payload to send to the NER endpoint.
        :return: The response from the NER endpoint.
        """
        payload = {
            "text": text,
            "labels": labels,
            "threshold": 0.5,
            "multi_label": False,
        }
        data = await self.post(Endpoint.NER, payload)
        if not data:
            return []
        return data

    async def predict_text(self, text):
        chunks = split_text_into_chunks(text)
        chunked = ""
        for c in chunks:
            # print(f"chunk: {c}")
            chunked += c[0]

        sum_entity = []
        for c in chunks:
            text_t = c[0]
            entities = await self._ner(text_t)
            entities = update_entity_positions(entities, c[1])
            sum_entity.extend(entities)

        return sum_entity


if __name__ == "__main__":
    import asyncio

    # Example usage
    gliner_service = GlinerModelService()
    text = "Bitcoin is a cryptocurrency."

    async def main():
        ner_results = await gliner_service.ner(text)
        print(ner_results)

    asyncio.run(main())
