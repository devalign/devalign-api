import asyncio
import os
import sys

# Add src to python path
sys.path.append(os.path.abspath("."))

from src.config import settings
from src.ml_engine.infrastructure.embeddings import get_embedding_service


async def main():
    print("Getting embedding service...")
    print(f"EMBEDDING_PROVIDER: {settings.EMBEDDING_PROVIDER}")
    print(f"EMBEDDING_MODEL: {settings.EMBEDDING_MODEL}")
    service = get_embedding_service()
    print(f"Service class: {service.__class__.__name__}")
    print("Embedding text...")
    try:
        vector = await asyncio.wait_for(service.embed_text("test"), timeout=30.0)
        print(f"Embedding success! Vector len: {len(vector)}")
    except TimeoutError:
        print("TIMEOUT: Embedding generation hung and took more than 30 seconds.")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e!s}")


if __name__ == "__main__":
    asyncio.run(main())
