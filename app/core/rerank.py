"""Cohere cross-encoder reranking client for precision document ranking."""

import logging
from typing import List, Optional
import httpx

from app.config import settings

logger = logging.getLogger(__name__)

COHERE_RERANK_URL = "https://api.cohere.com/v2/rerank"
COHERE_RERANK_V1_URL = "https://api.cohere.com/v1/rerank"


async def rerank_documents_with_cohere(
    query: str,
    documents: List[str],
    top_n: Optional[int] = None,
) -> List[str]:
    """Rerank candidate document chunks using Cohere's Cross-Encoder Rerank API.

    Args:
        query: The search query to compare candidate documents against.
        documents: List of retrieved context strings to be reranked.
        top_n: Maximum number of top documents to return after reranking.

    Returns:
        List of reranked document strings in descending order of relevance.
    """
    if not documents:
        return []

    target_top_n = top_n if top_n is not None else settings.COHERE_TOP_N
    target_top_n = min(target_top_n, len(documents))

    api_key = settings.COHERE_API_KEY.strip()
    if not api_key:
        logger.debug("Cohere API key not configured; returning un-reranked top candidate documents.")
        return documents[:target_top_n]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Client-Name": "MultiTenantAgenticRAG",
    }
    payload = {
        "model": settings.COHERE_RERANK_MODEL,
        "query": query,
        "documents": documents,
        "top_n": target_top_n,
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(COHERE_RERANK_URL, json=payload, headers=headers)
            if response.status_code in (400, 404):
                # Fallback to v1 endpoint if model or v2 payload differs
                response = await client.post(COHERE_RERANK_V1_URL, json=payload, headers=headers)

            if response.is_success:
                data = response.json()
                results = data.get("results", [])
                reranked_docs = []
                for item in results:
                    idx = item.get("index")
                    if idx is not None and 0 <= idx < len(documents):
                        reranked_docs.append(documents[idx])

                if reranked_docs:
                    logger.info(
                        "✅ Cohere Rerank successfully re-ordered %d documents.",
                        len(reranked_docs),
                    )
                    return reranked_docs
            else:
                logger.warning(
                    "Cohere Rerank API returned status %d: %s. Using default vector ordering.",
                    response.status_code,
                    response.text[:200],
                )
    except Exception as exc:
        logger.warning(
            "Cohere Rerank failed with exception: %s. Using default vector ordering.",
            exc,
        )

    return documents[:target_top_n]
