# modules/retriever.py

import numpy as np
import faiss
from typing import List
from google.genai import types  # make sure google-genai is installed


def _embed_query(query: str, client) -> np.ndarray:
    """Embed a single query string using Gemini embeddings."""
    # Single string or list both valid; here we use a single string [web:9]
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query,
        # Optional: reduce dimensionality if needed
        # config=types.EmbedContentConfig(output_dimensionality=768),
    )
    vec = np.array(response.embeddings[0].values, dtype="float32")
    # Normalize for cosine similarity with IndexFlatIP
    faiss.normalize_L2(vec.reshape(1, -1))
    return vec


def search_index(
    query: str,
    client,
    index: faiss.IndexFlatIP,
    texts: List[str],
    top_k: int = 3,
) -> List[str]:
    """
    Embed the query, search FAISS index, and return top_k text snippets.
    """
    if not texts:
        return []

    q_vec = _embed_query(query, client).reshape(1, -1)
    # FAISS search: distances (scores), indices
    scores, indices = index.search(q_vec, top_k)

    result_snippets = []
    for idx in indices[0]:
        if 0 <= idx < len(texts):
            result_snippets.append(texts[idx])

    return result_snippets

