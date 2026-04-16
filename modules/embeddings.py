# modules/embeddings.py

import numpy as np
import faiss
from typing import List


def embed_texts(texts: List[str], client) -> np.ndarray:
    """
    Generate embeddings for a list of texts using Gemini embeddings.

    Parameters
    ----------
    texts : list of str
        Text chunks to embed.
    client : genai.Client
        Shared Google Gen AI client from configure_gemini().

    Returns
    -------
    np.ndarray
        2D array of shape (len(texts), dim) with float32 embeddings.
    """
    if not texts:
        raise ValueError("embed_texts: 'texts' list is empty")

    # Ensure all elements are strings
    texts = [str(t) for t in texts]

    # Call Gemini embeddings API (batch)
    # Docs: ai.google.dev/gemini-api/docs/embeddings [web:1][web:6]
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts,
    )

    # response.embeddings is a list of embedding objects with .values [web:1][web:6]
    vectors = np.array(
        [emb.values for emb in response.embeddings],
        dtype="float32",
    )

    return vectors


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS index for cosine-similarity search on the embeddings.

    We use inner product + L2-normalization to approximate cosine similarity. [web:14][web:16]

    Parameters
    ----------
    embeddings : np.ndarray
        2D float32 array of shape (n_items, dim).

    Returns
    -------
    faiss.IndexFlatIP
        FAISS index ready for search.
    """
    if embeddings.ndim != 2:
        raise ValueError("build_faiss_index: embeddings must be 2D (n, dim)")

    # Normalize to unit length for cosine similarity
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product index [web:14]

    index.add(embeddings)
    return index

