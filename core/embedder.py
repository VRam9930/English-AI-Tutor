"""
core/embedder.py

Text embedding module for Linguist-OS using the all-MiniLM-L6-v2 model
(referred to as "OroMini" in the architecture).

This is STEP 2 of the pipeline: converts user text into a
384-dimensional vector embedding for semantic similarity search
in ChromaDB.

Architecture role:
  - Encodes user text into dense vectors
  - Powers RAG retrieval of relevant lessons
  - Enables semantic matching of past mistakes
"""

_model = None


def _get_model():
    """
    Lazy-load the sentence-transformers model to avoid startup delay
    if embeddings aren't needed immediately.

    Example:
        model = _get_model()
        # model is a SentenceTransformer instance

    Returns:
        SentenceTransformer model (all-MiniLM-L6-v2).
    """
    global _model
    if _model is None:
        print("[embedder] Loading OroMini (all-MiniLM-L6-v2) model...")
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        print("[embedder] OroMini model loaded successfully.")
    return _model


def embed_text(text: str) -> list:
    """
    Convert a text string to a 384-dimensional embedding vector.

    Example:
        >>> vec = embed_text("I went to the park yesterday")
        >>> len(vec)
        384
        >>> type(vec[0])
        <class 'float'>

    Args:
        text: The text to embed.

    Returns:
        List of 384 floats representing the text embedding.
    """
    print(f"[embedder] Embedding text: '{text[:60]}...'")
    model = _get_model()
    embedding = model.encode(text)
    result = embedding.tolist()
    print(f"[embedder] Generated {len(result)}-dimensional embedding.")
    return result


def embed_batch(texts: list) -> list:
    """
    Convert a batch of text strings to embedding vectors.

    Example:
        >>> vecs = embed_batch(["Hello world", "Goodbye world"])
        >>> len(vecs)
        2
        >>> len(vecs[0])
        384

    Args:
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors (each a list of 384 floats).
    """
    print(f"[embedder] Batch embedding {len(texts)} texts...")
    model = _get_model()
    embeddings = model.encode(texts)
    results = [e.tolist() for e in embeddings]
    print(f"[embedder] Generated {len(results)} embeddings.")
    return results
