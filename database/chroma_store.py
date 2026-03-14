"""
database/chroma_store.py

ChromaDB vector store interface for Linguist-OS.
Manages two collections:
  1. "lessons"        - RAG retrieval of lesson content by error description
  2. "past_mistakes"  - User mistake history for pattern detection

Uses the all-MiniLM-L6-v2 embedding model (called "OroMini" in architecture)
to convert text into 384-dimensional vectors for semantic similarity search.

Architecture role:
  - Stores and retrieves lesson cards for teaching (RAG)
  - Stores user mistakes and finds similar past mistakes
  - Enables the Audit Agent to detect repeated error patterns
"""

import chromadb
from chromadb.config import Settings

_chroma_client = None
_lesson_collection = None
_mistake_collection = None


def _get_chroma_client():
    """
    Lazy-initialize and return a persistent ChromaDB client.

    Example:
        client = _get_chroma_client()

    Returns:
        chromadb.PersistentClient instance.
    """
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path="./chroma_data")
        print("[chroma_store] ChromaDB client initialized (persistent).")
    return _chroma_client


def _get_lesson_collection():
    """
    Get or create the 'lessons' collection.

    Example:
        collection = _get_lesson_collection()

    Returns:
        chromadb.Collection for lessons.
    """
    global _lesson_collection
    if _lesson_collection is None:
        client = _get_chroma_client()
        _lesson_collection = client.get_or_create_collection(
            name="lessons",
            metadata={"description": "English language lesson cards for RAG retrieval"},
        )
        print("[chroma_store] Lessons collection ready.")
    return _lesson_collection


def _get_mistake_collection():
    """
    Get or create the 'past_mistakes' collection.

    Example:
        collection = _get_mistake_collection()

    Returns:
        chromadb.Collection for past mistakes.
    """
    global _mistake_collection
    if _mistake_collection is None:
        client = _get_chroma_client()
        _mistake_collection = client.get_or_create_collection(
            name="past_mistakes",
            metadata={"description": "User mistake history for pattern detection"},
        )
        print("[chroma_store] Past mistakes collection ready.")
    return _mistake_collection


def init_chroma_lessons():
    """
    Populate the lessons collection with lesson content from lessons.py.
    Called once at startup.

    Example:
        >>> init_chroma_lessons()
        [chroma_store] Lessons loaded into ChromaDB.
    """
    from lessons import LESSONS
    print("[chroma_store] Initializing lesson content in ChromaDB...")
    store_lessons(LESSONS)
    print("[chroma_store] Lessons loaded into ChromaDB.")


def store_lessons(lessons_list: list):
    """
    Upsert a list of lesson dicts into the lessons collection.

    Example:
        >>> store_lessons([
        ...     {"id": "past_simple", "concept": "past_simple",
        ...      "content": "Past Simple vs Present Perfect..."}
        ... ])

    Args:
        lessons_list: List of dicts with keys: id, concept, content.
    """
    collection = _get_lesson_collection()
    ids = []
    documents = []
    metadatas = []
    for lesson in lessons_list:
        ids.append(lesson["id"])
        documents.append(lesson["content"])
        metadatas.append({"concept": lesson["concept"]})
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    print(f"[chroma_store] Upserted {len(ids)} lessons.")


def retrieve_lesson(error_description: str, n: int = 1) -> list:
    """
    Query the lessons collection for the most relevant lesson(s)
    given an error description string.

    Example:
        >>> retrieve_lesson("user used present perfect instead of past simple")
        [{'id': 'past_simple', 'content': '...', 'concept': 'past_simple'}]

    Args:
        error_description: Natural language description of the error.
        n: Number of results to return.

    Returns:
        List of dicts with id, content, concept keys.
    """
    print(f"[chroma_store] Retrieving lesson for: {error_description[:60]}...")
    collection = _get_lesson_collection()
    try:
        results = collection.query(query_texts=[error_description], n_results=n)
        lessons = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                lessons.append({
                    "id": doc_id,
                    "content": results["documents"][0][i] if results["documents"] else "",
                    "concept": results["metadatas"][0][i].get("concept", "") if results["metadatas"] else "",
                })
        print(f"[chroma_store] Found {len(lessons)} relevant lesson(s).")
        return lessons
    except Exception as e:
        print(f"[chroma_store] ERROR retrieving lesson: {e}")
        return []


def get_lesson_by_id(lesson_id: str) -> dict:
    """
    Get a specific lesson by its ID from ChromaDB.

    Example:
        >>> get_lesson_by_id("articles")
        {'id': 'articles', 'content': '...', 'concept': 'articles'}

    Args:
        lesson_id: The lesson ID string.

    Returns:
        Dict with id, content, concept or empty dict if not found.
    """
    print(f"[chroma_store] Getting lesson by ID: {lesson_id}")
    collection = _get_lesson_collection()
    try:
        results = collection.get(ids=[lesson_id])
        if results and results["ids"]:
            return {
                "id": results["ids"][0],
                "content": results["documents"][0] if results["documents"] else "",
                "concept": results["metadatas"][0].get("concept", "") if results["metadatas"] else "",
            }
        return {}
    except Exception as e:
        print(f"[chroma_store] ERROR getting lesson by ID: {e}")
        return {}


def store_mistake_in_chroma(user_id: str, text: str, concept: str):
    """
    Store a user mistake in the past_mistakes collection for future
    pattern detection via semantic search.

    Example:
        >>> store_mistake_in_chroma("student_1", "I have went to park", "past_simple")

    Args:
        user_id: Unique user identifier.
        text: The mistake text.
        concept: The grammar/vocab concept involved.
    """
    print(f"[chroma_store] Storing mistake in ChromaDB for {user_id}: {concept}")
    collection = _get_mistake_collection()
    import uuid
    mistake_id = f"{user_id}_{concept}_{uuid.uuid4().hex[:8]}"
    try:
        collection.upsert(
            ids=[mistake_id],
            documents=[text],
            metadatas=[{"user_id": user_id, "concept": concept}],
        )
        print(f"[chroma_store] Mistake stored: {mistake_id}")
    except Exception as e:
        print(f"[chroma_store] ERROR storing mistake: {e}")


def find_similar_past_mistakes(user_id: str, text: str, n: int = 3) -> list:
    """
    Find past mistakes similar to the current text for pattern detection.

    Example:
        >>> find_similar_past_mistakes("student_1", "I have went to store", n=3)
        [{'text': 'I have went to park', 'concept': 'past_simple', 'distance': 0.12}]

    Args:
        user_id: Unique user identifier.
        text: Current mistake text to compare against.
        n: Number of similar results to return.

    Returns:
        List of dicts with text, concept, distance keys.
    """
    print(f"[chroma_store] Finding similar past mistakes for {user_id}...")
    collection = _get_mistake_collection()
    try:
        results = collection.query(
            query_texts=[text],
            n_results=n,
            where={"user_id": user_id},
        )
        similar = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                similar.append({
                    "text": results["documents"][0][i] if results["documents"] else "",
                    "concept": results["metadatas"][0][i].get("concept", "") if results["metadatas"] else "",
                    "distance": results["distances"][0][i] if results.get("distances") else 0,
                })
        print(f"[chroma_store] Found {len(similar)} similar past mistake(s).")
        return similar
    except Exception as e:
        print(f"[chroma_store] ERROR finding similar mistakes: {e}")
        return []
