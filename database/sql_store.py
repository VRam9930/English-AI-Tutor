"""
database/sql_store.py

Supabase PostgreSQL interface for Linguist-OS.
Handles all persistent relational data: users, mistakes, conversations,
and audit reports. Every function wraps Supabase calls in try/except
with sensible fallback values so the app never crashes on DB errors.

Architecture role:
  - Stores user profiles and mastery scores
  - Records every mistake for long-term tracking
  - Saves conversation history for context
  - Persists full audit reports for review
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

_client = None

DEFAULT_MASTERY = {
    "past_simple": 0.5,
    "present_perfect": 0.5,
    "articles": 0.5,
    "prepositions": 0.5,
    "vocabulary": 0.5,
    "subject_verb_agreement": 0.5,
    "irregular_verbs": 0.5,
}


def _get_client():
    """
    Lazy-initialize and return the Supabase client.

    Example:
        client = _get_client()
        # client is a supabase.Client instance

    Returns:
        supabase.Client or None if credentials are missing.
    """
    global _client
    if _client is not None:
        return _client
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("[sql_store] WARNING: Supabase credentials not set. Running in offline mode.")
        return None
    try:
        from supabase import create_client
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("[sql_store] Supabase client created successfully.")
        return _client
    except Exception as e:
        print(f"[sql_store] ERROR creating Supabase client: {e}")
        return None


def init_db():
    """
    Initialize the database connection and print a status message.

    Example:
        >>> init_db()
        [sql_store] Database connected and ready.
    """
    client = _get_client()
    if client:
        print("[sql_store] Database connected and ready.")
    else:
        print("[sql_store] Database running in offline/fallback mode.")


def get_user(user_id: str) -> dict:
    """
    Retrieve a user profile by ID. Creates the user if they don't exist.

    Example:
        >>> get_user("student_1")
        {'user_id': 'student_1', 'name': 'Learner', 'mastery': {...}}

    Args:
        user_id: Unique user identifier.

    Returns:
        dict with user_id, name, mastery keys.
    """
    print(f"[sql_store] Getting user: {user_id}")
    client = _get_client()
    if not client:
        return {"user_id": user_id, "name": "Learner", "mastery": DEFAULT_MASTERY.copy()}
    try:
        resp = client.table("users").select("*").eq("user_id", user_id).execute()
        if resp.data and len(resp.data) > 0:
            user = resp.data[0]
            if isinstance(user.get("mastery"), str):
                user["mastery"] = json.loads(user["mastery"])
            print(f"[sql_store] Found existing user: {user_id}")
            return user
        else:
            print(f"[sql_store] User not found, creating: {user_id}")
            return create_user(user_id)
    except Exception as e:
        print(f"[sql_store] ERROR getting user: {e}")
        return {"user_id": user_id, "name": "Learner", "mastery": DEFAULT_MASTERY.copy()}


def create_user(user_id: str) -> dict:
    """
    Insert a new user with default mastery scores.

    Example:
        >>> create_user("student_42")
        {'user_id': 'student_42', 'name': 'Learner', 'mastery': {...}}

    Args:
        user_id: Unique user identifier.

    Returns:
        dict with the new user record.
    """
    print(f"[sql_store] Creating user: {user_id}")
    client = _get_client()
    new_user = {
        "user_id": user_id,
        "name": "Learner",
        "mastery": json.dumps(DEFAULT_MASTERY),
    }
    if not client:
        return {"user_id": user_id, "name": "Learner", "mastery": DEFAULT_MASTERY.copy()}
    try:
        resp = client.table("users").insert(new_user).execute()
        result = resp.data[0] if resp.data else new_user
        if isinstance(result.get("mastery"), str):
            result["mastery"] = json.loads(result["mastery"])
        print(f"[sql_store] User created: {user_id}")
        return result
    except Exception as e:
        print(f"[sql_store] ERROR creating user: {e}")
        return {"user_id": user_id, "name": "Learner", "mastery": DEFAULT_MASTERY.copy()}


def update_mastery(user_id: str, concept: str, new_score: float) -> bool:
    """
    Update a single concept mastery score in the user's mastery JSONB.

    Example:
        >>> update_mastery("student_1", "articles", 0.35)
        True

    Args:
        user_id: Unique user identifier.
        concept: The mastery concept key (e.g., 'articles').
        new_score: New score value, clamped between 0.0 and 1.0.

    Returns:
        True if update succeeded, False otherwise.
    """
    new_score = max(0.0, min(1.0, new_score))
    print(f"[sql_store] Updating mastery for {user_id}: {concept} -> {new_score:.2f}")
    client = _get_client()
    if not client:
        return False
    try:
        user = get_user(user_id)
        mastery = user.get("mastery", DEFAULT_MASTERY.copy())
        if isinstance(mastery, str):
            mastery = json.loads(mastery)
        mastery[concept] = round(new_score, 2)
        client.table("users").update({"mastery": json.dumps(mastery)}).eq("user_id", user_id).execute()
        print(f"[sql_store] Mastery updated: {concept} = {new_score:.2f}")
        return True
    except Exception as e:
        print(f"[sql_store] ERROR updating mastery: {e}")
        return False


def save_mistake(user_id: str, original_text: str, error_dict: dict) -> bool:
    """
    Insert a mistake record into the mistakes table.

    Example:
        >>> save_mistake("student_1", "I have went", {
        ...     "wrong": "have went",
        ...     "correct": "went",
        ...     "rule": "past_simple",
        ...     "concept": "past_simple",
        ...     "severity": "critical"
        ... })
        True

    Args:
        user_id: Unique user identifier.
        original_text: The full original text from the user.
        error_dict: Dict with keys: wrong, correct, rule, concept, severity.

    Returns:
        True if insert succeeded, False otherwise.
    """
    print(f"[sql_store] Saving mistake for {user_id}: {error_dict.get('wrong', 'N/A')}")
    client = _get_client()
    if not client:
        return False
    try:
        row = {
            "user_id": user_id,
            "original_text": original_text,
            "wrong_part": error_dict.get("wrong", ""),
            "correct_part": error_dict.get("correct", ""),
            "rule_broken": error_dict.get("rule", ""),
            "concept": error_dict.get("concept", ""),
            "severity": error_dict.get("severity", "minor"),
        }
        client.table("mistakes").insert(row).execute()
        print(f"[sql_store] Mistake saved: {row['wrong_part']} -> {row['correct_part']}")
        return True
    except Exception as e:
        print(f"[sql_store] ERROR saving mistake: {e}")
        return False


def get_recent_mistakes(user_id: str, limit: int = 10) -> list:
    """
    Retrieve the most recent mistakes for a user.

    Example:
        >>> get_recent_mistakes("student_1", limit=5)
        [{'wrong_part': 'have went', 'correct_part': 'went', ...}, ...]

    Args:
        user_id: Unique user identifier.
        limit: Maximum number of mistakes to return.

    Returns:
        List of mistake dicts, newest first.
    """
    print(f"[sql_store] Getting recent mistakes for {user_id} (limit={limit})")
    client = _get_client()
    if not client:
        return []
    try:
        resp = (
            client.table("mistakes")
            .select("*")
            .eq("user_id", user_id)
            .order("timestamp", desc=True)
            .limit(limit)
            .execute()
        )
        print(f"[sql_store] Found {len(resp.data)} recent mistakes")
        return resp.data or []
    except Exception as e:
        print(f"[sql_store] ERROR getting mistakes: {e}")
        return []


def get_mistake_count_by_concept(user_id: str) -> dict:
    """
    Count how many mistakes the user has made per concept.

    Example:
        >>> get_mistake_count_by_concept("student_1")
        {'past_simple': 5, 'articles': 3, 'prepositions': 1}

    Args:
        user_id: Unique user identifier.

    Returns:
        Dict mapping concept names to counts.
    """
    print(f"[sql_store] Getting mistake counts for {user_id}")
    client = _get_client()
    if not client:
        return {}
    try:
        resp = (
            client.table("mistakes")
            .select("concept")
            .eq("user_id", user_id)
            .execute()
        )
        counts = {}
        for row in (resp.data or []):
            concept = row.get("concept", "unknown")
            counts[concept] = counts.get(concept, 0) + 1
        print(f"[sql_store] Mistake counts: {counts}")
        return counts
    except Exception as e:
        print(f"[sql_store] ERROR getting mistake counts: {e}")
        return {}


def save_audit_report(user_id: str, report_dict: dict) -> bool:
    """
    Insert an audit report record.

    Example:
        >>> save_audit_report("student_1", {
        ...     "analyzed_text": "I have went",
        ...     "verdict": "lesson",
        ...     "severity": "critical",
        ...     "overall_score": 0.45,
        ...     "errors_found": 1,
        ...     "primary_issue": "past_simple",
        ...     "ai_analysis": "User confused past simple with present perfect.",
        ...     "audit_summary": "Critical grammar error detected.",
        ...     "full_report": {}
        ... })
        True

    Args:
        user_id: Unique user identifier.
        report_dict: Dict with audit report fields.

    Returns:
        True if insert succeeded, False otherwise.
    """
    print(f"[sql_store] Saving audit report for {user_id}")
    client = _get_client()
    if not client:
        return False
    try:
        row = {
            "user_id": user_id,
            "analyzed_text": report_dict.get("analyzed_text", ""),
            "verdict": report_dict.get("verdict", "conversation"),
            "severity": report_dict.get("severity", "clean"),
            "overall_score": report_dict.get("overall_score", 1.0),
            "errors_found": report_dict.get("errors_found", 0),
            "primary_issue": report_dict.get("primary_issue", "none"),
            "ai_analysis": report_dict.get("ai_analysis", ""),
            "audit_summary": report_dict.get("audit_summary", ""),
            "full_report": json.dumps(report_dict.get("full_report", {})),
        }
        client.table("audit_reports").insert(row).execute()
        print(f"[sql_store] Audit report saved: verdict={row['verdict']}")
        return True
    except Exception as e:
        print(f"[sql_store] ERROR saving audit report: {e}")
        return False


def get_audit_history(user_id: str, limit: int = 10) -> list:
    """
    Retrieve recent audit reports for a user.

    Example:
        >>> get_audit_history("student_1", limit=5)
        [{'verdict': 'lesson', 'overall_score': 0.45, ...}, ...]

    Args:
        user_id: Unique user identifier.
        limit: Maximum number of reports to return.

    Returns:
        List of audit report dicts, newest first.
    """
    print(f"[sql_store] Getting audit history for {user_id} (limit={limit})")
    client = _get_client()
    if not client:
        return []
    try:
        resp = (
            client.table("audit_reports")
            .select("*")
            .eq("user_id", user_id)
            .order("timestamp", desc=True)
            .limit(limit)
            .execute()
        )
        data = resp.data or []
        for row in data:
            if isinstance(row.get("full_report"), str):
                try:
                    row["full_report"] = json.loads(row["full_report"])
                except Exception:
                    row["full_report"] = {}
        print(f"[sql_store] Found {len(data)} audit reports")
        return data
    except Exception as e:
        print(f"[sql_store] ERROR getting audit history: {e}")
        return []


def save_message(user_id: str, role: str, message: str) -> bool:
    """
    Insert a conversation message (user or tutor) into the conversations table.

    Example:
        >>> save_message("student_1", "user", "I have went to park")
        True

    Args:
        user_id: Unique user identifier.
        role: Either 'user' or 'tutor'.
        message: The message text.

    Returns:
        True if insert succeeded, False otherwise.
    """
    print(f"[sql_store] Saving message for {user_id} (role={role})")
    client = _get_client()
    if not client:
        return False
    try:
        row = {
            "user_id": user_id,
            "role": role,
            "message": message,
        }
        client.table("conversations").insert(row).execute()
        print(f"[sql_store] Message saved ({role})")
        return True
    except Exception as e:
        print(f"[sql_store] ERROR saving message: {e}")
        return False


def get_conversation_history(user_id: str, limit: int = 10) -> list:
    """
    Retrieve recent conversation messages for a user.

    Example:
        >>> get_conversation_history("student_1", limit=5)
        [{'role': 'user', 'message': 'Hello', ...}, ...]

    Args:
        user_id: Unique user identifier.
        limit: Maximum number of messages to return.

    Returns:
        List of message dicts, newest first.
    """
    print(f"[sql_store] Getting conversation history for {user_id} (limit={limit})")
    client = _get_client()
    if not client:
        return []
    try:
        resp = (
            client.table("conversations")
            .select("*")
            .eq("user_id", user_id)
            .order("timestamp", desc=True)
            .limit(limit)
            .execute()
        )
        print(f"[sql_store] Found {len(resp.data)} messages")
        return resp.data or []
    except Exception as e:
        print(f"[sql_store] ERROR getting conversation history: {e}")
        return []
