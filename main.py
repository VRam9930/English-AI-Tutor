"""
main.py

FastAPI application entry point for Linguist-OS.
Serves the frontend, handles chat API requests, and orchestrates
the multi-agent pipeline through the supervisor module.

Architecture role:
  - GET  /                           -> Serve index.html
  - POST /api/chat                   -> Main chat endpoint (pipeline)
  - GET  /api/user/{user_id}/profile -> User mastery + recent mistakes
  - GET  /api/user/{user_id}/audit-history -> Past audit reports
  - GET  /health                     -> Health check

On startup:
  - Initializes Supabase connection
  - Loads lesson content into ChromaDB
  - Prints "Linguist-OS ready!"
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from database.sql_store import init_db, get_user, get_recent_mistakes, get_audit_history
from database.chroma_store import init_chroma_lessons
from core.supervisor import process_message


class ChatRequest(BaseModel):
    """
    Request body for the /api/chat endpoint.

    Example:
        {"user_id": "student_1", "message": "I have went to park"}
    """
    user_id: str
    message: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown lifecycle for the FastAPI app.

    On startup:
      - Initialize Supabase database connection
      - Load lessons into ChromaDB
      - Print ready message

    Example:
        # Automatically called by FastAPI on startup
    """
    print("\n" + "=" * 60)
    print("  LINGUIST-OS STARTING UP")
    print("=" * 60)

    # Initialize database
    print("\n[main] Initializing database...")
    init_db()

    # Load lessons into ChromaDB
    print("\n[main] Loading lessons into ChromaDB...")
    init_chroma_lessons()

    print("\n" + "=" * 60)
    print("  Linguist-OS ready!")
    print("=" * 60 + "\n")

    yield

    print("\n[main] Linguist-OS shutting down.")


app = FastAPI(
    title="Linguist-OS",
    description="AI-Powered English Language Tutor with Multi-Agent Architecture",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS - allow all origins for deployment flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """
    Serve the single-page frontend (index.html).

    Example:
        GET / -> Returns the full HTML chat interface.
    """
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Linguist-OS</h1><p>index.html not found.</p>",
            status_code=404,
        )


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    """
    Main chat endpoint. Processes user message through the full
    5-agent pipeline and returns tutor response with analysis.

    Example request:
        POST /api/chat
        {"user_id": "student_1", "message": "I have went to park"}

    Example response:
        {
            "reply": "Hey! I noticed you said 'have went'...",
            "verdict": "lesson",
            "severity": "critical",
            "overall_score": 0.45,
            "grammar_errors": [...],
            "vocab_suggestions": [...],
            "cultural_errors": [...],
            "confidence": "medium",
            "prescription_stack": [...],
            "audit_summary": "...",
            "recommendation": "..."
        }

    Args:
        req: ChatRequest with user_id and message fields.

    Returns:
        JSONResponse with full pipeline results.
    """
    print(f"\n[main] Chat request from {req.user_id}: '{req.message}'")

    if not req.message.strip():
        return JSONResponse(
            content={
                "reply": "Please type something so I can help you practice!",
                "verdict": "conversation",
                "severity": "clean",
                "overall_score": 1.0,
                "grammar_errors": [],
                "vocab_suggestions": [],
                "cultural_errors": [],
                "confidence": "medium",
                "prescription_stack": [],
                "audit_summary": "Empty message received.",
                "recommendation": "Try typing a sentence in English!",
            }
        )

    try:
        result = process_message(req.user_id, req.message)
        return JSONResponse(content=result)
    except Exception as e:
        print(f"[main] ERROR processing message: {e}")
        return JSONResponse(
            content={
                "reply": "Oops, I had a small hiccup processing that. Could you try again?",
                "verdict": "conversation",
                "severity": "clean",
                "overall_score": 0.5,
                "grammar_errors": [],
                "vocab_suggestions": [],
                "cultural_errors": [],
                "confidence": "medium",
                "prescription_stack": [],
                "audit_summary": f"Processing error: {str(e)[:100]}",
                "recommendation": "Please try again.",
            },
            status_code=200,
        )


@app.get("/api/user/{user_id}/profile")
async def get_user_profile(user_id: str):
    """
    Get user profile with mastery scores and recent mistakes.

    Example:
        GET /api/user/student_1/profile
        -> {"user_id": "student_1", "mastery": {...}, "recent_mistakes": [...]}

    Args:
        user_id: Unique user identifier.

    Returns:
        JSONResponse with user profile data.
    """
    print(f"[main] Profile request for {user_id}")
    try:
        user = get_user(user_id)
        mistakes = get_recent_mistakes(user_id, limit=10)
        return JSONResponse(content={
            "user_id": user.get("user_id", user_id),
            "name": user.get("name", "Learner"),
            "mastery": user.get("mastery", {}),
            "recent_mistakes": mistakes,
        })
    except Exception as e:
        print(f"[main] ERROR getting profile: {e}")
        return JSONResponse(content={
            "user_id": user_id,
            "name": "Learner",
            "mastery": {},
            "recent_mistakes": [],
        })


@app.get("/api/user/{user_id}/audit-history")
async def get_user_audit_history(user_id: str):
    """
    Get past audit reports for a user.

    Example:
        GET /api/user/student_1/audit-history
        -> {"user_id": "student_1", "audit_history": [...]}

    Args:
        user_id: Unique user identifier.

    Returns:
        JSONResponse with audit history list.
    """
    print(f"[main] Audit history request for {user_id}")
    try:
        history = get_audit_history(user_id, limit=10)
        return JSONResponse(content={
            "user_id": user_id,
            "audit_history": history,
        })
    except Exception as e:
        print(f"[main] ERROR getting audit history: {e}")
        return JSONResponse(content={
            "user_id": user_id,
            "audit_history": [],
        })


@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring and deployment.

    Example:
        GET /health -> {"status": "ok"}

    Returns:
        JSONResponse with status.
    """
    return JSONResponse(content={"status": "ok"})
