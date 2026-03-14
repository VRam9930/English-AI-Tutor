"""
agents/cultural_agent.py

Cultural Bridge Agent for Linguist-OS.
Uses gemma2-9b-it via Groq for multilingual and cultural awareness.

Architecture role:
  - Flags idiomatic errors, tone mismatches, direct translation errors
  - Detects when learners translate literally from their native language
  - Returns structured JSON with errors, tone assessment, pragmatic score
  - Error types: "idiom", "tone", "direct_translation", "pragmatic"
"""

import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

CULTURAL_MODEL = "llama-3.1-8b-instant"

DEFAULT_CULTURAL_RESULT = {
    "has_cultural_error": False,
    "errors": [],
    "tone": "neutral",
    "pragmatic_score": 0.8,
}


def _parse_json_response(text: str) -> dict:
    """
    Parse JSON from an LLM response, handling markdown code blocks.

    Example:
        >>> _parse_json_response('{"has_cultural_error": false}')
        {'has_cultural_error': False}

    Args:
        text: Raw LLM response text.

    Returns:
        Parsed dict or DEFAULT_CULTURAL_RESULT on failure.
    """
    cleaned = text.strip()
    if "```" in cleaned:
        cleaned = cleaned.split("```")[1] if len(cleaned.split("```")) > 1 else cleaned
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start != -1 and end > start:
        cleaned = cleaned[start:end]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"[cultural_agent] JSON parse error: {e}")
        return DEFAULT_CULTURAL_RESULT.copy()


def run_cultural_agent(text: str) -> dict:
    """
    Run the Cultural Bridge agent to detect cultural and pragmatic errors.

    Example:
        >>> result = run_cultural_agent("I want you to give me the book")
        >>> result["has_cultural_error"]
        True
        >>> result["errors"][0]["type"]
        'tone'

    Args:
        text: The user's English text to analyze.

    Returns:
        Dict with keys: has_cultural_error (bool), errors (list),
        tone (str), pragmatic_score (float).
        Each error: {phrase, issue, correction, explanation, type}.
        Types: "idiom", "tone", "direct_translation", "pragmatic".
    """
    print(f"[cultural_agent] Analyzing cultural aspects: '{text[:60]}...'")

    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))

        system_prompt = "You are a cross-cultural communication expert who specializes in helping ESL learners understand English idioms, tone, and pragmatics. You can detect when someone is translating directly from another language, using idioms incorrectly, or using an inappropriate tone. You only flag genuine issues, not personal style choices. You return ONLY valid JSON."

        user_prompt = f"""Analyze this English text for cultural and pragmatic issues:

TEXT: "{text}"

Check for:
1. Incorrect idiom usage (using idioms wrong or mixing them up)
2. Tone issues (too direct, too formal, too casual for context)
3. Direct translation artifacts (phrases that sound translated from another language)
4. Pragmatic errors (inappropriate politeness level, missing context cues)

IMPORTANT: Only flag GENUINE cultural/pragmatic issues. Simple grammar errors are NOT your job.
Be conservative - if something sounds natural in English, do NOT flag it.

tone should be one of: "formal", "neutral", "casual", "too_direct", "appropriate"
pragmatic_score: 0.0-1.0 (1.0 = perfectly natural English)

Return ONLY valid JSON. No other text. No markdown.
Format:
{{
    "has_cultural_error": true/false,
    "errors": [
        {{
            "phrase": "the problematic phrase",
            "issue": "what's wrong",
            "correction": "better way to say it",
            "explanation": "why this is better",
            "type": "idiom/tone/direct_translation/pragmatic"
        }}
    ],
    "tone": "formal/neutral/casual/too_direct/appropriate",
    "pragmatic_score": 0.0-1.0
}}

If no cultural issues found, return: {{"has_cultural_error": false, "errors": [], "tone": "neutral", "pragmatic_score": 1.0}}"""

        response = client.chat.completions.create(
            model=CULTURAL_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=1024,
        )

        result_text = response.choices[0].message.content
        parsed = _parse_json_response(result_text)

        # Validate structure
        if "has_cultural_error" not in parsed:
            parsed["has_cultural_error"] = len(parsed.get("errors", [])) > 0
        if "errors" not in parsed:
            parsed["errors"] = []
        if "tone" not in parsed:
            parsed["tone"] = "neutral"
        if "pragmatic_score" not in parsed:
            parsed["pragmatic_score"] = 0.8

        print(f"[cultural_agent] Found {len(parsed['errors'])} cultural issues, "
              f"tone: {parsed['tone']}")
        return parsed

    except Exception as e:
        print(f"[cultural_agent] ERROR: {e}")
        return DEFAULT_CULTURAL_RESULT.copy()
