"""
agents/vocab_agent.py

Vocab Stylist Agent for Linguist-OS.
Uses llama3-8b-8192 via Groq for fast, lightweight vocabulary analysis.

Architecture role:
  - Assesses the user's current vocabulary level (A1-C2 CEFR scale)
  - Suggests higher-tier vocabulary alternatives
  - Returns structured JSON with level, suggestions, and scores
  - Each suggestion includes basic word, better alternatives, best fit, example
"""

import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

VOCAB_MODEL = "llama-3.1-8b-instant"

DEFAULT_VOCAB_RESULT = {
    "current_level": "B1",
    "suggestions": [],
    "vocab_score": 0.7,
    "diversity_score": 0.7,
}


def _parse_json_response(text: str) -> dict:
    """
    Parse JSON from an LLM response, handling markdown code blocks.

    Example:
        >>> _parse_json_response('```json\\n{"current_level": "A2"}\\n```')
        {'current_level': 'A2'}

    Args:
        text: Raw LLM response text.

    Returns:
        Parsed dict or DEFAULT_VOCAB_RESULT on failure.
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
        print(f"[vocab_agent] JSON parse error: {e}")
        return DEFAULT_VOCAB_RESULT.copy()


def run_vocab_agent(text: str) -> dict:
    """
    Run the Vocab Stylist agent to assess vocabulary and suggest upgrades.

    Example:
        >>> result = run_vocab_agent("The food was very good and I was very happy")
        >>> result["current_level"]
        'A2'
        >>> result["suggestions"][0]["basic_word"]
        'good'

    Args:
        text: The user's English text to analyze.

    Returns:
        Dict with keys: current_level (str), suggestions (list),
        vocab_score (float), diversity_score (float).
        Each suggestion: {basic_word, better_alternatives, best_fit, example}.
    """
    print(f"[vocab_agent] Analyzing vocabulary: '{text[:60]}...'")

    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))

        system_prompt = "You are a vocabulary specialist who helps English learners upgrade their word choices. You are an expert in the CEFR scale (A1 to C2) and know exactly which words belong to each level. You suggest practical, natural-sounding alternatives. You return ONLY valid JSON."

        user_prompt = f"""Analyze the vocabulary in this English text:

TEXT: "{text}"

TASK:
1. Determine the overall CEFR vocabulary level (A1, A2, B1, B2, C1, C2)
2. Find basic/simple words that could be upgraded to more advanced alternatives
3. Give a vocab_score (0.0-1.0) based on word sophistication
4. Give a diversity_score (0.0-1.0) based on vocabulary variety

Only suggest upgrades for 1-3 most impactful words. Do not suggest changes for every word.

Return ONLY valid JSON. No other text. No markdown.
Format:
{{
    "current_level": "A1/A2/B1/B2/C1/C2",
    "suggestions": [
        {{
            "basic_word": "the simple word used",
            "better_alternatives": ["word1", "word2", "word3"],
            "best_fit": "the single best replacement",
            "example": "example sentence using the best_fit word"
        }}
    ],
    "vocab_score": 0.0-1.0,
    "diversity_score": 0.0-1.0
}}

If vocabulary is already advanced, return empty suggestions list with high scores."""

        response = client.chat.completions.create(
            model=VOCAB_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=1024,
        )

        result_text = response.choices[0].message.content
        parsed = _parse_json_response(result_text)

        # Validate structure
        if "current_level" not in parsed:
            parsed["current_level"] = "B1"
        if "suggestions" not in parsed:
            parsed["suggestions"] = []
        if "vocab_score" not in parsed:
            parsed["vocab_score"] = 0.7
        if "diversity_score" not in parsed:
            parsed["diversity_score"] = 0.7

        print(f"[vocab_agent] Level: {parsed['current_level']}, "
              f"{len(parsed['suggestions'])} suggestions")
        return parsed

    except Exception as e:
        print(f"[vocab_agent] ERROR: {e}")
        return DEFAULT_VOCAB_RESULT.copy()
