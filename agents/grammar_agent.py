"""
agents/grammar_agent.py

Grammar Architect Agent for Linguist-OS.
Uses mixtral-8x7b-32768 via Groq for maximum grammar accuracy.

Architecture role:
  - Checks grammar, tense, and sentence structure
  - Returns structured JSON with errors, scores, and explanations
  - Each error includes wrong text, correction, rule, explanation, severity
  - Severity is "critical" or "minor"
  - False positive rate target: below 5%

This agent is the primary error detection engine and its output
directly drives the Audit Agent's verdict decision.
"""

import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GRAMMAR_MODEL = "llama-3.3-70b-versatile"

DEFAULT_GRAMMAR_RESULT = {
    "has_error": False,
    "errors": [],
    "grammar_score": 1.0,
}


def _parse_json_response(text: str) -> dict:
    """
    Parse JSON from an LLM response, handling markdown code blocks.

    Example:
        >>> _parse_json_response('```json\\n{"has_error": true}\\n```')
        {'has_error': True}

    Args:
        text: Raw LLM response text.

    Returns:
        Parsed dict or DEFAULT_GRAMMAR_RESULT on failure.
    """
    cleaned = text.strip()
    # Strip markdown code blocks
    if "```" in cleaned:
        cleaned = cleaned.split("```")[1] if len(cleaned.split("```")) > 1 else cleaned
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    # Find JSON boundaries
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start != -1 and end > start:
        cleaned = cleaned[start:end]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"[grammar_agent] JSON parse error: {e}")
        return DEFAULT_GRAMMAR_RESULT.copy()


def run_grammar_agent(text: str, lesson_context: str = "") -> dict:
    """
    Run the Grammar Architect agent on the user's text.

    Example:
        >>> result = run_grammar_agent("I have went to the park yesterday")
        >>> result["has_error"]
        True
        >>> result["errors"][0]["wrong"]
        'have went'

    Args:
        text: The user's English text to analyze.
        lesson_context: Optional relevant lesson content for context.

    Returns:
        Dict with keys: has_error (bool), errors (list), grammar_score (float).
        Each error: {wrong, correct, rule, explanation, severity}.
    """
    print(f"[grammar_agent] Analyzing: '{text[:60]}...'")

    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))

        system_prompt = "You are a world-class English grammar expert with 20 years of experience teaching ESL students. You have an extremely precise eye for grammar errors and never flag correct usage as wrong. You specialize in tense usage, subject-verb agreement, articles, prepositions, and irregular verbs. Your false positive rate is below 5%. You return ONLY valid JSON."

        user_prompt = f"""Analyze this English text for grammar errors:

TEXT: "{text}"

{f"LESSON CONTEXT: {lesson_context}" if lesson_context else ""}

Check for:
1. Tense errors (past simple vs present perfect, etc.)
2. Subject-verb agreement
3. Article usage (a, an, the)
4. Preposition errors
5. Irregular verb forms
6. Sentence structure issues

IMPORTANT RULES:
- Only flag GENUINE errors. Do NOT flag informal or conversational English as wrong.
- Be conservative. If unsure, do NOT flag it.
- grammar_score should be 0.0 to 1.0 (1.0 = perfect, 0.0 = many errors)
- severity must be "critical" for major errors, "minor" for small issues

Return ONLY valid JSON. No other text. No markdown.
Format:
{{
    "has_error": true/false,
    "errors": [
        {{
            "wrong": "the incorrect phrase",
            "correct": "the corrected phrase",
            "rule": "grammar rule name",
            "explanation": "brief explanation",
            "severity": "critical or minor"
        }}
    ],
    "grammar_score": 0.0-1.0
}}

If no errors found, return: {{"has_error": false, "errors": [], "grammar_score": 1.0}}"""

        response = client.chat.completions.create(
            model=GRAMMAR_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=1024,
        )

        result_text = response.choices[0].message.content
        parsed = _parse_json_response(result_text)

        # Validate structure
        if "has_error" not in parsed:
            parsed["has_error"] = len(parsed.get("errors", [])) > 0
        if "errors" not in parsed:
            parsed["errors"] = []
        if "grammar_score" not in parsed:
            parsed["grammar_score"] = 1.0 if not parsed["errors"] else 0.5

        print(f"[grammar_agent] Found {len(parsed['errors'])} errors, "
              f"score: {parsed['grammar_score']}")
        return parsed

    except Exception as e:
        print(f"[grammar_agent] ERROR: {e}")
        return DEFAULT_GRAMMAR_RESULT.copy()
