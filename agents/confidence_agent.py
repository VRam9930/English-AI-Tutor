"""
agents/confidence_agent.py

Confidence Coach Agent for Linguist-OS.
Uses llama3-70b-8192 via Groq for nuanced fluency assessment.

Architecture role:
  - Uses BOTH the user text AND metadata from the chunker
  - Measures fluency, hesitation patterns, sentence complexity
  - Returns structured JSON with fluency score, confidence level, and issues
  - Provides personalized encouragement
  - confidence_level: "low", "medium", "high"
"""

import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

CONFIDENCE_MODEL = "llama-3.3-70b-versatile"

DEFAULT_CONFIDENCE_RESULT = {
    "fluency_score": 0.7,
    "confidence_level": "medium",
    "hesitation_count": 0,
    "words_per_sentence": 0.0,
    "complexity_level": "intermediate",
    "issues": [],
    "encouragement": "Keep practicing! You're doing well.",
}


def _parse_json_response(text: str) -> dict:
    """
    Parse JSON from an LLM response, handling markdown code blocks.

    Example:
        >>> _parse_json_response('{"fluency_score": 0.8}')
        {'fluency_score': 0.8}

    Args:
        text: Raw LLM response text.

    Returns:
        Parsed dict or DEFAULT_CONFIDENCE_RESULT on failure.
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
        print(f"[confidence_agent] JSON parse error: {e}")
        return DEFAULT_CONFIDENCE_RESULT.copy()


def run_confidence_agent(text: str, metadata: dict) -> dict:
    """
    Run the Confidence Coach agent using text and chunker metadata.

    Example:
        >>> metadata = {
        ...     "word_count": 12, "sentence_count": 2,
        ...     "hesitation_count": 2, "hesitation_words_found": ["um", "like"],
        ...     "words_per_sentence": 6.0, "avg_word_length": 3.5,
        ...     "question_marks": 1
        ... }
        >>> result = run_confidence_agent("Um, I went to, like, the store?", metadata)
        >>> result["confidence_level"]
        'low'

    Args:
        text: The user's English text.
        metadata: Metadata dict from core/chunker.py.

    Returns:
        Dict with keys: fluency_score, confidence_level, hesitation_count,
        words_per_sentence, complexity_level, issues, encouragement.
    """
    print(f"[confidence_agent] Analyzing confidence: '{text[:60]}...'")

    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))

        system_prompt = "You are an empathetic language confidence coach with deep expertise in second language acquisition. You can detect hesitation patterns, sentence complexity issues, and overall fluency from text. You always provide warm, encouraging feedback. You return ONLY valid JSON."

        metadata_str = json.dumps(metadata, indent=2)

        user_prompt = f"""Analyze the fluency and confidence in this English text:

TEXT: "{text}"

TEXT METADATA (from analysis):
{metadata_str}

Key metadata to consider:
- word_count: {metadata.get('word_count', 0)} words total
- sentence_count: {metadata.get('sentence_count', 0)} sentences
- hesitation_count: {metadata.get('hesitation_count', 0)} hesitation words found
- hesitation_words_found: {metadata.get('hesitation_words_found', [])}
- words_per_sentence: {metadata.get('words_per_sentence', 0)} average
- avg_word_length: {metadata.get('avg_word_length', 0)}
- question_marks: {metadata.get('question_marks', 0)}

ASSESSMENT CRITERIA:
- fluency_score (0.0-1.0): Based on sentence flow, complexity, and naturalness
- confidence_level: "low" (many hesitations, very short), "medium" (some flow), "high" (natural, complex)
- complexity_level: "basic" (A1-A2), "intermediate" (B1-B2), "advanced" (C1-C2)
- issues: List specific fluency concerns
- encouragement: A warm, personalized encouragement message (1-2 sentences)

Return ONLY valid JSON. No other text. No markdown.
Format:
{{
    "fluency_score": 0.0-1.0,
    "confidence_level": "low/medium/high",
    "hesitation_count": number,
    "words_per_sentence": number,
    "complexity_level": "basic/intermediate/advanced",
    "issues": ["issue1", "issue2"],
    "encouragement": "warm encouragement message"
}}"""

        response = client.chat.completions.create(
            model=CONFIDENCE_MODEL,
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
        if "fluency_score" not in parsed:
            parsed["fluency_score"] = 0.7
        if "confidence_level" not in parsed:
            parsed["confidence_level"] = "medium"
        if "hesitation_count" not in parsed:
            parsed["hesitation_count"] = metadata.get("hesitation_count", 0)
        if "words_per_sentence" not in parsed:
            parsed["words_per_sentence"] = metadata.get("words_per_sentence", 0)
        if "complexity_level" not in parsed:
            parsed["complexity_level"] = "intermediate"
        if "issues" not in parsed:
            parsed["issues"] = []
        if "encouragement" not in parsed:
            parsed["encouragement"] = "Keep practicing! You're making great progress."

        print(f"[confidence_agent] Confidence: {parsed['confidence_level']}, "
              f"Fluency: {parsed['fluency_score']}")
        return parsed

    except Exception as e:
        print(f"[confidence_agent] ERROR: {e}")
        return DEFAULT_CONFIDENCE_RESULT.copy()
