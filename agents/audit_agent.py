"""
agents/audit_agent.py

Audit Agent (THE JUDGE) for Linguist-OS.
Uses llama3-8b-8192 via Groq for efficient decision-making.

Architecture role:
  - Receives ALL 4 other agent results
  - Also receives mistake_counts from Supabase and similar_mistakes from ChromaDB
  - Synthesizes everything into a FINAL VERDICT
  - Determines if the tutor should teach, correct, or just chat
  - Builds a prescription stack of lessons to complete

Decision Rules:
  REPEATED  -> same concept mistake 3+ times           -> "lesson"
  CRITICAL  -> critical severity + grammar_score < 0.7  -> "lesson"
  MODERATE  -> critical error but score >= 0.7           -> "soft_correction"
  MINOR     -> minor errors only + score < 0.8           -> "soft_correction"
  CLEAN     -> no errors                                 -> "conversation"

Verdict options: "lesson", "soft_correction", "conversation"
Severity options: "clean", "minor", "moderate", "critical", "repeated"
"""

import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

AUDIT_MODEL = "llama-3.1-8b-instant"

DEFAULT_AUDIT_RESULT = {
    "verdict": "conversation",
    "severity": "clean",
    "overall_score": 1.0,
    "primary_issue": "none",
    "concept_scores": {},
    "prescription_stack": [],
    "audit_summary": "No issues detected.",
    "recommendation": "Keep practicing!",
}


def _parse_json_response(text: str) -> dict:
    """
    Parse JSON from an LLM response, handling markdown code blocks.

    Example:
        >>> _parse_json_response('{"verdict": "lesson"}')
        {'verdict': 'lesson'}

    Args:
        text: Raw LLM response text.

    Returns:
        Parsed dict or DEFAULT_AUDIT_RESULT on failure.
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
        print(f"[audit_agent] JSON parse error: {e}")
        return DEFAULT_AUDIT_RESULT.copy()


def _apply_decision_rules(
    grammar_result: dict,
    mistake_counts: dict,
    similar_mistakes: list,
) -> dict:
    """
    Apply the hard-coded decision rules before running the LLM audit.
    This ensures deterministic verdicts based on clear thresholds.

    Example:
        >>> _apply_decision_rules(
        ...     {"has_error": True, "errors": [{"severity": "critical"}], "grammar_score": 0.4},
        ...     {"past_simple": 4},
        ...     [{"concept": "past_simple"}]
        ... )
        {'rule_verdict': 'lesson', 'rule_severity': 'repeated', 'reason': '...'}

    Args:
        grammar_result: Output from Grammar Architect.
        mistake_counts: Dict of concept -> count from Supabase.
        similar_mistakes: List of similar past mistakes from ChromaDB.

    Returns:
        Dict with rule_verdict, rule_severity, and reason.
    """
    errors = grammar_result.get("errors", [])
    grammar_score = grammar_result.get("grammar_score", 1.0)
    has_error = grammar_result.get("has_error", False)

    # Check for repeated mistakes (3+ times same concept)
    for error in errors:
        rule = error.get("rule", "").lower()
        for concept, count in mistake_counts.items():
            if concept.lower() in rule or rule in concept.lower():
                if count >= 3:
                    return {
                        "rule_verdict": "lesson",
                        "rule_severity": "repeated",
                        "reason": f"User has made {count} mistakes on '{concept}' - repeated pattern detected.",
                    }

    # Also check similar_mistakes for repeated patterns
    if similar_mistakes:
        concept_hits = {}
        for m in similar_mistakes:
            c = m.get("concept", "")
            if c:
                concept_hits[c] = concept_hits.get(c, 0) + 1
        for c, count in concept_hits.items():
            if count >= 2 and mistake_counts.get(c, 0) >= 2:
                return {
                    "rule_verdict": "lesson",
                    "rule_severity": "repeated",
                    "reason": f"Similar mistakes on '{c}' found in history - repeated pattern.",
                }

    # Check for critical errors
    has_critical = any(e.get("severity") == "critical" for e in errors)
    has_minor = any(e.get("severity") == "minor" for e in errors)

    if has_critical and grammar_score < 0.7:
        return {
            "rule_verdict": "lesson",
            "rule_severity": "critical",
            "reason": f"Critical grammar error with low score ({grammar_score}).",
        }

    if has_critical and grammar_score >= 0.7:
        return {
            "rule_verdict": "soft_correction",
            "rule_severity": "moderate",
            "reason": f"Critical error but reasonable score ({grammar_score}).",
        }

    if has_minor and grammar_score < 0.8:
        return {
            "rule_verdict": "soft_correction",
            "rule_severity": "minor",
            "reason": f"Minor errors with below-threshold score ({grammar_score}).",
        }

    if not has_error and not errors:
        return {
            "rule_verdict": "conversation",
            "rule_severity": "clean",
            "reason": "No errors detected. Clean message.",
        }

    # Default for minor errors with good score
    return {
        "rule_verdict": "conversation",
        "rule_severity": "clean",
        "reason": "Minor issues within acceptable range.",
    }


def run_audit_agent(
    grammar_result: dict,
    vocab_result: dict,
    cultural_result: dict,
    confidence_result: dict,
    mistake_counts: dict,
    similar_mistakes: list,
) -> dict:
    """
    Run the Audit Agent to synthesize all agent results into a final verdict.

    Example:
        >>> result = run_audit_agent(
        ...     grammar_result={"has_error": True, "errors": [...], "grammar_score": 0.4},
        ...     vocab_result={"current_level": "A2", "suggestions": [...]},
        ...     cultural_result={"has_cultural_error": False, "errors": []},
        ...     confidence_result={"confidence_level": "medium"},
        ...     mistake_counts={"past_simple": 4},
        ...     similar_mistakes=[...]
        ... )
        >>> result["verdict"]
        'lesson'

    Args:
        grammar_result: Output from Grammar Architect.
        vocab_result: Output from Vocab Stylist.
        cultural_result: Output from Cultural Bridge.
        confidence_result: Output from Confidence Coach.
        mistake_counts: Dict of concept -> count from Supabase.
        similar_mistakes: List of similar past mistakes from ChromaDB.

    Returns:
        Dict with keys: verdict, severity, overall_score, primary_issue,
        concept_scores, prescription_stack, audit_summary, recommendation.
    """
    print("[audit_agent] Running audit synthesis...")

    # Apply hard-coded decision rules first
    rules = _apply_decision_rules(grammar_result, mistake_counts, similar_mistakes)
    rule_verdict = rules["rule_verdict"]
    rule_severity = rules["rule_severity"]
    print(f"[audit_agent] Rule-based verdict: {rule_verdict} ({rule_severity})")
    print(f"[audit_agent] Reason: {rules['reason']}")

    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))

        system_prompt = "You are the chief auditor of an English language learning system. You receive analysis from 4 specialist agents and must synthesize scores, build prescription stacks, and provide summaries. You return ONLY valid JSON."

        user_prompt = f"""You are the final judge. Synthesize these agent results:

GRAMMAR ANALYSIS:
{json.dumps(grammar_result, indent=2)}

VOCABULARY ANALYSIS:
{json.dumps(vocab_result, indent=2)}

CULTURAL ANALYSIS:
{json.dumps(cultural_result, indent=2)}

CONFIDENCE ANALYSIS:
{json.dumps(confidence_result, indent=2)}

MISTAKE HISTORY (counts by concept):
{json.dumps(mistake_counts, indent=2)}

SIMILAR PAST MISTAKES:
{json.dumps(similar_mistakes, indent=2)}

RULE-BASED VERDICT (must be respected):
Verdict: {rule_verdict}
Severity: {rule_severity}
Reason: {rules['reason']}

IMPORTANT: The verdict MUST be "{rule_verdict}" and severity MUST be "{rule_severity}".
Your job is to provide the analysis, scores, and prescription stack.

Calculate:
- overall_score: weighted average (grammar 40%, vocab 20%, cultural 20%, confidence 20%)
- primary_issue: the single most important thing to address
- concept_scores: scores for each concept area
- prescription_stack: ordered list of lessons needed (max 3)
  Each: {{"priority": "urgent/high/medium", "concept": "concept_name", "reason": "why"}}
- audit_summary: 1-2 sentence summary of the analysis
- recommendation: actionable advice for the student

Return ONLY valid JSON. No other text. No markdown.
Format:
{{
    "verdict": "{rule_verdict}",
    "severity": "{rule_severity}",
    "overall_score": 0.0-1.0,
    "primary_issue": "concept name or none",
    "concept_scores": {{"grammar": 0.0-1.0, "vocabulary": 0.0-1.0, "cultural": 0.0-1.0, "confidence": 0.0-1.0}},
    "prescription_stack": [
        {{"priority": "urgent/high/medium", "concept": "concept_name", "reason": "why this needs attention"}}
    ],
    "audit_summary": "1-2 sentence summary",
    "recommendation": "actionable advice"
}}"""

        response = client.chat.completions.create(
            model=AUDIT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=1024,
        )

        result_text = response.choices[0].message.content
        parsed = _parse_json_response(result_text)

        # Enforce rule-based verdict (LLM can't override hard rules)
        parsed["verdict"] = rule_verdict
        parsed["severity"] = rule_severity

        # Validate and fill defaults
        if "overall_score" not in parsed:
            gs = grammar_result.get("grammar_score", 1.0)
            vs = vocab_result.get("vocab_score", 0.7)
            cs = cultural_result.get("pragmatic_score", 0.8)
            fs = confidence_result.get("fluency_score", 0.7)
            parsed["overall_score"] = round(gs * 0.4 + vs * 0.2 + cs * 0.2 + fs * 0.2, 2)
        if "primary_issue" not in parsed:
            parsed["primary_issue"] = "none"
        if "concept_scores" not in parsed:
            parsed["concept_scores"] = {}
        if "prescription_stack" not in parsed:
            parsed["prescription_stack"] = []
        if "audit_summary" not in parsed:
            parsed["audit_summary"] = rules["reason"]
        if "recommendation" not in parsed:
            parsed["recommendation"] = "Keep practicing!"

        print(f"[audit_agent] Final verdict: {parsed['verdict']} "
              f"(severity: {parsed['severity']}, score: {parsed['overall_score']})")
        return parsed

    except Exception as e:
        print(f"[audit_agent] ERROR: {e}")
        # Fall back to rule-based result
        gs = grammar_result.get("grammar_score", 1.0)
        vs = vocab_result.get("vocab_score", 0.7)
        cs = cultural_result.get("pragmatic_score", 0.8)
        fs = confidence_result.get("fluency_score", 0.7)
        overall = round(gs * 0.4 + vs * 0.2 + cs * 0.2 + fs * 0.2, 2)
        return {
            "verdict": rule_verdict,
            "severity": rule_severity,
            "overall_score": overall,
            "primary_issue": "grammar" if grammar_result.get("has_error") else "none",
            "concept_scores": {},
            "prescription_stack": [],
            "audit_summary": rules["reason"],
            "recommendation": "Keep practicing and pay attention to grammar rules!",
        }
