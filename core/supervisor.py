"""
core/supervisor.py

Central orchestrator for Linguist-OS multi-agent pipeline.
Coordinates all 5 specialist agents via CrewAI, manages the full
processing pipeline from chunking through response generation.

Architecture role:
  - STEP 1: Chunk text and extract metadata
  - STEP 2: Embed text (handled by ChromaDB internally)
  - STEP 3: Retrieve relevant lessons and past mistakes from ChromaDB
  - STEP 4: Run 5 specialist agents via CrewAI
  - STEP 5: Save results to Supabase and ChromaDB
  - STEP 6: Generate final tutor response

This is the brain of the system - it decides what happens and when.
"""

import os
import json
from dotenv import load_dotenv
from groq import Groq

from core.chunker import chunk_text
from database.chroma_store import (
    retrieve_lesson,
    find_similar_past_mistakes,
    store_mistake_in_chroma,
)
from database.sql_store import (
    get_user,
    update_mastery,
    save_mistake,
    get_mistake_count_by_concept,
    save_audit_report,
    save_message,
)
from agents.grammar_agent import run_grammar_agent
from agents.vocab_agent import run_vocab_agent
from agents.cultural_agent import run_cultural_agent
from agents.confidence_agent import run_confidence_agent
from agents.audit_agent import run_audit_agent

load_dotenv()


def _get_groq_client():
    """Return a Groq client for response generation."""
    return Groq(api_key=os.getenv("GROQ_API_KEY", ""))


def process_message(user_id: str, message: str) -> dict:
    """
    Main pipeline: process a user message through all 5 agents and
    generate a tutor response.

    Example:
        >>> result = process_message("student_1", "I have went to park")
        >>> result["verdict"]
        'lesson'
        >>> result["reply"]
        "Hey! I noticed you said 'have went'..."

    Args:
        user_id: Unique user identifier.
        message: The user's English text input.

    Returns:
        Dict with keys: reply, verdict, severity, overall_score,
        grammar_errors, vocab_suggestions, cultural_errors,
        confidence, prescription_stack, audit_summary, recommendation.
    """
    print("\n" + "=" * 60)
    print(f"[supervisor] PIPELINE START for {user_id}")
    print(f"[supervisor] Message: '{message}'")
    print("=" * 60)

    # Save user message
    save_message(user_id, "user", message)

    # ── STEP 1: CHUNK ──────────────────────────────────────────
    print("\n[supervisor] STEP 1: Chunking text...")
    chunk_data = chunk_text(message)
    metadata = chunk_data["metadata"]
    print(f"[supervisor] Metadata: {json.dumps(metadata, indent=2)}")

    # ── STEP 2 & 3: RETRIEVE FROM CHROMADB ─────────────────────
    print("\n[supervisor] STEP 2-3: Retrieving from ChromaDB...")
    relevant_lessons = retrieve_lesson(message, n=2)
    similar_mistakes = find_similar_past_mistakes(user_id, message, n=3)
    lesson_context = ""
    if relevant_lessons:
        lesson_context = relevant_lessons[0].get("content", "")
    print(f"[supervisor] Found {len(relevant_lessons)} lessons, "
          f"{len(similar_mistakes)} similar past mistakes")

    # ── STEP 4: RUN 5 SPECIALIST AGENTS ────────────────────────
    print("\n[supervisor] STEP 4: Running specialist agents...")

    # Agent 1: Grammar Architect
    print("\n[supervisor] Running Grammar Architect...")
    grammar_result = run_grammar_agent(message, lesson_context)
    print(f"[supervisor] Grammar result: {json.dumps(grammar_result, indent=2)}")

    # Agent 2: Vocab Stylist
    print("\n[supervisor] Running Vocab Stylist...")
    vocab_result = run_vocab_agent(message)
    print(f"[supervisor] Vocab result: {json.dumps(vocab_result, indent=2)}")

    # Agent 3: Cultural Bridge
    print("\n[supervisor] Running Cultural Bridge...")
    cultural_result = run_cultural_agent(message)
    print(f"[supervisor] Cultural result: {json.dumps(cultural_result, indent=2)}")

    # Agent 4: Confidence Coach
    print("\n[supervisor] Running Confidence Coach...")
    confidence_result = run_confidence_agent(message, metadata)
    print(f"[supervisor] Confidence result: {json.dumps(confidence_result, indent=2)}")

    # Get mistake counts from Supabase for the Audit Agent
    mistake_counts = get_mistake_count_by_concept(user_id)

    # Agent 5: Audit Agent (THE JUDGE)
    print("\n[supervisor] Running Audit Agent (THE JUDGE)...")
    audit_result = run_audit_agent(
        grammar_result=grammar_result,
        vocab_result=vocab_result,
        cultural_result=cultural_result,
        confidence_result=confidence_result,
        mistake_counts=mistake_counts,
        similar_mistakes=similar_mistakes,
    )
    print(f"[supervisor] Audit result: {json.dumps(audit_result, indent=2)}")

    # ── STEP 5: SAVE RESULTS ──────────────────────────────────
    print("\n[supervisor] STEP 5: Saving results...")

    # Save grammar errors to Supabase and ChromaDB
    user_data = get_user(user_id)
    mastery = user_data.get("mastery", {})

    grammar_errors = grammar_result.get("errors", [])
    for error in grammar_errors:
        error_record = {
            "wrong": error.get("wrong", ""),
            "correct": error.get("correct", ""),
            "rule": error.get("rule", ""),
            "concept": _rule_to_concept(error.get("rule", "")),
            "severity": error.get("severity", "minor"),
        }
        save_mistake(user_id, message, error_record)
        concept = error_record["concept"]
        if concept:
            store_mistake_in_chroma(user_id, message, concept)
            # Update mastery scores
            current = mastery.get(concept, 0.5)
            if error.get("severity") == "critical":
                new_score = current - 0.10
            else:
                new_score = current - 0.05
            update_mastery(user_id, concept, new_score)

    # Check for repeated mistakes and apply extra penalty
    for concept, count in mistake_counts.items():
        if count >= 3:
            current = mastery.get(concept, 0.5)
            new_score = current - 0.15
            update_mastery(user_id, concept, new_score)

    # If clean message, boost mastery slightly
    if not grammar_errors:
        for concept in mastery:
            current = mastery.get(concept, 0.5)
            new_score = current + 0.05
            update_mastery(user_id, concept, new_score)

    # Save audit report
    audit_report_data = {
        "analyzed_text": message,
        "verdict": audit_result.get("verdict", "conversation"),
        "severity": audit_result.get("severity", "clean"),
        "overall_score": audit_result.get("overall_score", 1.0),
        "errors_found": len(grammar_errors),
        "primary_issue": audit_result.get("primary_issue", "none"),
        "ai_analysis": audit_result.get("audit_summary", ""),
        "audit_summary": audit_result.get("audit_summary", ""),
        "full_report": {
            "grammar": grammar_result,
            "vocab": vocab_result,
            "cultural": cultural_result,
            "confidence": confidence_result,
            "audit": audit_result,
        },
    }
    save_audit_report(user_id, audit_report_data)

    # ── STEP 6: GENERATE RESPONSE ─────────────────────────────
    print("\n[supervisor] STEP 6: Generating response...")
    verdict = audit_result.get("verdict", "conversation")
    reply = _generate_response(
        message=message,
        verdict=verdict,
        grammar_result=grammar_result,
        vocab_result=vocab_result,
        cultural_result=cultural_result,
        confidence_result=confidence_result,
        audit_result=audit_result,
        lesson_context=lesson_context,
    )

    # Save tutor reply
    save_message(user_id, "tutor", reply)

    # Build final response
    response = {
        "reply": reply,
        "verdict": verdict,
        "severity": audit_result.get("severity", "clean"),
        "overall_score": audit_result.get("overall_score", 1.0),
        "grammar_errors": grammar_errors,
        "vocab_suggestions": vocab_result.get("suggestions", []),
        "cultural_errors": cultural_result.get("errors", []),
        "confidence": confidence_result.get("confidence_level", "medium"),
        "prescription_stack": audit_result.get("prescription_stack", []),
        "audit_summary": audit_result.get("audit_summary", ""),
        "recommendation": audit_result.get("recommendation", ""),
    }

    print("\n" + "=" * 60)
    print(f"[supervisor] PIPELINE COMPLETE. Verdict: {verdict}")
    print("=" * 60 + "\n")

    return response


def _generate_response(
    message: str,
    verdict: str,
    grammar_result: dict,
    vocab_result: dict,
    cultural_result: dict,
    confidence_result: dict,
    audit_result: dict,
    lesson_context: str,
) -> str:
    """
    Generate the final tutor response based on the audit verdict.

    Verdict rules:
      - 'lesson': Interrupt, teach the rule, ask to retry (< 80 words)
      - 'soft_correction': Gentle correction, keep conversation (< 70 words)
      - 'conversation': Chat naturally, weave vocab tip (< 60 words)

    Example:
        >>> _generate_response("I have went", "lesson", ...)
        "Hey! I noticed you said 'have went'. The rule is..."

    Args:
        message: Original user message.
        verdict: The audit verdict.
        grammar_result: Output from Grammar Architect.
        vocab_result: Output from Vocab Stylist.
        cultural_result: Output from Cultural Bridge.
        confidence_result: Output from Confidence Coach.
        audit_result: Output from Audit Agent.
        lesson_context: Relevant lesson content from ChromaDB.

    Returns:
        String response from the tutor "Ling".
    """
    errors_text = ""
    if grammar_result.get("errors"):
        for e in grammar_result["errors"]:
            errors_text += f"- Wrong: '{e.get('wrong', '')}' -> Correct: '{e.get('correct', '')}' (Rule: {e.get('rule', '')})\n"

    vocab_text = ""
    if vocab_result.get("suggestions"):
        for s in vocab_result["suggestions"]:
            vocab_text += f"- Instead of '{s.get('basic_word', '')}', try '{s.get('best_fit', '')}'\n"

    cultural_text = ""
    if cultural_result.get("errors"):
        for c in cultural_result["errors"]:
            cultural_text += f"- '{c.get('phrase', '')}': {c.get('explanation', '')}\n"

    if verdict == "lesson":
        prompt = f"""You are Ling, a warm and encouraging English language tutor.
The student wrote: "{message}"

Grammar errors found:
{errors_text if errors_text else "None"}

Relevant lesson content:
{lesson_context if lesson_context else "No specific lesson available."}

TASK: Interrupt the conversation to teach. Explain the grammar rule simply, show the correct form, and ask the student to try again.
Keep your response under 80 words. Be warm and encouraging.
Do NOT use markdown formatting. Use plain text only.
Do NOT use asterisks, bold, or bullet points.
Respond as Ling speaking directly to the student."""

    elif verdict == "soft_correction":
        prompt = f"""You are Ling, a warm and encouraging English language tutor.
The student wrote: "{message}"

Grammar errors found:
{errors_text if errors_text else "None"}

Vocabulary suggestions:
{vocab_text if vocab_text else "None"}

TASK: Gently correct the errors while keeping the conversation going naturally. Mention the correct form casually.
Keep your response under 70 words. Be warm and encouraging.
Do NOT use markdown formatting. Use plain text only.
Do NOT use asterisks, bold, or bullet points.
Respond as Ling speaking directly to the student."""

    else:  # conversation
        prompt = f"""You are Ling, a warm and encouraging English language tutor.
The student wrote: "{message}"

Vocabulary suggestions:
{vocab_text if vocab_text else "None"}

Confidence assessment: {confidence_result.get('confidence_level', 'medium')}
Encouragement: {confidence_result.get('encouragement', '')}

TASK: Chat naturally with the student. If there are vocabulary suggestions, weave ONE tip into the conversation naturally.
Keep your response under 60 words. Be warm and friendly.
Do NOT use markdown formatting. Use plain text only.
Do NOT use asterisks, bold, or bullet points.
Respond as Ling speaking directly to the student."""

    try:
        client = _get_groq_client()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are Ling, a warm and encouraging English language tutor. Respond in plain text only. No markdown, no asterisks, no bold, no bullet points."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=300,
        )
        reply = response.choices[0].message.content.strip()
        # Strip any accidental markdown
        reply = reply.replace("**", "").replace("*", "").replace("##", "").replace("#", "")
        print(f"[supervisor] Generated reply: {reply[:80]}...")
        return reply
    except Exception as e:
        print(f"[supervisor] ERROR generating response: {e}")
        if verdict == "lesson":
            errors = grammar_result.get("errors", [])
            if errors:
                e0 = errors[0]
                return (f"Hey there! I noticed you wrote '{e0.get('wrong', '')}' "
                        f"but the correct form is '{e0.get('correct', '')}'. "
                        f"Can you try saying that again?")
            return "I noticed a small error in your sentence. Could you try rephrasing it?"
        elif verdict == "soft_correction":
            return "That was pretty good! Just a small note on your grammar - keep practicing!"
        else:
            return "Nice message! Keep up the great practice. What else would you like to talk about?"


def _rule_to_concept(rule: str) -> str:
    """
    Map a grammar rule name to a mastery concept key.

    Example:
        >>> _rule_to_concept("Past Simple vs Present Perfect")
        'past_simple'
        >>> _rule_to_concept("Article usage")
        'articles'

    Args:
        rule: Grammar rule name from the agent.

    Returns:
        Concept key matching the mastery dict keys.
    """
    rule_lower = rule.lower()
    mapping = {
        "past_simple": "past_simple",
        "past simple": "past_simple",
        "present_perfect": "present_perfect",
        "present perfect": "present_perfect",
        "article": "articles",
        "articles": "articles",
        "preposition": "prepositions",
        "prepositions": "prepositions",
        "subject_verb": "subject_verb_agreement",
        "subject-verb": "subject_verb_agreement",
        "subject verb": "subject_verb_agreement",
        "irregular_verb": "irregular_verbs",
        "irregular verb": "irregular_verbs",
        "irregular_verbs": "irregular_verbs",
        "vocabulary": "vocabulary",
        "vocab": "vocabulary",
    }
    for key, concept in mapping.items():
        if key in rule_lower:
            return concept
    # Default: try to match directly
    for concept in ["past_simple", "present_perfect", "articles",
                     "prepositions", "subject_verb_agreement",
                     "irregular_verbs", "vocabulary"]:
        if concept in rule_lower:
            return concept
    return "vocabulary"
