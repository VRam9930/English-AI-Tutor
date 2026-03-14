"""
core/chunker.py

Text chunking and metadata extraction for Linguist-OS.
This is STEP 1 of the pipeline: it breaks user text into chunks and
extracts linguistic metadata used by downstream agents (especially
the Confidence Coach).

Architecture role:
  - First module to process user input
  - Extracts word_count, hesitation_words, sentence_count,
    avg_word_length, question_marks
  - Hesitation detection powers the Confidence Coach's fluency analysis
"""

import re

HESITATION_WORDS = [
    "um", "uh", "er", "hmm", "like", "you know",
    "i mean", "sort of", "kind of", "basically", "literally",
]


def chunk_text(text: str) -> dict:
    """
    Break user text into chunks and extract metadata for the pipeline.

    Example:
        >>> chunk_text("Um, I have went to the park yesterday. Like, it was nice?")
        {
            'original': 'Um, I have went to the park yesterday. Like, it was nice?',
            'chunks': ['Um, I have went to the park yesterday.', 'Like, it was nice?'],
            'metadata': {
                'word_count': 13,
                'sentence_count': 2,
                'avg_word_length': 3.46,
                'question_marks': 1,
                'hesitation_count': 2,
                'hesitation_words_found': ['um', 'like'],
                'words_per_sentence': 6.5
            }
        }

    Args:
        text: Raw user input text.

    Returns:
        Dict with original text, chunks list, and metadata dict.
    """
    print(f"[chunker] Processing text: '{text[:60]}...'")

    # Clean whitespace
    cleaned = text.strip()
    if not cleaned:
        return {
            "original": text,
            "chunks": [],
            "metadata": _empty_metadata(),
        }

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', cleaned)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        sentences = [cleaned]

    # Word-level analysis
    words = cleaned.split()
    word_count = len(words)

    # Average word length (exclude punctuation)
    clean_words = [re.sub(r'[^\w]', '', w) for w in words]
    clean_words = [w for w in clean_words if w]
    avg_word_length = round(
        sum(len(w) for w in clean_words) / max(len(clean_words), 1), 2
    )

    # Question marks
    question_marks = cleaned.count("?")

    # Hesitation detection
    lower_text = cleaned.lower()
    hesitation_found = []
    for h in HESITATION_WORDS:
        pattern = r'\b' + re.escape(h) + r'\b'
        if re.search(pattern, lower_text):
            hesitation_found.append(h)

    hesitation_count = len(hesitation_found)

    # Words per sentence
    words_per_sentence = round(word_count / max(len(sentences), 1), 2)

    metadata = {
        "word_count": word_count,
        "sentence_count": len(sentences),
        "avg_word_length": avg_word_length,
        "question_marks": question_marks,
        "hesitation_count": hesitation_count,
        "hesitation_words_found": hesitation_found,
        "words_per_sentence": words_per_sentence,
    }

    print(f"[chunker] Extracted metadata: {word_count} words, "
          f"{len(sentences)} sentences, {hesitation_count} hesitations")

    return {
        "original": cleaned,
        "chunks": sentences,
        "metadata": metadata,
    }


def _empty_metadata() -> dict:
    """
    Return a zeroed-out metadata dict for empty input.

    Example:
        >>> _empty_metadata()
        {'word_count': 0, 'sentence_count': 0, ...}

    Returns:
        Dict with all metadata fields set to zero/empty.
    """
    return {
        "word_count": 0,
        "sentence_count": 0,
        "avg_word_length": 0.0,
        "question_marks": 0,
        "hesitation_count": 0,
        "hesitation_words_found": [],
        "words_per_sentence": 0.0,
    }
