# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

import re
from typing import List, Dict, Tuple, Optional

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS

# ASCII emoji sequences mapped to token strings for scoring
ASCII_EMOJI_MAP = {
    ":)": "emoji_smile",
    ":-)": "emoji_smile",
    ":D": "emoji_grin",
    ":-D": "emoji_grin",
    ";)": "emoji_wink",
    ";-)": "emoji_wink",
    ":P": "emoji_tongue",
    ":-P": "emoji_tongue",
    ":(": "emoji_sad",
    ":-(": "emoji_sad",
    ":'(": "emoji_cry",
    ":O": "emoji_surprise",
    ":-O": "emoji_surprise",
    ":/": "emoji_skeptical",
    ":-/": "emoji_skeptical",
    ":|": "emoji_neutral",
    ":-|": "emoji_neutral",
}

# Unicode emoji characters mapped to token strings for scoring
UNICODE_EMOJI_MAP = {
    "😂": "emoji_joy",
    "🥲": "emoji_bittersweet",
    "💀": "emoji_skull",
    "❤️": "emoji_heart",
    "❤": "emoji_heart",
    "😊": "emoji_smile",
    "😢": "emoji_sad",
    "😭": "emoji_cry",
    "😡": "emoji_angry",
    "🤩": "emoji_excited",
    "😴": "emoji_tired",
    "🙄": "emoji_eye_roll",
}

# Build a compiled regex for ASCII emoji matching. Longest keys first to avoid
# partial matches (e.g., ":-)" tried before ":)").
_ASCII_EMOJI_PATTERN = re.compile(
    "|".join(re.escape(k) for k in sorted(ASCII_EMOJI_MAP, key=len, reverse=True))
)


def _replace_ascii_emojis(text: str) -> str:
    """Replace ASCII emoji sequences with their token equivalents, padded with spaces."""
    return _ASCII_EMOJI_PATTERN.sub(
        lambda m: " " + ASCII_EMOJI_MAP[m.group()] + " ", text
    )


def _replace_unicode_emojis(text: str) -> str:
    """Replace Unicode emoji characters with their token equivalents, padded with spaces."""
    for emoji_char, token in sorted(
        UNICODE_EMOJI_MAP.items(), key=lambda x: len(x[0]), reverse=True
    ):
        text = text.replace(emoji_char, " " + token + " ")
    return text


def _normalize_repeated_chars(text: str) -> str:
    """Collapse runs of 3+ identical characters down to 2 (e.g., 'soooo' -> 'soo')."""
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


def _strip_punctuation(token: str) -> str:
    """Remove all punctuation from a token (leading, trailing, and internal)."""
    # Remove all punctuation characters, keeping only word characters and spaces
    return re.sub(r"[^\w]", "", token)


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of clean, normalized tokens.

        Pipeline (in order):
          1. Strip leading/trailing whitespace
          2. Replace ASCII emojis (":)", ":(", etc.) with token strings
          3. Replace Unicode emojis ("😂", "💀", etc.) with token strings
          4. Lowercase everything
          5. Normalize repeated characters ("soooo" -> "soo")
          6. Split on whitespace
          7. Strip punctuation from each token
          8. Remove empty tokens

        This ensures that emojis and punctuation-attached words are handled
        before they can interfere with tokenization.
        """
        # Step 1: Strip leading/trailing whitespace
        cleaned = text.strip()

        # Step 2: Replace ASCII emojis BEFORE lowercasing (":D" != ":d")
        cleaned = _replace_ascii_emojis(cleaned)

        # Step 3: Replace Unicode emojis
        cleaned = _replace_unicode_emojis(cleaned)

        # Step 4: Lowercase everything
        cleaned = cleaned.lower()

        # Step 5: Normalize repeated characters ("soooo" -> "soo")
        cleaned = _normalize_repeated_chars(cleaned)

        # Step 6 + 7: Split on whitespace, then strip punctuation from each token
        tokens = [_strip_punctuation(t) for t in cleaned.split()]

        # Step 8: Remove empty tokens (can occur if a token was only punctuation)
        tokens = [t for t in tokens if t]

        return tokens

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def score_text(self, text: str) -> int:
        """
        Compute a numeric "mood score" for the given text.

        Positive words increase the score.
        Negative words decrease the score.

        Handles negation: "not happy" or "never fun" flips sentiment.
        """
        tokens = self.preprocess(text)
        score = 0
        negation_words = {"not", "never", "no", "dont", "cannot", "can't"}

        for i, token in enumerate(tokens):
            is_negated = i > 0 and tokens[i - 1] in negation_words

            if token in self.positive_words:
                score += -1 if is_negated else 1
            elif token in self.negative_words:
                score += 1 if is_negated else -1

        return score

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the numeric score for a piece of text into a mood label.

        Mapping:
          - score > 0  -> "positive"
          - score < 0  -> "negative"
          - score == 0 -> "neutral"
        """
        score = self.score_text(text)
        if score > 0:
            return "positive"
        elif score < 0:
            return "negative"
        else:
            return "neutral"

    # ---------------------------------------------------------------------
    # Explanations (optional but recommended)
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label.

        TODO:
          - Look at the tokens and identify which ones counted as positive
            and which ones counted as negative.
          - Show the final score.
          - Return a short human readable explanation.

        Example explanation (your exact wording can be different):
          'Score = 2 (positive words: ["love", "great"]; negative words: [])'

        The current implementation is a placeholder so the code runs even
        before you implement it.
        """
        tokens = self.preprocess(text)

        positive_hits: List[str] = []
        negative_hits: List[str] = []
        score = 0

        for token in tokens:
            if token in self.positive_words:
                positive_hits.append(token)
                score += 1
            if token in self.negative_words:
                negative_hits.append(token)
                score -= 1

        return (
            f"Score = {score} "
            f"(positive: {positive_hits or '[]'}, "
            f"negative: {negative_hits or '[]'})"
        )
