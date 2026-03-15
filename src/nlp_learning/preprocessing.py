import re
from typing import Iterable, List

try:
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
except ImportError:  # pragma: no cover
    stopwords = None
    word_tokenize = None


def clean_text(text: str) -> str:
    """Lowercase text, remove punctuation, and collapse extra spaces."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> List[str]:
    """Tokenize text with NLTK when available; fallback to split."""
    cleaned = clean_text(text)
    if word_tokenize is not None:
        try:
            return [token for token in word_tokenize(cleaned) if token.strip()]
        except LookupError:
            pass
    return cleaned.split()


def remove_stopwords(tokens: Iterable[str]) -> List[str]:
    """Remove common English stopwords, with a minimal fallback list."""
    token_list = [t.lower() for t in tokens]

    if stopwords is not None:
        try:
            stop_words = set(stopwords.words("english"))
        except LookupError:
            stop_words = {"the", "is", "a", "an", "and", "or", "to", "of", "in", "on"}
    else:
        stop_words = {"the", "is", "a", "an", "and", "or", "to", "of", "in", "on"}

    return [token for token in token_list if token not in stop_words]
