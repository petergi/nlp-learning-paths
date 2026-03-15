"""Text-cleaning utilities: stemming, lemmatization, and regex extraction."""

import re
from typing import List

try:
    from nltk.stem import PorterStemmer
except ImportError:  # pragma: no cover
    PorterStemmer = None

try:
    from nltk.stem import WordNetLemmatizer
except ImportError:  # pragma: no cover
    WordNetLemmatizer = None


def stem_words(words: List[str]) -> List[str]:
    """Apply Porter stemming to a list of words.

    Falls back to returning the words unchanged if NLTK is unavailable.
    """
    if PorterStemmer is None:
        return list(words)
    stemmer = PorterStemmer()
    return [stemmer.stem(w) for w in words]


def lemmatize_words(words: List[str]) -> List[str]:
    """Lemmatize words using the WordNet lemmatizer.

    Falls back to returning the words unchanged if NLTK is unavailable.
    """
    if WordNetLemmatizer is None:
        return list(words)
    lemmatizer = WordNetLemmatizer()
    try:
        return [lemmatizer.lemmatize(w) for w in words]
    except LookupError:
        raise LookupError(
            "WordNet data not found. Download it with: "
            "python -m nltk.downloader wordnet omw-1.4"
        )


def extract_patterns(text: str, pattern: str) -> List[str]:
    """Return all non-overlapping matches of *pattern* in *text*.

    This is a thin wrapper around ``re.findall`` that compiles the pattern
    first so callers get a clear error on invalid regex.
    """
    compiled = re.compile(pattern)
    return compiled.findall(text)
