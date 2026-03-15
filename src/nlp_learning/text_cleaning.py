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

    Raises:
        LookupError: If NLTK is installed but the WordNet corpus data is
            missing.  The error message includes download instructions.
    """
    if WordNetLemmatizer is None:
        return list(words)
    lemmatizer = WordNetLemmatizer()
    try:
        return [lemmatizer.lemmatize(w) for w in words]
    except LookupError as e:
        raise LookupError(
            "WordNet data not found. Download it with: "
            "python -m nltk.downloader wordnet omw-1.4"
        ) from e


def extract_patterns(text: str, pattern: str) -> List[str]:
    """Return all non-overlapping full matches of *pattern* in *text*.

    Uses ``re.finditer`` so the return type is always ``List[str]``
    (full match strings), regardless of whether *pattern* contains
    capturing groups.
    """
    compiled = re.compile(pattern)
    return [m.group(0) for m in compiled.finditer(text)]
