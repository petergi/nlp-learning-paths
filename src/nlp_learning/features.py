"""Feature-extraction utilities: Bag-of-Words and TF-IDF matrices."""

from typing import List, Tuple

from scipy.sparse import spmatrix

try:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
except ImportError:  # pragma: no cover
    CountVectorizer = None
    TfidfVectorizer = None


def build_bow_matrix(corpus: List[str]) -> Tuple[spmatrix, List[str]]:
    """Build a Bag-of-Words matrix from a list of documents.

    Returns:
        (matrix, feature_names) where *matrix* is a sparse document-term
        matrix and *feature_names* is a list of vocabulary terms.

    Raises:
        RuntimeError: If scikit-learn is not installed.
        ValueError: If *corpus* is empty or contains only whitespace documents.
    """
    if CountVectorizer is None:
        raise RuntimeError("scikit-learn is required for build_bow_matrix")
    if not corpus or not any(doc.strip() for doc in corpus):
        raise ValueError("corpus must contain at least one non-empty document")
    vectorizer = CountVectorizer()
    try:
        matrix = vectorizer.fit_transform(corpus)
    except ValueError as e:
        raise ValueError(
            "corpus produced an empty vocabulary — documents may contain "
            "only stop-words or single-character tokens"
        ) from e
    return matrix, vectorizer.get_feature_names_out().tolist()


def build_tfidf_matrix(corpus: List[str]) -> Tuple[spmatrix, List[str]]:
    """Build a TF-IDF matrix from a list of documents.

    Returns:
        (matrix, feature_names) where *matrix* is a sparse TF-IDF matrix
        and *feature_names* is a list of vocabulary terms.

    Raises:
        RuntimeError: If scikit-learn is not installed.
        ValueError: If *corpus* is empty or produces an empty vocabulary.
    """
    if TfidfVectorizer is None:
        raise RuntimeError("scikit-learn is required for build_tfidf_matrix")
    if not corpus or not any(doc.strip() for doc in corpus):
        raise ValueError("corpus must contain at least one non-empty document")
    vectorizer = TfidfVectorizer()
    try:
        matrix = vectorizer.fit_transform(corpus)
    except ValueError as e:
        raise ValueError(
            "corpus produced an empty vocabulary — documents may contain "
            "only stop-words or single-character tokens"
        ) from e
    return matrix, vectorizer.get_feature_names_out().tolist()
