"""Feature-extraction utilities: Bag-of-Words and TF-IDF matrices."""

from typing import List, Tuple

try:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
except ImportError:  # pragma: no cover
    CountVectorizer = None
    TfidfVectorizer = None


def build_bow_matrix(corpus: List[str]) -> Tuple:
    """Build a Bag-of-Words matrix from a list of documents.

    Returns:
        (matrix, feature_names) where *matrix* is a sparse document-term
        matrix and *feature_names* is a list of vocabulary terms.

    Raises:
        RuntimeError: If scikit-learn is not installed.
    """
    if CountVectorizer is None:
        raise RuntimeError("scikit-learn is required for build_bow_matrix")
    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform(corpus)
    return matrix, vectorizer.get_feature_names_out().tolist()


def build_tfidf_matrix(corpus: List[str]) -> Tuple:
    """Build a TF-IDF matrix from a list of documents.

    Returns:
        (matrix, feature_names) where *matrix* is a sparse TF-IDF matrix
        and *feature_names* is a list of vocabulary terms.

    Raises:
        RuntimeError: If scikit-learn is not installed.
    """
    if TfidfVectorizer is None:
        raise RuntimeError("scikit-learn is required for build_tfidf_matrix")
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(corpus)
    return matrix, vectorizer.get_feature_names_out().tolist()
