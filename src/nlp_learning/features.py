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
    matrix = vectorizer.fit_transform(corpus)
    return matrix, vectorizer.get_feature_names_out().tolist()


def build_tfidf_matrix(corpus: List[str]) -> Tuple[spmatrix, List[str]]:
    """Build a TF-IDF matrix from a list of documents.

    Returns:
        (matrix, feature_names) where *matrix* is a sparse TF-IDF matrix
        and *feature_names* is a list of vocabulary terms.

    Raises:
        RuntimeError: If scikit-learn is not installed.
        ValueError: If *corpus* is empty or contains only whitespace documents.
    """
    if TfidfVectorizer is None:
        raise RuntimeError("scikit-learn is required for build_tfidf_matrix")
    if not corpus or not any(doc.strip() for doc in corpus):
        raise ValueError("corpus must contain at least one non-empty document")
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(corpus)
    return matrix, vectorizer.get_feature_names_out().tolist()
