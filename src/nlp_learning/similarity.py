"""Similarity utilities: TF-IDF cosine similarity for document matching."""

from typing import List, Tuple

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:  # pragma: no cover
    TfidfVectorizer = None
    cosine_similarity = None


def find_best_match(query: str, candidates: List[str]) -> Tuple[int, float]:
    """Find the candidate most similar to *query* using TF-IDF + cosine similarity.

    Args:
        query: The input text to match against.
        candidates: A list of candidate strings to compare with.

    Returns:
        (best_index, score) — the index of the best-matching candidate and
        the cosine-similarity score (0.0 to 1.0).

    Raises:
        RuntimeError: If scikit-learn is not installed.
        ValueError: If *candidates* is empty.
    """
    if TfidfVectorizer is None or cosine_similarity is None:
        raise RuntimeError("scikit-learn is required for find_best_match")
    if not candidates:
        raise ValueError("candidates must not be empty")

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(candidates + [query])

    # The query vector is the last row; candidates are all preceding rows.
    query_vec = tfidf_matrix[-1]
    candidate_vecs = tfidf_matrix[:-1]

    scores = cosine_similarity(query_vec, candidate_vecs).flatten()
    best_index = int(scores.argmax())
    return best_index, float(scores[best_index])
