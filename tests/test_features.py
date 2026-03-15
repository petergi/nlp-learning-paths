import pytest

from nlp_learning.features import build_bow_matrix, build_tfidf_matrix


def test_bow_matrix_shape_and_features() -> None:
    corpus = ["the cat sat", "the dog sat", "the cat lay"]
    matrix, features = build_bow_matrix(corpus)
    assert matrix.shape[0] == 3
    assert "cat" in features
    assert "dog" in features


def test_tfidf_matrix_shape_and_features() -> None:
    corpus = ["the cat sat", "the dog sat", "the cat lay"]
    matrix, features = build_tfidf_matrix(corpus)
    assert matrix.shape[0] == 3
    assert "cat" in features


def test_tfidf_values_are_normalized() -> None:
    corpus = ["hello world", "hello there"]
    matrix, _ = build_tfidf_matrix(corpus)
    # Each row in a TF-IDF matrix (default L2 norm) should have norm ~1.0
    import numpy as np

    for i in range(matrix.shape[0]):
        row_norm = np.sqrt(matrix[i].multiply(matrix[i]).sum())
        assert abs(row_norm - 1.0) < 1e-6


# --- BoW error-condition tests ---


def test_bow_raises_on_empty_corpus() -> None:
    with pytest.raises(ValueError, match="at least one non-empty document"):
        build_bow_matrix([])


def test_bow_raises_on_whitespace_only_corpus() -> None:
    with pytest.raises(ValueError, match="at least one non-empty document"):
        build_bow_matrix(["", "   ", "\t"])


def test_bow_raises_on_empty_vocabulary() -> None:
    # Single-character tokens are stripped by CountVectorizer's default token pattern.
    with pytest.raises(ValueError, match="empty vocabulary"):
        build_bow_matrix(["a", "b", "c"])


# --- TF-IDF error-condition tests ---


def test_tfidf_raises_on_empty_corpus() -> None:
    with pytest.raises(ValueError, match="at least one non-empty document"):
        build_tfidf_matrix([])


def test_tfidf_raises_on_whitespace_only_corpus() -> None:
    with pytest.raises(ValueError, match="at least one non-empty document"):
        build_tfidf_matrix(["", "   "])


def test_tfidf_raises_on_empty_vocabulary() -> None:
    with pytest.raises(ValueError, match="empty vocabulary"):
        build_tfidf_matrix(["a", "b", "c"])
