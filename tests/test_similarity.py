import pytest

from nlp_learning.similarity import find_best_match


def test_find_best_match_returns_correct_index() -> None:
    candidates = [
        "I love playing football",
        "Python is a programming language",
        "The weather is sunny today",
    ]
    idx, score = find_best_match("Tell me about Python programming", candidates)
    assert idx == 1


def test_find_best_match_score_is_positive() -> None:
    candidates = ["hello world", "goodbye moon"]
    idx, score = find_best_match("hello", candidates)
    assert idx == 0
    assert score > 0.0


def test_find_best_match_raises_on_empty_candidates() -> None:
    with pytest.raises(ValueError, match="candidates must not be empty"):
        find_best_match("hello", [])


def test_find_best_match_raises_on_whitespace_only_query() -> None:
    with pytest.raises(ValueError, match="query must not be empty or whitespace"):
        find_best_match("   ", ["hello world"])


def test_find_best_match_raises_on_whitespace_only_candidates() -> None:
    with pytest.raises(ValueError, match="at least one non-empty string"):
        find_best_match("hello", ["", "   ", "\t"])


def test_find_best_match_raises_on_empty_vocabulary() -> None:
    # Single-character tokens are stripped by TfidfVectorizer's default token pattern,
    # producing an empty vocabulary even though the strings are non-whitespace.
    with pytest.raises(ValueError, match="empty vocabulary"):
        find_best_match("x", ["a", "b", "c"])
