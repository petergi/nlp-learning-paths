from nlp_learning.text_cleaning import extract_patterns, lemmatize_words, stem_words


def test_stem_words_reduces_to_stems() -> None:
    result = stem_words(["running", "flies", "happily"])
    assert "run" in result
    assert "fli" in result  # Porter stemmer output for "flies"


def test_lemmatize_words_produces_base_forms() -> None:
    result = lemmatize_words(["cats", "running", "better"])
    assert "cat" in result


def test_extract_patterns_finds_emails() -> None:
    text = "Contact us at hello@example.com or support@test.org"
    emails = extract_patterns(text, r"[\w.+-]+@[\w-]+\.[\w.]+")
    assert len(emails) == 2
    assert "hello@example.com" in emails


def test_extract_patterns_returns_empty_on_no_match() -> None:
    assert extract_patterns("no numbers here", r"\d+") == []
