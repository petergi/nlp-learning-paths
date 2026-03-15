from nlp_learning.preprocessing import clean_text, remove_stopwords, tokenize


def test_clean_text_normalizes_text() -> None:
    assert clean_text("Hello, NLP!!!") == "hello nlp"


def test_tokenize_returns_tokens() -> None:
    tokens = tokenize("Tokenize this sentence.")
    assert "tokenize" in tokens
    assert "sentence" in tokens


def test_remove_stopwords_filters_common_words() -> None:
    result = remove_stopwords(["this", "is", "a", "test"])
    assert "is" not in result
    assert "a" not in result
    assert "test" in result
