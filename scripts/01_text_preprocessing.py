from nlp_learning.preprocessing import clean_text, remove_stopwords, tokenize


def main() -> None:
    sample = "NLP is amazing! It helps computers understand human language."

    cleaned = clean_text(sample)
    tokens = tokenize(sample)
    tokens_no_stopwords = remove_stopwords(tokens)

    print("Original:", sample)
    print("Cleaned:", cleaned)
    print("Tokens:", tokens)
    print("Without stopwords:", tokens_no_stopwords)


if __name__ == "__main__":
    main()
