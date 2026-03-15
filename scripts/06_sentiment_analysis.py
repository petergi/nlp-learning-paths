"""Sentiment analysis comparison: VADER, TextBlob, and sklearn."""

from __future__ import annotations


def main() -> None:
    test_sentences: list[str] = [
        "This movie was absolutely wonderful and I loved every minute!",
        "The food was terrible, I will never go back to that restaurant.",
        "The weather today is okay, nothing special.",
        "I am so excited about the upcoming concert, it will be amazing!",
        "The product broke after one day, worst purchase ever.",
    ]

    # --- VADER ---
    try:
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        nltk.download("vader_lexicon", quiet=True)
        vader = SentimentIntensityAnalyzer()
        vader_scores: list[float] = [
            vader.polarity_scores(s)["compound"] for s in test_sentences
        ]
    except ImportError:
        print("NLTK not installed. Run: pip install nltk")
        return

    # --- TextBlob ---
    try:
        from textblob import TextBlob

        blob_scores: list[float] = [
            round(TextBlob(s).sentiment.polarity, 4) for s in test_sentences
        ]
    except ImportError:
        print("TextBlob not installed. Run: pip install textblob")
        return

    # --- sklearn Logistic Regression ---
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        print("scikit-learn not installed. Run: pip install scikit-learn")
        return

    train_texts: list[str] = [
        "I love this, it is great",
        "Fantastic experience, highly recommend",
        "Best thing ever, so happy",
        "Really enjoyed it, wonderful time",
        "This is awful and terrible",
        "Horrible experience, very disappointing",
        "Worst thing I have ever seen",
        "I hate this, completely useless",
        "It was fine, nothing remarkable",
        "Average at best, not impressed",
    ]
    train_labels: list[int] = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  # 1=pos, 0=neg

    vectorizer = TfidfVectorizer()
    x_train = vectorizer.fit_transform(train_texts)
    x_test = vectorizer.transform(test_sentences)

    clf = LogisticRegression(max_iter=200)
    clf.fit(x_train, train_labels)
    sklearn_labels: list[str] = [
        "pos" if p == 1 else "neg" for p in clf.predict(x_test)
    ]

    # --- Comparison Table ---
    print("=" * 80)
    print("SENTIMENT ANALYSIS COMPARISON")
    print("=" * 80)
    print(
        f"\n{'#':<4} {'VADER':<10} {'TextBlob':<10} {'sklearn':<10} {'Sentence'}"
    )
    print("-" * 80)
    for i, sentence in enumerate(test_sentences):
        short: str = sentence[:40] + "..." if len(sentence) > 40 else sentence
        print(
            f"{i+1:<4} {vader_scores[i]:<10.4f} {blob_scores[i]:<10.4f} "
            f"{sklearn_labels[i]:<10} {short}"
        )

    print("\nVADER/TextBlob: score > 0 = positive, < 0 = negative")
    print("sklearn: trained on small labeled set (pos/neg classification)")


if __name__ == "__main__":
    main()
