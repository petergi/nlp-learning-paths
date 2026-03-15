"""Sentiment analysis comparison: VADER, TextBlob, and sklearn."""

from __future__ import annotations

import csv
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sentiment_reviews.csv"


def _load_training_data(path: Path) -> tuple[list[str], list[int]]:
    """Load labeled reviews from a CSV with ``text`` and ``label`` columns.

    Skips rows with missing or blank ``text`` fields.  Raises
    ``ValueError`` on rows where ``label`` is not an integer.
    """
    texts: list[str] = []
    labels: list[int] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "text" not in reader.fieldnames:
            raise ValueError(f"{path}: missing required 'text' column")
        if "label" not in reader.fieldnames:
            raise ValueError(f"{path}: missing required 'label' column")
        for lineno, row in enumerate(reader, start=2):
            text = (row.get("text") or "").strip()
            if not text:
                continue
            try:
                label = int(row["label"])
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    f"{path} line {lineno}: invalid label {row.get('label')!r}"
                ) from exc
            texts.append(text)
            labels.append(label)
    if not texts:
        raise ValueError(f"{path}: no valid training rows found")
    return texts, labels


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

    try:
        train_texts, train_labels = _load_training_data(DATA_PATH)
    except OSError:
        print(f"Training data not found: {DATA_PATH}")
        print("Make sure you're running from the repo root.")
        return
    except ValueError as exc:
        print(f"Bad training data: {exc}")
        return

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
