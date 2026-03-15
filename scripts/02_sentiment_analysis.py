import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


def main() -> None:
    samples = pd.DataFrame(
        {
            "text": [
                "I love this product",
                "This is fantastic",
                "Absolutely terrible experience",
                "I hate it",
                "Very happy with the results",
                "Worst purchase ever",
            ],
            "label": [1, 1, 0, 0, 1, 0],
        }
    )

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(samples["text"])
    y = samples["label"]

    model = LogisticRegression(max_iter=300)
    model.fit(X, y)

    test_phrases = ["I love it", "This is the worst"]
    X_test = vectorizer.transform(test_phrases)
    preds = model.predict(X_test)

    for phrase, pred in zip(test_phrases, preds):
        sentiment = "positive" if pred == 1 else "negative"
        print(f"{phrase!r} -> {sentiment}")


if __name__ == "__main__":
    main()
